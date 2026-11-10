"""度量：把一个量化后的模型变成一条**四维读数**。

## 四维里只有一维是「测出来的」，另外三维是「算出来的」

| 维 | 来源 | 口径 |
| --- | --- | --- |
| Quality | **真的在测试集上跑一遍** | 准确率（512 条，与校准集**逐行不重叠**） |
| Latency | 成本模型 | `固定开销 + 访存时间 × 反量化惩罚 / kernel 效率` |
| Memory | 成本模型 | `体积 + 激活/KV + 运行时固定开销` |
| Size | 成本模型 | `部署参数量 × 有效位宽 / 8` |

**这句话必须印在报告里。** 把成本模型当成实测，是量化类报告最常见的一种谎
（而且它不来自造假，来自「表格长得像实测」）。成本模型的每一个系数都在
`configs/default.json` 的 `runtime` / `deployment` 段里，改系数不改代码。

## 两个尺度必须显式映射

本地模型只有 **8,840** 个参数（30 秒内跑完整个设计空间），而部署代价按
`deployment.base_params_b = 7.0`（七十亿）算：

```
size_gb = base_params_b × effective_bits / 8
```

也就是「7B 模型在 `effective_bits` 位宽下的权重体积」。**质量的排序能迁移，
绝对数值不能** —— 本地 0.74 → 0.85 的那段差距，在真模型上不是同一个数。

⚠️ 注意这里用的是 **`effective_bits`（含元数据）而不是 `nominal_bits`**。
同一个「INT4」在 `rtn` 下是 5.0 有效位、在 `kquant` 下是 5.0882 ——
用名义位宽算体积，等于假装元数据不占空间。

## 第四维与第二维是共线的（本项目要证明的结论之一）

```
memory = size + (activation_mb + fixed_memory_mb) / 1024
       = size + 0.87890625          ← 常数，与量化方案无关
```

于是 `corr(memory, size) = 1.000000`，**四维实际是三维**。
保留四维是验收标准的要求；证明它冗余是本项目的诚实结论
（`expectations.memory_is_an_independent_axis` 预期被推翻）。
`acceptance.max_memory_axis_frontier_delta = 0` 把这件事变成一条可断言的不变量：
**把 Memory 维换成 Size 维，前沿逐点不变。**

## 反量化惩罚为什么挂在分组上

分组越小 ⇒ 块数越多 ⇒ 每个权重都要多读一次 scale/zero ⇒ 访存之外的额外开销。
惩罚写成「块数相对参考分组翻倍多少次，就乘多少次 `(1+p)`」：

```
penalty = (1 + 0.06) ** log2(n_blocks / n_blocks_ref)
```

`g16 / g32 / g64` 相对 `base_group_for_dequant=128` 分别是 `8× / 4× / 2×`，
于是惩罚是 `1.191 / 1.124 / 1.060`。**group 这一个旋钮同时动体积与延迟**，
而且动延迟的幅度远大于动体积 —— 这是「小分组不一定划算」的来源。
"""

from __future__ import annotations

import math
import time
from typing import Any, Mapping

from .model import Model, accuracy
from .quantize import QuantResult
from .schema import QuantSpec, Reading
from .world import World

__all__ = [
    "FP16",
    "deployment_scale",
    "size_gb_of",
    "memory_overhead_gb",
    "memory_gb_of",
    "dequant_penalty",
    "latency_ms_of",
    "within_budget",
    "measure_quantized",
    "measure_fp16",
]

# 未量化基线的设计点。`group=0` ⇒ 没有块 ⇒ `n_blocks=0` ⇒ 反量化惩罚恒为 1。
FP16 = QuantSpec(algorithm="fp16", precision="fp16", group=0, calib=0)


# --------------------------------------------------------------------------- #
# 成本模型
# --------------------------------------------------------------------------- #


def deployment_scale(cfg: Mapping[str, Any], n_params_local: int) -> float:
    """本地参数 → 部署参数的放大倍数。

    单独留一个函数是为了让「两个尺度」这件事在代码里**看得见**：
    任何把本地读数当部署读数的写法，都必须先经过这里。
    """
    if n_params_local <= 0:
        raise ValueError(f"本地参数量必须为正，得到 {n_params_local}")
    return float(cfg["deployment"]["base_params_b"]) * 1e9 / n_params_local


def size_gb_of(cfg: Mapping[str, Any], effective_bits: float, n_params_local: int) -> float:
    """权重体积（GB）。**用有效位宽，不用名义位宽。**"""
    scale = deployment_scale(cfg, n_params_local)
    return n_params_local * scale * float(effective_bits) / 8.0 / 1e9


def memory_overhead_gb(cfg: Mapping[str, Any]) -> float:
    """`memory − size` 的那个常数。**它必须与量化方案无关，这是共线的根源。**

    单独暴露出来，是因为「Memory 是独立一维」这个期望的推翻，
    靠的就是「这个常数对所有候选都相同」这一条。
    """
    r = cfg["runtime"]
    return (float(r["activation_mb"]) + float(r["fixed_memory_mb"])) / 1024.0


def memory_gb_of(cfg: Mapping[str, Any], size_gb: float) -> float:
    """峰值显存（GB）= 权重体积 + 常数开销。"""
    return float(size_gb) + memory_overhead_gb(cfg)


def dequant_penalty(cfg: Mapping[str, Any], n_blocks: int, n_blocks_ref: int) -> float:
    """反量化惩罚。`n_blocks <= 0`（未量化）时恒为 1 —— 没有块就没有反量化。"""
    r = cfg["runtime"]
    p = float(r["dequant_penalty_per_block"])
    if n_blocks <= 0 or n_blocks_ref <= 0:
        return 1.0
    return (1.0 + p) ** math.log2(n_blocks / n_blocks_ref)


def latency_ms_of(
    cfg: Mapping[str, Any],
    size_gb: float,
    n_blocks: int,
    n_blocks_ref: int,
    algorithm: str,
) -> float:
    """单 token 延迟（ms）。**访存主导**，再乘反量化惩罚、除 kernel 效率。"""
    r = cfg["runtime"]
    eff = float(r["kernel_efficiency"][algorithm])
    if eff <= 0.0:
        raise ValueError(f"kernel_efficiency[{algorithm!r}] 必须为正，得到 {eff}")
    access_ms = float(size_gb) * 1e9 / (float(r["bandwidth_gbps"]) * 1e9) * 1000.0
    return float(r["fixed_latency_ms"]) + access_ms * dequant_penalty(
        cfg, n_blocks, n_blocks_ref
    ) / eff


def within_budget(cfg: Mapping[str, Any], memory_gb: float, latency_ms: float) -> bool:
    """部署约束。**只用来做选型，不用来过滤候选** —— 被约束挡掉的候选也要进矩阵，
    否则「这张表里没有」与「这个方案不可行」就分不清了。"""
    d = cfg["deployment"]
    return (
        memory_gb <= float(d["memory_limit_gb"]) + 1e-12
        and latency_ms <= float(d["latency_limit_ms"]) + 1e-12
    )


# --------------------------------------------------------------------------- #
# 读数
# --------------------------------------------------------------------------- #


def _timed_accuracy(model: Model, X: list[list[float]], Y: list[int]) -> tuple[float, float]:
    t0 = time.perf_counter()
    acc = accuracy(model, X, Y)
    return acc, time.perf_counter() - t0


def measure_quantized(
    world: World,
    cfg: Mapping[str, Any],
    spec: QuantSpec,
    qres: QuantResult,
    *,
    baseline_quality: float,
    n_blocks_ref: int,
    quantize_s: float = 0.0,
) -> Reading:
    """量化候选 → 四维读数。

    `baseline_quality` 是 fp16 基座在**同一个测试集**上的准确率 ——
    传进来而不是自己算，是为了保证「全部候选共用同一个基线」这件事
    在代码里是显然的（自己算的话，某天有人在候选循环里换了测试集，
    质量损失这一列就会静默变成两个不同基线的差）。

    `quantize_s` 是**量化那一步**的墙钟时间（调用方测），与这里测的
    测试集前向时间相加，合成 `candidate_s`。**它不进产物** —— 见 `schema.Reading`。
    """
    quality, secs = _timed_accuracy(qres.model, world.test_x, world.test_y)
    size = size_gb_of(cfg, qres.effective_bits, qres.model.n_params)
    memory = memory_gb_of(cfg, size)
    latency = latency_ms_of(cfg, size, qres.n_blocks, n_blocks_ref, spec.algorithm)

    notes = list(qres.notes)
    if not within_budget(cfg, memory, latency):
        d = cfg["deployment"]
        why = []
        if memory > float(d["memory_limit_gb"]):
            why.append(f"显存 {memory:.4f} > {d['memory_limit_gb']} GB")
        if latency > float(d["latency_limit_ms"]):
            why.append(f"延迟 {latency:.4f} > {d['latency_limit_ms']} ms")
        notes.append("超出部署预算：" + "；".join(why))

    return Reading(
        spec=spec,
        quality=quality,
        latency_ms=latency,
        memory_gb=memory,
        size_gb=size,
        nominal_bits=qres.nominal_bits,
        effective_bits=qres.effective_bits,
        metadata_bits_per_weight=qres.metadata_bits_per_weight,
        n_blocks=qres.n_blocks,
        n_params=qres.model.n_params,
        quality_loss=float(baseline_quality) - quality,
        weight_rel_error=qres.weight_rel_error,
        weight_max_abs_error=qres.weight_max_abs_error,
        layer_bits=dict(qres.layer_bits),
        candidate_s=float(quantize_s) + secs,
        notes=notes,
    )


def measure_fp16(
    world: World,
    cfg: Mapping[str, Any],
    model: Model,
    *,
    n_blocks_ref: int,
) -> Reading:
    """未量化基线。**它就是 Quality 那一维的原点。**

    `effective_bits = 16`（fp16 没有元数据、没有块、没有层混合），
    于是它的体积是全部候选里最大的那个 —— 这正是「量化换来了什么」的参照。
    """
    quality, secs = _timed_accuracy(model, world.test_x, world.test_y)
    eff = 16.0
    size = size_gb_of(cfg, eff, model.n_params)
    memory = memory_gb_of(cfg, size)
    latency = latency_ms_of(cfg, size, 0, n_blocks_ref, "fp16")

    return Reading(
        spec=FP16,
        quality=quality,
        latency_ms=latency,
        memory_gb=memory,
        size_gb=size,
        nominal_bits=eff,
        effective_bits=eff,
        metadata_bits_per_weight=0.0,
        n_blocks=0,
        n_params=model.n_params,
        quality_loss=0.0,
        weight_rel_error=0.0,
        weight_max_abs_error=0.0,
        layer_bits={role: eff for role in model.roles},
        candidate_s=secs,
        notes=[
            "未量化基线：四维里的 Quality 以它为原点，所有候选的质量损失都相对它算",
            "体积 14.0 GB（7B × 16 位）⇒ 在任何 8 GB 卡上都不可部署 —— 这就是量化的动机",
        ],
    )
