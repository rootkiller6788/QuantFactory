"""配置：口径与门槛。**代码里不硬编码任何阈值。**

## 这个项目的口径是「精度设计空间」

设计空间是四个轴的笛卡尔积：

```
algorithm  ∈ {rtn, gptq, awq, kquant}          误差补偿机制
precision  ∈ {int2 … int8}                      名义位宽
group      ∈ {16, 32, 64}                       每组多少个权重
calib      ∈ {none, 64, 256}                    校准集规模（rtn / kquant 不用）
```

`rtn` 与 `kquant` **校准无关**（`calib = "none"`），`gptq` / `awq` 需要校准集。
这不是省事，是事实：RTN 是逐组 absmax 舍入，K-quant 是超块缩放，
两者都不读激活统计；GPTQ 要 Hessian、AWQ 要逐通道激活幅度。

## 两个尺度必须显式映射

本地模型只有几千个参数（30 秒内跑完整个设计空间），而部署代价按
`deployment.base_params_b = 7.0`（七十亿参数）算。**质量的排序能迁移，
绝对数值不能** —— 这句话会印在报告的显眼位置。

## 四维目标里有一维是冗余的（本项目要证明的结论之一）

```
size    = 权重字节（含元数据）
memory  = size + 激活/KV（与量化无关）+ 运行时固定开销
        ⇒ memory = size + 常数   ⇒ **与 Size 完全共线**
```

保留四维是验收标准要求的；**证明第四维冗余**是本项目的诚实结论。
见 `expectations.memory_is_an_independent_axis`（预期被推翻）。
"""

from __future__ import annotations

import copy
import json
from typing import Any, Mapping

from .rng import digest_of
from .schema import QuantSpec

__all__ = [
    "DEFAULTS",
    "PROTOCOL_KEYS",
    "protocol_digest",
    "version_digest",
    "default_config",
    "enumerate_specs",
    "calib_levels",
    "merge_config",
    "validate_config",
    "load_config",
    "dump_config",
]

# 协议指纹覆盖的键：**改这些 = 换一个项目**（产物全部失效）。
# 不含 seed（那是「同一项目的另一次抽样」）与 acceptance / expectations（那是判据）。
PROTOCOL_KEYS: tuple[str, ...] = (
    "world",
    "train",
    "space",
    "quant",
    "runtime",
    "deployment",
    "matrix",
    "acceptance",
    "expectations",
)


DEFAULTS: dict[str, Any] = {
    # ---- 世界：一个**有结构**的合成分类任务 ----------------------------------
    # 「有结构」是必须的：如果标签是纯噪声，量化误差不会造成可分辨的质量差，
    # 整张矩阵就会退化成一片噪声。结构来自「若干隐含原型 + 每类一个原型」。
    "seed": 20261003,
    "world": {
        "n_train": 2400,          # 训练集（用来训 fp16 基座）
        "n_test": 512,            # 测试集（只用来测质量；**与校准集互不重叠**）
        "n_calib_pool": 384,      # 校准集池（最大档 256 从这里抽）
        "d_in": 64,
        "n_classes": 8,
        "hidden": [64, 64],       # 两个隐藏层；三层权重矩阵的 in 维都是 64
        "n_prototypes": 24,       # 隐含原型数（3 个/类）
        "noise": 1.10,            # 观测噪声（**按通道尺度缩放**，见下）
        # 通道尺度的谱。**这三个数是这个世界最重要的结构。**
        #
        # 尺度是**对数正态**的（sigma 控制离散度），而不是所有通道一样大。
        # 为什么必须离散：权重幅度 ∝ 1/输入尺度（要把同样大小的信号放大到同样的
        # logit 量级），所以**尺度谱越宽，权重谱就越宽** —— 而权重谱的宽度正是
        # 逐组量化的痛点（离群权重把整组的 scale 撑大，同组的小权重被压成 0）。
        # 尺度谱窄 ⇒ 组内 max/median 只有 2.7× ⇒ int3 以上完全不掉点 ⇒ 矩阵没信息。
        #
        #   narrow：尺度极小 ⇒ 读它的权重必须放得很大 ⇒ **权重离群**
        #   wide  ：尺度极大 ⇒ 激活离群 ⇒ Hessian 病态、AWQ 的缩放有的可做
        #
        # 噪声按同一尺度缩放，于是每个通道的信噪比相同 —— 否则小尺度通道会退化成
        # 纯噪声，模型直接忽略它们，权重离群就不会出现。
        "channel_scale_sigma": 0.9,
        "narrow_dims": [7, 41],
        "narrow_scale": 0.18,
        "wide_dims": [19, 53],
        "wide_scale": 3.2,
        # **信号与幅度必须解耦。** `wide_spread_ratio` 让宽通道「幅度大但不携带判别信号」，
        # 于是模型给它们的权重 ≈ 0 —— 而这一条正是 AWQ 有效的**前提**：
        # AWQ 把 `W·s` 的组最大值 `M = max_k |W_k|·act_k^α` 压下去，靠的是
        # 「激活大的通道权重小」。若宽通道反而权重最大，`M` 随 α 上升，
        # 缩放就变成纯粹放大离群值 —— 实测 α 从 0 到 1.5 单调变差（0.7715 → 0.5488）。
        "wide_spread_ratio": 0.05,
    },
    # ---- 训练：手写 SGD，只为了拿到一个「真的学到了东西」的 fp16 基座 --------
    "train": {
        "steps": 550,
        "batch": 48,
        "lr": 0.20,
        "momentum": 0.90,
        "weight_decay": 1e-3,
        "grad_clip": 4.0,
        "init_scale": 0.30,       # 初始化 U(−s, s)
    },
    # ---- 设计空间 ------------------------------------------------------------
    "space": {
        "algorithms": ["rtn", "gptq", "awq", "kquant"],
        "precisions": ["int2", "int3", "int4", "int5", "int6", "int8"],
        "groups": [16, 32, 64],
        # 每个算法各自允许的校准档位。`0` = 不需要校准（"none"）。
        "calib_sizes": {
            "rtn": [0],
            "kquant": [0],
            "gptq": [64, 256],
            "awq": [64, 256],
        },
        "include_fp16_baseline": True,
    },
    # ---- 量化算子 ------------------------------------------------------------
    "quant": {
        "bit_widths": {
            "int2": 2, "int3": 3, "int4": 4,
            "int5": 5, "int6": 6, "int8": 8,
        },
        # 每组要额外存多少位元数据。**这就是「4-bit 其实不是 4-bit」的来源。**
        # rtn / gptq 用非对称量化 ⇒ 每组一个 scale + 一个 zero。
        # awq 用对称量化 + 逐输入通道一个缩放因子。
        # kquant 用对称量化 + 量化过的子块 scale + 超块 fp16 scale。
        "metadata_bits": {
            "rtn": {"per_group_scale": 16, "per_group_zero": 16, "per_channel_scale": 0},
            "gptq": {"per_group_scale": 16, "per_group_zero": 16, "per_channel_scale": 0},
            "awq": {"per_group_scale": 16, "per_group_zero": 0, "per_channel_scale": 16},
            "kquant": {"per_group_scale": 8, "per_group_zero": 0, "per_channel_scale": 0},
        },
        "kquant": {
            "super_block": 128,        # 超块大小（跨超块存一个 fp16 scale）
            "super_scale_bits": 16,
            # 层敏感度启发式：**首层与末层更敏感**，给它们多分位宽。
            # 这是声明式的（改这里就改行为），也是本项目的一条期望。
            "layer_mix": {"layer0": 1, "layer1": 0, "head": 2},
        },
        "gptq": {
            "damp": 0.01,              # Hessian 对角阻尼 λ·mean(diag)
        },
        "awq": {
            "alpha": 0.50,             # 缩放指数 s = act^alpha
            "grid": 21,                # 缩放搜索的网格点数（在 alpha 邻域细扫）
        },
        "eps": 1e-12,
    },
    # ---- 运行时成本模型 -------------------------------------------------------
    # 延迟 = 固定开销 + 访存时间 × 反量化惩罚 / kernel 效率
    # 关键：**反量化惩罚随分组变小而上升**（块数变多），所以 group 这个旋钮
    # 同时动体积与延迟，而且**动延迟的幅度远大于动体积**。
    "runtime": {
        "bandwidth_gbps": 200.0,            # 4060 级显卡的可用带宽
        "fixed_latency_ms": 3.0,            # 与模型无关的固定开销
        "base_group_for_dequant": 128,      # 惩罚的参考分组
        "dequant_penalty_per_block": 0.06,  # 分组减半一次，惩罚 +6%
        "kernel_efficiency": {              # < 1 = 更慢
            "rtn": 1.00, "gptq": 0.94, "awq": 0.96, "kquant": 0.90, "fp16": 1.00,
        },
        "activation_mb": 420.0,             # 激活 + KV cache（**不随量化变化**）
        "fixed_memory_mb": 480.0,           # 运行时 + 上下文
    },
    # ---- 部署约束（用来做选型，不用来过滤候选）--------------------------------
    "deployment": {
        "base_params_b": 7.0,       # 部署规模：七十亿参数
        "memory_limit_gb": 8.0,     # 4060 8G
        "latency_limit_ms": 120.0,
    },
    # ---- 矩阵与前沿 -----------------------------------------------------------
    "matrix": {
        "knee_axes": [0, 3],        # 拐点算在 (Quality, Size) 平面上
        "hv_ref_margin": 0.10,      # 超体积参考点：各维最差再外推 10%
        "diff_axes": ["precision", "group", "algorithm", "calib"],
        "sensitivity_axis": "calib",
    },
    # ---- 验收门槛（12 条）-----------------------------------------------------
    # 每条门槛都必须是**可失败的**。两条反例（本项目第一版就写错了）：
    #
    # ① `max_quality_above_fp16 = 0` —— 写「零个候选可以高于 fp16 基线」。
    #    这在统计上**不可达**：测试集 512 条、108 个候选取最大值，
    #    一定有候选偶然高过基线。实测最大超出 +0.0137，而二项标准误是 0.0155
    #    （< 1 SE，纯噪声）。于是门槛改成「按标准误的倍数算」：
    #    `max_quality_above_fp16_se`。**拍一个 0 与拍一个 0.02 都不如拍一个 SE 倍数**，
    #    因为后者会随测试集规模自动缩放。
    #
    # ② `min_test_calib_overlap = 0` —— 名字里是 `min`，值却是 0，
    #    于是判据成了「重叠数 ≥ 0」，**恒真**。一条恒真的门槛与没有门槛无法区分。
    #    改名成 `max_test_calib_overlap`（重叠数必须 ≤ 0，也就是「必须不重叠」）。
    "acceptance": {
        "min_candidates": 100,
        "max_matrix_gaps": 0,
        "max_frontier_dominated": 0,
        "min_frontier_size": 6,
        "min_frontier_precisions": 3,
        "min_frontier_algorithms": 3,
        "max_effective_bits_violations": 0,
        "max_idempotence_violations": 0,
        "max_fp16_invariance_violations": 0,
        "max_rtn_calibration_violations": 0,
        "max_quality_above_fp16_se": 1.5,
        "max_test_calib_overlap": 0,
        "max_memory_axis_frontier_delta": 0,
    },
    # ---- 设计期望（15 条，允许被推翻）-----------------------------------------
    "expectations": {
        "lower_bits_lose_more_quality": True,
        "gptq_beats_rtn_at_low_bits": True,
        "awq_beats_gptq_at_int2": True,
        "kquant_has_the_smallest_metadata": True,
        "algorithm_gap_widens_at_low_bits": True,
        "more_calibration_helps": True,
        "smaller_group_improves_quality": True,
        "smaller_group_costs_latency": True,
        "memory_is_an_independent_axis": True,
        "knee_point_is_the_best_deployment_choice": True,
        "int8_is_almost_free": True,
        "int2_is_unusable": True,
        "frontier_uses_every_algorithm": True,
        "calibration_free_algorithms_are_competitive": True,
        "metadata_dominates_at_small_groups": True,
    },
}


def default_config() -> dict[str, Any]:
    return copy.deepcopy(DEFAULTS)


# --------------------------------------------------------------------------- #
# 设计空间
# --------------------------------------------------------------------------- #


def enumerate_specs(cfg: Mapping[str, Any]) -> list[QuantSpec]:
    """设计空间 = 四个轴的笛卡尔积。

    **枚举顺序被钉死**（算法 → 位宽 → 分组 → 校准），因为产物里的行顺序就是它 ——
    改顺序会让所有历史产物失效（内容不变，字节变了）。

    `rtn` / `kquant` 的 `calib_sizes` 是 `[0]`（**不需要**校准），
    所以它们只展开出一份；`gptq` / `awq` 有两个档位，各展开两份。
    这不是「少测了」，是事实：那两个算法根本不读激活统计。
    """
    space = cfg["space"]
    out: list[QuantSpec] = []
    for alg in space["algorithms"]:
        for prec in space["precisions"]:
            for g in space["groups"]:
                for c in space["calib_sizes"][alg]:
                    out.append(QuantSpec(
                        algorithm=str(alg), precision=str(prec),
                        group=int(g), calib=int(c),
                    ))
    return out


def calib_levels(cfg: Mapping[str, Any]) -> list[int]:
    """全部**正的**校准档位（升序）。校准统计按档位算一次，全部候选共用。"""
    return sorted({
        int(c)
        for a in cfg["space"]["algorithms"]
        for c in cfg["space"]["calib_sizes"][a]
        if int(c) > 0
    })


def _deep_merge(base: dict[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并。**注意：这不会删除 `base` 里多出来的键。**

    `DistillFactory` 那条坑（220）说的就是这个：从 DEFAULTS 里删掉的键，
    会从已落盘的旧配置里活下来，而代码里的删除传不到磁盘。
    所以改完 DEFAULTS **必须**重跑 `dump-config`。
    """
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_merge(dict(out[k]), v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def merge_config(over: Mapping[str, Any] | None = None) -> dict[str, Any]:
    cfg = default_config()
    return _deep_merge(cfg, over) if over else cfg


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        over = json.load(fh)
    return merge_config(over)


def dump_config(cfg: Mapping[str, Any], path: str) -> str:
    text = json.dumps(cfg, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    return path


# --------------------------------------------------------------------------- #
# 校验
# --------------------------------------------------------------------------- #

def validate_config(cfg: Mapping[str, Any]) -> list[str]:
    """返回问题列表（空 = 通过）。**每条检查都必须被真的调用到。**"""
    bad: list[str] = []

    space = cfg["space"]
    quant = cfg["quant"]
    algs = list(space["algorithms"])
    precs = list(space["precisions"])
    groups = list(space["groups"])

    if not algs:
        bad.append("space.algorithms 不能为空")
    if not precs:
        bad.append("space.precisions 不能为空")
    if not groups:
        bad.append("space.groups 不能为空")

    # 每个精度都要有位宽定义 —— 否则会在枚举中途 KeyError（离原因很远）
    for p in precs:
        if p not in quant["bit_widths"]:
            bad.append(f"quant.bit_widths 缺少精度 {p!r}")
        elif not 1 < int(quant["bit_widths"][p]) <= 16:
            bad.append(f"精度 {p!r} 的位宽必须在 (1, 16]，得到 {quant['bit_widths'][p]}")

    # 每个算法都要有元数据定义与 kernel 效率
    for a in algs:
        if a not in quant["metadata_bits"]:
            bad.append(f"quant.metadata_bits 缺少算法 {a!r}")
        if a not in cfg["runtime"]["kernel_efficiency"]:
            bad.append(f"runtime.kernel_efficiency 缺少算法 {a!r}")

    # 每个算法都要有校准档位；档位必须是 0（不校准）或正偶数
    for a in algs:
        levels = space["calib_sizes"].get(a)
        if not levels:
            bad.append(f"space.calib_sizes 缺少算法 {a!r} 的档位")
            continue
        for lv in levels:
            if int(lv) < 0:
                bad.append(f"算法 {a!r} 的校准档位不能为负：{lv}")
            elif int(lv) > int(cfg["world"]["n_calib_pool"]):
                bad.append(
                    f"算法 {a!r} 的校准档位 {lv} 超过校准池 "
                    f"{cfg['world']['n_calib_pool']}"
                )

    # 分组必须能整除（或小于）最窄的权重矩阵的 in 维，否则分组语义会静默退化
    d_in = int(cfg["world"]["d_in"])
    for g in groups:
        if int(g) <= 0:
            bad.append(f"分组必须为正，得到 {g}")
        elif int(g) > d_in:
            bad.append(
                f"分组 {g} 超过 d_in={d_in}，该轴上会退化成「整行一组」"
                "（不是错误，但会让不同分组取值撞在一起）"
            )

    # 测试集与校准池必须能装下，且训练集非空
    w = cfg["world"]
    if int(w["n_train"]) <= 0:
        bad.append("world.n_train 必须为正")
    if int(w["n_test"]) <= 0:
        bad.append("world.n_test 必须为正")
    if int(w["n_calib_pool"]) <= 0:
        bad.append("world.n_calib_pool 必须为正")
    if int(w["n_prototypes"]) < int(w["n_classes"]):
        bad.append(
            f"world.n_prototypes={w['n_prototypes']} 少于类别数 "
            f"{w['n_classes']} —— 至少每类一个原型"
        )
    if len(list(w["hidden"])) < 1:
        bad.append("world.hidden 至少要有一层")

    # 隐藏层必须都能被最大分组整除，否则 kquant 的块结构对不齐
    gmax = max(int(g) for g in groups) if groups else 0
    if gmax:
        for h in list(w["hidden"]):
            if int(h) % gmax != 0:
                bad.append(
                    f"隐藏层宽度 {h} 不能被最大分组 {gmax} 整除 —— "
                    "块边界会跨行，量化语义不再逐组独立"
                )
    if d_in % gmax != 0:
        bad.append(f"d_in={d_in} 不能被最大分组 {gmax} 整除")

    # kquant 的超块必须能装下最大分组
    kb = int(quant["kquant"]["super_block"])
    if kb < gmax:
        bad.append(f"kquant.super_block={kb} 小于最大分组 {gmax}")
    if kb % gmax != 0:
        bad.append(f"kquant.super_block={kb} 不能被最大分组 {gmax} 整除")

    # 验收门槛的键必须存在（`every_threshold_is_wired` 的静态版）
    for k in DEFAULTS["acceptance"]:
        if k not in cfg["acceptance"]:
            bad.append(f"acceptance 缺少门槛 {k!r}")
    for k in DEFAULTS["expectations"]:
        if k not in cfg["expectations"]:
            bad.append(f"expectations 缺少期望 {k!r}")

    return bad


# --------------------------------------------------------------------------- #
# 指纹
# --------------------------------------------------------------------------- #

def protocol_digest(cfg: Mapping[str, Any]) -> str:
    """协议指纹：只覆盖 `PROTOCOL_KEYS`。**改这些 = 换一个项目。**"""
    return digest_of({k: cfg[k] for k in PROTOCOL_KEYS}, length=16)


def version_digest(cfg: Mapping[str, Any]) -> str:
    """配置版本：协议 + 种子 + 世界 + 模型规模。用来回答「这份产物对应哪份配置」。"""
    return "v1-" + digest_of(
        {
            "protocol": protocol_digest(cfg),
            "seed": cfg["seed"],
            "world": cfg["world"],
            "train": cfg["train"],
        },
        length=12,
    )
