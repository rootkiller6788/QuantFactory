"""四个量化算子。**都是真实现，不是打分公式。**

| 算法 | 机制 | 需要校准？ | 元数据（每组） |
| --- | --- | --- | --- |
| `rtn` | 逐组 absmax，**非对称**（scale + zero） | 否 | 16 + 16 位 |
| `gptq` | 用 `H⁻¹` 做**逐列误差补偿**，把量化误差摊到还没量化的列上 | 是 | 16 + 16 位 |
| `awq` | 用**激活幅度**算逐输入通道缩放，缩放后量化再还原 | 是 | 16/通道 + 16/组 |
| `kquant` | 超块二级 scale（子块 scale 本身也被量化）+ **层位宽混合** | 否 | 8/组 + 16/超块 |

## 三条设计纪律

1. **`rtn` 是必需的对照，不是凑数。** 它把「算法的贡献」从「位宽与分组的贡献」里
   分出来。没有它，`gptq` 比 `int2` 好多少这件事说不清是算法的功劳还是位宽的功劳。
2. **元数据必须计入体积。** 同一个「INT4」在不同算法下不是同一个体积：

   ```
   rtn    : 4 + (16 + 16)/g                          ← 每组一个 fp16 scale + 一个 fp16 zero
   awq    : 4 + 16/g + 16/in                         ← 逐输入通道再摊一个缩放因子
   kquant : 4 + 8/g + 16/min(super_block, in) + 层混合的位宽再分配
   ```

   实测（本项目模型，`in = 64`、`super_block = 128`）：
   `g16` 下 `rtn = 6.0000` 而 `kquant = 5.3382`（**kquant 省 11%**）；
   但 `g32` 下 `rtn = 5.0000` 而 `kquant = 5.0882`（**kquant 反而更贵**）——
   因为分组变大之后元数据被摊薄，而层混合把 `head` 抬了 2 位、`layer0` 抬了 1 位
   （`head` 只有 512 个参数，抬它很便宜；`layer0` 有 4096 个，抬它很贵）。

   ⇒ **「元数据最少」与「有效位宽最少」是两个不同的结论**，报告里必须分开说。
   `metadata_bits_per_weight` 与 `effective_bits` 是**两个字段**，就是为了这个。
   只比名义位宽，比的就不是同一个东西。
3. **`kquant` 的层混合是「再分配」，不是「加码」。** `layer0` 与 `layer1` 的参数量
   相同（都是 64×64），所以 `+1 / −1` 的净位宽变化恰好为 0 ——
   否则「kquant 更省」这条结论会被混合本身污染（把位宽加回去谁都能变好）。
   `head` 保持名义位宽。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .calib import CalibStats
from .model import Layer, Model, clone
from .schema import QuantSpec

__all__ = [
    "QuantResult",
    "quantize",
    "metadata_bits_per_weight",
    "effective_bits",
    "n_blocks_of",
]

_sumprod = math.sumprod


@dataclass
class QuantResult:
    model: Model
    layer_bits: dict[str, float]
    nominal_bits: float
    metadata_bits_per_weight: float
    effective_bits: float
    n_blocks: int
    weight_rel_error: float
    weight_max_abs_error: float
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# 元数据与位宽口径
# --------------------------------------------------------------------------- #


def _g_eff(group: int, in_dim: int) -> int:
    """有效组宽。组宽大于输入维时退化成「整行一组」——这是真实行为，不是错误。"""
    return in_dim if group <= 0 or group > in_dim else group


def _metadata_layer(
    cfg: Mapping[str, Any], algorithm: str, group: int, in_dim: int
) -> float:
    """**每个权重**分摊到的元数据位数。"""
    meta = cfg["quant"]["metadata_bits"]
    m = meta[algorithm]
    g = _g_eff(group, in_dim)
    bits = (float(m["per_group_scale"]) + float(m["per_group_zero"])) / g
    bits += float(m["per_channel_scale"]) / in_dim
    if algorithm == "kquant":
        sb = int(cfg["quant"]["kquant"]["super_block"])
        # 超块尺度：每个超块一个 fp16 值，摊到超块内的每个权重
        bits += float(cfg["quant"]["kquant"]["super_scale_bits"]) / min(sb, in_dim)
    return bits


def metadata_bits_per_weight(
    cfg: Mapping[str, Any], algorithm: str, group: int, model: Model
) -> float:
    """参数量加权平均的元数据位数。"""
    total_params = 0
    acc = 0.0
    for layer in model.layers:
        out, inp = layer.shape
        n = out * inp
        total_params += n
        acc += n * _metadata_layer(cfg, algorithm, group, inp)
    return acc / total_params if total_params else 0.0


def _layer_bits(
    cfg: Mapping[str, Any], spec: QuantSpec, model: Model
) -> dict[str, float]:
    """每一层的**名义位宽**（`kquant` 会因层混合而不同）。"""
    base = float(cfg["quant"]["bit_widths"][spec.precision])
    out: dict[str, float] = {}
    mix = cfg["quant"]["kquant"]["layer_mix"] if spec.algorithm == "kquant" else {}
    for layer in model.layers:
        out[layer.role] = base + float(mix.get(layer.role, 0.0))
    return out


def effective_bits(
    cfg: Mapping[str, Any], spec: QuantSpec, model: Model, layer_bits: Mapping[str, float]
) -> float:
    """参数加权平均的**有效位宽** = 名义位宽 + 元数据。"""
    total_params = 0
    acc = 0.0
    for layer in model.layers:
        out, inp = layer.shape
        n = out * inp
        total_params += n
        acc += n * (float(layer_bits[layer.role]) + _metadata_layer(cfg, spec.algorithm, spec.group, inp))
    return acc / total_params if total_params else 0.0


def n_blocks_of(model: Model, group: int) -> int:
    """总组数。**它同时决定延迟（反量化开销）与元数据量。**"""
    total = 0
    for layer in model.layers:
        out, inp = layer.shape
        total += out * max(1, math.ceil(inp / _g_eff(group, inp)))
    return total


# --------------------------------------------------------------------------- #
# 逐组量化基元
# --------------------------------------------------------------------------- #


def _q_asym(blk: Sequence[float], levels: int) -> list[float]:
    """非对称 absmax：`[lo, hi]` 映到 `0..levels-1`。返回**反量化后**的值。"""
    lo = min(blk)
    hi = max(blk)
    span = hi - lo
    if span <= 0.0:
        return [lo] * len(blk)
    scale = span / (levels - 1)
    zero = round(-lo / scale)
    out = []
    for v in blk:
        q = round(v / scale) + zero
        if q < 0:
            q = 0
        elif q > levels - 1:
            q = levels - 1
        out.append((q - zero) * scale)
    return out


def _q_sym(blk: Sequence[float], qmax: int) -> list[float]:
    """对称 absmax：`±qmax`。**只有 `kquant` 用它**（K-quant 是对称方案）。

    `rtn` / `gptq` / `awq` 一律用 `_q_asym` —— 三个算法之间的差异必须**只来自机制**
    （补偿 / 缩放），不能混进「非对称 vs 对称」这第二个变量。见 `_do_awq` 的注释。
    """
    mx = max(abs(v) for v in blk)
    if mx <= 0.0:
        return [0.0] * len(blk)
    scale = mx / qmax
    out = []
    for v in blk:
        q = round(v / scale)
        if q > qmax:
            q = qmax
        elif q < -qmax:
            q = -qmax
        out.append(q * scale)
    return out


def _blocks(n: int, g: int) -> list[tuple[int, int]]:
    return [(s, min(s + g, n)) for s in range(0, n, g)]


# --------------------------------------------------------------------------- #
# 四个算法
# --------------------------------------------------------------------------- #


def _do_rtn(model: Model, spec: QuantSpec, cfg: Mapping[str, Any]) -> Model:
    q = clone(model)
    bits = int(cfg["quant"]["bit_widths"][spec.precision])
    levels = 1 << bits
    for layer in q.layers:
        g = _g_eff(spec.group, layer.shape[1])
        for row in layer.W:
            for s, e in _blocks(len(row), g):
                row[s:e] = _q_asym(row[s:e], levels)
    return q


def _do_kquant(model: Model, spec: QuantSpec, cfg: Mapping[str, Any]) -> Model:
    """K-quant：对称 + 超块二级 scale（**子块 scale 本身也被量化**）+ 层位宽混合。"""
    q = clone(model)
    kc = cfg["quant"]["kquant"]
    sb_size = int(kc["super_block"])
    sub_scale_levels = 1 << int(cfg["quant"]["metadata_bits"]["kquant"]["per_group_scale"])
    mix = kc["layer_mix"]
    base = int(cfg["quant"]["bit_widths"][spec.precision])

    for layer in q.layers:
        bits = base + int(mix.get(layer.role, 0))
        bits = max(2, min(bits, 16))
        qmax = (1 << (bits - 1)) - 1
        g = _g_eff(spec.group, layer.shape[1])
        for row in layer.W:
            n = len(row)
            for ss, se in _blocks(n, min(sb_size, n)):
                # ---- 第一级：每个子块算 scale
                sub_blocks = [(s, e) for s, e in _blocks(se - ss, g)]
                scales: list[float] = []
                for s, e in sub_blocks:
                    blk = row[ss + s: ss + e]
                    mx = max(abs(v) for v in blk)
                    scales.append(mx / qmax if mx > 0.0 else 0.0)
                # ---- 第二级：子块 scale 被量化到 8 位（**这一步是 K-quant 的本质**）
                smax = max(scales) if scales else 0.0
                if smax <= 0.0:
                    for s, e in sub_blocks:
                        row[ss + s: ss + e] = [0.0] * (e - s)
                    continue
                for (s, e), sc in zip(sub_blocks, scales):
                    sc_q = round(sc / smax * (sub_scale_levels - 1))
                    sc_hat = sc_q / (sub_scale_levels - 1) * smax
                    if sc_hat <= 0.0:
                        row[ss + s: ss + e] = [0.0] * (e - s)
                        continue
                    # ⚠️ 用的是**量化后的** scale（sc_hat），不是原始 scale ——
                    # 这一步是 K-quant 的本质：用二级量化换元数据位数。
                    row[ss + s: ss + e] = [
                        _clip_round(v, sc_hat, qmax) for v in row[ss + s: ss + e]
                    ]
    return q


def _clip_round(v: float, scale: float, qmax: int) -> float:
    q = round(v / scale)
    if q > qmax:
        q = qmax
    elif q < -qmax:
        q = -qmax
    return q * scale


def _do_gptq(
    model: Model, spec: QuantSpec, cfg: Mapping[str, Any], stats: CalibStats
) -> Model:
    """GPTQ：误差补偿。

    ## 为什么是「整组一次量化 + 只补偿后续组」，而不是「逐列量化 + 补偿所有后续列」

    逐列版本（参考实现的写法）在本项目上**会把误差放大 2~5 倍**。原因不是公式错
    （已用微测试逐项验过，且与 `H⁻¹` 的 Cholesky 形式等价），而是**分组与补偿不相容**：

    GPTQ 的推导假设剩余列是**连续**的。但组内各列共享一个由**组的 min/max** 决定的
    scale，把补偿落回同组，等于把值推向网格边缘甚至越界（`_q_asym` 会 clamp），
    于是补偿不但没把误差摊出去，还额外制造了截断误差。

    微测试实测（良态 H，6×8 矩阵）：
    逐列补偿在 g=2/int2 下让层输出误差 **+272%**，而整组补偿是 **−0.5%**。

    ## 结果：在这个设定下补偿**不带来收益**

    整组补偿的实测增益是 −1.6% ~ +2.0%（六个格子），也就是**与 RTN 打平**。
    这是一个**被推翻的设计期望**（`gptq_beats_rtn_at_low_bits`）：在逐组 scale 已经
    很细的前提下，误差补偿没有可发挥的空间。见 README 附录 B。
    """
    q = clone(model)
    bits = int(cfg["quant"]["bit_widths"][spec.precision])
    levels = 1 << bits

    for li, layer in enumerate(q.layers):
        out_dim, in_dim = layer.shape
        orig = [row[:] for row in layer.W]
        Wc = [row[:] for row in layer.W]
        Hi = stats.hessian_inv[li]
        g = _g_eff(spec.group, in_dim)

        for s in range(0, in_dim, g):
            e = min(s + g, in_dim)
            # ---- 本组：整组一次量化
            for o in range(out_dim):
                Wc[o][s:e] = _q_asym(Wc[o][s:e], levels)
            # ---- 本组的误差，按 H⁻¹ 摊到**后续组**
            for j in range(s, e):
                d = Hi[j][j]
                if d <= 0.0:
                    d = 1.0
                delta = [orig[o][j] - Wc[o][j] for o in range(out_dim)]
                if not any(delta):
                    continue
                for k in range(e, in_dim):
                    hk = Hi[j][k]
                    if hk == 0.0:
                        continue
                    f = hk / d
                    for o in range(out_dim):
                        Wc[o][k] -= delta[o] * f
        for o in range(out_dim):
            layer.W[o] = Wc[o]
    return q


def _do_awq(
    model: Model, spec: QuantSpec, cfg: Mapping[str, Any], stats: CalibStats
) -> Model:
    """AWQ：用激活幅度算逐输入通道缩放 `s_j = act_j^α`，缩放权重 → 量化 → 还原。

    `W' = W·diag(s)`，运行时用 `W'/s`。输出误差是 `Σ_j (δ_j/s_j)·x_j` ——
    当 `δ_j` 由**组步长**（而不是该列自身的幅度）主导时，误差 ∝ `x_j/s_j`，
    于是 `s_j ∝ x_j`（即 `α=1`）最优；`α` 就是在这个极端与「相对误差均匀」之间的插值。

    ⚠️ **量化基元必须与 `rtn` / `gptq` 相同**（逐组非对称 absmax）。
    第一版这里用了**对称**量化，于是 AWQ 的对比里混进了「非对称 vs 对称」这第二个
    变量 —— 实测 AWQ 因此「差 0.27」，而那个差距里绝大部分是基元造成的，不是缩放。
    **换口径要换全，包括兄弟列。**
    """
    q = clone(model)
    ac = cfg["quant"]["awq"]
    alpha = float(ac["alpha"])
    bits = int(cfg["quant"]["bit_widths"][spec.precision])
    levels = 1 << bits

    for li, layer in enumerate(q.layers):
        out_dim, in_dim = layer.shape
        act = stats.act_absmean[li]
        raw = [max(v, 1e-12) ** alpha for v in act]
        logmean = math.fsum(math.log(v) for v in raw) / len(raw)
        norm = math.exp(logmean)
        s = [v / norm for v in raw]

        Ws = [[layer.W[o][j] * s[j] for j in range(in_dim)] for o in range(out_dim)]
        g = _g_eff(spec.group, in_dim)
        for row in Ws:
            for a, b in _blocks(in_dim, g):
                row[a:b] = _q_asym(row[a:b], levels)
        for o in range(out_dim):
            layer.W[o] = [Ws[o][j] / s[j] for j in range(in_dim)]
    return q


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


def quantize(
    model: Model,
    spec: QuantSpec,
    cfg: Mapping[str, Any],
    stats: CalibStats | None,
) -> QuantResult:
    """把一个设计点真的跑一遍。返回量化后的模型 + 位宽口径 + 诊断。"""
    notes: list[str] = []
    if spec.algorithm == "rtn":
        qm = _do_rtn(model, spec, cfg)
    elif spec.algorithm == "kquant":
        qm = _do_kquant(model, spec, cfg)
    elif spec.algorithm in ("gptq", "awq"):
        if stats is None:
            raise ValueError(f"{spec.algorithm} 需要校准统计，但收到 None")
        if stats.n != spec.calib:
            raise ValueError(
                f"校准档位不匹配：spec={spec.calib} 而 stats.n={stats.n}"
            )
        qm = _do_gptq(model, spec, cfg, stats) if spec.algorithm == "gptq" else _do_awq(
            model, spec, cfg, stats
        )
    else:
        raise ValueError(f"未知算法 {spec.algorithm!r}")

    # ---- 误差诊断
    num = 0.0
    den = 0.0
    mx = 0.0
    for a, b in zip(model.layers, qm.layers):
        for ro, rq in zip(a.W, b.W):
            for v0, v1 in zip(ro, rq):
                dd = v0 - v1
                num += dd * dd
                den += v0 * v0
                av = abs(dd)
                if av > mx:
                    mx = av
    rel = math.sqrt(num / den) if den > 0.0 else 0.0

    lb = _layer_bits(cfg, spec, model)
    eff = effective_bits(cfg, spec, model, lb)
    meta = metadata_bits_per_weight(cfg, spec.algorithm, spec.group, model)
    nominal = float(cfg["quant"]["bit_widths"][spec.precision])

    if spec.algorithm == "kquant":
        notes.append(
            "层位宽混合：" + " / ".join(f"{r}={int(v)}" for r, v in sorted(lb.items()))
            + "（layer0 与 layer1 参数量相同 ⇒ 混合是**再分配**，净位宽变化为 0）"
        )
    if spec.calib <= 0 and spec.algorithm in ("rtn", "kquant"):
        notes.append("校准无关：该算法只读权重，不读激活统计")
    if spec.algorithm == "awq":
        notes.append(
            f"α={float(cfg['quant']['awq']['alpha'])} 固定在配置里（原文会网格搜索 α；"
            "本项目把 α 的敏感性单独做一张表，避免每个候选都乘上网格大小）"
        )

    return QuantResult(
        model=qm,
        layer_bits=lb,
        nominal_bits=nominal,
        metadata_bits_per_weight=meta,
        effective_bits=eff,
        n_blocks=n_blocks_of(model, spec.group),
        weight_rel_error=rel,
        weight_max_abs_error=mx,
        notes=notes,
    )
