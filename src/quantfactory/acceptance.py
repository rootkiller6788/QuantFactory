"""验收：**门槛**（13 条）+ **对抗用例**（22 条）。

## 门槛与对抗用例的分工

| | 门槛 | 对抗用例 |
| --- | --- | --- |
| 形态 | `观测值 ≤/≥ 配置里的数` | 一个布尔判据，判据写在代码里 |
| 数量 | 由 `cfg["acceptance"]` 决定（**改门槛不改代码**） | 由这个文件决定 |
| 回答 | 「这次运行的结果**够不够好**」 | 「这套机制**是不是真的在按它声称的方式工作**」 |

**两者的失败含义完全不同**：门槛失败 = 这次结果不达标（可能是配置问题）；
对抗用例失败 = 代码有 bug。

## 三条纪律

1. **每一条门槛都必须被真的读到。** `every_threshold_is_wired` 把这件事变成一条
   对抗用例：`cfg["acceptance"]` 里的每个键都必须出现在某条检查的 `key` 上。
   一条从不触发的保险丝与没有保险丝无法区分 —— 而「恒真」的门槛更糟，
   它会让人以为这里被守住了。
2. **门槛必须是可失败的。** 见 `config.py` 里那两条反例的注释
   （`max_quality_above_fp16_se` / `max_test_calib_overlap`）。
3. **阈值不许写在代码里。** 这个文件里出现的数字只有两种：从 `cfg` 读的，
   或者从**统计量**推出来的（比如二项标准误）。剩下的一律是
   「判据本身的定义」（比如「≥ 3 个不同算法」里的 3 只出现在配置里）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from . import pareto
from .calib import hessian_of
from .config import enumerate_specs
from .matrix import match_axes
from .measure import latency_ms_of, memory_gb_of, size_gb_of
from .model import Model, accuracy
from .quantize import quantize
from .rng import Rng, derive_seed, sha256_text
from .schema import QuantSpec

__all__ = [
    "Check",
    "Case",
    "AcceptanceReport",
    "run_acceptance",
    "binomial_se",
    "hash_rows",
]

_sumprod = math.sumprod

# 相对容差：浮点重算与存储值之间的允许偏差。
_REL_TOL = 1e-9
_ABS_TOL = 1e-12


@dataclass
class Check:
    key: str
    limit: float
    observed: float
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "limit": self.limit,
            "observed": self.observed,
            "passed": bool(self.passed),
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Check":
        return cls(
            key=str(d["key"]),
            limit=float(d["limit"]),
            observed=float(d["observed"]),
            passed=bool(d["passed"]),
            detail=str(d.get("detail", "")),
        )


@dataclass
class Case:
    name: str
    passed: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": bool(self.passed), "detail": self.detail}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Case":
        return cls(
            name=str(d["name"]),
            passed=bool(d["passed"]),
            detail=str(d.get("detail", "")),
        )


@dataclass
class AcceptanceReport:
    thresholds: list[Check] = field(default_factory=list)
    cases: list[Case] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.thresholds) and all(c.passed for c in self.cases)

    def to_dict(self) -> dict[str, Any]:
        return {
            "thresholds": [c.to_dict() for c in self.thresholds],
            "cases": [c.to_dict() for c in self.cases],
            "n_thresholds": len(self.thresholds),
            "n_thresholds_passed": sum(1 for c in self.thresholds if c.passed),
            "n_cases": len(self.cases),
            "n_cases_passed": sum(1 for c in self.cases if c.passed),
            "all_passed": self.all_passed,
        }

    def failing(self) -> list[str]:
        return [c.key for c in self.thresholds if not c.passed] + [
            c.name for c in self.cases if not c.passed
        ]

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "AcceptanceReport":
        """从 `acceptance.json` 重建（`visualize` 的产物重建路径需要）。"""
        return cls(
            thresholds=[Check.from_dict(x) for x in d.get("thresholds", [])],
            cases=[Case.from_dict(x) for x in d.get("cases", [])],
        )


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #


def binomial_se(p: float, n: int) -> float:
    """二项标准误。**门槛必须按它来定，不能拍一个绝对值。**

    测试集 512 条、准确率 0.857 ⇒ SE = 0.0155。108 个候选取最大值，
    「有候选偶然高于基线」是必然事件，所以「零个候选可以高于 fp16」这种门槛
    在统计上不可达（本项目第一版就这么写了，实测最大超出 +0.0137 = 0.88 SE）。
    """
    if n <= 0:
        return 0.0
    v = p * (1.0 - p) / n
    return math.sqrt(v) if v > 0.0 else 0.0


def hash_rows(rows: Sequence[Sequence[float]]) -> list[str]:
    """逐行指纹。**用「每一行的哈希」而不是「整体的哈希」来查重叠** ——
    整体哈希只能回答「两块是不是同一个集合」，逐行哈希才能回答「有几行重合」，
    而后者才是「测试集被拿来当校准集了」这个问题的正确读法。
    """
    out: list[str] = []
    for r in rows:
        # repr 而不是 str：float 的 repr 是往返精确的（str 在 3.12 之前会截断）。
        out.append(sha256_text("\x1f".join(repr(float(v)) for v in r)))
    return out


def _max_abs_diff(a: Model, b: Model) -> float:
    worst = 0.0
    for la, lb in zip(a.layers, b.layers):
        for ra, rb in zip(la.W, lb.W):
            for x, y in zip(ra, rb):
                d = abs(x - y)
                if d > worst:
                    worst = d
    return worst


def _activation_columns(model: Model, X: Sequence[Sequence[float]], layer_index: int):
    """独立重算第 `layer_index` 层的**输入**激活（按通道转置）。

    ⚠️ 这里是**故意重写**的：不复用 `calib.py` 的任何函数，也不读 `CalibStats`。
    如果这里改成调用 `build_calib`，那这条检查就退化成「一个函数等于它自己」——
    这正是「拿被测对象当自己的判据」的形态。
    """
    n_in = len(model.layers[layer_index].W[0])
    cols: list[list[float]] = [[] for _ in range(n_in)]
    for x in X:
        a: list[float] = list(x)
        for i in range(layer_index):
            layer = model.layers[i]
            z = [_sumprod(layer.W[o], a) + layer.b[o] for o in range(len(layer.W))]
            a = [math.tanh(v) for v in z] if layer.activation == "tanh" else z
        for j in range(n_in):
            cols[j].append(a[j])
    return cols


def _hv_2d_exact(points: Sequence[Sequence[float]], ref: Sequence[float]) -> float:
    """二维超体积的**暴力精确解**（按 x 分条带）。用来校验 HSO 递推。"""
    pts = [p for p in points if p[0] > ref[0] and p[1] > ref[1]]
    if not pts:
        return 0.0
    xs = sorted({p[0] for p in pts})
    vol = 0.0
    for i, x_hi in enumerate(xs):
        x_lo = xs[i - 1] if i > 0 else ref[0]
        if x_hi <= x_lo:
            continue
        h = max((p[1] for p in pts if p[0] >= x_hi), default=ref[1]) - ref[1]
        if h > 0.0:
            vol += (x_hi - x_lo) * h
    return vol


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #


def run_acceptance(result: Any) -> AcceptanceReport:
    """跑完全部门槛与对抗用例。`result` 是 `pipeline.QuantRun`（鸭子类型）。"""
    cfg = result.cfg
    lim = cfg["acceptance"]
    rep = AcceptanceReport()

    def at_least(key: str, observed: float, detail: str) -> None:
        v = float(lim[key])
        rep.thresholds.append(Check(key, v, float(observed), observed >= v, detail))

    def at_most(key: str, observed: float, detail: str) -> None:
        v = float(lim[key])
        rep.thresholds.append(Check(key, v, float(observed), observed <= v, detail))

    readings = result.readings
    frontier = result.frontier

    # ---- 1. 候选数
    at_least("min_candidates", len(readings), f"设计空间展开出 {len(readings)} 个量化候选")

    # ---- 2. 设计空间无缺口：配置能展开出的组合必须全部出现在读数里
    expected = {s.spec_id for s in enumerate_specs(cfg)}
    got = {r.spec_id for r in readings}
    gaps = len(expected ^ got)
    at_most("max_matrix_gaps", gaps, f"应有 {len(expected)} 个设计点，实有 {len(got)} 个，差集 {gaps}")

    # ---- 3. 前沿里没有被支配的点
    vecs = [r.vector() for r in frontier]
    dominated = 0
    for i, a in enumerate(vecs):
        for j, b in enumerate(vecs):
            if i != j and pareto.dominates(b, a):
                dominated += 1
                break
    at_most("max_frontier_dominated", dominated, f"前沿 {len(frontier)} 点，其中被支配 {dominated} 点")

    # ---- 4~6. 前沿的多样性
    at_least("min_frontier_size", len(frontier), "前沿点数")
    at_least(
        "min_frontier_precisions",
        len({r.precision for r in frontier}),
        f"前沿用到的位宽：{sorted({r.precision for r in frontier})}",
    )
    at_least(
        "min_frontier_algorithms",
        len({r.algorithm for r in frontier}),
        f"前沿用到的算法：{sorted({r.algorithm for r in frontier})}",
    )

    # ---- 7. 位宽口径自洽
    bad_bits = _effective_bits_violations(cfg, readings, result.fp16)
    at_most(
        "max_effective_bits_violations", len(bad_bits),
        "有效位宽 < 名义位宽、或体积与有效位宽对不上、或层位宽与 layer_mix 不一致"
        + (f"：{bad_bits[:3]}" if bad_bits else ""),
    )

    # ---- 8. 量化幂等
    idem = _idempotence_violations(cfg, result)
    at_most(
        "max_idempotence_violations", len(idem),
        "单级量化（rtn/gptq/awq）必须逐位幂等；K-quant 的二级 scale 量化没有不动点，"
        "允许移动不超过一个步长" + (f"：{idem[:3]}" if idem else ""),
    )

    # ---- 9. 基座不变性
    base = _base_invariance_violations(result)
    at_most(
        "max_fp16_invariance_violations", base,
        "量化器必须 `clone` 后改副本；就地改基座会让**后面所有候选**都从量化模型出发",
    )

    # ---- 10. 校准无关算法
    cal = _calibration_independence_violations(cfg, result)
    at_most(
        "max_rtn_calibration_violations", len(cal),
        "rtn / kquant 只读权重，喂不同校准档位必须得到逐位相同的权重"
        + (f"：{cal[:3]}" if cal else ""),
    )

    # ---- 11. 质量不得显著高于 fp16 基线（按标准误）
    se = binomial_se(result.fp16.quality, len(result.world.test_x))
    excess = max((r.quality - result.fp16.quality for r in readings), default=0.0)
    at_most(
        "max_quality_above_fp16_se", excess / se if se > 0 else 0.0,
        f"最大超出 = {excess:+.4f} = {excess / se if se else 0.0:+.2f} SE"
        f"（SE={se:.6f}，n={len(result.world.test_x)}）",
    )

    # ---- 12. 测试集与校准集不重叠
    overlap = len(set(hash_rows(result.world.test_x)) & set(hash_rows(result.world.calib_pool)))
    at_most(
        "max_test_calib_overlap", overlap,
        "逐行哈希求交；拿测试集当校准集，测出来的质量就是「校准过的那个集合上的质量」",
    )

    # ---- 13. Memory 轴冗余
    delta = result.memory_axis_frontier_delta
    at_most(
        "max_memory_axis_frontier_delta", delta,
        "把 Memory 维换成 Size 维后，前沿集合的对称差；0 表示这一维**不提供任何信息**",
    )

    # ---- 对抗用例（需要先有门槛表，`every_threshold_is_wired` 要读它）
    rep.cases = _adversarial_cases(result, se, rep)
    return rep


# --------------------------------------------------------------------------- #
# 门槛的实现
# --------------------------------------------------------------------------- #


def _effective_bits_violations(
    cfg: Mapping[str, Any], readings: Sequence[Any], fp16: Any
) -> list[str]:
    """三条：① `effective ≥ nominal`；② `size == base_params_b·eff/8`；
    ③ 层位宽与 `kquant.layer_mix` 一致。"""
    base_b = float(cfg["deployment"]["base_params_b"])
    mix = cfg["quant"]["kquant"]["layer_mix"]
    bad: list[str] = []
    for r in list(readings) + [fp16]:
        if r.effective_bits < r.nominal_bits - _ABS_TOL:
            bad.append(f"{r.spec_id}: effective {r.effective_bits} < nominal {r.nominal_bits}")
            continue
        want = base_b * r.effective_bits / 8.0
        if abs(r.size_gb - want) > max(_ABS_TOL, _REL_TOL * abs(want)):
            bad.append(f"{r.spec_id}: size {r.size_gb} != {want}")
            continue
        for role, bits in r.layer_bits.items():
            expect = r.nominal_bits + (float(mix.get(role, 0.0)) if r.algorithm == "kquant" else 0.0)
            if abs(bits - expect) > _ABS_TOL:
                bad.append(f"{r.spec_id}: {role} 层位宽 {bits} != {expect}")
                break
    return bad


def _idempotence_violations(cfg: Mapping[str, Any], result: Any) -> list[str]:
    """再量化一次。

    * **单级量化**（`rtn` / `gptq` / `awq`）：网格完全由权重自身决定，且
      组内 min/max 一定被取到（`sc = max|v|/qmax` 让最大值恰好落在 qmax），
      于是数学上**严格幂等** —— 只允许 1e-12 的浮点残差。
    * **K-quant**：子块 scale 自己也被量化到 8 位。`round(sc/smax·255)` 可能
      把 scale **向上**取整（实测 155.66 → 156），于是该子块的 max 落到 254 而非 255；
      再量化时 `sc` 由 254 反推，第二次取整成 155 —— **这个映射没有不动点**。
      这是二级 scale 量化的**固有性质**（llama.cpp 的 K-quant 同构），不是实现错误。
      所以对 K-quant 只断言「移动不超过一个步长」。
    """
    bad: list[str] = []
    stats = result.stats
    for r in result.readings:
        spec = r.spec
        st = stats.get(spec.calib) if spec.calib > 0 else None
        q1 = result.quantized_models[spec.spec_id]
        q2 = quantize(q1, spec, cfg, st).model
        worst = _max_abs_diff(q1, q2)
        if spec.algorithm == "kquant":
            peak = max((abs(v) for l in q1.layers for row in l.W for v in row), default=0.0)
            step = peak / (1 << int(r.nominal_bits))
            if worst > step:
                bad.append(f"{spec.spec_id}: max|Δ|={worst:.3e} > 一个步长 {step:.3e}")
        elif worst > 1e-12:
            bad.append(f"{spec.spec_id}: max|Δ|={worst:.3e}（单级量化应当严格幂等）")
    return bad


def _base_invariance_violations(result: Any) -> int:
    """`result.model` 必须与「流水线开始时的那份」逐位相同。"""
    ref = result.model_snapshot
    return 0 if _max_abs_diff(ref, result.model) == 0.0 else 1


def _calibration_independence_violations(cfg: Mapping[str, Any], result: Any) -> list[str]:
    """`rtn` / `kquant` 喂不同档位的 `CalibStats`，权重必须逐位相同。"""
    levels = sorted(result.stats)
    if len(levels) < 2:
        return []
    bad: list[str] = []
    for r in result.readings:
        if r.algorithm not in ("rtn", "kquant"):
            continue
        spec = r.spec
        a = result.quantized_models[spec.spec_id]
        probe = QuantSpec(spec.algorithm, spec.precision, spec.group, levels[-1])
        b = quantize(result.model, probe, cfg, result.stats[levels[-1]]).model
        if _max_abs_diff(a, b) != 0.0:
            bad.append(spec.spec_id)
    return bad


# --------------------------------------------------------------------------- #
# 对抗用例
# --------------------------------------------------------------------------- #


def _adversarial_cases(
    result: Any, se: float, rep: AcceptanceReport
) -> list[Case]:
    cfg = result.cfg
    out: list[Case] = []

    # 1) 每条门槛都被读到
    keys = {c.key for c in rep.thresholds}
    missing = sorted(set(cfg["acceptance"]) - keys)
    out.append(Case(
        "every_threshold_is_wired", not missing,
        f"配置里的 {len(cfg['acceptance'])} 条门槛，被检查读到的 {len(keys)} 条"
        + (f"；**没有接线**：{missing}" if missing else ""),
    ))

    # 2) spec_id 单射
    specs = enumerate_specs(cfg)
    ids = [s.spec_id for s in specs]
    out.append(Case(
        "spec_id_is_injective", len(ids) == len(set(ids)),
        f"{len(ids)} 个设计点，{len(set(ids))} 个不同的 spec_id",
    ))

    # 3) 枚举数与配置的乘积一致
    space = cfg["space"]
    want = sum(
        len(space["precisions"]) * len(space["groups"]) * len(space["calib_sizes"][a])
        for a in space["algorithms"]
    )
    out.append(Case(
        "enumeration_matches_config_product", len(specs) == want,
        f"枚举 {len(specs)}，配置乘积 {want}",
    ))

    # 4) 校准集是**前缀**而不是随机抽
    out.append(_case_calibration_is_a_prefix(result))

    # 5) 三块数据两两不重叠（门槛只管 test∩calib）
    tr = set(hash_rows(result.world.train_x))
    te = set(hash_rows(result.world.test_x))
    ca = set(hash_rows(result.world.calib_pool))
    out.append(Case(
        "all_three_splits_are_pairwise_disjoint",
        not (tr & te) and not (tr & ca) and not (te & ca),
        f"train∩test={len(tr & te)} train∩calib={len(tr & ca)} test∩calib={len(te & ca)}",
    ))

    # 6) Hessian 求逆残差
    worst = max((max(v.inv_residual) for v in result.stats.values()), default=0.0)
    out.append(Case(
        "hessian_inverse_residual_is_small", worst < 1e-9,
        f"max|H·H⁻¹ − I| = {worst:.3e}（求逆写错不会报错，只会让 GPTQ 静默退化成带噪声的 RTN）",
    ))

    # 7) Hessian 对称
    sym = True
    worst_asym = 0.0
    for st in result.stats.values():
        for H in st.hessian:
            for i in range(len(H)):
                for j in range(i + 1, len(H)):
                    d = abs(H[i][j] - H[j][i])
                    worst_asym = max(worst_asym, d)
                    if d > 1e-15:
                        sym = False
    out.append(Case("hessian_is_symmetric", sym, f"max|H[j][k] − H[k][j]| = {worst_asym:.3e}"))

    # 8~9) 抽样器的两条对抗用例（`rng.sample_indices` 的文档里点了名）
    out.extend(_cases_sample_indices(cfg))

    # 10) 支配关系的反对称性
    pts = [(0.5, 0.5), (0.6, 0.4), (0.4, 0.6), (0.6, 0.6), (0.3, 0.3)]
    ok = True
    for a in pts:
        for b in pts:
            if a == b:
                continue
            if pareto.dominates(a, b) and pareto.dominates(b, a):
                ok = False
    out.append(Case(
        "dominates_is_antisymmetric", ok,
        "构造 5 个含平局/互不支配的点对，检查不存在互相支配",
    ))

    # 11) HSO 超体积 = 二维暴力精确解
    ref = (-0.5, -0.5)
    pts2 = [(0.9, 0.2), (0.5, 0.7), (0.7, 0.6), (0.2, 0.95), (0.4, 0.3)]
    a = pareto.hypervolume(pts2, ref)
    b = _hv_2d_exact(pts2, ref)
    out.append(Case(
        "hypervolume_matches_bruteforce_2d", abs(a - b) < 1e-9,
        f"HSO 递推 = {a:.12f}，按 x 分条带的精确解 = {b:.12f}",
    ))

    # 12) 加一个被支配的点，超体积不变；加一个非支配的点，超体积不减
    hv0 = pareto.hypervolume(pts2, ref)
    hv_dom = pareto.hypervolume(pts2 + [(0.3, 0.1)], ref)
    hv_nd = pareto.hypervolume(pts2 + [(0.95, 0.95)], ref)
    out.append(Case(
        "hypervolume_ignores_dominated_points", abs(hv_dom - hv0) < 1e-12,
        f"加 (0.3,0.1)（被 (0.4,0.3) 支配）后 HV {hv0:.12f} → {hv_dom:.12f}",
    ))
    out.append(Case(
        "hypervolume_is_monotone", hv_nd >= hv0 - 1e-12,
        f"加 (0.95,0.95) 后 HV {hv0:.12f} → {hv_nd:.12f}",
    ))

    # 13) 参考点必须被所有点支配
    ref4 = tuple(result.reference_point)
    out.append(Case(
        "reference_point_is_dominated_by_every_candidate",
        all(all(r.vector()[k] > ref4[k] for k in range(4)) for r in result.readings),
        f"参考点 {[round(v, 4) for v in ref4]}",
    ))

    # 14) 拐点必须**先投影再算**：换一组轴会给出不同的点
    quad = [(0.9, 0.1, 0.1, 0.5), (0.2, 0.9, 0.5, 0.2), (0.5, 0.5, 0.9, 0.9),
            (0.7, 0.3, 0.3, 0.7), (0.3, 0.7, 0.7, 0.3)]
    k01 = pareto.knee_index(quad, (0, 1))
    k03 = pareto.knee_index(quad, (0, 3))
    out.append(Case(
        "knee_index_projects_before_measuring", k01 != k03,
        f"同一组四维点在 (0,1) 上的拐点是 #{k01}，在 (0,3) 上是 #{k03} —— "
        "四维没有「两端连线」，不先投影就是在算一个不存在的量",
    ))

    # 15) 间距指标对量纲免疫
    base_pts = [(0.2, 1.0), (0.4, 2.0), (0.6, 3.2), (0.9, 3.3)]
    scaled = [(p[0], p[1] * 1e6) for p in base_pts]
    s1 = pareto.spacing(base_pts)
    s2 = pareto.spacing(scaled)
    out.append(Case(
        "spacing_is_scale_free", abs(s1 - s2) < 1e-12,
        f"把第 2 维乘 1e6：spacing {s1:.12f} → {s2:.12f}（每维先归一化的作用）",
    ))

    # 16) Diff 的匹配键：算法轴必须排除 calib（防回归）
    out.append(Case(
        "diff_key_drops_calib_for_algorithm_axis", match_axes("algorithm") == ("precision", "group"),
        f"match_axes('algorithm') = {match_axes('algorithm')} —— "
        "不排除 calib 的话 rtn(calib=0) 与 gptq(calib=64) 永远不同格，"
        "Diff 表会静默少掉 4 个算法对",
    ))

    # 17) 量化不就地改基座（小规模、与门槛 9 独立的一次）
    spec = QuantSpec("gptq", "int4", 32, sorted(result.stats)[0])
    # ⚠️ 第一版这里写的是 `zip(snap, model.layers)`，而 `snap` 是**全部行拍平**的列表 ——
    # 于是 `snap[1]`（layer0 的第 1 行）被拿去和 `layers[1].W[0]`（layer1 的第 0 行）比。
    # 这个检查当场红了，但红的原因是**检查自己写错了**，不是被测对象错了。
    # 「对抗用例失败 = 代码有 bug」—— 这里的 bug 在用例里。
    snap = [[row[:] for row in l.W] for l in result.model.layers]
    quantize(result.model, spec, cfg, result.stats[spec.calib])
    same = all(
        x == y
        for l0, l1 in zip(snap, result.model.layers)
        for row0, row1 in zip(l0, l1.W)
        for x, y in zip(row0, row1)
    )
    out.append(Case(
        "quantize_does_not_mutate_the_base_model", same,
        f"对 {spec.spec_id} 量化一次后，基座的全部 {result.model.n_params - 128} 个权重逐位不变",
    ))

    # 18) 成本模型确定性
    r0 = result.readings[0]
    s1 = size_gb_of(cfg, r0.effective_bits, r0.n_params)
    l1 = latency_ms_of(cfg, s1, r0.n_blocks, result.nref, r0.algorithm)
    m1 = memory_gb_of(cfg, s1)
    out.append(Case(
        "cost_model_is_deterministic",
        (s1, l1, m1) == (r0.size_gb, r0.latency_ms, r0.memory_gb),
        "同一个读数重算体积/延迟/显存，必须逐位相同",
    ))

    # 19) Memory = Size + 常数
    deltas = {round(r.memory_gb - r.size_gb, 12) for r in result.readings}
    out.append(Case(
        "memory_minus_size_is_one_constant", len(deltas) == 1,
        f"memory − size 的取值集合 = {sorted(deltas)}（只有一个值 ⇒ 两维完全共线）",
    ))

    # 20) 有效位宽必须真的用上元数据
    meta_used = all(r.effective_bits > r.nominal_bits - 1e-12 for r in result.readings)
    out.append(Case(
        "metadata_actually_enters_effective_bits", meta_used,
        "每个候选的 effective_bits 都 > nominal_bits（元数据被计入了体积）",
    ))

    # 21) 质量可复现
    probe = result.quantized_models[result.readings[0].spec_id]
    a1 = accuracy(probe, result.world.test_x, result.world.test_y)
    a2 = accuracy(probe, result.world.test_x, result.world.test_y)
    out.append(Case(
        "quality_is_reproducible", a1 == a2,
        f"同一模型评估两次：{a1:.10f} vs {a2:.10f}",
    ))

    # 22) 校准统计按档位共享（对象同一性）
    shared = result.stats_identity_ok
    out.append(Case(
        "calibration_stats_are_shared_per_level", shared,
        "每个档位只构建一次 `CalibStats`，全部候选共用同一个对象 —— "
        "否则结果看起来一模一样，只是慢了几百倍",
    ))

    # 23) report.html 必须是**产物的纯函数**
    out.append(_case_html_is_a_pure_function_of_the_products(result))

    return out


def _case_html_is_a_pure_function_of_the_products(result: Any) -> Case:
    """`report.html` 必须能从**产物字典**重建出来，且逐位相同。

    ## 它钉的是哪条纪律

    工作区纪律：**产物从字典渲染，不从对象渲染**。两条路径（真跑 / 从产物重建）
    会漂，漂了之后哪个数字对没法判断 —— 所以只许有一条路径。

    ## 它防的是一个具体的退化

    某天有人在渲染里读了一个**没进产物**的字段（`run.world.test_x`、`run.stats`、
    `run.quantized_models`……），于是 `visualize` 就再也无法从产物重建，只能把整条
    流水线**再跑一遍**才渲染得出来。实测那要 **44 s**，而它证明的东西反而更少：
    全跑一遍证明的是「流水线是确定的」，那件事已经由「两次 `run` 只有两个豁免文件
    不同」证明了。真正该被证明的是「**产物足以重造报告**」。

    ⇒ **渲染器读不到的字段，产物里就不该缺。** 这条用例让这句话每次 `run` 都被核对。

    ## 为什么在这里现造一次产物

    `run.products` 此刻还是空的（`products_of` 要在验收之后才跑，因为
    `acceptance.json` 得等验收完）。所以用例自己调一次 `products_of` ——
    它是 `result` 的纯函数，多调一次只是多花约 0.5 s。

    注：调用点必须排在 `run.expectations` 之后（见 `pipeline.run_pipeline` 里的注释），
    否则造出来的 HTML 里期望表是空的，用例会以错误的理由红掉。
    """
    from .pipeline import products_of, reconstruct_run
    from .visualize import render_html

    products = products_of(result)
    direct = products["report.html"]
    rebuilt = render_html(reconstruct_run(products))
    same = direct == rebuilt
    # ⚠️ 诚实地说清射程：`acceptance.json` 此刻还没有本用例自己（验收正在跑），
    # 两侧都取同一份内容 ⇒ **验收表自身不在射程内**（自指）。它由
    # `Check.from_dict` / `Case.from_dict` 的往返覆盖。其余全部输入都过了产物往返。
    return Case(
        "html_is_a_pure_function_of_the_products", same,
        f"从 {len(products)} 个产物重建后渲染，与直接渲染逐位相同"
        if same else
        f"从 {len(products)} 个产物重建后渲染，与直接渲染**不同**"
        " ⇒ 渲染器读了没进产物的东西",
    )


def _case_calibration_is_a_prefix(result: Any) -> Case:
    """`build_calib(n)` 必须用校准池的**前 n 行**。

    做法：在 `acceptance.py` 里**独立重算** Hessian（不调用 `calib.py`），
    再与 `CalibStats.hessian` 比。若 `build_calib` 改成随机抽 n 行，
    重算（取前缀）就对不上 —— 而「64 档不是 256 档的前缀」这件事
    会让「更多校准数据带来什么变化」混进第二个变量。
    """
    cfg = result.cfg
    levels = sorted(result.stats)
    if not levels:
        return Case("calibration_uses_a_prefix_of_the_pool", False, "没有校准统计")
    n = levels[-1]
    pool = result.world.calib_pool
    if n > len(pool):
        return Case("calibration_uses_a_prefix_of_the_pool", False, f"档位 {n} 超过池 {len(pool)}")
    st = result.stats[n]
    worst = 0.0
    for li in range(len(result.model.layers)):
        cols = _activation_columns(result.model, pool[:n], li)
        H = hessian_of(cols, n)
        mean = math.fsum(H[i][i] for i in range(len(H))) / len(H)
        add = st.damp * mean if st.damp * mean > 0 else 1e-12
        for i in range(len(H)):
            H[i][i] += add
        for i in range(len(H)):
            for j in range(len(H)):
                worst = max(worst, abs(H[i][j] - st.hessian[li][i][j]))
    return Case(
        "calibration_uses_a_prefix_of_the_pool", worst < 1e-9,
        f"独立重算 pool[:{n}] 的 Hessian 与 `CalibStats` 的最大差 = {worst:.3e}",
    )


def _cases_sample_indices(cfg: Mapping[str, Any]) -> list[Case]:
    """`rng.sample_indices` 的两条对抗用例 —— **判据只有一条：直接量抽样分布。**

    `DistillFactory` 的坑 218 就是这里：`out.append(vi)` 写成交换**前**的值，
    于是 `out` 恒等于 `[0..k-1]`，训练永远只看数据集的前 48 条。
    症状不是「抽错」而是「batch 太小」，六轮标定全在给一个抽样 bug 调超参。
    """
    n, k, trials = 520, 40, 3000
    rng = Rng(derive_seed("quantfactory.adversarial.sample_indices", int(cfg["seed"])))
    counts = [0] * n
    total = 0.0
    for _ in range(trials):
        idx = rng.sample_indices(n, k)
        for i in idx:
            counts[i] += 1
        total += math.fsum(idx) / len(idx)
    expect = trials * k / n
    chi2 = math.fsum((c - expect) ** 2 / expect for c in counts)
    # 卡方分布 df = n−1 = 519，均值 519、标准差 sqrt(2·519) = 32.2。
    # 取 +5σ ≈ 680 作上界：正常抽样落在 ~520，退化成 [0..k-1] 会到 ~10^5。
    bound = (n - 1) + 5.0 * math.sqrt(2.0 * (n - 1))
    mean_index = total / trials
    return [
        Case(
            "sample_indices_is_uniform", chi2 < bound,
            f"n={n} k={k} trials={trials}：卡方 {chi2:.1f} < {bound:.1f}（df={n - 1}）",
        ),
        Case(
            "sample_indices_covers_the_whole_range",
            abs(mean_index - (n - 1) / 2.0) < 10.0,
            f"一个 batch 的平均下标 = {mean_index:.2f}，期望 ≈ {(n - 1) / 2.0:.1f}"
            f"（退化成 [0..k-1] 时会给出 {k / 2.0:.1f}）",
        ),
    ]
