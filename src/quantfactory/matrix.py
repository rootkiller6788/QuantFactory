"""矩阵：候选 → 对比表 / 前沿 / 选型 / Diff / 敏感性。

## 六张表各自回答一个问题（**不要合并**）

| 表 | 回答的问题 | 谁读它 |
| --- | --- | --- |
| `matrix_rows` | 每个候选的完整读数（**最细的一张**） | `matrix.csv` |
| `quality_table` | 哪个算法在哪个位宽 / 分组上最好 | `quality_table.csv`、报告主表 |
| `frontier_of` | 哪些候选是**不可支配**的 | `frontier.csv` |
| `selections` | 给定部署预算，**选谁** | `selection.csv` |
| `pairwise_diff` | 单独拧一个旋钮，质量动多少 | `diff.csv` |
| `calibration_sensitivity` | 校准规模翻倍，动的是**什么** | `sensitivity.csv` |

## 三条纪律

1. **被约束挡掉的候选也要进矩阵。** `deployment` 只用来做选型，不用来过滤候选 ——
   过滤掉之后，「这张表里没有」与「这个方案不可行」就分不清了。可行性是**一列**，
   不是一道闸门。
2. **Diff 必须落在「匹配格」上。** 拿「g16 的最好候选」减「g64 的最差候选」，
   得到的不是分组的影响。所以 `pairwise_diff` 固定其余各轴，只在同格里比。
3. **前沿必须是同一个参考点下的超体积。** 参考点由**全部候选**（不只前沿）
   的最差值再外推 `hv_ref_margin` 得到 —— 否则「多加一个烂点，超体积反而变大」。
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from . import pareto
from .measure import within_budget
from .schema import Reading, Selection

__all__ = [
    "MATRIX_COLUMNS",
    "QUALITY_TABLE_COLUMNS",
    "DIFF_COLUMNS",
    "SENSITIVITY_COLUMNS",
    "AXES",
    "matrix_rows",
    "quality_table",
    "frontier_of",
    "reference_point",
    "frontier_summary",
    "selections",
    "match_axes",
    "pairwise_diff",
    "diff_rows",
    "calibration_sensitivity",
    "frontier_delta_with_axis_swapped",
    "corr",
    "by_spec_id",
]

AXES: tuple[str, ...] = ("algorithm", "precision", "group", "calib")

MATRIX_COLUMNS: tuple[str, ...] = (
    "spec_id", "algorithm", "precision", "group", "calib", "calib_label",
    "quality", "quality_loss", "latency_ms", "memory_gb", "size_gb",
    "nominal_bits", "effective_bits", "metadata_bits_per_weight",
    "n_blocks", "weight_rel_error", "weight_max_abs_error", "feasible",
)

QUALITY_TABLE_COLUMNS: tuple[str, ...] = (
    "precision", "group", "algorithm", "calib", "spec_id",
    "quality", "quality_loss", "effective_bits", "size_gb", "latency_ms", "feasible",
)

DIFF_COLUMNS: tuple[str, ...] = (
    "axis", "left", "right", "n_cells",
    "mean_delta_quality", "min_delta_quality", "max_delta_quality",
    "n_up", "n_down", "n_tie",
    "mean_delta_size_gb", "mean_delta_latency_ms", "mean_delta_effective_bits",
)

SENSITIVITY_COLUMNS: tuple[str, ...] = (
    "algorithm", "precision", "group", "calib_left", "calib_right",
    "quality_left", "quality_right", "delta_quality",
    "rel_error_left", "rel_error_right", "delta_weight_rel_error",
    "delta_effective_bits",
)


# --------------------------------------------------------------------------- #
# 小工具
# --------------------------------------------------------------------------- #


def by_spec_id(readings: Sequence[Reading]) -> dict[str, Reading]:
    return {r.spec_id: r for r in readings}


def corr(a: Sequence[float], b: Sequence[float]) -> float:
    """皮尔逊相关系数。**本项目用它把「Memory 与 Size 共线」变成一个数。**

    样本方差为 0 时返回 0.0（而不是抛异常或 NaN）—— 上游是「全部候选的
    memory 都相同」这种退化情形，此时相关系数**没有定义**，返回 NaN 会让
    它悄悄流进产物（`allow_nan=False` 会当场炸，但报错位置离原因很远）。
    """
    n = len(a)
    if n < 2 or n != len(b):
        return 0.0
    ma = sum(a) / n
    mb = sum(b) / n
    da = [x - ma for x in a]
    db = [x - mb for x in b]
    va = sum(x * x for x in da)
    vb = sum(x * x for x in db)
    if va <= 0.0 or vb <= 0.0:
        return 0.0
    return sum(x * y for x, y in zip(da, db)) / math.sqrt(va * vb)


def _round(v: float | None, nd: int = 10) -> float | None:
    return None if v is None else round(float(v), nd)


# --------------------------------------------------------------------------- #
# 1. 主表
# --------------------------------------------------------------------------- #


def matrix_rows(readings: Sequence[Reading], cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """每个候选一行。**顺序 = 枚举顺序**（不是按质量排序）——

    按质量排序会让「同一个算法的候选散落在各处」，读者没法沿着某一列看趋势。
    """
    out: list[dict[str, Any]] = []
    for r in readings:
        out.append({
            "spec_id": r.spec_id,
            "algorithm": r.algorithm,
            "precision": r.precision,
            "group": r.group,
            "calib": r.calib,
            "calib_label": r.spec.calib_label,
            "quality": round(r.quality, 10),
            "quality_loss": round(r.quality_loss, 10),
            "latency_ms": round(r.latency_ms, 10),
            "memory_gb": round(r.memory_gb, 10),
            "size_gb": round(r.size_gb, 10),
            "nominal_bits": r.nominal_bits,
            "effective_bits": round(r.effective_bits, 10),
            "metadata_bits_per_weight": round(r.metadata_bits_per_weight, 10),
            "n_blocks": r.n_blocks,
            "weight_rel_error": round(r.weight_rel_error, 12),
            "weight_max_abs_error": round(r.weight_max_abs_error, 12),
            "feasible": within_budget(cfg, r.memory_gb, r.latency_ms),
        })
    return out


# --------------------------------------------------------------------------- #
# 2. 主结论表
# --------------------------------------------------------------------------- #


def quality_table(
    readings: Sequence[Reading], cfg: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """每个 `(precision, group, algorithm)` 一行，取该格上**质量最高**的候选。

    `(precision, group, algorithm)` 这一格上可能有多条候选（GPTQ / AWQ 有两个
    校准档位），取质量最高的那条 —— 于是这张表读的是「每个算法在每个位宽上
    **能拿到的最好结果**」，而不是某个特定校准档位的结果。
    校准档位本身的影响单独在 `calibration_sensitivity` 里看。
    """
    cells: dict[tuple[str, int, str], Reading] = {}
    for r in readings:
        k = (r.precision, r.group, r.algorithm)
        cur = cells.get(k)
        if cur is None or (r.quality, -r.effective_bits) > (cur.quality, -cur.effective_bits):
            cells[k] = r
    rows: list[dict[str, Any]] = []
    for (prec, grp, alg), r in cells.items():
        rows.append({
            "precision": prec,
            "group": grp,
            "algorithm": alg,
            "calib": r.calib,
            "spec_id": r.spec_id,
            "quality": round(r.quality, 10),
            "quality_loss": round(r.quality_loss, 10),
            "effective_bits": round(r.effective_bits, 10),
            "size_gb": round(r.size_gb, 10),
            "latency_ms": round(r.latency_ms, 10),
            "feasible": within_budget(cfg, r.memory_gb, r.latency_ms),
        })
    rows.sort(key=lambda d: (d["precision"], d["group"], d["algorithm"]))
    return rows


def pivot_quality(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """把 `quality_table` 的 72 行折成报告里那张「行 = 位宽×分组，列 = 算法」的表。"""
    algorithms = sorted({str(d["algorithm"]) for d in rows})
    keys = sorted({(str(d["precision"]), int(d["group"])) for d in rows})
    cells: dict[str, dict[str, Any]] = {}
    for d in rows:
        cells[f"{d['precision']}|g{d['group']}|{d['algorithm']}"] = d
    return {"algorithms": algorithms, "row_keys": keys, "cells": cells}


# --------------------------------------------------------------------------- #
# 3. 前沿
# --------------------------------------------------------------------------- #


def frontier_of(readings: Sequence[Reading]) -> list[Reading]:
    """四维非支配集。**排序钉死**（体积升序 → 延迟 → spec_id），
    于是「拐点下标」在两台机器上指的是同一个点。"""
    nd = pareto.non_dominated_of(list(readings), lambda r: r.vector())
    nd.sort(key=lambda r: (r.size_gb, r.latency_ms, r.spec_id))
    return nd


def reference_point(
    readings: Sequence[Reading], cfg: Mapping[str, Any]
) -> tuple[float, float, float, float]:
    """超体积参考点：**全部候选**（不只前沿）各维最差再外推 `hv_ref_margin`。

    必须被所有点支配 —— 否则会出现「多加一个烂点，超体积反而变大」。
    """
    margin = float(cfg["matrix"]["hv_ref_margin"])
    vecs = [r.vector() for r in readings]
    d = len(vecs[0])
    lo = tuple(min(v[k] for v in vecs) for k in range(d))
    hi = tuple(max(v[k] for v in vecs) for k in range(d))
    return tuple(lo[k] - margin * (hi[k] - lo[k]) for k in range(d))  # type: ignore[return-value]


def frontier_summary(
    frontier: Sequence[Reading], readings: Sequence[Reading], cfg: Mapping[str, Any]
) -> dict[str, Any]:
    """前沿的结构性读数。**每条结论都要能从这些数里读出来。**"""
    vecs = [r.vector() for r in frontier]
    ref = reference_point(readings, cfg)
    axes = tuple(int(a) for a in cfg["matrix"]["knee_axes"])
    ki = pareto.knee_index(vecs, axes) if vecs else 0
    return {
        "size": len(frontier),
        "spec_ids": [r.spec_id for r in frontier],
        "precisions": sorted({r.precision for r in frontier}),
        "algorithms": sorted({r.algorithm for r in frontier}),
        "groups": sorted({r.group for r in frontier}),
        "quality_min": _round(min((r.quality for r in frontier), default=0.0)),
        "quality_max": _round(max((r.quality for r in frontier), default=0.0)),
        "size_gb_min": _round(min((r.size_gb for r in frontier), default=0.0)),
        "size_gb_max": _round(max((r.size_gb for r in frontier), default=0.0)),
        "hypervolume": _round(pareto.hypervolume(vecs, ref), 10) if vecs else 0.0,
        "reference_point": [_round(v, 10) for v in ref],
        "spacing": _round(pareto.spacing(vecs), 10) if vecs else 0.0,
        "knee_axes": list(axes),
        "knee_index": ki,
        "knee_spec_id": frontier[ki].spec_id if frontier else None,
        "n_dominated": len(readings) - len(frontier),
    }


def frontier_delta_with_axis_swapped(readings: Sequence[Reading], i: int, j: int) -> int:
    """把第 `i` 维换成第 `j` 维之后，前沿集合的**对称差**大小。

    用来把「Memory 与 Size 共线」这件事从「一个相关系数」升级成
    「一条不变量」：`frontier_delta_with_axis_swapped(readings, 2, 3) == 0`
    的含义是「把 Memory 维删掉，前沿逐点不变」。
    """

    def vec_swapped(r: Reading) -> tuple[float, ...]:
        v = list(r.vector())
        v[i] = v[j]
        return tuple(v)

    a = {r.spec_id for r in pareto.non_dominated_of(list(readings), lambda r: r.vector())}
    b = {r.spec_id for r in pareto.non_dominated_of(list(readings), vec_swapped)}
    return len(a ^ b)


# --------------------------------------------------------------------------- #
# 4. 选型
# --------------------------------------------------------------------------- #


def _sel(r: Reading, rule: str, why: str, cfg: Mapping[str, Any]) -> Selection:
    return Selection(
        rule=rule,
        spec_id=r.spec_id,
        quality=r.quality,
        latency_ms=r.latency_ms,
        memory_gb=r.memory_gb,
        size_gb=r.size_gb,
        effective_bits=r.effective_bits,
        feasible=within_budget(cfg, r.memory_gb, r.latency_ms),
        why=why,
    )


def _empty(rule: str, why: str) -> Selection:
    return Selection(
        rule=rule, spec_id=None, quality=None, latency_ms=None, memory_gb=None,
        size_gb=None, effective_bits=None, feasible=False, why=why,
    )


def selections(
    readings: Sequence[Reading], frontier: Sequence[Reading], cfg: Mapping[str, Any]
) -> list[Selection]:
    """三条选型规则。**规则是声明的（写在这里），不是散在报告文字里的。**"""
    d = cfg["deployment"]
    budget = [r for r in readings if within_budget(cfg, r.memory_gb, r.latency_ms)]
    out: list[Selection] = []

    # ---- 规则 1：预算内质量最高
    if budget:
        best = min(budget, key=lambda r: (-r.quality, r.size_gb, r.spec_id))
        out.append(_sel(
            best, "max_quality_under_budget",
            f"在 ≤{d['memory_limit_gb']} GB / ≤{d['latency_limit_ms']} ms 的预算内"
            f"质量最高：{best.quality:.4f}（有效位宽 {best.effective_bits:.4f}，"
            f"体积 {best.size_gb:.4f} GB）",
            cfg,
        ))
    else:
        out.append(_empty(
            "max_quality_under_budget",
            f"没有任何候选满足 ≤{d['memory_limit_gb']} GB / ≤{d['latency_limit_ms']} ms",
        ))

    # ---- 规则 2：预算内「每 GB 质量」最高
    # 分子减掉**全局最差质量**（floor）。不减的话，「质量/体积」里那个 ~0.5 的
    # 常数项会随体积单调变化，于是这条规则退化成「永远选最小的那个」——
    # 看起来像个正经规则，其实是个常数。
    floor = min((r.quality for r in readings), default=0.0)
    if budget:
        def per_gb(r: Reading) -> tuple[float, float, str]:
            return (-((r.quality - floor) / r.size_gb), r.spec_id)

        best = min(budget, key=per_gb)
        ratio = (best.quality - floor) / best.size_gb
        out.append(_sel(
            best, "max_quality_per_gb_under_budget",
            f"预算内「每 GB 质量」最高：({best.quality:.4f} − {floor:.4f}) / "
            f"{best.size_gb:.4f} = {ratio:.6f} 每 GB（分子已减全局最差质量）",
            cfg,
        ))
    else:
        out.append(_empty("max_quality_per_gb_under_budget", "预算内无候选"))

    # ---- 规则 3：前沿拐点
    if frontier:
        axes = tuple(int(a) for a in cfg["matrix"]["knee_axes"])
        ki = pareto.knee_index([r.vector() for r in frontier], axes)
        r = frontier[ki]
        out.append(_sel(
            r, "knee_point",
            f"前沿在 (第{axes[0]}维, 第{axes[1]}维) 平面上的拐点（到两端连线垂直距离"
            f"最大）：质量 {r.quality:.4f}，体积 {r.size_gb:.4f} GB",
            cfg,
        ))
    else:
        out.append(_empty("knee_point", "前沿为空"))

    return out


# --------------------------------------------------------------------------- #
# 5. 两两 Diff
# --------------------------------------------------------------------------- #


def _cell_index(axis: str) -> int:
    try:
        return AXES.index(axis)
    except ValueError:
        raise ValueError(f"未知轴 {axis!r}，可选：{AXES}") from None


def match_axes(axis: str) -> tuple[str, ...]:
    """比较 `axis` 时要固定的**其余**各轴。**这里有一个会让整张表静默变空的坑。**

    ⚠️ 比较**算法**时必须把 `calib` 从匹配键里**排除**。

    理由：校准档位是**算法的属性**，不是与算法正交的一个轴 ——
    `rtn` / `kquant` 恒为 0（它们不读激活统计），`gptq` / `awq` 恒为正。
    如果按「其余各轴」机械地拼匹配键，`(precision, group, calib)` 里
    `calib=0` 的格只装得下 `rtn`/`kquant`，`calib=64` 的格只装得下
    `gptq`/`awq` —— 于是 **`gptq` 与 `rtn` 永远不会落进同一个格**，
    Diff 表里只会剩下 `awq→gptq` 与 `kquant→rtn` 两行，
    而「算法到底值不值它多花的元数据与校准成本」这个问题**一行都没有**。

    实测（本项目第一版）：`algorithm` 轴的 6 个两两组合里只有 2 个出得来。
    **它不报错、不警告，只是少了几行** —— 而少的那几行恰好是最该看的那几行。
    """
    if axis == "algorithm":
        return ("precision", "group")
    return tuple(a for a in AXES if a != axis)


def pairwise_diff(readings: Sequence[Reading], axis: str) -> list[dict[str, Any]]:
    """固定其余各轴（见 `match_axes`），只让 `axis` 变，然后**两两**比较。

    同一个（格, 轴值）上若有多条候选（例如算法轴上的两个校准档位），
    取**质量最高**的那条 —— 于是这条 Diff 读的是「每个取值能拿到的最好结果」，
    而不是某个特定校准档位的结果。

    `n_up / n_down / n_tie` 是**符号的分布**，不是装饰：
    平均差 +0.02 里如果一半格 +0.06、一半格 −0.02，那和「一致 +0.02」
    是两个完全不同的结论。只报均值会把它们抹平。
    """
    _cell_index(axis)  # 校验轴名
    others = match_axes(axis)
    cells: dict[tuple[Any, ...], dict[Any, Reading]] = {}
    for r in readings:
        k = tuple(getattr(r, a) for a in others)
        v = getattr(r, axis)
        cur = cells.setdefault(k, {}).get(v)
        if cur is None or r.quality > cur.quality:
            cells[k][v] = r

    values = sorted({v for d in cells.values() for v in d})
    rows: list[dict[str, Any]] = []
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            a, b = values[i], values[j]
            dq: list[float] = []
            dsz: list[float] = []
            dlat: list[float] = []
            dbit: list[float] = []
            up = down = tie = 0
            for d in cells.values():
                ra = d.get(a)
                rb = d.get(b)
                if ra is None or rb is None:
                    continue
                delta = rb.quality - ra.quality
                dq.append(delta)
                dsz.append(rb.size_gb - ra.size_gb)
                dlat.append(rb.latency_ms - ra.latency_ms)
                dbit.append(rb.effective_bits - ra.effective_bits)
                if delta > 0.0:
                    up += 1
                elif delta < 0.0:
                    down += 1
                else:
                    tie += 1
            if not dq:
                continue
            rows.append({
                "axis": axis,
                "left": a,
                "right": b,
                "n_cells": len(dq),
                "mean_delta_quality": round(sum(dq) / len(dq), 10),
                "min_delta_quality": round(min(dq), 10),
                "max_delta_quality": round(max(dq), 10),
                "n_up": up,
                "n_down": down,
                "n_tie": tie,
                "mean_delta_size_gb": round(sum(dsz) / len(dsz), 10),
                "mean_delta_latency_ms": round(sum(dlat) / len(dlat), 10),
                "mean_delta_effective_bits": round(sum(dbit) / len(dbit), 10),
            })
    return rows


def diff_rows(
    readings: Sequence[Reading], cfg: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """把 `matrix.diff_axes` 里每个轴都过一遍。"""
    out: list[dict[str, Any]] = []
    for axis in cfg["matrix"]["diff_axes"]:
        out.extend(pairwise_diff(readings, axis))
    return out


# --------------------------------------------------------------------------- #
# 6. 校准规模敏感性
# --------------------------------------------------------------------------- #


def calibration_sensitivity(
    readings: Sequence[Reading], cfg: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """校准档位之间的差距，**拆成两列**：下游质量 vs 权重误差。

    为什么必须拆：

    * `delta_weight_rel_error` 是**机制**上的读数 —— 量化器真的把权重贴得更准了吗？
    * `delta_quality` 是**下游**读数 —— 贴得更准有没有变成更好的分类？

    只报后者，会把「校准集过拟合」误读成「校准没用」（权重误差降了、质量没动）；
    只报前者，会把「权重误差与质量脱钩」这件事藏起来（本项目 GPTQ 的实测就是这样：
    补偿把权重误差压下去，质量却没动 —— 因为逐组 scale 已经很细，
    权重误差已经不是质量的主导项了）。

    另一个必然为 0 的列：`delta_effective_bits`。**校准档位不该改变位宽** ——
    它非 0 就说明校准统计被误接进了位宽计算。这是一条便宜的哨兵。
    """
    levels = {
        str(a): sorted(int(x) for x in v)
        for a, v in cfg["space"]["calib_sizes"].items()
    }
    by_key: dict[tuple[str, str, int], dict[int, Reading]] = {}
    for r in readings:
        if len(levels.get(r.algorithm, [])) < 2:
            continue
        by_key.setdefault((r.algorithm, r.precision, r.group), {})[r.calib] = r

    rows: list[dict[str, Any]] = []
    for (alg, prec, grp), d in sorted(by_key.items()):
        lv = [x for x in levels[alg] if x in d]
        for i in range(len(lv) - 1):
            a, b = lv[i], lv[i + 1]
            ra, rb = d[a], d[b]
            rows.append({
                "algorithm": alg,
                "precision": prec,
                "group": grp,
                "calib_left": a,
                "calib_right": b,
                "quality_left": round(ra.quality, 10),
                "quality_right": round(rb.quality, 10),
                "delta_quality": round(rb.quality - ra.quality, 10),
                "rel_error_left": round(ra.weight_rel_error, 12),
                "rel_error_right": round(rb.weight_rel_error, 12),
                "delta_weight_rel_error": round(
                    rb.weight_rel_error - ra.weight_rel_error, 12
                ),
                "delta_effective_bits": round(
                    rb.effective_bits - ra.effective_bits, 10
                ),
            })
    return rows


def sensitivity_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """按算法聚合敏感性：平均质量差、平均权重误差差、以及**符号一致性**。"""
    by_alg: dict[str, list[Mapping[str, Any]]] = {}
    for d in rows:
        by_alg.setdefault(str(d["algorithm"]), []).append(d)
    out: list[dict[str, Any]] = []
    for alg in sorted(by_alg):
        rs = by_alg[alg]
        n = len(rs)
        dq = [float(d["delta_quality"]) for d in rs]
        dr = [float(d["delta_weight_rel_error"]) for d in rs]
        out.append({
            "algorithm": alg,
            "cells": n,
            "mean_delta_quality": round(sum(dq) / n, 10),
            "n_quality_up": sum(1 for v in dq if v > 0.0),
            "n_quality_down": sum(1 for v in dq if v < 0.0),
            "n_quality_tie": sum(1 for v in dq if v == 0.0),
            "mean_delta_weight_rel_error": round(sum(dr) / n, 12),
            "n_relerr_down": sum(1 for v in dr if v < 0.0),
        })
    return out
