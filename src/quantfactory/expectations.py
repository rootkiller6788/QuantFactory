"""设计期望：15 条「我以为会怎样」。**允许被推翻，被推翻的那条最值钱。**

## 一条期望必须同时有三样东西

| 字段 | 作用 |
| --- | --- |
| `statement` | 自然语言的期望（人读的） |
| `operationalization` | **判据是怎么被操作化的**（机器读的） |
| `reading` | 实测读数（**每个数字都是从本次运行的数据生成的，一个都不许写死**） |

第二样是必须的：**「A 比 B 好」不是一个可判定的命题**，除非说清在哪一档、
比的是均值还是逐格、允许多大容差。同一条自然语言期望，换一种操作化就能换一个结论 ——
所以操作化必须写在报告里，读者才能判断这个「被推翻」是不是他自己的那个问题。

## 为什么报告里的数字不许写死

`L4-rewardfactory` 踩过三次：报告 note 里出现过硬编码的「1.13 倍」「correctness 占 58%」——
改一次配置这三句就全成谎话，而**报告里的谎话比缺一句话危险得多**（读者没法从报告内部发现它）。
所以这里的每一个数字都是插值出来的。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .acceptance import binomial_se
from .matrix import by_spec_id, pivot_quality, quality_table
from .schema import Reading

__all__ = ["Expectation", "run_expectations", "expectation_summary"]


@dataclass
class Expectation:
    name: str
    statement: str
    operationalization: str
    expected: bool
    observed: bool
    reading: str

    @property
    def holds(self) -> bool:
        return self.expected == self.observed

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "statement": self.statement,
            "operationalization": self.operationalization,
            "expected": bool(self.expected),
            "observed": bool(self.observed),
            "holds": bool(self.holds),
            "reading": self.reading,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Expectation":
        """从 `expectations.json` 重建（`visualize` 的产物重建路径需要）。"""
        expected = bool(d["expected"])
        observed = bool(d["observed"])
        return cls(
            name=str(d["name"]),
            statement=str(d.get("statement", "")),
            operationalization=str(d.get("operationalization", "")),
            expected=expected,
            observed=observed,
            reading=str(d.get("reading", "")),
        )


def expectation_summary(items: Sequence[Expectation]) -> dict[str, Any]:
    refuted = [e for e in items if not e.holds]
    return {
        "n": len(items),
        "n_holds": len(items) - len(refuted),
        "n_refuted": len(refuted),
        "refuted": [e.name for e in refuted],
        "note": (
            "**被推翻的期望不是失败，是产出。** 每一条都带着它的操作化一起读 —— "
            "同一条自然语言期望换一种操作化就可能换一个结论。"
        ),
    }


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #


def _best(rs: Sequence[Reading], **flt: Any) -> Reading | None:
    sel = [r for r in rs if all(getattr(r, k) == v for k, v in flt.items())]
    if not sel:
        return None
    return max(sel, key=lambda r: (r.quality, -r.effective_bits))


def _q(rs: Sequence[Reading], **flt: Any) -> float | None:
    r = _best(rs, **flt)
    return None if r is None else r.quality


def _diff(rows: Sequence[Mapping[str, Any]], axis: str, left: Any, right: Any):
    for d in rows:
        if d["axis"] == axis and d["left"] == left and d["right"] == right:
            return d
    return None


def _fmt(x: float | None, nd: int = 4) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


# --------------------------------------------------------------------------- #
# 主入口
# --------------------------------------------------------------------------- #


def run_expectations(result: Any) -> list[Expectation]:
    cfg = result.cfg
    exp = cfg["expectations"]
    rs = result.readings
    fp16 = result.fp16
    n_test = len(result.world.test_x)
    se = binomial_se(fp16.quality, n_test)
    diffs = result.diff_rows
    algs = list(cfg["space"]["algorithms"])
    groups = list(cfg["space"]["groups"])
    precs = list(cfg["space"]["precisions"])
    pv = pivot_quality(quality_table(rs, cfg))
    out: list[Expectation] = []

    def cell(prec: str, grp: int, alg: str) -> Mapping[str, Any] | None:
        return pv["cells"].get(f"{prec}|g{grp}|{alg}")

    def add(name: str, statement: str, op: str, observed: bool, reading: str) -> None:
        out.append(Expectation(
            name=name, statement=statement, operationalization=op,
            expected=bool(exp[name]), observed=bool(observed), reading=reading,
        ))

    # ---- 1
    b2, b4, b8 = _best(rs, precision="int2"), _best(rs, precision="int4"), _best(rs, precision="int8")
    d28 = _diff(diffs, "precision", "int2", "int8")
    obs = (
        b2 is not None and b4 is not None and b8 is not None
        and b2.quality < b4.quality < b8.quality
        and d28 is not None and d28["n_down"] == 0
    )
    tail = (
        f"int2→int8 的 {d28['n_cells']} 格：升 {d28['n_up']} / 降 {d28['n_down']} / 平 {d28['n_tie']}"
        if d28 else "缺少 int2→int8 的匹配格"
    )
    add(
        "lower_bits_lose_more_quality",
        "位宽越低，质量掉得越多",
        "① 跨分组的**最好质量**必须满足 q(int2) < q(int4) < q(int8)；"
        "② `int2 → int8` 的匹配格里，没有任何一格质量下降",
        obs,
        f"q(int2)={_fmt(b2.quality if b2 else None)} < q(int4)={_fmt(b4.quality if b4 else None)}"
        f" < q(int8)={_fmt(b8.quality if b8 else None)}；{tail}",
    )

    # ---- 2
    wins = []
    for g in groups:
        qg, qr = cell("int2", g, "gptq"), cell("int2", g, "rtn")
        if qg and qr:
            wins.append((g, qg["quality"], qr["quality"]))
    obs = bool(wins) and all(a >= b for _, a, b in wins)
    add(
        "gptq_beats_rtn_at_low_bits",
        "在最低位宽（int2）上，GPTQ 的误差补偿能胜过 RTN",
        "int2 的**每一个分组**上，`q(gptq) ≥ q(rtn)`（不劣，允许平局）",
        obs,
        "；".join(f"g{g}: gptq {a:.4f} vs rtn {b:.4f}（{a - b:+.4f}）" for g, a, b in wins)
        + f" ⇒ {sum(1 for _, a, b in wins if a >= b)}/{len(wins)} 个分组成立",
    )

    # ---- 3
    awq_rows = []
    for g in groups:
        qa, qg = cell("int2", g, "awq"), cell("int2", g, "gptq")
        if qa and qg:
            awq_rows.append((g, qa["quality"], qg["quality"]))
    obs = bool(awq_rows) and all(a > b for _, a, b in awq_rows)
    add(
        "awq_beats_gptq_at_int2",
        "AWQ 的激活感知缩放比 GPTQ 的误差补偿更适合 int2",
        "int2 的**每一个分组**上，`q(awq) > q(gptq)`",
        obs,
        "；".join(f"g{g}: awq {a:.4f} vs gptq {b:.4f}（{a - b:+.4f}）" for g, a, b in awq_rows)
        + f" ⇒ {sum(1 for _, a, b in awq_rows if a > b)}/{len(awq_rows)} 个分组成立",
    )

    # ---- 4
    meta_rows = []
    for p in precs:
        for g in groups:
            vals = {}
            for a in algs:
                r = _best(rs, algorithm=a, precision=p, group=g)
                if r:
                    vals[a] = r.metadata_bits_per_weight
            if vals:
                meta_rows.append((p, g, vals))
    obs = bool(meta_rows) and all(
        v["kquant"] <= min(x for k, x in v.items() if k != "kquant") + 1e-12
        for _, _, v in meta_rows
    )
    worst = max(
        (v["kquant"] - min(x for k, x in v.items() if k != "kquant") for _, _, v in meta_rows),
        default=0.0,
    )
    sample = next((v for p, g, v in meta_rows if p == "int4" and g == 32), {})
    add(
        "kquant_has_the_smallest_metadata",
        "K-quant 的二级 scale 方案让它每个权重分摊的元数据最少",
        "在**每一个** (位宽, 分组) 上，`metadata(kquant)` 都不大于其它算法的最小值",
        obs,
        f"（int4,g32）kquant={sample.get('kquant', 0):.4f} vs "
        + " / ".join(f"{a}={v:.4f}" for a, v in sorted(sample.items()) if a != "kquant")
        + f"；全部 {len(meta_rows)} 个格子里最大超出 {worst:+.6f}",
    )

    # ---- 5
    gaps = []
    for g in groups:
        row = []
        for p in ("int2", "int8"):
            vals = [cell(p, g, a)["quality"] for a in algs if cell(p, g, a)]
            row.append(max(vals) - min(vals) if vals else None)
        if None not in row:
            gaps.append((g, row[0], row[1]))
    obs = bool(gaps) and all(a > b for _, a, b in gaps)
    add(
        "algorithm_gap_widens_at_low_bits",
        "位宽越低，「选哪个算法」越重要",
        "在**每一个**分组上，`max−min` 的算法间质量极差满足 spread(int2) > spread(int8)",
        obs,
        "；".join(f"g{g}: int2 {a:.4f} > int8 {b:.4f}" for g, a, b in gaps),
    )

    # ---- 6
    sens = result.sensitivity_summary
    obs = bool(sens) and all(s["mean_delta_quality"] > 0.0 for s in sens)
    add(
        "more_calibration_helps",
        "校准集从 64 加到 256，质量会变好",
        "每个**需要校准的**算法，其 `Δquality(c256 − c64)` 的**均值 > 0**"
        "（只看均值，不看符号一致性 —— 符号分布单独在报告里列）",
        obs,
        "；".join(
            f"{s['algorithm']}: 均值 {s['mean_delta_quality']:+.4f}"
            f"（升 {s['n_quality_up']} / 降 {s['n_quality_down']} / 平 {s['n_quality_tie']}）"
            for s in sens
        ),
    )

    # ---- 7
    d1664 = _diff(diffs, "group", 16, 64)
    obs = d1664 is not None and d1664["mean_delta_quality"] < 0.0
    add(
        "smaller_group_improves_quality",
        "分组越小，质量越好（组内共享的 scale 更贴合）",
        "`Δquality(g64 − g16)` 在 36 个匹配格上的**均值 < 0**",
        obs,
        f"均值 {d1664['mean_delta_quality']:+.4f}"
        f"（升 {d1664['n_up']} / 降 {d1664['n_down']} / 平 {d1664['n_tie']}）"
        f"，区间 [{d1664['min_delta_quality']:+.4f}, {d1664['max_delta_quality']:+.4f}]"
        if d1664 else "缺少 group 16→64 的匹配格",
    )

    # ---- 8
    obs = d1664 is not None and d1664["mean_delta_latency_ms"] < 0.0
    add(
        "smaller_group_costs_latency",
        "分组越小，延迟越高（块数变多，反量化开销上升）",
        "`Δlatency(g64 − g16)` 在 36 个匹配格上的**均值 < 0**",
        obs,
        f"均值 {d1664['mean_delta_latency_ms']:+.4f} ms（同时体积 {d1664['mean_delta_size_gb']:+.4f} GB）"
        if d1664 else "缺少 group 16→64 的匹配格",
    )

    # ---- 9
    c = result.corr_memory_size
    obs = c < 1.0 - 1e-12 and result.memory_axis_frontier_delta > 0
    add(
        "memory_is_an_independent_axis",
        "Memory（峰值显存）与 Size（磁盘体积）是两个独立的维度",
        "① `corr(memory, size) < 1`；② 把 Memory 维换成 Size 维后，"
        "非支配前沿的**对称差 > 0**（也就是前沿真的变了）",
        obs,
        f"corr = {c:.12f}；前沿对称差 = {result.memory_axis_frontier_delta}"
        f"（memory − size 恒等于 {result.memory_overhead_gb:.8f} GB）",
    )

    # ---- 10
    knee = by_spec_id(rs).get(result.frontier_summary["knee_spec_id"] or "")
    best_budget = next(
        (s for s in result.selections if s.rule == "max_quality_under_budget"), None
    )
    gap = (
        (best_budget.quality - knee.quality)
        if knee is not None and best_budget is not None and best_budget.quality is not None
        else None
    )
    obs = (
        knee is not None and gap is not None and knee.memory_gb <= float(cfg["deployment"]["memory_limit_gb"])
        and gap <= 1.5 * se
    )
    add(
        "knee_point_is_the_best_deployment_choice",
        "前沿拐点就是最好的部署选择（质量与体积的折中）",
        "拐点① 在部署预算内；② 它的质量与「预算内最高质量」的差 ≤ **1.5 个二项标准误**"
        f"（1.5·SE = {1.5 * se:.4f}）—— 「最好」被操作化成「统计上不可区分」",
        obs,
        f"拐点 {result.frontier_summary['knee_spec_id']} q={_fmt(knee.quality if knee else None)}"
        f" size={_fmt(knee.size_gb if knee else None, 4)} GB；"
        f"预算内最高 {best_budget.spec_id if best_budget else '-'} "
        f"q={_fmt(best_budget.quality if best_budget else None)}；差 {_fmt(gap)} "
        f"= {gap / se:.2f} SE（SE={se:.6f}）" if gap is not None else "缺拐点或选型结果",
    )

    # ---- 11
    b8 = _best(rs, precision="int8")
    qgap = abs(b8.quality - fp16.quality) if b8 else None
    ratio = (b8.size_gb / fp16.size_gb) if b8 else None
    obs = qgap is not None and ratio is not None and qgap <= se and ratio <= 0.6
    add(
        "int8_is_almost_free",
        "INT8 几乎是免费的（质量与 fp16 无差别，体积小一半以上）",
        "① `|q(int8) − q(fp16)| ≤ 1 SE`；② `size(int8) / size(fp16) ≤ 0.6`。"
        "**注意这里只判「质量」与「体积」——它仍然可能装不进显存**",
        obs,
        f"q(int8)={_fmt(b8.quality if b8 else None)} vs fp16={_fmt(fp16.quality)}"
        f"（差 {_fmt(qgap)} = {qgap / se:.2f} SE）；"
        f"体积 {_fmt(b8.size_gb if b8 else None, 4)} / {fp16.size_gb:.4f} = {ratio:.4f}"
        if qgap is not None and ratio is not None else "缺少 int8 候选",
    )

    # ---- 12
    b2 = _best(rs, precision="int2")
    b4 = _best(rs, precision="int4")
    drop = (b4.quality - b2.quality) if (b2 and b4) else None
    obs = drop is not None and drop > 0.03
    add(
        "int2_is_unusable",
        "INT2 不可用（掉点太多）",
        "`q(int4) − q(int2) > 0.03`（3 个百分点的绝对准确率），"
        "比的是**跨分组的最好结果**",
        obs,
        f"int4 最好 {_fmt(b4.quality if b4 else None)} − int2 最好 {_fmt(b2.quality if b2 else None)}"
        f" = {_fmt(drop)}"
        + (
            f"；而 int2 只省下 {(1 - b2.size_gb / b4.size_gb) * 100:.1f}% 体积"
            f"（{b2.size_gb:.4f} vs {b4.size_gb:.4f} GB）"
            if b2 and b4 else ""
        ),
    )

    # ---- 13
    used = sorted({r.algorithm for r in result.frontier})
    obs = len(used) == len(algs)
    add(
        "frontier_uses_every_algorithm",
        "每个算法都在前沿上占有一席之地（没有算法被完全支配）",
        "前沿上的算法集合大小 == `space.algorithms` 的长度",
        obs,
        f"前沿 {len(result.frontier)} 个点，用到算法 {used}（配置里有 {len(algs)} 个）",
    )

    # ---- 14
    free_best = max((r.quality for r in rs if r.algorithm in ("rtn", "kquant")), default=None)
    need_best = max((r.quality for r in rs if r.algorithm in ("gptq", "awq")), default=None)
    obs = (
        free_best is not None and need_best is not None and free_best >= need_best - se
    )
    add(
        "calibration_free_algorithms_are_competitive",
        "不需要校准的算法（RTN / K-quant）与需要校准的（GPTQ / AWQ）打得有来有回",
        "`best(rtn ∪ kquant) ≥ best(gptq ∪ awq) − 1 SE`",
        obs,
        f"校准无关 {_fmt(free_best)} vs 需要校准 {_fmt(need_best)}"
        f"（差 {_fmt((free_best - need_best) if free_best is not None and need_best is not None else None)}"
        f" = {(free_best - need_best) / se:+.2f} SE）"
        if free_best is not None and need_best is not None else "缺少候选",
    )

    # ---- 15
    worst = []
    for g in groups:
        for p in precs:
            r = _best(rs, algorithm="rtn", precision=p, group=g)
            if r:
                worst.append((p, g, r.metadata_bits_per_weight, r.nominal_bits))
    obs = any(m >= n - 1e-12 for _, _, m, n in worst)
    hit = max(worst, key=lambda t: t[2] / t[3]) if worst else None
    add(
        "metadata_dominates_at_small_groups",
        "分组小到一定程度，元数据比权重本身还占地方",
        "存在某个 (位宽, 分组) 使 `metadata_bits_per_weight ≥ nominal_bits`（以 rtn 为准，"
        "它的元数据最直白：每组一个 scale + 一个 zero）",
        obs,
        f"最极端的是 rtn/{hit[0]}/g{hit[1]}：元数据 {hit[2]:.4f} 位 vs 名义位宽 {hit[3]:.1f} 位"
        f"（占 {hit[2] / hit[3] * 100:.1f}%）" if hit else "缺少 rtn 候选",
    )

    return out
