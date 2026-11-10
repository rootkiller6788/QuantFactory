"""性能读数：每一段花了多少时间。

## 为什么这个文件单独存在

「30 秒能跑完」不是一句感觉，而是一条**可核对的读数**。
量化型项目特别容易在这件事上自欺：一个算法的误差补偿如果慢 10 倍，
「它没比 RTN 好多少」就可能只是因为它没跑完 —— 而这件事在结果表里看不出来。

所以 `benchmark` 记两件事：
1. **分段耗时**（世界 / 建模 / 校准 / 量化 / 度量 / 前沿与选型 / Diff 与敏感性）
2. **每个算法的累计耗时与平均每个候选的耗时**

## 一条纪律

**耗时只进 `benchmark.json`，不进 `report.md` 的正文数字里。**
理由与 `created_at` 一样：耗时依赖机器，印进正文会让「两份相同输入的报告
逐位相同」这条核对失效。报告里可以引用它，但必须标明「本机实测、随机器变化」。
"""

from __future__ import annotations

from typing import Any, Sequence

__all__ = ["benchmark"]

# 分段的顺序就是流水线的执行顺序。**报告里也按这个顺序列**，
# 否则读者没法把「哪一段慢」和「哪一段做了什么事」对上。
STAGE_ORDER = (
    "world",
    "model",
    "calibration",
    "quantize",
    "measure",
    "frontier_and_selection",
    "diff_and_sensitivity",
)

STAGE_NOTE = {
    "world": "造合成任务 + 训练/测试/校准三块**互不重叠**的分片（每块各自 fork 一条 RNG）",
    "model": "训基座 MLP（手写 SGD，math.sumprod 内积）—— 全部量化候选共用这一个 fp16 模型",
    "calibration": "校准集前向 → Hessian 对角 + 逐通道激活幅度。**这一份被全部候选共享**，只算一次",
    "quantize": "枚举整个设计空间，每个候选真跑一遍量化（RTN / GPTQ / AWQ / K-quant）",
    "measure": "四维读数：Quality / Latency / Memory / Size（含名义位宽 vs 有效位宽）",
    "frontier_and_selection": "四维非支配集 + 超体积 + 拐点 + 三条选型规则",
    "diff_and_sensitivity": "同预算两两 Diff + 校准规模敏感性表（拆成两列差距）",
}


def benchmark(result: Any) -> dict[str, Any]:
    timings = dict(getattr(result, "timings", {}))
    stages = [
        {
            "stage": name,
            "seconds": round(float(timings.get(name, 0.0)), 4),
            "note": STAGE_NOTE.get(name, ""),
        }
        for name in STAGE_ORDER
    ]
    accounted = sum(s["seconds"] for s in stages)
    total = float(getattr(result, "elapsed_s", accounted))

    return {
        "stages": stages,
        "accounted_seconds": round(accounted, 4),
        "total_seconds": round(total, 4),
        "unaccounted_seconds": round(max(0.0, total - accounted), 4),
        "by_algorithm": _per_algorithm(result),
        "counts": {
            "candidates": len(result.readings),
            "distinct_specs": result.space_meta["n_distinct_specs"],
            "evaluations": result.space_meta["n_evaluations"],
            "frontier": len(result.frontier),
        },
        "note": "耗时**随机器变化**，所以它只进本文件、不进 report 正文的数字里",
    }


def _per_algorithm(result: Any) -> list[dict[str, Any]]:
    """按算法聚合：候选数、平均每个候选的耗时、平均有效位宽。

    ⚠️ 这里的「耗时」是**可靠**的（每个候选都真的跑了一遍量化 + 一次全测试集前向），
    不像搜索型项目那样被缓存吃掉。所以它可以直接用来横向比算法的开销。
    """
    names: Sequence[str] = result.cfg["space"]["algorithms"]
    out: list[dict[str, Any]] = []
    for name in names:
        rs = [r for r in result.readings if r.spec.algorithm == name]
        if not rs:
            out.append({"algorithm": name, "candidates": 0})
            continue
        n = len(rs)
        out.append({
            "algorithm": name,
            "candidates": n,
            "candidate_seconds": round(sum(r.candidate_s for r in rs), 4),
            "mean_candidate_ms": round(1000.0 * sum(r.candidate_s for r in rs) / n, 4),
            "mean_effective_bits": round(sum(r.effective_bits for r in rs) / n, 6),
            "mean_metadata_bits_per_weight": round(
                sum(r.metadata_bits_per_weight for r in rs) / n, 6
            ),
        })
    return out
