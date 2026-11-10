"""流水线：把各模块串成一次运行，并定义**产物契约**。

## 产物的两条分类

| 类别 | 文件 | 纪律 |
| --- | --- | --- |
| **版本化**（`VERSIONED_FILES`） | 其余全部 | 同一份配置跑两次必须**逐位相同** |
| **豁免**（`UNVERSIONED_FILES`） | `version.json`、`benchmark.json` | 允许不同 |

豁免的原因各一个：

* `version.json` 含 `created_at`。**这是全项目唯一允许出现 `created_at` 的地方** ——
  它印进 `report.md` 正文，两份相同输入的报告就不逐位相同了。
* `benchmark.json` 含耗时。耗时随机器变化，同样不能进正文（见 `benchmark.py`）。

同理，`Reading.candidate_s`（每候选的墙钟时间）**不进 `to_dict()`** ——
否则 `readings.json` 会带上耗时，两次运行就差在那几个数字上。

## 只读路径

`pipeline.run_pipeline` 是唯一的真跑路径。只读命令走 `products_of` 读盘，
**绝不写盘**；产物缺失时在内存里重跑并明说「本次是内存重跑，未写盘」。

## 一处必须钉住的共享

`build_calib` 按**档位**算一次，全部候选共用同一个 `CalibStats` 对象。
这里用 `stats_identity_ok` 把「对象同一性」记进产物 —— 只写文档是守不住的：
某天有人把 `build_calib` 挪进候选循环，**结果看起来完全一样**，只是慢了几百倍。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from . import matrix
from .acceptance import AcceptanceReport, run_acceptance
from .benchmark import benchmark
from .calib import CalibStats, build_calib
from .config import enumerate_specs, protocol_digest, version_digest
from .emit import csv_text, dump_json, write_outputs
from .expectations import Expectation, expectation_summary, run_expectations
from .lineage import to_dot
from .measure import memory_overhead_gb, measure_fp16, measure_quantized
from .model import Layer, Model
from .quantize import n_blocks_of, quantize
from .schema import Reading, Selection
from .train import TrainStats, train_model
from .world import World, build_world

__all__ = [
    "OUT_FILES",
    "VERSIONED_FILES",
    "UNVERSIONED_FILES",
    "SELECTION_COLUMNS",
    "QuantRun",
    "run_pipeline",
    "products_of",
    "write_products",
    "read_products",
    "missing_products",
    "dump_default_config",
]

SELECTION_COLUMNS: tuple[str, ...] = (
    "rule", "spec_id", "quality", "latency_ms", "memory_gb", "size_gb",
    "effective_bits", "feasible", "why",
)

# 产物清单。**顺序即报告的阅读顺序。**
OUT_FILES: tuple[str, ...] = (
    "config.json",          # 有效配置（= DEFAULTS ⊕ 覆盖）
    "world.json",           # 世界读数
    "train.json",           # 训练读数
    "calibration.json",     # 校准统计（按档位）
    "readings.json",        # 全部候选的四维读数
    "matrix.csv",           # 每个候选一行（最细的一张）
    "quality_table.csv",    # 主结论表（行=位宽×分组，列=算法）
    "frontier.csv",         # 非支配集
    "selection.csv",        # 三条选型规则
    "diff.csv",             # 两两 Diff（匹配格）
    "sensitivity.csv",      # 校准规模敏感性
    "acceptance.json",      # 门槛 + 对抗用例
    "expectations.json",    # 15 条设计期望
    "summary.json",         # 一页结论（报告的数字来源）
    "lineage.dot",          # 血缘图
    "report.md",            # 报告
    "report.html",          # 可视化
    "version.json",         # 含 created_at（豁免）
    "benchmark.json",       # 含耗时（豁免）
)

UNVERSIONED_FILES: tuple[str, ...] = ("version.json", "benchmark.json")
VERSIONED_FILES: tuple[str, ...] = tuple(f for f in OUT_FILES if f not in UNVERSIONED_FILES)


@dataclass
class QuantRun:
    cfg: dict[str, Any]
    protocol_digest: str
    version_digest: str
    world: World
    train_stats: TrainStats
    # `world_d` / `train_d` 是 `world.json` / `train.json` 的**内容本身**。
    # 渲染层（`report.py`）读这两个字典，**不读上面的对象** ——
    # 于是「真跑一次」与「从产物重建」两条路径喂给渲染器的是同一样东西，
    # 没有第二条会漂的路径。
    world_d: dict[str, Any]
    train_d: dict[str, Any]
    model: Model
    model_snapshot: Model
    stats: dict[int, CalibStats]
    stats_identity_ok: bool
    nref: int
    fp16: Reading
    readings: list[Reading]
    quantized_models: dict[str, Model]
    frontier: list[Reading]
    frontier_summary: dict[str, Any]
    reference_point: tuple[float, ...]
    quality_rows: list[dict[str, Any]]
    diff_rows: list[dict[str, Any]]
    sensitivity_rows: list[dict[str, Any]]
    sensitivity_summary: list[dict[str, Any]]
    selections: list[Selection]
    acceptance: AcceptanceReport
    expectations: list[Expectation]
    corr_memory_size: float
    memory_overhead_gb: float
    memory_axis_frontier_delta: int
    # 下面两个字段的唯一用途是**让 `render_html` 不必碰 world / stats 对象** ——
    # 于是 HTML 可以纯粹从产物重建（`reconstruct_run`），`visualize` 不必全跑一遍。
    n_test: int = 0
    calib_levels: tuple[int, ...] = ()
    timings: dict[str, float] = field(default_factory=dict)
    elapsed_s: float = 0.0
    products: dict[str, Any] = field(default_factory=dict)

    @property
    def space_meta(self) -> dict[str, int]:
        return {
            "n_distinct_specs": len({r.spec_id for r in self.readings}),
            "n_evaluations": len(self.readings),
        }


def run_pipeline(cfg: Mapping[str, Any]) -> QuantRun:
    """真跑一遍。**这是唯一会做计算的入口。**"""
    t_all = time.perf_counter()
    timings: dict[str, float] = {}

    # ---- 1) 世界
    t0 = time.perf_counter()
    world = build_world(cfg)
    timings["world"] = time.perf_counter() - t0

    # ---- 2) 基座
    t0 = time.perf_counter()
    model, tstats = train_model(world, cfg)
    timings["model"] = time.perf_counter() - t0
    # 快照：验收要断言「量化没有就地改基座」。**必须在这里取** ——
    # 等候选跑完再取，快照本身可能已经是脏的。
    snapshot = Model(layers=[
        Layer(role=l.role, W=[row[:] for row in l.W], b=l.b[:], activation=l.activation)
        for l in model.layers
    ])

    # ---- 3) 校准（按档位，各一次）
    t0 = time.perf_counter()
    levels = sorted({
        int(c)
        for a in cfg["space"]["algorithms"]
        for c in cfg["space"]["calib_sizes"][a]
        if int(c) > 0
    })
    stats: dict[int, CalibStats] = {}
    for n in levels:
        stats[n] = build_calib(model, world, cfg, n)
    timings["calibration"] = time.perf_counter() - t0

    nref = n_blocks_of(model, int(cfg["runtime"]["base_group_for_dequant"]))

    # ---- 4) 基线
    fp16 = measure_fp16(world, cfg, model, n_blocks_ref=nref)

    # ---- 5) 枚举 + 量化 + 度量
    specs = enumerate_specs(cfg)
    readings: list[Reading] = []
    models: dict[str, Model] = {}
    # 「每个档位只算一次」的**对象同一性**：候选循环里每个档位只许见到一个对象。
    # 写成 `stats[n] is stats[n]` 是恒真式（一个东西当然是它自己）——
    # 一条从不触发的保险丝与没有保险丝无法区分。必须从**候选循环内部**观测。
    seen_objects: dict[int, set[int]] = {}
    t_quant = 0.0
    t_meas = 0.0
    for spec in specs:
        st = stats.get(spec.calib) if spec.calib > 0 else None
        if st is not None:
            seen_objects.setdefault(spec.calib, set()).add(id(st))
        t0 = time.perf_counter()
        qres = quantize(model, spec, cfg, st)
        qt = time.perf_counter() - t0
        t_quant += qt
        t0 = time.perf_counter()
        r = measure_quantized(
            world, cfg, spec, qres,
            baseline_quality=fp16.quality, n_blocks_ref=nref,
            quantize_s=qt,
        )
        t_meas += time.perf_counter() - t0
        readings.append(r)
        models[spec.spec_id] = qres.model
    timings["quantize"] = t_quant
    timings["measure"] = t_meas
    stats_identity_ok = (
        len(seen_objects) == len(levels)
        and all(len(ids) == 1 for ids in seen_objects.values())
    )

    # ---- 6) 前沿与选型
    t0 = time.perf_counter()
    frontier = matrix.frontier_of(readings)
    fsummary = matrix.frontier_summary(frontier, readings, cfg)
    ref = matrix.reference_point(readings, cfg)
    selections = matrix.selections(readings, frontier, cfg)
    timings["frontier_and_selection"] = time.perf_counter() - t0

    # ---- 7) Diff 与敏感性
    t0 = time.perf_counter()
    quality_rows = matrix.quality_table(readings, cfg)
    diff_rows = matrix.diff_rows(readings, cfg)
    sens_rows = matrix.calibration_sensitivity(readings, cfg)
    sens_summary = matrix.sensitivity_summary(sens_rows)
    timings["diff_and_sensitivity"] = time.perf_counter() - t0

    corr = matrix.corr([r.memory_gb for r in readings], [r.size_gb for r in readings])
    delta = matrix.frontier_delta_with_axis_swapped(readings, 2, 3)

    run = QuantRun(
        cfg=dict(cfg),
        protocol_digest=protocol_digest(cfg),
        version_digest=version_digest(cfg),
        world=world, train_stats=tstats,
        world_d=world.to_dict(), train_d=tstats.to_dict(),
        model=model, model_snapshot=snapshot,
        stats=stats, stats_identity_ok=stats_identity_ok, nref=nref,
        fp16=fp16, readings=readings, quantized_models=models,
        frontier=frontier, frontier_summary=fsummary, reference_point=ref,
        quality_rows=quality_rows, diff_rows=diff_rows,
        sensitivity_rows=sens_rows, sensitivity_summary=sens_summary,
        selections=selections,
        acceptance=AcceptanceReport(), expectations=[],
        corr_memory_size=corr, memory_overhead_gb=memory_overhead_gb(cfg),
        memory_axis_frontier_delta=delta,
        n_test=len(world.test_x), calib_levels=tuple(levels),
        timings=timings,
    )
    # 验收要读 `result` 的很多字段，所以先建 run、再跑验收。
    # **顺序是有意义的**：期望先于验收 —— 对抗用例 `html_is_a_pure_function_of_the_products`
    # 要在验收里造一次完整产物（含 `expectations.json`），若期望还是空的，
    # 造出来的 HTML 与最终的那一份不同，用例会**以错误的理由**红掉。
    run.expectations = run_expectations(run)
    run.acceptance = run_acceptance(run)
    run.elapsed_s = time.perf_counter() - t_all
    run.products = products_of(run)
    return run


# --------------------------------------------------------------------------- #
# 产物
# --------------------------------------------------------------------------- #


def reconstruct_run(products: Mapping[str, Any]) -> QuantRun:
    """从 `products` 重建一个**够渲染用**的 `QuantRun`。**这个函数只有字典。**

    ## 它为什么存在

    `visualize` 的语义是「重渲染 `out/report.html`」。如果它走 `run_pipeline`，
    就要把世界重造、基座重训、108 个候选重量化一遍（实测 **44 s**），而它证明的
    只是「流水线是确定的」—— 那件事已经由「两次 `run` 只有两个豁免文件不同」
    证明了。真正该被证明的是另一件事：

    > **`out/` 里那 17 个版本化产物，足以把 `report.html` 一字不差地重造出来。**

    这就是工作区纪律「**产物从字典渲染，不从对象渲染**」的可执行形式，也是
    对抗用例 `html_is_a_pure_function_of_the_products` 的落点。

    ## 哪些字段留空，以及为什么

    `world` / `train_stats` / `model` / `model_snapshot` / `stats` /
    `quantized_models` 全部留空 —— 它们是 `run_pipeline` 的**中间对象**，不是产物。
    凡是渲染要用到的东西都必须能从产物算出来；`QuantRun.n_test`、`calib_levels`
    与 `readings.json["n_blocks_ref"]` 三个字段就是为此加的。

    ⚠️ 拿重建出来的 run 去跑 `run_acceptance` 会失败（那些用例要摸模型），
    这是**故意的** —— 重建路径的用途只有一个：渲染。
    """
    cfg = dict(products["config.json"])
    rd = products["readings.json"]
    readings = [Reading.from_dict(d) for d in rd["candidates"]]
    fp16 = Reading.from_dict(rd["baseline"])

    frontier = matrix.frontier_of(readings)
    sens_rows = matrix.calibration_sensitivity(readings, cfg)

    return QuantRun(
        cfg=cfg,
        protocol_digest=protocol_digest(cfg),
        version_digest=version_digest(cfg),
        world=None,            # type: ignore[arg-type]
        train_stats=None,      # type: ignore[arg-type]
        world_d=dict(products["world.json"]),
        train_d=dict(products["train.json"]),
        model=None,            # type: ignore[arg-type]
        model_snapshot=None,   # type: ignore[arg-type]
        stats={},              # type: ignore[arg-type]
        stats_identity_ok=bool(
            products["calibration.json"].get("shared_across_candidates", False)
        ),
        nref=int(rd["n_blocks_ref"]),
        fp16=fp16,
        readings=readings,
        quantized_models={},
        frontier=frontier,
        frontier_summary=matrix.frontier_summary(frontier, readings, cfg),
        reference_point=matrix.reference_point(readings, cfg),
        quality_rows=matrix.quality_table(readings, cfg),
        diff_rows=matrix.diff_rows(readings, cfg),
        sensitivity_rows=sens_rows,
        sensitivity_summary=matrix.sensitivity_summary(sens_rows),
        selections=matrix.selections(readings, frontier, cfg),
        acceptance=AcceptanceReport.from_dict(products["acceptance.json"]),
        expectations=[
            Expectation.from_dict(d) for d in products["expectations.json"]["items"]
        ],
        corr_memory_size=matrix.corr(
            [r.memory_gb for r in readings], [r.size_gb for r in readings]
        ),
        memory_overhead_gb=memory_overhead_gb(cfg),
        memory_axis_frontier_delta=matrix.frontier_delta_with_axis_swapped(readings, 2, 3),
        n_test=int(products["world.json"]["n_test"]),
        calib_levels=tuple(int(x) for x in products["calibration.json"]["levels"]),
        products=dict(products),
    )


def products_of(run: QuantRun) -> dict[str, Any]:
    """把一次运行折成「相对路径 → 文本或对象」。**唯一的产物生成入口。**"""
    cfg = run.cfg
    from .report import render_markdown
    from .visualize import render_html

    rows = matrix.matrix_rows(run.readings, cfg)
    frontier_ids = {r.spec_id for r in run.frontier}
    frontier_rows = [d for d in rows if d["spec_id"] in frontier_ids]

    products: dict[str, Any] = {
        "config.json": cfg,
        "world.json": run.world_d,
        "train.json": run.train_d,
        "calibration.json": {
            "levels": sorted(run.stats),
            "by_level": {str(n): st.to_dict() for n, st in sorted(run.stats.items())},
            "shared_across_candidates": run.stats_identity_ok,
            "note": "每个档位只构建一次；该档位的全部候选共用同一个对象",
        },
        "readings.json": {
            "baseline": run.fp16.to_dict(),
            "candidates": [r.to_dict() for r in run.readings],
            "n_candidates": len(run.readings),
            # 全部读数的成本模型共用这一个「反量化参考块数」。
            # 它是**模型形状的函数**，所以不进 readings 的每一行，只记一次；
            # 记在这里是为了让 `reconstruct_run` 不必重建模型对象。
            "n_blocks_ref": run.nref,
        },
        "matrix.csv": csv_text(matrix.MATRIX_COLUMNS, [
            [d[c] for c in matrix.MATRIX_COLUMNS] for d in rows
        ]),
        "quality_table.csv": csv_text(matrix.QUALITY_TABLE_COLUMNS, [
            [d[c] for c in matrix.QUALITY_TABLE_COLUMNS] for d in run.quality_rows
        ]),
        "frontier.csv": csv_text(matrix.MATRIX_COLUMNS, [
            [d[c] for c in matrix.MATRIX_COLUMNS] for d in frontier_rows
        ]),
        "selection.csv": csv_text(SELECTION_COLUMNS, [
            [s.to_dict()[c] for c in SELECTION_COLUMNS] for s in run.selections
        ]),
        "diff.csv": csv_text(matrix.DIFF_COLUMNS, [
            [d[c] for c in matrix.DIFF_COLUMNS] for d in run.diff_rows
        ]),
        "sensitivity.csv": csv_text(matrix.SENSITIVITY_COLUMNS, [
            [d[c] for c in matrix.SENSITIVITY_COLUMNS] for d in run.sensitivity_rows
        ]),
        "acceptance.json": run.acceptance.to_dict(),
        "expectations.json": {
            "items": [e.to_dict() for e in run.expectations],
            "summary": expectation_summary(run.expectations),
        },
        "summary.json": _summary(run),
        "lineage.dot": to_dot(run),
        "report.md": render_markdown(run),
        "report.html": render_html(run),
    }
    products["version.json"] = {
        "version_digest": run.version_digest,
        "protocol_digest": run.protocol_digest,
        "seed": int(cfg["seed"]),
        "n_candidates": len(run.readings),
        "n_frontier": len(run.frontier),
        "all_passed": run.acceptance.all_passed,
        # ⚠️ `created_at` **只允许出现在这个文件里**。见模块文档。
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": "本文件与 benchmark.json 是**唯一允许两次运行不同**的两个产物",
    }
    products["benchmark.json"] = benchmark(run)
    return products


def _summary(run: QuantRun) -> dict[str, Any]:
    cfg = run.cfg
    d = cfg["deployment"]
    return {
        "version_digest": run.version_digest,
        "protocol_digest": run.protocol_digest,
        "n_candidates": len(run.readings),
        "n_frontier": len(run.frontier),
        "baseline": {
            "spec_id": run.fp16.spec_id,
            "quality": round(run.fp16.quality, 10),
            "size_gb": round(run.fp16.size_gb, 10),
            "memory_gb": round(run.fp16.memory_gb, 10),
            "latency_ms": round(run.fp16.latency_ms, 10),
        },
        "world": run.world.to_dict(),
        "train": run.train_stats.to_dict(),
        "frontier": run.frontier_summary,
        "selections": [s.to_dict() for s in run.selections],
        "sensitivity_summary": run.sensitivity_summary,
        "collinearity": {
            "corr_memory_size": round(run.corr_memory_size, 12),
            "memory_minus_size_gb": round(run.memory_overhead_gb, 12),
            "frontier_delta_with_memory_replaced_by_size": run.memory_axis_frontier_delta,
        },
        "acceptance": run.acceptance.to_dict(),
        "expectations": expectation_summary(run.expectations),
        "deployment": {
            "base_params_b": d["base_params_b"],
            "memory_limit_gb": d["memory_limit_gb"],
            "latency_limit_ms": d["latency_limit_ms"],
            "n_infeasible": sum(
                1 for r in run.readings
                if r.memory_gb > float(d["memory_limit_gb"])
                or r.latency_ms > float(d["latency_limit_ms"])
            ),
        },
        "cost_model_note": (
            "四维里**只有 Quality 是实测的**（测试集前向），"
            "Latency / Memory / Size 全部来自成本模型；"
            "系数都在 config.json 的 runtime / deployment 段里。"
        ),
    }


def write_products(out_dir: str | Path, products: Mapping[str, Any]) -> list[str]:
    return write_outputs(out_dir, products)


def read_products(out_dir: str | Path) -> dict[str, Any]:
    """读回产物。`.json` 解析成对象，其余按文本。**缺的文件不进结果。**"""
    root = Path(out_dir)
    out: dict[str, Any] = {}
    for rel in OUT_FILES:
        p = root / rel
        if not p.is_file():
            continue
        text = p.read_text(encoding="utf-8")
        out[rel] = json.loads(text) if rel.endswith(".json") else text
    return out


def missing_products(out_dir: str | Path) -> list[str]:
    root = Path(out_dir)
    return [rel for rel in OUT_FILES if not (root / rel).is_file()]


def dump_default_config(path: str | Path) -> str:
    """把 `DEFAULTS` 落成 `configs/default.json`。**改完 DEFAULTS 必须重跑它。**"""
    from .config import default_config

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dump_json(default_config()), encoding="utf-8", newline="\n")
    return str(p)
