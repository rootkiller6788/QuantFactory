"""血缘图：这次运行的每一份产物是从哪来的。

## 为什么这张图必须**由代码生成**，不能手画

手画的血缘图会随着代码演进悄悄变成谎话，而且**没有任何东西会发现它过期了** ——
它不在任何断言里。所以这里的每个节点、每条边、每个数字都从 `result` 里读出来。

## 图里刻意保留的两条边

1. `calib(c64) → gptq|*|c64` 与 `calib(c256) → gptq|*|c256` 是**扇出**的：
   每个档位的 `CalibStats` 只算一次，被该档位的全部候选共用。
   `acceptance.calibration_stats_are_shared_per_level` 把这件事钉住 ——
   否则结果**看起来一模一样**，只是慢了几百倍，而没有任何断言会发现。
2. `model → quantize` 是**只读**的：全部候选共用同一个 fp16 基座。
   所以边上标的是「clone 后改副本」，而不是「修改」。

## 形状

```
seed ─→ world ─┬─→ train ──→ model ─┬─→ calib(64)  ─┐
               ├─→ test            ├─→ calib(256) ─┤
               └─→ calib_pool      └───────────────┴─→ quantize ×108 ─→ measure ─→ readings
                                                                                     │
                                        ┌────────────────────────────────────────────┤
                                        ↓            ↓              ↓                ↓
                                    frontier    quality_table   diff.csv     sensitivity.csv
                                        ↓
                                   selection
```
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

__all__ = ["to_dot", "node_count", "edge_count"]


def _esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


class _Graph:
    def __init__(self) -> None:
        self.nodes: list[tuple[str, str, list[str]]] = []
        self.edges: list[tuple[str, str, str]] = []
        self._seen: set[str] = set()

    def node(self, nid: str, label: str, attrs: Sequence[str] = ()) -> str:
        if nid not in self._seen:
            self._seen.add(nid)
            self.nodes.append((nid, label, list(attrs)))
        return nid

    def edge(self, src: str, dst: str, label: str = "") -> None:
        self.edges.append((src, dst, label))

    def render(self) -> str:
        lines = [
            "digraph quantfactory {",
            '  graph [rankdir=LR, bgcolor="white", fontname="Helvetica"];',
            '  node [shape=box, style="rounded,filled", fillcolor="#f5f5f5",'
            ' fontname="Helvetica", fontsize=10];',
            '  edge [fontname="Helvetica", fontsize=9, color="#666666"];',
        ]
        for nid, label, attrs in self.nodes:
            extra = (", " + ", ".join(attrs)) if attrs else ""
            lines.append(f'  "{_esc(nid)}" [label="{_esc(label)}"{extra}];')
        for src, dst, label in self.edges:
            lab = f' [label="{_esc(label)}"]' if label else ""
            lines.append(f'  "{_esc(src)}" -> "{_esc(dst)}"{lab};')
        lines.append("}")
        return "\n".join(lines) + "\n"


def to_dot(result: Any) -> str:
    cfg = result.cfg
    g = _Graph()

    seed = int(cfg["seed"])
    g.node("seed", f"seed\\n{seed}", ['fillcolor="#fff2cc"'])

    w = result.world
    g.node(
        "world",
        "world\\n"
        f"d_in={w.d_in} hidden={w.hidden} classes={w.n_classes}\\n"
        f"原型 {w.n_prototypes} 个 / 类间最小距 {w.margin:.4f}",
    )
    g.node("train_split", f"train\\n{len(w.train_x)} 条")
    g.node("test_split", f"test\\n{len(w.test_x)} 条\\n（只用来测质量）")
    g.node("calib_pool", f"calib_pool\\n{len(w.calib_pool)} 条\\n（按档位取前 n 行）")

    ts = result.train_stats
    g.node(
        "model",
        "fp16 基座（全部候选共用）\\n"
        f"{ts.steps} 步 × B{ts.batch}\\n"
        f"train {ts.train_accuracy:.4f} / test {ts.test_accuracy:.4f}\\n"
        f"裁剪触发 {ts.grad_clip_hits} 次",
        ['fillcolor="#d9ead3"'],
    )

    for n in sorted(result.stats):
        st = result.stats[n]
        g.node(
            f"calib{n}",
            f"CalibStats(n={n})\\n"
            f"cond={[round(v, 1) for v in st.diag_condition]}\\n"
            f"求逆残差 {max(st.inv_residual):.1e}",
        )

    space = cfg["space"]
    n_specs = len(result.readings)
    g.node(
        "space",
        "设计空间（笛卡尔积）\\n"
        + "\\n".join([
            f"algorithm ∈ {list(space['algorithms'])}",
            f"precision ∈ {list(space['precisions'])}",
            f"group ∈ {list(space['groups'])}",
            f"calib ∈ {sorted(result.stats)}（rtn/kquant 恒为 none）",
            f"⇒ {n_specs} 个候选",
        ]),
        ['fillcolor="#e6f2ff"'],
    )

    g.node("quantize", f"quantize ×{n_specs}\\n（clone 后改副本，基座只读）")
    g.node(
        "measure",
        "measure（四维）\\n"
        f"Quality 实测（{len(w.test_x)} 条）\\n"
        "Latency / Memory / Size 由成本模型算出\\n"
        f"部署尺度 {cfg['deployment']['base_params_b']}B"
        f"（本地模型只有 {result.fp16.n_params} 个参数）",
    )
    g.node("readings", f"readings\\n{n_specs} 个候选 × 4 维", ['fillcolor="#e6f2ff"'])
    g.node("fp16", f"fp16 基线\\nq={result.fp16.quality:.4f} size={result.fp16.size_gb:.2f} GB")

    g.node("frontier", f"前沿\\n{len(result.frontier)} 个非支配点")
    g.node("quality_table", f"主结论表\\n{len(result.quality_rows)} 行")
    g.node("diff", f"两两 Diff\\n{len(result.diff_rows)} 行")
    g.node("sensitivity", f"校准敏感性\\n{len(result.sensitivity_rows)} 行")
    g.node("selection", "选型（3 条规则）")
    g.node(
        "acceptance",
        f"验收\\n{len(result.acceptance.thresholds)} 门槛 + "
        f"{len(result.acceptance.cases)} 对抗用例",
    )
    g.node("expectations", f"设计期望\\n{len(result.expectations)} 条")
    g.node("report", "report.md / report.html", ['fillcolor="#fce5cd"'])

    g.edge("seed", "world", "根种子：只用 seed + 世界结构参数")
    g.edge("world", "train_split")
    g.edge("world", "test_split")
    g.edge("world", "calib_pool")
    g.edge("train_split", "model", "手写 SGD")
    for n in sorted(result.stats):
        g.edge("model", f"calib{n}", "前向取激活")
        g.edge(f"calib{n}", "quantize", f"档位 {n} 的全部候选共用这一份")
    g.edge("model", "quantize", "只读（clone 后改副本）")
    g.edge("space", "quantize", "枚举顺序")
    g.edge("quantize", "measure")
    g.edge("model", "fp16", "不量化")
    g.edge("fp16", "measure", "基线质量")
    g.edge("measure", "readings")
    g.edge("readings", "frontier", "四维非支配")
    g.edge("readings", "quality_table")
    g.edge("readings", "diff", "匹配格两两比")
    g.edge("readings", "sensitivity")
    g.edge("frontier", "selection", "拐点 + 预算")
    g.edge("readings", "selection")
    g.edge("readings", "acceptance")
    g.edge("frontier", "acceptance")
    g.edge("readings", "expectations")
    g.edge("acceptance", "report")
    g.edge("expectations", "report")
    g.edge("selection", "report")
    g.edge("frontier", "report")

    return g.render()


def node_count(dot: str) -> int:
    """节点数：有 `label=` 且没有 `->` 的行。**只用来做产物的自洽断言。**"""
    return sum(1 for line in dot.splitlines() if "[label=" in line and "->" not in line)


def edge_count(dot: str) -> int:
    return sum(1 for line in dot.splitlines() if "->" in line)
