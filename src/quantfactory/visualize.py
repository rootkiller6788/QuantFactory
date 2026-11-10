"""可视化：`out/report.html`。**手写 SVG，零外部资源，浅色主题。**

## 为什么手写而不是引图表库

1. **零依赖**是这个项目的底线（`requirements.txt` 里没有一行）。
2. 引一个库就要把它的版本钉进产物 —— 于是「同一份输入得到逐位相同的产物」
   变成「同一份输入 + 同一个库版本」。手写 SVG 只依赖 Python 的 f-string。
3. 图里要标的东西（拐点、预算线、元数据占比）比通用图表 API 能表达的多。

## 四张图各自回答什么

| 图 | 问题 | 为什么单独画 |
| --- | --- | --- |
| 质量 × 位宽 | 位宽买到多少质量？在哪一档饱和？ | 折线才能看出「饱和点」，柱状看不出来 |
| 前沿 | 质量与体积的取舍边界在哪？ | 散点+阶梯线才能看见「被支配的点在哪」 |
| int2 上的算法 × 分组 | 算法差异到底有多大？ | **算法差异几乎全部集中在 int2**，混着画会被噪声淹没 |
| 名义 vs 有效位宽 | 「INT4 其实是多少位？」 | 一根柱子上叠一段元数据，比一张表直观 |

## 一条纪律

**图里不许出现 `created_at` 或耗时。** 它们是 HTML 产物的一部分，
而 HTML 产物必须是逐位可复现的。
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from . import matrix
from .acceptance import binomial_se
from .pipeline import UNVERSIONED_FILES, VERSIONED_FILES

__all__ = ["render_html"]

# ---- 浅色主题的调色板（与 IDE 的 light theme 一致）
BG = "#ffffff"
PANEL = "#f8fafc"
INK = "#1f2937"
INK_SOFT = "#6b7280"
GRID = "#e5e7eb"
AXIS = "#9ca3af"
SERIES = {
    "rtn": "#2563eb",
    "gptq": "#16a34a",
    "awq": "#dc2626",
    "kquant": "#9333ea",
}
FP16_COLOR = "#6b7280"
KNEE_COLOR = "#ea580c"


def _esc(s: Any) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


class _Chart:
    """极小的 SVG 画布：数据域 → 像素。**只做四张图需要的事。**"""

    def __init__(
        self,
        w: int,
        h: int,
        x_domain: tuple[float, float],
        y_domain: tuple[float, float],
        pad: tuple[int, int, int, int] = (64, 18, 34, 44),
    ) -> None:
        self.w, self.h = w, h
        self.x0, self.x1 = x_domain
        self.y0, self.y1 = y_domain
        self.pl, self.pr, self.pt, self.pb = pad
        self.parts: list[str] = []

    # ---- 坐标
    def px(self, v: float) -> float:
        span = (self.x1 - self.x0) or 1.0
        return self.pl + (v - self.x0) / span * (self.w - self.pl - self.pr)

    def py(self, v: float) -> float:
        span = (self.y1 - self.y0) or 1.0
        return self.h - self.pb - (v - self.y0) / span * (self.h - self.pt - self.pb)

    # ---- 图元
    def rect(self, x: float, y: float, w: float, h: float, fill: str, **kw: Any) -> None:
        extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in kw.items())
        self.parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(0.0, w):.2f}" '
            f'height="{max(0.0, h):.2f}" fill="{fill}"{extra}/>'
        )

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str,
             width: float = 1.0, dash: str = "") -> None:
        d = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" '
            f'stroke="{stroke}" stroke-width="{width}"{d}/>'
        )

    def polyline(self, pts: Sequence[tuple[float, float]], stroke: str, width: float = 2.0) -> None:
        if len(pts) < 2:
            return
        d = " ".join(f"{x:.2f},{y:.2f}" for x, y in pts)
        self.parts.append(
            f'<polyline points="{d}" fill="none" stroke="{stroke}" '
            f'stroke-width="{width}" stroke-linejoin="round"/>'
        )

    def circle(self, x: float, y: float, r: float, fill: str, stroke: str = "",
               width: float = 0.0, opacity: float = 1.0) -> None:
        s = f' stroke="{stroke}" stroke-width="{width}"' if stroke else ""
        o = f' opacity="{opacity}"' if opacity != 1.0 else ""
        self.parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{r:.2f}" fill="{fill}"{s}{o}/>')

    def text(self, x: float, y: float, s: Any, fill: str = INK, size: int = 11,
             anchor: str = "start", weight: str = "normal") -> None:
        w = f' font-weight="{weight}"' if weight != "normal" else ""
        self.parts.append(
            f'<text x="{x:.2f}" y="{y:.2f}" fill="{fill}" font-size="{size}" '
            f'text-anchor="{anchor}"{w} font-family="system-ui,Segoe UI,Helvetica,sans-serif">'
            f"{_esc(s)}</text>"
        )

    def grid_y(self, ticks: Sequence[float], fmt: str = "{:.2f}") -> None:
        for t in ticks:
            y = self.py(t)
            self.line(self.pl, y, self.w - self.pr, y, GRID, 1.0)
            self.text(self.pl - 8, y + 4, fmt.format(t), INK_SOFT, 10, "end")

    def frame(self) -> None:
        self.line(self.pl, self.pt, self.pl, self.h - self.pb, AXIS, 1.2)
        self.line(self.pl, self.h - self.pb, self.w - self.pr, self.h - self.pb, AXIS, 1.2)

    def svg(self, title: str, caption: str = "") -> str:
        cap = (f'<figcaption>{_esc(caption)}</figcaption>' if caption else "")
        return (
            f'<figure><svg viewBox="0 0 {self.w} {self.h}" width="100%" '
            f'role="img" aria-label="{_esc(title)}" '
            f'style="max-width:{self.w}px;background:{BG}">'
            f'<title>{_esc(title)}</title>{"".join(self.parts)}</svg>{cap}</figure>'
        )


def _ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if hi <= lo:
        return [lo]
    step = (hi - lo) / n
    return [lo + step * i for i in range(n + 1)]


# --------------------------------------------------------------------------- #
# 图 1：质量 × 位宽
# --------------------------------------------------------------------------- #


def _fig_quality_vs_precision(run: Any) -> str:
    cfg = run.cfg
    precs = list(cfg["space"]["precisions"])
    bits = [float(cfg["quant"]["bit_widths"][p]) for p in precs]
    pv = matrix.pivot_quality(run.quality_rows)
    algs = pv["algorithms"]

    vals: dict[str, list[float | None]] = {}
    lo = run.fp16.quality
    hi = run.fp16.quality
    for a in algs:
        row: list[float | None] = []
        for p in precs:
            qs = [pv["cells"][f"{p}|g{g}|{a}"]["quality"]
                  for g in cfg["space"]["groups"] if f"{p}|g{g}|{a}" in pv["cells"]]
            v = max(qs) if qs else None
            row.append(v)
            if v is not None:
                lo = min(lo, v)
                hi = max(hi, v)
        vals[a] = row

    span = (hi - lo) or 0.01
    ch = _Chart(680, 300, (bits[0] - 0.4, bits[-1] + 0.4), (lo - span * 0.15, hi + span * 0.15))
    ch.grid_y(_ticks(lo - span * 0.15, hi + span * 0.15, 5), "{:.3f}")
    for i, p in enumerate(precs):
        x = ch.px(bits[i])
        ch.line(x, ch.pt, x, ch.h - ch.pb, GRID, 1.0)
        ch.text(x, ch.h - ch.pb + 16, f"{p}\n({int(bits[i])}b)", INK_SOFT, 10, "middle")
    ch.text((ch.pl + ch.w - ch.pr) / 2, ch.h - 6, "名义位宽", INK_SOFT, 11, "middle")

    # fp16 基线
    yb = ch.py(run.fp16.quality)
    ch.line(ch.pl, yb, ch.w - ch.pr, yb, FP16_COLOR, 1.4, "5 4")
    ch.text(ch.w - ch.pr - 4, yb - 5, f"fp16 {run.fp16.quality:.4f}", FP16_COLOR, 10, "end")

    for a in algs:
        pts = [(ch.px(bits[i]), ch.py(v)) for i, v in enumerate(vals[a]) if v is not None]
        ch.polyline(pts, SERIES[a], 2.2)
        for x, y in pts:
            ch.circle(x, y, 3.0, SERIES[a])
    ch.frame()
    # 图例
    for i, a in enumerate(algs):
        x = ch.pl + 8 + i * 74
        ch.line(x, ch.pt + 8, x + 16, ch.pt + 8, SERIES[a], 2.4)
        ch.text(x + 21, ch.pt + 12, a, INK, 11)
    return ch.svg(
        "质量 × 名义位宽（每格取该算法跨分组、跨校准档位的最好质量）",
        "折线才看得出**饱和点**：int4 之后曲线压平，与 fp16 虚线重合。"
        "这正是「位宽买到多少质量」的答案。",
    )


# --------------------------------------------------------------------------- #
# 图 2：前沿
# --------------------------------------------------------------------------- #


def _fig_frontier(run: Any) -> str:
    cfg = run.cfg
    d = cfg["deployment"]
    rs = run.readings
    xs = [r.size_gb for r in rs]
    ys = [r.quality for r in rs]
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    sx = (xhi - xlo) or 1.0
    sy = (yhi - ylo) or 0.01
    ch = _Chart(680, 340, (xlo - sx * 0.05, xhi + sx * 0.05), (ylo - sy * 0.12, yhi + sy * 0.12))
    ch.grid_y(_ticks(ylo - sy * 0.12, yhi + sy * 0.12, 5), "{:.3f}")
    for t in _ticks(xlo - sx * 0.05, xhi + sx * 0.05, 5):
        x = ch.px(t)
        ch.line(x, ch.pt, x, ch.h - ch.pb, GRID, 1.0)
        ch.text(x, ch.h - ch.pb + 16, f"{t:.1f}", INK_SOFT, 10, "middle")
    ch.text((ch.pl + ch.w - ch.pr) / 2, ch.h - 6, "体积 size_gb（GB，7B 部署尺度）", INK_SOFT, 11, "middle")

    # 全部候选
    for r in rs:
        ch.circle(ch.px(r.size_gb), ch.py(r.quality), 2.4, "#cbd5e1", opacity=0.9)
    # fp16
    ch.circle(ch.px(run.fp16.size_gb), ch.py(run.fp16.quality), 4.0, FP16_COLOR, BG, 1.5)
    ch.text(ch.px(run.fp16.size_gb) - 6, ch.py(run.fp16.quality) - 8, "fp16", FP16_COLOR, 10, "end")

    # 预算边界
    if float(d["memory_limit_gb"]) < xhi:
        xb = ch.px(float(d["memory_limit_gb"]) - run.memory_overhead_gb)
        ch.line(xb, ch.pt, xb, ch.h - ch.pb, "#f59e0b", 1.4, "6 4")
        ch.text(xb + 4, ch.pt + 12, f"预算 {d['memory_limit_gb']}GB（减常数开销）",
                "#b45309", 10, "start")

    # 前沿阶梯
    fr = sorted(run.frontier, key=lambda r: r.size_gb)
    ch.polyline([(ch.px(r.size_gb), ch.py(r.quality)) for r in fr], "#0f172a", 2.4)
    for r in fr:
        ch.circle(ch.px(r.size_gb), ch.py(r.quality), 3.6, SERIES.get(r.algorithm, INK))

    knee_id = run.frontier_summary["knee_spec_id"]
    knee = matrix.by_spec_id(run.readings).get(knee_id or "")
    if knee is not None:
        kx, ky = ch.px(knee.size_gb), ch.py(knee.quality)
        ch.circle(kx, ky, 7.0, "none", KNEE_COLOR, 2.2)
        ch.text(kx + 10, ky - 8, f"拐点 {knee_id}", KNEE_COLOR, 10, "start", "bold")
    ch.frame()
    return ch.svg(
        "四维非支配前沿（投影到 质量 × 体积）",
        "浅灰点是全部候选，深色折线是前沿。**被折线压在左下方的点都是被支配的** —— "
        "它们要么更慢、要么更大、要么更差，没有任何理由选。",
    )


# --------------------------------------------------------------------------- #
# 图 3：int2 上的算法 × 分组
# --------------------------------------------------------------------------- #


def _fig_int2_algorithm_group(run: Any) -> str:
    cfg = run.cfg
    pv = matrix.pivot_quality(run.quality_rows)
    algs = pv["algorithms"]
    groups = list(cfg["space"]["groups"])
    vals: dict[tuple[str, int], float | None] = {}
    lo, hi = run.fp16.quality, run.fp16.quality
    for a in algs:
        for g in groups:
            c = pv["cells"].get(f"int2|g{g}|{a}")
            v = c["quality"] if c else None
            vals[(a, g)] = v
            if v is not None:
                lo = min(lo, v)
                hi = max(hi, v)
    span = (hi - lo) or 0.01
    ch = _Chart(680, 300, (0.0, float(len(groups))), (lo - span * 0.25, hi + span * 0.12))
    ch.grid_y(_ticks(lo - span * 0.25, hi + span * 0.12, 5), "{:.3f}")
    bw = (ch.w - ch.pl - ch.pr) / (len(groups) * len(algs) + 1)
    for gi, g in enumerate(groups):
        for ai, a in enumerate(algs):
            v = vals[(a, g)]
            if v is None:
                continue
            x = ch.pl + bw * (1 + gi * len(algs) + ai)
            y = ch.py(v)
            ch.rect(x, y, bw * 0.78, ch.h - ch.pb - y, SERIES[a], opacity="0.85")
            ch.text(x + bw * 0.39, y - 4, f"{v:.3f}", INK_SOFT, 9, "middle")
        ch.text(ch.pl + bw * (1 + gi * len(algs) + len(algs) / 2 - 0.1),
                ch.h - ch.pb + 16, f"g{g}", INK, 11, "middle")
    yb = ch.py(run.fp16.quality)
    ch.line(ch.pl, yb, ch.w - ch.pr, yb, FP16_COLOR, 1.4, "5 4")
    ch.text(ch.w - ch.pr - 4, yb - 5, f"fp16 {run.fp16.quality:.4f}", FP16_COLOR, 10, "end")
    ch.frame()
    for i, a in enumerate(algs):
        x = ch.pl + 8 + i * 74
        ch.rect(x, ch.pt + 4, 10, 8, SERIES[a])
        ch.text(x + 14, ch.pt + 12, a, INK, 11)
    return ch.svg(
        "INT2 上「算法 × 分组」的质量（算法差异最大的地方）",
        "**算法之间的差异几乎全部集中在 int2。** 在 int4 以上把它们画在一起，"
        "柱子会长得一模一样 —— 那不是「算法都很好」，是「差异被噪声淹没了」。",
    )


# --------------------------------------------------------------------------- #
# 图 4：名义位宽 vs 有效位宽
# --------------------------------------------------------------------------- #


def _fig_effective_bits(run: Any) -> str:
    cfg = run.cfg
    by_id = matrix.by_spec_id(run.readings)
    algs = list(cfg["space"]["algorithms"])
    groups = list(cfg["space"]["groups"])
    prec = "int4"
    nominal = float(cfg["quant"]["bit_widths"][prec])
    # 只取「最大校准档位」这一个信息 —— 所以读 `calib_levels` 而不是 `stats` 对象，
    # 于是整张图可以从产物重建（`pipeline.reconstruct_run`）。
    calib = max(run.calib_levels) if run.calib_levels else 0
    rows: list[tuple[str, float, float]] = []
    for g in groups:
        for a in algs:
            sid = f"{a}|{prec}|g{g}|" + (f"c{calib}" if a in ("gptq", "awq") else "none")
            r = by_id.get(sid)
            if r:
                rows.append((f"{a} g{g}", nominal, r.effective_bits))
    hi = max((e for _, _, e in rows), default=nominal + 1)
    ch = _Chart(680, 300, (0.0, float(len(rows))), (0.0, hi * 1.16))
    ch.grid_y(_ticks(0.0, hi * 1.16, 5), "{:.1f}")
    bw = (ch.w - ch.pl - ch.pr) / (len(rows) + 1)
    for i, (label, nom, eff) in enumerate(rows):
        x = ch.pl + bw * (1 + i)
        y_nom = ch.py(nom)
        ch.rect(x, y_nom, bw * 0.8, ch.h - ch.pb - y_nom, SERIES.get(label.split()[0], INK),
                opacity="0.9")
        if eff > nom + 1e-9:
            y_eff = ch.py(eff)
            ch.rect(x, y_eff, bw * 0.8, y_nom - y_eff, "#f59e0b", opacity="0.85")
            ch.text(x + bw * 0.4, y_eff - 4, f"{eff:.2f}", "#b45309", 9, "middle")
        else:
            ch.text(x + bw * 0.4, y_nom - 4, f"{eff:.2f}", INK_SOFT, 9, "middle")
        ch.text(x + bw * 0.4, ch.h - ch.pb + 14, label.split()[0][:2], INK_SOFT, 9, "middle")
        ch.text(x + bw * 0.4, ch.h - ch.pb + 25, label.split()[1], INK_SOFT, 9, "middle")
    ch.frame()
    ch.text(ch.pl + 8, ch.pt + 12, f"深色 = 名义 {nominal:.0f} 位　"
                                   f"橙色 = 元数据（scale / zero / 超块 / 层混合）",
            INK, 11)
    return ch.svg(
        f"{prec.upper()} 的名义位宽 vs 有效位宽",
        "**同一个「INT4」在不同算法下不是同一个体积。** 橙色段就是元数据 —— "
        "它随分组变小而变长（每组一个 scale + 一个 zero），"
        "而 K-quant 用「量化过的子块 scale」把这一段压到最短，"
        "代价是它的有效位宽可能被层混合顶回去。",
    )


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #


def _tbl(header: Sequence[str], rows: Sequence[Sequence[Any]], cls: str = "") -> str:
    h = "".join(f"<th>{_esc(c)}</th>" for c in header)
    body = "".join(
        "<tr>" + "".join(f"<td>{_esc(c)}</td>" for c in r) + "</tr>" for r in rows
    )
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{body}</tbody></table>'


def render_html(run: Any) -> str:
    cfg = run.cfg
    d = cfg["deployment"]
    # `n_test` 走字段而不是 `len(run.world.test_x)` —— 见 `QuantRun.n_test` 的注释。
    n_test = run.n_test
    se = binomial_se(run.fp16.quality, n_test)
    pv = matrix.pivot_quality(run.quality_rows)
    algs = pv["algorithms"]
    refuted = [e for e in run.expectations if not e.holds]
    acc = run.acceptance.to_dict()

    qrows = []
    for prec, grp in pv["row_keys"]:
        cells = {a: pv["cells"].get(f"{prec}|g{grp}|{a}") for a in algs}
        got = {a: c["quality"] for a, c in cells.items() if c}
        if not got:
            continue
        besta = max(got, key=lambda k: got[k])
        qrows.append([prec, f"g{grp}"]
                     + [("**" if a == besta else "") + f"{got[a]:.4f}" for a in algs]
                     + [f"{max(got.values()) - min(got.values()):.4f}", besta])

    thr_rows = [[c.key, f"{c.limit:g}", f"{c.observed:g}",
                 "通过" if c.passed else "**未通过**", c.detail]
                for c in run.acceptance.thresholds]
    case_rows = [[c.name, "通过" if c.passed else "**未通过**", c.detail]
                 for c in run.acceptance.cases]
    exp_rows = [[e.name, "是" if e.expected else "否", "是" if e.observed else "否",
                 "成立" if e.holds else "**被推翻**", e.reading]
                for e in run.expectations]
    sel_rows = [[s.rule, s.spec_id or "—", f"{s.quality:.4f}" if s.quality is not None else "—",
                 f"{s.size_gb:.4f}" if s.size_gb is not None else "—",
                 f"{s.latency_ms:.2f}" if s.latency_ms is not None else "—",
                 "是" if s.feasible else "否"] for s in run.selections]
    diff_rows = [[x["axis"], x["left"], x["right"], x["n_cells"],
                  f"{x['mean_delta_quality']:+.4f}",
                  f"[{x['min_delta_quality']:+.4f}, {x['max_delta_quality']:+.4f}]",
                  f"{x['n_up']}/{x['n_down']}/{x['n_tie']}",
                  f"{x['mean_delta_size_gb']:+.4f}", f"{x['mean_delta_latency_ms']:+.2f}"]
                 for x in run.diff_rows]
    fr_rows = [[x.spec_id, f"{x.quality:.4f}", f"{x.effective_bits:.4f}",
                f"{x.size_gb:.4f}", f"{x.latency_ms:.2f}",
                "是" if (x.memory_gb <= float(d["memory_limit_gb"])
                         and x.latency_ms <= float(d["latency_limit_ms"])) else "否"]
               for x in run.frontier]

    css = f"""
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; padding: 28px 20px 60px; background: {BG}; color: {INK};
  font: 14px/1.7 system-ui, "Segoe UI", "Microsoft YaHei", Helvetica, sans-serif; }}
.wrap {{ max-width: 1080px; margin: 0 auto; }}
h1 {{ font-size: 24px; margin: 0 0 4px; }}
h2 {{ font-size: 18px; margin: 34px 0 10px; padding-bottom: 6px; border-bottom: 2px solid {GRID}; }}
h3 {{ font-size: 15px; margin: 20px 0 6px; }}
.sub {{ color: {INK_SOFT}; font-size: 13px; }}
code {{ background: {PANEL}; padding: 1px 5px; border-radius: 4px;
  font-family: ui-monospace, Consolas, monospace; font-size: 12.5px; }}
.kpis {{ display: flex; flex-wrap: wrap; gap: 10px; margin: 16px 0 4px; }}
.kpi {{ flex: 1 1 170px; background: {PANEL}; border: 1px solid {GRID};
  border-radius: 10px; padding: 10px 14px; }}
.kpi .v {{ font-size: 21px; font-weight: 600; }}
.kpi .k {{ color: {INK_SOFT}; font-size: 12px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0 18px; font-size: 12.5px; }}
th, td {{ border: 1px solid {GRID}; padding: 5px 8px; text-align: left; }}
th {{ background: {PANEL}; font-weight: 600; }}
tbody tr:nth-child(even) {{ background: #fcfcfd; }}
figure {{ margin: 14px 0 22px; }}
figcaption {{ color: {INK_SOFT}; font-size: 12.5px; margin-top: 6px; }}
.note {{ background: #fff7ed; border-left: 4px solid {KNEE_COLOR};
  padding: 10px 14px; border-radius: 0 8px 8px 0; margin: 14px 0; }}
.ok {{ color: #15803d; }} .bad {{ color: #b91c1c; font-weight: 600; }}
ul {{ padding-left: 22px; }}
"""

    def fig(title: str, svg: str) -> str:
        return f"<h3>{_esc(title)}</h3>{svg}"

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>QuantFactory 报告</title>
<style>{css}</style>
</head>
<body><div class="wrap">

<h1>QuantFactory —— 精度设计空间的一次完整测量</h1>
<p class="sub">版本摘要 <code>{_esc(run.version_digest)}</code>　
协议指纹 <code>{_esc(run.protocol_digest)}</code>　
seed <code>{cfg['seed']}</code>　候选 {len(run.readings)} 个　前沿 {len(run.frontier)} 个点</p>

<div class="note"><strong>先读这一句：</strong>四维目标里
<strong>只有 Quality 是实测的</strong>（{n_test} 条测试集前向），
Latency / Memory / Size 全部来自成本模型；系数在
<code>config.json</code> 的 <code>runtime</code> / <code>deployment</code> 段里，
改系数不改代码。</div>

<div class="kpis">
  <div class="kpi"><div class="v">{run.fp16.quality:.4f}</div><div class="k">fp16 基线质量</div></div>
  <div class="kpi"><div class="v">{max(r.quality for r in run.readings):.4f}</div>
    <div class="k">量化后最高质量</div></div>
  <div class="kpi"><div class="v">{len(run.readings)}</div><div class="k">候选数</div></div>
  <div class="kpi"><div class="v">{len(run.frontier)}</div><div class="k">前沿点数</div></div>
  <div class="kpi"><div class="v">{acc['n_thresholds_passed']}/{acc['n_thresholds']}</div>
    <div class="k">门槛通过</div></div>
  <div class="kpi"><div class="v">{acc['n_cases_passed']}/{acc['n_cases']}</div>
    <div class="k">对抗用例通过</div></div>
  <div class="kpi"><div class="v">{len(refuted)}/{len(run.expectations)}</div>
    <div class="k">期望被推翻</div></div>
</div>

<h2>1. 质量 × 位宽：位宽买到多少质量</h2>
{_fig_quality_vs_precision(run)}

<h2>2. 主结论表</h2>
<p class="sub">每格是该 (位宽, 分组, 算法) 上质量最高的候选（跨校准档位取最好）。
加粗 = 该行最好。fp16 = {run.fp16.quality:.4f}，二项标准误 {se:.6f}。</p>
{_tbl(["位宽", "分组", *algs, "spread", "最好"], qrows)}

<h2>3. 前沿</h2>
{_fig_frontier(run)}
{_tbl(["spec_id", "质量", "有效位宽", "体积 GB", "延迟 ms", "在预算内"], fr_rows)}

<h2>4. 选型</h2>
{_tbl(["规则", "选中", "质量", "体积 GB", "延迟 ms", "可行"], sel_rows)}

<h2>5. 算法差异集中在 int2</h2>
{_fig_int2_algorithm_group(run)}

<h2>6. 名义位宽 ≠ 有效位宽</h2>
{_fig_effective_bits(run)}

<h2>7. 两两 Diff（匹配格）</h2>
<p class="sub">固定其余各轴，只让一个轴变，在同格里两两比。
<code>升/降/平</code> 是符号分布 —— 只报均值会把「一半格大涨一半格大跌」抹平成「一致小涨」。</p>
{_tbl(["轴", "左", "右", "格数", "Δ质量均值", "区间", "升/降/平", "Δ体积", "Δ延迟 ms"], diff_rows)}

<h2>8. 验收</h2>
<h3>8.1 门槛（{acc['n_thresholds']} 条，阈值全部在 config.json 里）</h3>
{_tbl(["门槛", "阈值", "实测", "结果", "说明"], thr_rows)}
<h3>8.2 对抗用例（{acc['n_cases']} 条，判据写在代码里）</h3>
<p class="sub">门槛失败 = 这次结果不达标；<strong>对抗用例失败 = 代码有 bug</strong>。
两者的含义完全不同。</p>
{_tbl(["用例", "结果", "说明"], case_rows)}

<h2>9. 设计期望（{len(run.expectations)} 条，被推翻 {len(refuted)} 条）</h2>
{_tbl(["期望", "期望值", "实测", "结果", "实测读数"], exp_rows)}
<div class="note"><strong>被推翻的期望不是失败，是产出。</strong>
每条都带着它的<strong>操作化</strong>一起读 —— 同一条自然语言期望换一种操作化
就可能换一个结论。</div>

<h2>10. 这些结论在哪里失效</h2>
<ul>
<li><strong>绝对数值不能迁移到真模型。</strong>本地 {run.fp16.n_params} 个参数，
部署按 {d['base_params_b']}B 算；排序能迁移，数值不能。</li>
<li><strong>三维成本是模型算的，不是测的。</strong>带宽 {cfg['runtime']['bandwidth_gbps']} GB/s、
反量化惩罚 {cfg['runtime']['dequant_penalty_per_block']}/块、各算法 kernel 效率全是假设。</li>
<li><strong>AWQ 在这个世界上没赢，而这不是实现错误。</strong>
它的收益来自真实 LLM 的强激活离群（20~100×），本项目为保持 fp16 可学把宽通道限制在
{cfg['world']['wide_scale']}×（信号再乘 {cfg['world']['wide_spread_ratio']}）。</li>
<li><strong>位宽 ≥ int5 之后的一切差异都在噪声里。</strong>二项标准误 {se:.6f}，
比它小的差异不该被当成结论。</li>
<li><strong>只有一个种子、一个世界。</strong>seed = {cfg['seed']}。</li>
<li><strong>K-quant 不是幂等的</strong>（二级 scale 量化没有不动点），
单级量化（RTN / GPTQ / AWQ）严格幂等。</li>
</ul>

<h2>11. 产物</h2>
<p class="sub"><code>version.json</code>（含 <code>created_at</code>）与
<code>benchmark.json</code>（含耗时）是<strong>唯一允许两次运行不同</strong>的
{len(UNVERSIONED_FILES)} 个产物；
其余 {len(VERSIONED_FILES)} 个文件在同一份配置下必须逐位相同。</p>

</div></body></html>
"""
    return html
