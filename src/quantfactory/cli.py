"""命令行：15 个子命令（**12 只读 + 3 写**）。

## 三条硬约定

1. **只读命令绝不写盘。** `run` 是唯一会做计算的入口；只读命令优先读 `out/`，
   产物缺失或版本对不上时**在内存里重跑，并在 stderr 上明说「本次是内存重跑，未写盘」**。
   （对不上也重跑：拿一份旧配置的产物回答新配置的问题，比慢 30 秒危险得多。）
2. **`--json` 时 stdout 上只许有一段 JSON。** 任何散文（包括「内存重跑」这句提示）
   都走 stderr。手法是 `_quiet_prose` 把 `sys.stdout` 换成缓冲区，`_emit_json`
   直接写真正的 stdout。**不这么做的话 `| jq` 会被前面的说明文字噎死。**
3. **子命令上的 `--config/--out/--json` 用 `default=argparse.SUPPRESS`。**
   子解析器与主解析器共用同一个 namespace；不抑制默认值的话，
   `main.py --json check` 里的 `--json` 会被子解析器的默认值覆写回 `False`。

## 只读路径为什么能从盘上重建对象

`readings.json` 里存的是 `Reading.to_dict()` 的完整字段（**不含耗时**），
`schema.Reading.from_dict` 是它的逆。于是只读命令可以重建出 `Reading` 对象，
直接复用 `matrix.*` 的函数 —— 而不是再写一套解析 CSV 的实现（那两条实现一定会漂移）。
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from . import matrix
from .config import load_config, merge_config, validate_config, version_digest
from .pipeline import (
    VERSIONED_FILES,
    dump_default_config,
    missing_products,
    read_products,
    reconstruct_run,
    run_pipeline,
    write_products,
)
from .schema import Reading
from .visualize import render_html

__all__ = ["main", "build_parser", "READ_ONLY_COMMANDS", "WRITE_COMMANDS"]

DEFAULT_OUT = "out"
DEFAULT_CONFIG_PATH = "configs/default.json"

#: 项目根（`<root>/src/quantfactory/cli.py` 往上三层）。
#: **相对路径一律相对它解析，不是相对 cwd** —— 见 `_anchor`。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _anchor(p: str | Path) -> Path:
    """把相对路径锚定到**项目根**。绝对路径原样返回。

    ## 为什么必须有这个函数

    `--config` / `--out` 的默认值是 cwd 相对的 `"configs/default.json"` / `"out"`。
    从项目目录里跑时它们是对的（所有文档写的都是 `cd <project> && python main.py run`），
    于是这个缺陷**长期隐形**。一旦从别处调用 —— 例如跨项目的批量核对
    `python L5-quantfactory/main.py check` —— 两个默认值就都指向了 cwd 下不存在的位置：
    `out/` 读不到 ⇒ **静默地内存重跑一遍**（实测 22.8 s，而正常只要 0.4 s），
    并在 stdout 上说一句「本次是内存重跑，未写盘」。

    它不崩，所以**更难发现**。相邻的 `L5-distillfactory` 走的是严格 `load_config`
    （文件不存在就抛异常）⇒ 那边是直接崩 —— 崩溃反而更容易被看见。

    ## 语义

    相对路径 ⇒ 相对项目根（与 `pytest` 的 rootdir / `cargo` / `npm --prefix` 一致）。
    绝对路径 ⇒ 原样使用。
    """
    q = Path(p)
    return q if q.is_absolute() else _PROJECT_ROOT / q

# 只读：只回答「已有的产物说明了什么」
READ_ONLY_COMMANDS: tuple[str, ...] = (
    "check",        # 门槛 + 对抗用例
    "expectations", # 15 条设计期望
    "quality",      # 主结论表
    "frontier",     # 非支配集
    "selection",    # 三条选型规则
    "diff",         # 两两 Diff
    "sensitivity",  # 校准规模敏感性
    "candidates",   # 候选读数（可过滤）
    "cost",         # 成本模型分解
    "summary",      # 一页结论
    "lineage",      # 血缘图（DOT）
    "report",       # 打印 report.md
)

# 写：会产生/修改磁盘上的东西
WRITE_COMMANDS: tuple[str, ...] = ("run", "dump-config", "visualize")


# --------------------------------------------------------------------------- #
# 输出管道
# --------------------------------------------------------------------------- #


@contextlib.contextmanager
def _quiet_prose():
    """把 `sys.stdout` 换成缓冲区。**`--json` 时用它把散文吞掉。**"""
    buf = io.StringIO()
    real = sys.stdout
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = real


def _emit_json(obj: Any) -> None:
    """走**真正的** stdout，且只写一段 JSON。"""
    sys.stdout.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _note(msg: str) -> None:
    """给 `--json` 用的旁白通道：stderr。"""
    sys.stderr.write(msg + "\n")


# --------------------------------------------------------------------------- #
# 视图：从产物里重建
# --------------------------------------------------------------------------- #


def _readings(products: Mapping[str, Any]) -> list[Reading]:
    rd = products.get("readings.json") or {}
    return [Reading.from_dict(d) for d in rd.get("candidates", [])]


def _baseline(products: Mapping[str, Any]) -> Reading | None:
    rd = products.get("readings.json") or {}
    b = rd.get("baseline")
    return Reading.from_dict(b) if b else None


def _cfg(products: Mapping[str, Any], fallback: Mapping[str, Any]) -> dict[str, Any]:
    return dict(products.get("config.json") or fallback)


def _load_view(
    cfg: Mapping[str, Any], out_dir: str, *, force_memory: bool = False
) -> tuple[dict[str, Any], bool]:
    """返回 `(products, from_disk)`。

    **只读路径唯一的取数入口。** 缺产物或版本对不上 → 内存重跑，**不落盘**。
    """
    if not force_memory:
        missing = missing_products(out_dir)
        if not missing:
            products = read_products(out_dir)
            ver = products.get("version.json") or {}
            if ver.get("version_digest") == version_digest(cfg):
                return products, True
    return run_pipeline(cfg).products, False


# --------------------------------------------------------------------------- #
# 各命令的 payload
# --------------------------------------------------------------------------- #


def _payload_check(products: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return products.get("acceptance.json") or {"error": "缺少 acceptance.json"}


def _payload_expectations(products, args) -> dict[str, Any]:
    return products.get("expectations.json") or {"error": "缺少 expectations.json"}


def _payload_quality(products, args) -> dict[str, Any]:
    cfg = _cfg(products, {})
    rs = _readings(products)
    rows = matrix.quality_table(rs, cfg)
    pv = matrix.pivot_quality(rows)
    return {
        "algorithms": pv["algorithms"],
        "rows": [{"precision": p, "group": g,
                  "quality": {a: (pv["cells"][f"{p}|g{g}|{a}"]["quality"]
                                  if f"{p}|g{g}|{a}" in pv["cells"] else None)
                              for a in pv["algorithms"]},
                  "spread": _spread(pv, p, g)}
                 for p, g in pv["row_keys"]],
        "n_cells": len(rows),
        "baseline_quality": (_baseline(products).quality if _baseline(products) else None),
    }


def _spread(pv: Mapping[str, Any], prec: str, grp: int) -> float | None:
    vals = [pv["cells"][k]["quality"] for k in pv["cells"]
            if k.startswith(f"{prec}|g{grp}|")]
    return (max(vals) - min(vals)) if vals else None


def _payload_frontier(products, args) -> dict[str, Any]:
    cfg = _cfg(products, {})
    rs = _readings(products)
    fr = matrix.frontier_of(rs)
    return {
        "summary": matrix.frontier_summary(fr, rs, cfg),
        "points": [r.to_dict() for r in fr],
    }


def _payload_selection(products, args) -> dict[str, Any]:
    cfg = _cfg(products, {})
    rs = _readings(products)
    fr = matrix.frontier_of(rs)
    return {"selections": [s.to_dict() for s in matrix.selections(rs, fr, cfg)]}


def _payload_diff(products, args) -> dict[str, Any]:
    cfg = _cfg(products, {})
    rs = _readings(products)
    rows = matrix.diff_rows(rs, cfg)
    if args.axis:
        rows = [d for d in rows if d["axis"] == args.axis]
    return {"axes": list(cfg["matrix"]["diff_axes"]), "rows": rows}


def _payload_sensitivity(products, args) -> dict[str, Any]:
    cfg = _cfg(products, {})
    rs = _readings(products)
    rows = matrix.calibration_sensitivity(rs, cfg)
    return {"rows": rows, "summary": matrix.sensitivity_summary(rows)}


def _payload_candidates(products, args) -> dict[str, Any]:
    cfg = _cfg(products, {})
    rs = _readings(products)
    if args.algorithm:
        rs = [r for r in rs if r.algorithm == args.algorithm]
    if args.precision:
        rs = [r for r in rs if r.precision == args.precision]
    if args.group:
        rs = [r for r in rs if r.group == args.group]
    rows = matrix.matrix_rows(rs, cfg)
    if args.sort:
        rows.sort(key=lambda d: d[args.sort], reverse=bool(args.desc))
    if args.top:
        rows = rows[: args.top]
    return {"n": len(rows), "filters": _filters(args), "rows": rows}


def _filters(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "algorithm": getattr(args, "algorithm", None),
        "precision": getattr(args, "precision", None),
        "group": getattr(args, "group", None),
        "top": getattr(args, "top", None),
        "sort": getattr(args, "sort", None),
    }


def _payload_cost(products, args) -> dict[str, Any]:
    cfg = _cfg(products, {})
    rs = _readings(products)
    base = _baseline(products)
    if args.algorithm:
        rs = [r for r in rs if r.algorithm == args.algorithm]
    if args.precision:
        rs = [r for r in rs if r.precision == args.precision]
    out = []
    for r in rs:
        out.append({
            "spec_id": r.spec_id,
            "nominal_bits": r.nominal_bits,
            "effective_bits": r.effective_bits,
            "metadata_bits_per_weight": r.metadata_bits_per_weight,
            "n_blocks": r.n_blocks,
            "size_gb": r.size_gb,
            "memory_gb": r.memory_gb,
            "latency_ms": r.latency_ms,
            "metadata_share_of_nominal": (
                r.metadata_bits_per_weight / r.nominal_bits if r.nominal_bits else None
            ),
        })
    return {
        "runtime": cfg.get("runtime", {}),
        "deployment": cfg.get("deployment", {}),
        "memory_minus_size_gb": (base.memory_gb - base.size_gb) if base else None,
        "baseline": base.to_dict() if base else None,
        "rows": out,
    }


def _payload_summary(products, args) -> dict[str, Any]:
    return products.get("summary.json") or {"error": "缺少 summary.json"}


def _payload_lineage(products, args) -> dict[str, Any]:
    dot = products.get("lineage.dot")
    return {"dot": dot} if dot else {"error": "缺少 lineage.dot"}


def _payload_report(products, args) -> dict[str, Any]:
    md = products.get("report.md")
    return {"markdown": md} if md else {"error": "缺少 report.md"}


# --------------------------------------------------------------------------- #
# 各命令的散文输出
# --------------------------------------------------------------------------- #


def _prose_check(p: Mapping[str, Any]) -> None:
    if "error" in p:
        print(p["error"])
        return
    print(f"门槛 {p['n_thresholds_passed']}/{p['n_thresholds']}　"
          f"对抗用例 {p['n_cases_passed']}/{p['n_cases']}　"
          f"全部通过：{'是' if p['all_passed'] else '否'}")
    print()
    print("门槛（阈值全部在 config.json 的 acceptance 段里）")
    for c in p["thresholds"]:
        mark = "OK  " if c["passed"] else "FAIL"
        print(f"  [{mark}] {c['key']:34} 阈值 {c['limit']:<12g} 实测 {c['observed']:<12g} {c['detail']}")
    print()
    print("对抗用例（判据写在代码里；失败 = 代码有 bug）")
    for c in p["cases"]:
        mark = "OK  " if c["passed"] else "FAIL"
        print(f"  [{mark}] {c['name']}")
        print(f"         {c['detail']}")


def _prose_expectations(p: Mapping[str, Any]) -> None:
    if "error" in p:
        print(p["error"])
        return
    s = p["summary"]
    print(f"设计期望 {s['n']} 条：成立 {s['n_holds']}，被推翻 {s['n_refuted']}")
    print()
    for e in p["items"]:
        mark = "成立    " if e["holds"] else "**被推翻**"
        print(f"  [{mark}] {e['name']}")
        print(f"      期望 {e['statement']}")
        print(f"      判据 {e['operationalization']}")
        print(f"      实测 {e['reading']}")
        print()


def _prose_quality(p: Mapping[str, Any]) -> None:
    if "error" in p:
        print(p["error"])
        return
    algs = p["algorithms"]
    print(f"fp16 基线质量 = {p['baseline_quality']}")
    print("位宽   分组 " + "".join(f"{a:>10}" for a in algs) + "     spread")
    for row in p["rows"]:
        line = f"{row['precision']:6} {row['group']:5} "
        for a in algs:
            v = row["quality"][a]
            line += f"{v:10.4f}" if v is not None else f"{'-':>10}"
        line += f"  {row['spread']:9.4f}"
        print(line)


def _prose_frontier(p: Mapping[str, Any]) -> None:
    if "error" in p:
        print(p["error"])
        return
    s = p["summary"]
    print(f"前沿 {s['size']} 个点（被支配 {s['n_dominated']}）；超体积 {s['hypervolume']}；"
          f"间距 {s['spacing']}")
    print(f"参考点 {s['reference_point']}　拐点 {s['knee_spec_id']}"
          f"（投影到第 {s['knee_axes']} 维）")
    print(f"用到的算法 {s['algorithms']}　位宽 {s['precisions']}　分组 {s['groups']}")
    print()
    print(f"{'spec_id':30} {'质量':>9} {'有效位':>9} {'体积GB':>9} {'延迟ms':>9}")
    for x in p["points"]:
        print(f"{x['spec_id']:30} {x['quality']:9.4f} {x['effective_bits']:9.4f} "
              f"{x['size_gb']:9.4f} {x['latency_ms']:9.2f}")


def _prose_selection(p: Mapping[str, Any]) -> None:
    if "error" in p:
        print(p["error"])
        return
    for s in p["selections"]:
        print(f"{s['rule']}")
        print(f"  选中 {s['spec_id']}　质量 {s['quality']}　体积 {s['size_gb']} GB　"
              f"延迟 {s['latency_ms']} ms　可行 {'是' if s['feasible'] else '否'}")
        print(f"  理由 {s['why']}")


def _prose_diff(p: Mapping[str, Any]) -> None:
    if "error" in p:
        print(p["error"])
        return
    for d in p["rows"]:
        print(f"{d['axis']:10} {str(d['left']):>8} → {str(d['right']):<8} "
              f"格数 {d['n_cells']:3}  Δ质量 {d['mean_delta_quality']:+.4f} "
              f"[{d['min_delta_quality']:+.4f}, {d['max_delta_quality']:+.4f}]  "
              f"升/降/平 {d['n_up']}/{d['n_down']}/{d['n_tie']}  "
              f"Δ体积 {d['mean_delta_size_gb']:+.4f}  Δ延迟 {d['mean_delta_latency_ms']:+.2f}")


def _prose_sensitivity(p: Mapping[str, Any]) -> None:
    if "error" in p:
        print(p["error"])
        return
    for d in p["rows"]:
        print(f"{d['algorithm']:7} {d['precision']:5} g{d['group']:<3} "
              f"c{d['calib_left']}→c{d['calib_right']}  "
              f"Δ质量 {d['delta_quality']:+.4f}  "
              f"Δ相对误差 {d['delta_weight_rel_error']:+.6f}  "
              f"Δ有效位宽 {d['delta_effective_bits']:+.10f}")
    print()
    for s in p["summary"]:
        print(f"[{s['algorithm']}] {s['cells']} 格：Δ质量均值 {s['mean_delta_quality']:+.4f}"
              f"（升 {s['n_quality_up']}/降 {s['n_quality_down']}/平 {s['n_quality_tie']}）"
              f"　Δ相对误差均值 {s['mean_delta_weight_rel_error']:+.6f}"
              f"　相对误差下降 {s['n_relerr_down']}/{s['cells']}")


def _prose_candidates(p: Mapping[str, Any]) -> None:
    if "error" in p:
        print(p["error"])
        return
    print(f"{p['n']} 个候选（过滤：{p['filters']}）")
    print(f"{'spec_id':30} {'质量':>9} {'质量损失':>9} {'有效位':>8} {'体积GB':>8} "
          f"{'延迟ms':>8} {'可行':>4}")
    for r in p["rows"]:
        print(f"{r['spec_id']:30} {r['quality']:9.4f} {r['quality_loss']:9.4f} "
              f"{r['effective_bits']:8.4f} {r['size_gb']:8.4f} {r['latency_ms']:8.2f} "
              f"{'是' if r['feasible'] else '否':>4}")


def _prose_cost(p: Mapping[str, Any]) -> None:
    if "error" in p:
        print(p["error"])
        return
    d = p["deployment"]
    print(f"部署：{d.get('base_params_b')}B　显存上限 {d.get('memory_limit_gb')} GB　"
          f"延迟上限 {d.get('latency_limit_ms')} ms")
    print(f"memory − size 恒等于 {p['memory_minus_size_gb']} GB（与量化方案无关 ⇒ 两维共线）")
    print()
    print(f"{'spec_id':30} {'名义':>5} {'有效':>8} {'元数据':>8} {'元数据/名义':>11} "
          f"{'块数':>6} {'体积GB':>8} {'延迟ms':>8}")
    for r in p["rows"]:
        share = r["metadata_share_of_nominal"]
        print(f"{r['spec_id']:30} {r['nominal_bits']:5.0f} {r['effective_bits']:8.4f} "
              f"{r['metadata_bits_per_weight']:8.4f} "
              f"{(f'{share * 100:.1f}%' if share is not None else '-'):>11} "
              f"{r['n_blocks']:6d} {r['size_gb']:8.4f} {r['latency_ms']:8.2f}")


def _prose_summary(p: Mapping[str, Any]) -> None:
    if "error" in p:
        print(p["error"])
        return
    print(json.dumps(p, ensure_ascii=False, indent=2, sort_keys=True))


def _prose_lineage(p: Mapping[str, Any]) -> None:
    print(p.get("dot") or p.get("error", ""))


def _prose_report(p: Mapping[str, Any]) -> None:
    print(p.get("markdown") or p.get("error", ""))


# --------------------------------------------------------------------------- #
# 命令表
# --------------------------------------------------------------------------- #

_VIEWS: dict[str, tuple[Callable[..., dict[str, Any]], Callable[[Mapping[str, Any]], None]]] = {
    "check": (_payload_check, _prose_check),
    "expectations": (_payload_expectations, _prose_expectations),
    "quality": (_payload_quality, _prose_quality),
    "frontier": (_payload_frontier, _prose_frontier),
    "selection": (_payload_selection, _prose_selection),
    "diff": (_payload_diff, _prose_diff),
    "sensitivity": (_payload_sensitivity, _prose_sensitivity),
    "candidates": (_payload_candidates, _prose_candidates),
    "cost": (_payload_cost, _prose_cost),
    "summary": (_payload_summary, _prose_summary),
    "lineage": (_payload_lineage, _prose_lineage),
    "report": (_payload_report, _prose_report),
}


# --------------------------------------------------------------------------- #
# 解析器
# --------------------------------------------------------------------------- #


def _add_common(sp: argparse.ArgumentParser) -> None:
    """子命令上的公共选项。**`default=argparse.SUPPRESS` 不是可选项。**"""
    sp.add_argument("--config", default=argparse.SUPPRESS, help="配置文件路径")
    sp.add_argument("--out", default=argparse.SUPPRESS, help="产物目录")
    sp.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                    help="只输出一段 JSON（旁白走 stderr）")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="quantfactory",
        description="QuantFactory —— 量化精度设计空间的测量流水线（零第三方依赖）",
    )
    p.add_argument("--config", default=DEFAULT_CONFIG_PATH,
                   help="配置文件路径（相对路径按项目根解析）")
    p.add_argument("--out", default=DEFAULT_OUT,
                   help="产物目录（相对路径按项目根解析）")
    p.add_argument("--json", action="store_true", help="只输出一段 JSON")
    sub = p.add_subparsers(dest="command", required=True)

    for name in READ_ONLY_COMMANDS:
        sp = sub.add_parser(name, help=f"[只读] {name}")
        _add_common(sp)
        if name == "diff":
            sp.add_argument("--axis", choices=list(matrix.AXES), default=None)
        if name == "candidates":
            sp.add_argument("--algorithm", default=None)
            sp.add_argument("--precision", default=None)
            sp.add_argument("--group", type=int, default=None)
            sp.add_argument("--sort", default=None,
                            choices=["quality", "size_gb", "latency_ms", "effective_bits"])
            sp.add_argument("--desc", action="store_true", default=False)
            sp.add_argument("--top", type=int, default=None)
        if name == "cost":
            sp.add_argument("--algorithm", default=None)
            sp.add_argument("--precision", default=None)

    sp = sub.add_parser("run", help="[写] 跑完整流水线并落盘全部产物")
    _add_common(sp)
    sp.add_argument("--allow-fail", action="store_true", default=False,
                    help="即使验收未全通过也返回 0（默认返回 1，便于 CI 用）")

    sp = sub.add_parser("dump-config", help="[写] 把 DEFAULTS 落成 configs/default.json")
    sp.add_argument("--path", default=DEFAULT_CONFIG_PATH)
    sp.add_argument("--json", action="store_true", default=argparse.SUPPRESS)

    sp = sub.add_parser("visualize", help="[写] 从产物重建并重渲染 out/report.html（只写这一个文件）")
    _add_common(sp)
    return p


# --------------------------------------------------------------------------- #
# 入口
# --------------------------------------------------------------------------- #


def _resolve_cfg(args: argparse.Namespace) -> dict[str, Any]:
    path = _anchor(getattr(args, "config", None) or DEFAULT_CONFIG_PATH)
    if path.is_file():
        cfg = load_config(str(path))
    else:
        cfg = merge_config()
    bad = validate_config(cfg)
    if bad:
        for b in bad:
            _note(f"[配置] {b}")
    return cfg


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    out_dir = str(_anchor(getattr(args, "out", None) or DEFAULT_OUT))
    want_json = bool(getattr(args, "json", False))

    # ---- 写命令：dump-config
    if args.command == "dump-config":
        path = dump_default_config(_anchor(args.path))
        if want_json:
            _emit_json({"written": path})
        else:
            print(f"已写出 {path}")
            print("**改完 DEFAULTS 必须重跑这个命令** —— 否则已落盘的旧配置里"
                  "多出来的键会活下来（代码里的删除传不到磁盘）。")
        return 0

    cfg = _resolve_cfg(args)

    # ---- 写命令：run
    if args.command == "run":
        run = run_pipeline(cfg)
        written = write_products(out_dir, run.products)
        acc = run.acceptance.to_dict()
        if want_json:
            _emit_json({
                "version_digest": run.version_digest,
                "written": written,
                "n_candidates": len(run.readings),
                "n_frontier": len(run.frontier),
                "acceptance": acc,
                "failing": run.acceptance.failing(),
            })
        else:
            print(f"版本摘要 {run.version_digest}　候选 {len(run.readings)}　"
                  f"前沿 {len(run.frontier)}")
            print(f"写出 {len(written)} 个产物到 {out_dir}/"
                  f"（其中 {len(VERSIONED_FILES)} 个必须逐位可复现）")
            print(f"门槛 {acc['n_thresholds_passed']}/{acc['n_thresholds']}　"
                  f"对抗用例 {acc['n_cases_passed']}/{acc['n_cases']}")
            if run.acceptance.failing():
                print("未通过：" + ", ".join(run.acceptance.failing()))
        if run.acceptance.failing() and not args.allow_fail:
            return 1
        return 0

    # ---- 写命令：visualize
    # **从产物重建再渲染**，不全跑流水线。理由见 `pipeline.reconstruct_run`：
    # 全跑一遍（实测 44 s）证明的只是「流水线是确定的」，而这里要证明的是
    # 「out/ 里那些产物足以把 report.html 一字不差地重造出来」——
    # 后者才是工作区纪律「产物从字典渲染」的可执行形式，而且只要 ~0.5 s。
    if args.command == "visualize":
        with _quiet_prose():
            products, from_disk = _load_view(cfg, out_dir)
        if not from_disk:
            _note(f"[visualize] out/ 缺少产物或版本对不上 —— "
                  "本次是内存重跑，未写盘")
        rendered = render_html(reconstruct_run(products))
        target = Path(out_dir) / "report.html"
        before = target.read_text(encoding="utf-8") if target.is_file() else None
        write_products(out_dir, {"report.html": rendered})
        same = before is not None and before == rendered
        if want_json:
            _emit_json({"written": str(target).replace("\\", "/"),
                        "byte_identical_to_previous": bool(same),
                        "rebuilt_from_products": True})
        else:
            print(f"已重写 {target}")
            if before is None:
                print("（此前没有 report.html，无法比较）")
            else:
                print(f"与重渲染前逐位相同：{'是' if same else '否'}")
        return 0 if (before is None or same) else 1

    # ---- 只读命令
    payload_fn, prose_fn = _VIEWS[args.command]
    if want_json:
        with _quiet_prose():
            products, from_disk = _load_view(cfg, out_dir)
            payload = payload_fn(products, args)
        if not from_disk:
            _note(f"[{args.command}] out/ 缺少产物或版本对不上 —— "
                  "本次是内存重跑，未写盘")
        _emit_json(payload)
    else:
        products, from_disk = _load_view(cfg, out_dir)
        if not from_disk:
            print(f"（out/ 缺少产物或版本对不上 —— 本次是内存重跑，未写盘）")
            print()
        prose_fn(payload_fn(products, args))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
