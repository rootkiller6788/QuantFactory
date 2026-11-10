"""落盘：JSON / CSV / DOT / MD / HTML 的写出。**三条硬约定。**

1. **换行一律 `\\n`。** Windows 上 `open(..., "w")` 默认会把 `\\n` 翻成 `\\r\\n`，
   于是同一份配置在两台机器上产出的字节不同 —— 而「逐位可复现」是底线。
   所有写出都显式 `newline="\\n"`。
2. **`None` 与 `0` 在 CSV 里必须能区分。** `nan` 表示「没测到」，`0` 表示「测到了 0」。
   把两者都渲染成空串或都渲染成 `0`，报告里那一列就再也读不出「没测到」这件事。
3. **JSON 用 `sort_keys=True` + `ensure_ascii=False` + `indent=2`。**
   键序稳定是为了 diff 可读；`ensure_ascii=False` 是为了让中文在产物里是人话。
   **不做 `allow_nan`** —— 默认的 `True` 会把 `NaN` 写进 JSON，而那是非法 JSON。

`str` 值按文本写，其它值按 JSON 写。这条规则在实践里没有歧义：`.json` 永远不是
`str`，而 `.csv` / `.dot` / `.md` / `.html` 永远是。
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = ["dump_json", "csv_text", "is_text", "render", "write_outputs", "cell"]


def dump_json(obj: Any) -> str:
    return (
        json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def cell(v: Any) -> str:
    """CSV 单元格。**`None` → 空串，`0` → `"0"`。** 见模块文档第 2 条。"""
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return ""
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return repr(round(v, 8))
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _escape(s: str) -> str:
    return '"' + s.replace('"', '""') + '"' if any(c in s for c in ',"\n') else s


def csv_text(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    """手写 CSV，不用 `csv` 模块 —— 只是为了把换行与转义的两条约定放在明处。"""
    out = [",".join(_escape(str(h)) for h in header)]
    for r in rows:
        out.append(",".join(_escape(cell(v)) for v in r))
    return "\n".join(out) + "\n"


def is_text(v: Any) -> bool:
    return isinstance(v, str)


def render(v: Any) -> str:
    return v if isinstance(v, str) else dump_json(v)


def write_outputs(out_dir: str | Path, products: Mapping[str, Any]) -> list[str]:
    """写出全部产物，返回**实际写下的相对路径**（升序）。

    **每个文件先写同目录临时文件，再 `os.replace` 原子改名。**
    产物契约说版本化文件「要么是这一次运行的完整结果，要么不存在」；直接
    `write_text` 在写到一半被打断时，会留下一个字节不全、却仍然能通过
    文件名与 JSON 前缀看起来合法的产物 —— 下一次只读命令会把它当成有效结果。
    临时文件名带 pid，避免两个进程同时写同一个 `out/` 时互相覆盖。
    """
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for rel in sorted(products):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        text = render(products[rel])
        tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
        try:
            tmp.write_text(text, encoding="utf-8", newline="\n")
            os.replace(tmp, p)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        written.append(str(rel).replace("\\", "/"))
    return written
