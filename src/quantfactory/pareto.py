"""Pareto 前沿：支配 / 非支配排序 / 拥挤度 / 超体积 / 间距 / 拐点。

**全部按「最大化」口径。** 最小化的目标（体积 / 内存 / 延迟）在进这个模块之前
先取负号 —— 一个模块里混着两种方向，是「支配关系写反」这类 bug 的温床，
而它的症状是「前沿看起来很正常，只是选出来的方案总是最差的那个」。

本项目的目标有**四维**（Quality / Latency / Memory / Size），所以这里每个函数
都必须对 `d` 维通用，不能只写死二维。

## 超体积（hypervolume）

用 **HSO**（Hypervolume by Slicing Objectives）：沿最后一维切片，
每片的高度乘以该片内 `(d−1)` 维的 HV。

```
vol = Σ_i (p_i[k] − p_{i+1}[k]) · HV_{d−1}({p_1..p_i 投影}, ref[:-1])
```

**所有候选必须共享同一个参考点**，否则会出现「多一个烂点，超体积反而变大」——
参考点必须被所有点支配（取最差质量再外推、最高成本再外推）。

## 拐点（knee）

取「到两端连线距离最大」的点。**四维下没有「两端连线」这个概念** ——
所以拐点必须**先在指定的两个轴上投影**再算（本项目用 `(Quality, Size)` 平面，
也就是部署选型时最常用的那张二维图）。传 `axes=(0, 1)` 时退化成经典的二维拐点。
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Sequence

__all__ = [
    "dominates",
    "non_dominated_of",
    "dedupe",
    "fast_non_dominated_sort",
    "crowding_distance",
    "hypervolume",
    "spacing",
    "knee_index",
]


def dominates(a: Sequence[float], b: Sequence[float]) -> bool:
    """`a` 支配 `b`（全部按最大化）：`a` 每维不差于 `b`，且至少一维严格更好。"""
    better = False
    for x, y in zip(a, b):
        if x < y:
            return False
        if x > y:
            better = True
    return better


def dedupe(items: Iterable, key: Callable[[object], str]) -> list:
    """按 `key` 去重，**保留首次出现**。跨算法/跨档位收拢时用。"""
    seen: set[str] = set()
    out = []
    for it in items:
        k = key(it)
        if k in seen:
            continue
        seen.add(k)
        out.append(it)
    return out


def non_dominated_of(items: Sequence, vec: Callable[[object], Sequence[float]]) -> list:
    """非支配子集。O(n²)，n 是候选数（本项目 n 在几百以内，够用）。"""
    vecs = [tuple(vec(it)) for it in items]
    keep: list = []
    for i, it in enumerate(items):
        dominated = False
        for j, other in enumerate(items):
            if i == j:
                continue
            if dominates(vecs[j], vecs[i]):
                dominated = True
                break
        if not dominated:
            keep.append(it)
    return keep


def fast_non_dominated_sort(items: Sequence, vec: Callable[[object], Sequence[float]]) -> list[list]:
    """NSGA-II 式分层。返回 `[front0, front1, ...]`，`front0` 是最优层。"""
    n = len(items)
    vecs = [tuple(vec(it)) for it in items]
    dominated_by: list[list[int]] = [[] for _ in range(n)]
    dom_count = [0] * n
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dominates(vecs[i], vecs[j]):
                dominated_by[i].append(j)
            elif dominates(vecs[j], vecs[i]):
                dom_count[i] += 1
    fronts: list[list[int]] = []
    cur = [i for i in range(n) if dom_count[i] == 0]
    while cur:
        fronts.append(cur)
        nxt: list[int] = []
        for i in cur:
            for j in dominated_by[i]:
                dom_count[j] -= 1
                if dom_count[j] == 0:
                    nxt.append(j)
        cur = nxt
    return [[items[i] for i in f] for f in fronts]


def crowding_distance(items: Sequence, vec: Callable[[object], Sequence[float]]) -> list[float]:
    """拥挤度。边界点给 `inf`（永远保留），其余按相邻间距之和。"""
    n = len(items)
    if n == 0:
        return []
    if n <= 2:
        return [float("inf")] * n
    vecs = [tuple(vec(it)) for it in items]
    d = len(vecs[0])
    dist = [0.0] * n
    for k in range(d):
        order = sorted(range(n), key=lambda i: vecs[i][k])
        lo = vecs[order[0]][k]
        hi = vecs[order[-1]][k]
        span = hi - lo
        dist[order[0]] = float("inf")
        dist[order[-1]] = float("inf")
        if span <= 0.0:
            continue
        for t in range(1, n - 1):
            i = order[t]
            dist[i] += (vecs[order[t + 1]][k] - vecs[order[t - 1]][k]) / span
    return dist


# --------------------------------------------------------------------------- #
# 超体积
# --------------------------------------------------------------------------- #


def _hv_rec(pts: list[tuple[float, ...]], ref: tuple[float, ...]) -> float:
    if not pts:
        return 0.0
    d = len(ref)
    if d == 1:
        return max(0.0, max(p[0] for p in pts) - ref[0])
    pts = sorted(pts, key=lambda p: p[-1], reverse=True)
    vol = 0.0
    for i, p in enumerate(pts):
        top = p[-1]
        bottom = pts[i + 1][-1] if i + 1 < len(pts) else ref[-1]
        if top <= bottom:
            continue
        sub = [q[:-1] for q in pts[: i + 1]]
        vol += (top - bottom) * _hv_rec(sub, ref[:-1])
    return vol


def hypervolume(points: Sequence[Sequence[float]], ref: Sequence[float]) -> float:
    """被 `points` 支配、且被 `ref` 支配的那块体积（全部最大化口径）。

    不支配 `ref` 的点会被剔除 —— 它们对体积没有贡献，
    留着会让结果依赖「有没有加一个烂点」。
    """
    ref_t = tuple(float(x) for x in ref)
    pts = [tuple(float(x) for x in p) for p in points]
    pts = [p for p in pts if all(p[i] > ref_t[i] for i in range(len(ref_t)))]
    if not pts:
        return 0.0
    pts = [tuple(p) for p in non_dominated_of(pts, lambda p: p)]
    return _hv_rec(pts, ref_t)


def spacing(points: Sequence[Sequence[float]]) -> float:
    """间距指标：前沿上相邻点距离的标准差 / 平均距离。越小越均匀。

    **每一维先按本点集的 min–max 归一化到 [0,1]** —— 否则「体积（GB）」这种
    量纲大的维度会单独支配距离，而质量（0~1）那一维几乎不贡献。
    四维前沿上不做这一步，间距读数就只是体积读数的复述。
    """
    pts = [tuple(float(x) for x in p) for p in points]
    if len(pts) < 2:
        return 0.0
    d = len(pts[0])
    lo = [min(p[k] for p in pts) for k in range(d)]
    hi = [max(p[k] for p in pts) for k in range(d)]
    span = [(hi[k] - lo[k]) or 1.0 for k in range(d)]
    norm = [tuple((p[k] - lo[k]) / span[k] for k in range(d)) for p in pts]
    norm = sorted(norm)
    ds = [math.dist(norm[i], norm[i + 1]) for i in range(len(norm) - 1)]
    mean = sum(ds) / len(ds)
    if mean <= 0.0:
        return 0.0
    var = sum((dd - mean) ** 2 for dd in ds) / len(ds)
    return math.sqrt(var) / mean


def knee_index(points: Sequence[Sequence[float]], axes: tuple[int, int] = (0, 1)) -> int:
    """拐点下标：**在 `axes` 指定的两个轴上投影**后，到「两端连线」垂直距离最大的点。

    四维前沿没有「两端连线」这个概念，所以必须先投影。默认 `(0, 1)`；
    本项目传 `(0, 3)`（Quality × Size）—— 部署选型最常用的那张二维图。

    两点或更少时返回 0。**全部最大化口径**，所以「两端」是投影后
    第一维最小与最大的那两个点。
    """
    i0, i1 = axes
    pts = [(float(p[i0]), float(p[i1])) for p in points]
    n = len(pts)
    if n <= 2:
        return 0
    lo = (min(p[0] for p in pts), min(p[1] for p in pts))
    hi = (max(p[0] for p in pts), max(p[1] for p in pts))
    span = ((hi[0] - lo[0]) or 1.0, (hi[1] - lo[1]) or 1.0)
    norm = [((p[0] - lo[0]) / span[0], (p[1] - lo[1]) / span[1]) for p in pts]

    order = sorted(range(n), key=lambda i: norm[i][0])
    a = norm[order[0]]
    b = norm[order[-1]]
    ab = (b[0] - a[0], b[1] - a[1])
    abn = math.hypot(ab[0], ab[1])
    if abn <= 0.0:
        return order[0]
    best_i, best_d = order[0], -1.0
    for idx in order[1:-1]:
        p = norm[idx]
        ap = (p[0] - a[0], p[1] - a[1])
        cross = ap[0] * ab[1] - ap[1] * ab[0]
        dist = abs(cross) / abn
        if dist > best_d:
            best_d, best_i = dist, idx
    return best_i
