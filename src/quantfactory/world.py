"""世界：一个有结构、**权重离群**、**激活也离群**的合成分类任务。

## 为什么任务必须同时造出两种离群

如果权重是均匀的小数，那么所有量化器（RTN / GPTQ / AWQ / K-quant）的表现
几乎一样 —— 整张对比矩阵会退化成一片噪声，而这个项目就只是在跑脚本。
**离群值是 AWQ 与 GPTQ 存在的理由**，所以世界必须把它们造出来。

两条通道各自负责一件事：

| 通道 | 尺度 | 它制造什么 | 谁受影响 |
| --- | --- | --- | --- |
| `narrow_dims` | 小（0.25） | **权重离群** —— 输入小，读它的权重必须放大 | RTN 被同组的大权重撑爆 scale |
| `wide_dims` | 大（4.0） | **激活离群** —— 输入大，`H = E[xxᵀ]` 病态 | GPTQ 的补偿更值钱；AWQ 的缩放有的可做 |

其余通道尺度 1.0。**噪声按同一尺度缩放**，于是每个通道的信噪比相同 ——
否则窄通道会退化成纯噪声，模型直接忽略它们，权重离群就不会出现
（第一版就是踩了这个：离群通道的权重只比普通通道大 2 倍，不是 6 倍）。

## 三块数据必须互不重叠

| 分片 | 用途 | 谁读它 |
| --- | --- | --- |
| train | 训 fp16 基座 | `train.py` |
| test | 测质量（四维里的 Quality 轴） | `measure.py` |
| calib_pool | 校准集池，按档位抽前 k 行 | `calib.py` |

三者各自 fork 一条独立的 RNG（见 `rng.Rng.fork`），并且**校准集绝不从测试集里抽**。
理由很直接：拿测试集当校准集，测出来的质量就是「校准过的那个集合上的质量」，
这个读数没有意义。这条不变量由 `acceptance.test_and_calibration_sets_are_disjoint`
逐行哈希核对。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .rng import Rng, derive_seed, digest_of

__all__ = [
    "World",
    "build_world",
    "world_digest",
    "channel_scales",
    "channel_spreads",
    "layer_roles",
]


@dataclass
class World:
    d_in: int
    n_classes: int
    hidden: list[int]
    n_prototypes: int
    prototypes: list[list[float]]
    proto_class: list[int]
    train_x: list[list[float]]
    train_y: list[int]
    test_x: list[list[float]]
    test_y: list[int]
    calib_pool: list[list[float]]
    # 世界的结构性读数（进产物，用来解释「为什么这个任务可解 / 为什么量化会掉点」）
    scales: list[float]
    narrow_dims: list[int]
    wide_dims: list[int]
    margin: float
    mean_pairwise_dist: float
    class_balance: list[int]

    @property
    def n_calib_pool(self) -> int:
        return len(self.calib_pool)

    def to_dict(self) -> dict[str, Any]:
        return {
            "d_in": self.d_in,
            "n_classes": self.n_classes,
            "hidden": list(self.hidden),
            "n_prototypes": self.n_prototypes,
            "narrow_dims": list(self.narrow_dims),
            "wide_dims": list(self.wide_dims),
            "channel_scale_summary": {
                "narrow": round(self.scales[self.narrow_dims[0]], 10) if self.narrow_dims else None,
                "wide": round(self.scales[self.wide_dims[0]], 10) if self.wide_dims else None,
                "default": 1.0,
            },
            "n_train": len(self.train_x),
            "n_test": len(self.test_x),
            "n_calib_pool": len(self.calib_pool),
            "margin": round(self.margin, 10),
            "mean_pairwise_dist": round(self.mean_pairwise_dist, 10),
            "class_balance": list(self.class_balance),
        }

    # 大数组不进指纹（它们由 seed 决定，seed 已经在版本摘要里了）
    def digest(self) -> str:
        return digest_of(self.to_dict(), length=16)


def world_digest(world: World) -> str:
    return world.digest()


def channel_scales(cfg: Mapping[str, Any]) -> list[float]:
    """每个输入通道的**幅度**（激活的量级）。**世界的结构，不是超参。**

    基础谱是**对数正态**（`sigma` 控制离散度），再在两个通道上各钉一个极值
    （narrow 极小 / wide 极大）。对数正态而不是均匀：均匀谱下组内
    `max/median` 只有 2.7×，int3 以上完全不掉点，整张矩阵没有信息。

    种子只来自 `(seed, d_in, sigma)` —— 改 narrow/wide 的具体数值**不会**
    重抽整个谱，只会覆盖那两个位置。这样「调一个旋钮」的因果是干净的。
    """
    w = cfg["world"]
    d = int(w["d_in"])
    sigma = float(w["channel_scale_sigma"])
    rng = Rng(derive_seed("quantfactory.scales", int(cfg["seed"]), d, sigma))
    s = [math.exp(sigma * rng.normal()) for _ in range(d)]
    for j in w["narrow_dims"]:
        if 0 <= int(j) < d:
            s[int(j)] = float(w["narrow_scale"])
    for j in w["wide_dims"]:
        if 0 <= int(j) < d:
            s[int(j)] = float(w["wide_scale"])
    return s


def channel_spreads(cfg: Mapping[str, Any], scales: Sequence[float]) -> list[float]:
    """每个通道的**判别信号**幅度（原型在该通道上的散布）。

    与 `channel_scales` **解耦**是关键：

    | 通道 | 幅度 `scale` | 信号 `spread` | 后果 |
    | --- | --- | --- | --- |
    | narrow | 极小 | = scale | 信息必须靠**大权重**放大 ⇒ 权重离群 |
    | wide | 极大 | = scale × 0.05 | **只制造激活离群**，权重 ≈ 0 |
    | 其余 | 对数正态 | = scale | 均匀信噪比 |

    宽通道若也携带信号，模型就会给它**大权重**，于是 `|W|` 与激活**正相关** ——
    而 AWQ 的前提是**负相关**。第一版就是这样，结果 AWQ 的 α 从 0 到 1.5
    单调变差（0.7715 → 0.5488），看起来像 AWQ 实现错了，其实是世界的相关性反了。
    """
    w = cfg["world"]
    ratio = float(w["wide_spread_ratio"])
    out = list(scales)
    for j in w["wide_dims"]:
        if 0 <= int(j) < len(out):
            out[int(j)] = out[int(j)] * ratio
    return out


def _make_prototypes(rng: Rng, n: int, scales: Sequence[float]) -> list[list[float]]:
    """造 `n` 个原型：`proto[k][j] ~ N(0, scales[j]²)`。**不做归一化** ——
    归一化会把刚造出来的尺度差抹掉，而尺度差正是这个世界的核心。"""
    return [[rng.normal(0.0, scales[j]) for j in range(len(scales))] for _ in range(n)]


def _min_pairwise_distance(protos: Sequence[Sequence[float]]) -> float:
    best = float("inf")
    for i in range(len(protos)):
        for j in range(i + 1, len(protos)):
            best = min(best, math.dist(protos[i], protos[j]))
    return best


def _mean_pairwise_distance(protos: Sequence[Sequence[float]]) -> float:
    ds = [
        math.dist(protos[i], protos[j])
        for i in range(len(protos))
        for j in range(i + 1, len(protos))
    ]
    return sum(ds) / len(ds) if ds else 0.0


def _sample(rng: Rng, protos: Sequence[Sequence[float]], labels: Sequence[int],
            n: int, noise: float, scales: Sequence[float]) -> tuple[list[list[float]], list[int]]:
    """抽 `n` 条：先均匀挑一个原型，再叠加**按通道尺度缩放**的高斯噪声。

    噪声缩放是关键（见模块文档）：不缩放的话窄通道全是噪声，模型会忽略它们，
    「窄通道 ⇒ 大权重」这条因果链就断了。
    """
    k = len(protos)
    xs: list[list[float]] = []
    ys: list[int] = []
    for _ in range(n):
        p = int(rng.random() * k)
        proto = protos[p]
        xs.append([proto[j] + noise * scales[j] * rng.normal() for j in range(len(proto))])
        ys.append(labels[p])
    return xs, ys


def build_world(cfg: Mapping[str, Any]) -> World:
    """造世界。**根种子只用任务参数** —— 不含 seed 之外的任何东西。

    `DistillFactory` 那条坑（246）与 `ModelOptimizationFactory` 那条坑（239）
    是同一个：根种子带上一个「只影响下游」的参数，改它就会连带改掉上游数据，
    于是「改一个不相干的配置」会让所有历史结论失效。这里根种子只用
    `cfg["seed"]` 与世界的结构性参数（维度 / 层宽 / 原型数 / 窄宽通道定义）。
    """
    w = cfg["world"]
    d_in = int(w["d_in"])
    n_classes = int(w["n_classes"])
    hidden = [int(h) for h in w["hidden"]]
    n_protos = int(w["n_prototypes"])
    scales = channel_scales(cfg)

    # 根种子：**只用 seed + 世界的结构性参数**。
    # 噪声、样本数都是「读数参数」，不进种子 —— 否则改一个样本数会让所有历史结论失效。
    root = derive_seed(
        "quantfactory.world",
        int(cfg["seed"]),
        d_in,
        n_classes,
        tuple(hidden),
        n_protos,
        tuple(w["narrow_dims"]),
        tuple(w["wide_dims"]),
    )

    protos = _make_prototypes(
        Rng(root).fork("prototypes"), n_protos, channel_spreads(cfg, scales)
    )
    # 每类至少一个原型；余下的按轮转分配（保证类别大致均衡）
    labels = [k % n_classes for k in range(n_protos)]

    noise = float(w["noise"])
    train_x, train_y = _sample(
        Rng(root).fork("train"), protos, labels, int(w["n_train"]), noise, scales
    )
    test_x, test_y = _sample(
        Rng(root).fork("test"), protos, labels, int(w["n_test"]), noise, scales
    )
    calib_pool, _ = _sample(
        Rng(root).fork("calib"), protos, labels, int(w["n_calib_pool"]), noise, scales
    )

    balance = [0] * n_classes
    for y in train_y:
        balance[y] += 1

    return World(
        d_in=d_in,
        n_classes=n_classes,
        hidden=hidden,
        n_prototypes=n_protos,
        prototypes=protos,
        proto_class=labels,
        train_x=train_x,
        train_y=train_y,
        test_x=test_x,
        test_y=test_y,
        calib_pool=calib_pool,
        scales=scales,
        narrow_dims=[int(j) for j in w["narrow_dims"]],
        wide_dims=[int(j) for j in w["wide_dims"]],
        margin=_min_pairwise_distance(protos),
        mean_pairwise_dist=_mean_pairwise_distance(protos),
        class_balance=balance,
    )


def layer_roles(world: World) -> list[str]:
    """权重矩阵的角色名，顺序 = 前向顺序。

    三层：`layer0`（输入→隐1）、`layer1`（隐1→隐2）、`head`（隐2→logits）。
    `kquant` 的层敏感度混合就作用在这三个名字上（首层与末层更敏感）。
    """
    return ["layer0"] + [f"layer{i}" for i in range(1, len(world.hidden))] + ["head"]

