"""训练：手写 SGD（动量 + 全局范数裁剪 + weight decay）。

只有一个目的：**拿到一个真的学到了东西的 fp16 基座**，然后全部量化候选
都在它上面跑。如果基座本身是随机权重，量化误差就没有可损伤的结构，
整张对比矩阵会退化成噪声。

## 三条纪律

1. **`batch_grads` 返回的是「和」，这里统一除 `1/B`。** 口径只在一处生效。
   （`DistillFactory` 坑 219：上游梯度被多除一次，声明的权重静默缩水 40 倍。）
2. **裁剪必须计数。** `grad_clip_hits` 是这条保险丝的实际触发次数 ——
   **一条从不触发的保险丝与没有保险丝无法区分**，所以它必须被算出来、被断言。
3. **训练过程只用一条 RNG。** 它从 `derive_seed("quantfactory.train", seed)` 出来，
   不含任何量化参数 —— 否则改一个量化配置就会改掉基座。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .model import Model, accuracy, batch_grads, grad_norm, init_model, transpose_weights
from .rng import Rng, derive_seed
from .world import World

__all__ = ["TrainStats", "train_model", "cross_entropy"]


@dataclass
class TrainStats:
    steps: int
    batch: int
    final_loss: float
    initial_loss: float
    train_accuracy: float
    test_accuracy: float
    grad_norm_max: float
    grad_norm_mean: float
    grad_clip_hits: int
    loss_curve: list[float] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "steps": self.steps,
            "batch": self.batch,
            "initial_loss": round(self.initial_loss, 10),
            "final_loss": round(self.final_loss, 10),
            "train_accuracy": round(self.train_accuracy, 10),
            "test_accuracy": round(self.test_accuracy, 10),
            "grad_norm_max": round(self.grad_norm_max, 10),
            "grad_norm_mean": round(self.grad_norm_mean, 10),
            "grad_clip_hits": self.grad_clip_hits,
            "loss_curve": [round(v, 10) for v in self.loss_curve],
            "note": self.note,
        }


def cross_entropy(logits: Sequence[float], y: int) -> float:
    m = max(logits)
    es = [math.exp(v - m) for v in logits]
    s = math.fsum(es)
    return (m + math.log(s)) - logits[y]


def _momentum_step(
    model: Model,
    dW: Sequence[Sequence[Sequence[float]]],
    db: Sequence[Sequence[float]],
    vW: list[list[list[float]]],
    vb: list[list[float]],
    *,
    lr: float,
    inv_batch: float,
    clip: float,
    momentum: float,
    weight_decay: float,
) -> None:
    """`v ← μ·v − lr·(g/B)·s`；`θ ← θ + v − lr·wd·θ`。就地改 `model` 与动量缓冲。"""
    norm = grad_norm(dW, db, inv_batch=inv_batch)
    scale = (clip / norm) if (norm > clip > 0.0) else 1.0
    k = lr * inv_batch * scale

    for l in range(len(model.layers)):
        layer = model.layers[l]
        for o in range(len(layer.W)):
            row = layer.W[o]
            vrow = vW[l][o]
            drow = dW[l][o]
            for j in range(len(row)):
                v = momentum * vrow[j] - k * drow[j]
                vrow[j] = v
                row[j] += v - lr * weight_decay * row[j]
            v = momentum * vb[l][o] - k * db[l][o]
            vb[l][o] = v
            layer.b[o] += v - lr * weight_decay * layer.b[o]


def train_model(world: World, cfg: Mapping[str, Any]) -> tuple[Model, TrainStats]:
    t = cfg["train"]
    steps = int(t["steps"])
    batch = int(t["batch"])
    lr = float(t["lr"])
    mu = float(t["momentum"])
    wd = float(t["weight_decay"])
    clip = float(t["grad_clip"])

    model = init_model(world, cfg)
    rng = Rng(derive_seed("quantfactory.train", int(cfg["seed"])))

    vW = [[[0.0] * len(l.W[0]) for _ in range(len(l.W))] for l in model.layers]
    vb = [[0.0] * len(l.b) for l in model.layers]

    n_train = len(world.train_x)
    b = min(batch, n_train)
    inv_batch = 1.0 / b

    curve: list[float] = []
    norms: list[float] = []
    clip_hits = 0
    first_loss = 0.0
    last_loss = 0.0

    sample_every = max(1, steps // 24)

    for step in range(steps):
        idx = rng.sample_indices(n_train, b)
        X = [world.train_x[i] for i in idx]
        Y = [world.train_y[i] for i in idx]

        W_T = transpose_weights(model)
        dW, db = batch_grads(model, W_T, X, Y)

        norm = grad_norm(dW, db, inv_batch=inv_batch)
        norms.append(norm)
        if clip > 0.0 and norm > clip:
            clip_hits += 1

        if step == 0 or step % sample_every == 0 or step == steps - 1:
            from .model import logits_of

            ls = [cross_entropy(logits_of(model, x), y) for x, y in zip(X, Y)]
            cur = math.fsum(ls) / len(ls)
            curve.append(cur)
            if step == 0:
                first_loss = cur
            last_loss = cur

        _momentum_step(
            model, dW, db, vW, vb,
            lr=lr, inv_batch=inv_batch, clip=clip,
            momentum=mu, weight_decay=wd,
        )

    return model, TrainStats(
        steps=steps,
        batch=b,
        initial_loss=first_loss,
        final_loss=last_loss,
        train_accuracy=accuracy(model, world.train_x, world.train_y),
        test_accuracy=accuracy(model, world.test_x, world.test_y),
        grad_norm_max=max(norms) if norms else 0.0,
        grad_norm_mean=(math.fsum(norms) / len(norms)) if norms else 0.0,
        grad_clip_hits=clip_hits,
        loss_curve=curve,
        note="基座只训一次；全部量化候选共用它。裁剪触发次数是这条保险丝的实测读数。",
    )
