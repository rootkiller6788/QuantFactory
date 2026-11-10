"""模型：三层 MLP，手写前向 / 反向，内积一律走 `math.sumprod`。

```
x(64) --W0--> h1(64) --tanh--> --W1--> h2(64) --tanh--> --W2--> logits(8)
```

三层而不是两层，是为了让 `kquant` 的**层敏感度混合**有地方可去：
两层的话「给首层加位宽、给末层加位宽」就等于给所有层加位宽，那不叫混合。

## 性能：三处刻意的写法

1. **前向按样本算**（`math.sumprod(W[o], a)`）。批量化在纯 Python 里没有收益 ——
   循环次数一样多，只是多了一层索引。
2. **反向先转置**。`dH[b][j] = Σ_o dZ[b][o]·W[o][j]` 里的 `W[o][j]` 如果每次现取，
   每步要多做 `B×in` 次列表构造（实测这一步比内积本身还贵）。所以反向用
   `W_T[j][o] = W[o][j]`，转置一次、用整批。
3. **`dW` 用「列优先 + 内积」**：`dW[o][j] = Σ_b dZ[b][o]·A[b][j]`。
   把整批转置成 `dZ_T[o][b]` / `A_T[j][b]` 之后，每个 `dW[o][j]` 就是一次
   `sumprod`，**调用次数固定为 `out×in`，与批大小无关** —— 于是批越大越划算。

实测（3.13.12，B=64）：一步 ≈ 23 ms，其中前向 8.7k 次内积、反向 17k 次。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .rng import Rng, derive_seed
from .world import World, layer_roles

__all__ = [
    "Layer",
    "Model",
    "init_model",
    "forward",
    "forward_logits",
    "logits_of",
    "predict",
    "accuracy",
    "batch_grads",
    "grad_norm",
    "param_count",
    "model_digest",
    "clone",
    "transpose_weights",
]

_sumprod = math.sumprod


@dataclass
class Layer:
    """一层。`W` 形状 `(out, in)`，`b` 长度 `out`，`role` 是它在 `kquant` 里的名字。"""

    role: str
    W: list[list[float]]
    b: list[float]
    activation: str = "tanh"  # "tanh" 或 "none"（末层）

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.W), len(self.W[0]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "shape": list(self.shape),
            "activation": self.activation,
            "n_params": len(self.W) * len(self.W[0]) + len(self.b),
        }


@dataclass
class Model:
    layers: list[Layer]

    @property
    def roles(self) -> list[str]:
        return [l.role for l in self.layers]

    @property
    def n_params(self) -> int:
        return sum(len(l.W) * len(l.W[0]) + len(l.b) for l in self.layers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "layers": [l.to_dict() for l in self.layers],
            "n_params": self.n_params,
            "roles": self.roles,
        }


def param_count(model: Model) -> int:
    return model.n_params


def clone(model: Model) -> Model:
    return Model(
        layers=[
            Layer(role=l.role, W=[row[:] for row in l.W], b=l.b[:], activation=l.activation)
            for l in model.layers
        ]
    )


def model_digest(model: Model) -> str:
    """模型指纹：**只用结构 + 参数量**，不用参数值。

    参数值由 seed 决定，seed 已经在版本摘要里 —— 把几万个浮点折进指纹
    只会让指纹变慢，不增加任何信息。
    """
    from .rng import digest_of

    return digest_of(
        {"roles": model.roles, "shapes": [list(l.shape) for l in model.layers],
         "n_params": model.n_params},
        length=16,
    )


# --------------------------------------------------------------------------- #
# 初始化
# --------------------------------------------------------------------------- #


def init_model(world: World, cfg: Mapping[str, Any]) -> Model:
    """Xavier 式初始化：`U(−s, s)`，`s = init_scale / sqrt(in)`。

    根种子只用「模型结构」这一件事（`d_in` / `hidden` / `n_classes` / seed），
    **不含任何量化或评测参数** —— 否则改一个量化配置就会改掉基座模型，
    「所有候选共用同一个 fp16 基座」这个前提会静默失效。
    """
    s = float(cfg["train"]["init_scale"])
    root = derive_seed(
        "quantfactory.model",
        int(cfg["seed"]),
        world.d_in,
        tuple(world.hidden),
        world.n_classes,
    )
    rng = Rng(root)

    dims = [world.d_in] + list(world.hidden) + [world.n_classes]
    roles = layer_roles(world)
    layers: list[Layer] = []
    for i in range(len(dims) - 1):
        out, inp = dims[i + 1], dims[i]
        lim = s / math.sqrt(inp)
        W = [[rng.uniform(-lim, lim) for _ in range(inp)] for _ in range(out)]
        b = [0.0] * out
        layers.append(
            Layer(
                role=roles[i],
                W=W,
                b=b,
                activation="none" if i == len(dims) - 2 else "tanh",
            )
        )
    return Model(layers=layers)


# --------------------------------------------------------------------------- #
# 前向
# --------------------------------------------------------------------------- #


def _forward(
    model: Model, x: Sequence[float], *, keep_acts: bool
) -> tuple[list[float], list[list[float]] | None]:
    """前向的唯一实现。`keep_acts=False` 时不保留逐层激活，供纯推理使用。

    局部绑定 `sumprod` / `tanh` 是为了让热点循环里的名称解析发生在函数入口，
    而不是每次内积、每个激活元素各查一次模块全局。数值与顺序完全不变。
    """
    sumprod = _sumprod
    tanh = math.tanh
    a: list[float] = list(x)
    acts: list[list[float]] | None = [a] if keep_acts else None
    for layer in model.layers:
        b = layer.b
        z = [sumprod(row, a) + b[o] for o, row in enumerate(layer.W)]
        a = [tanh(v) for v in z] if layer.activation == "tanh" else z
        if keep_acts:
            acts.append(a)  # type: ignore[union-attr]
    return a, acts


def forward(model: Model, x: Sequence[float]) -> tuple[list[float], list[list[float]]]:
    """返回 `(logits, acts)`，`acts[0] == x`，`acts[i]` 是第 i 层输出。

    **`acts` 是校准需要的**（GPTQ 的 Hessian 与 AWQ 的激活幅度都从它来），
    所以这里一并返回，不要在校准里再跑一遍前向。
    """
    a, acts = _forward(model, x, keep_acts=True)
    return a, acts  # type: ignore[return-value]


def forward_logits(model: Model, x: Sequence[float]) -> list[float]:
    """只算 logits，**不构造 `acts`**。

    度量阶段每个候选要跑 512 条测试样本；那里只需要 argmax，保留逐层激活
    是纯浪费（每个样本多 4 次列表分配与全部激活元素的 append）。
    """
    return _forward(model, x, keep_acts=False)[0]


def logits_of(model: Model, x: Sequence[float]) -> list[float]:
    return forward_logits(model, x)


def predict(model: Model, x: Sequence[float]) -> int:
    lg = logits_of(model, x)
    best, bv = 0, lg[0]
    for i in range(1, len(lg)):
        if lg[i] > bv:
            best, bv = i, lg[i]
    return best


def accuracy(model: Model, X: Sequence[Sequence[float]], Y: Sequence[int]) -> float:
    if not X:
        return 0.0
    hit = 0
    for x, y in zip(X, Y):
        if predict(model, x) == y:
            hit += 1
    return hit / len(X)


# --------------------------------------------------------------------------- #
# 反向
# --------------------------------------------------------------------------- #


def _softmax(z: Sequence[float]) -> list[float]:
    m = max(z)
    es = [math.exp(v - m) for v in z]
    s = math.fsum(es)
    return [e / s for e in es]


def transpose_weights(model: Model) -> list[list[list[float]]]:
    """`W_T[i][j][o] = W[i][o][j]`。**反向里必须用它**，见模块文档第 2 条。"""
    return [[[l.W[o][j] for o in range(len(l.W))] for j in range(len(l.W[0]))]
            for l in model.layers]


def batch_grads(
    model: Model,
    W_T: Sequence[Sequence[Sequence[float]]],
    X: Sequence[Sequence[float]],
    Y: Sequence[int],
) -> tuple[list[list[list[float]]], list[list[float]]]:
    """一批样本的梯度（求和，不取平均）。

    ⚠️ **求和口径**是刻意的：调用方（`train.py`）最后统一 `scale(1/B)`。
    如果这里取平均、调用方又除一次，学习率会静默缩水 B 倍 —— 这正是
    `DistillFactory` 坑 219 的形态（上游梯度被多除一次，声明的权重缩水 40 倍）。
    口径写在函数名旁边，比写在注释里管用。

    返回 `(dW, db)`，形状与 `model.layers` 一致。

    ## `dW` 也必须走 `sumprod`（第一版这里是纯 Python 三重循环）

    第一版写的是 `for o: for j: dW[o][j] += d·a[j]` —— 每步 `B×out×in`
    次 Python 级乘加，B=64 时是 **55 万次**，实测占单步的 **一半**（22 ms）。
    改成「把整批转置成 `dZ_T[o][b]` / `A_T[j][b]`，再 `sumprod`」之后，
    调用次数固定为 `out×in`（与批大小无关），单步从 46 ms 降到 **20 ms**。
    转置本身的 `out×B + in×B` 次拷贝只有 1.5 ms —— 值得。
    """
    L = len(model.layers)
    B = len(X)

    # ---- 1) 前向，把每层的输入激活与输出 z 都留下（反向要用）
    As: list[list[list[float]]] = [[] for _ in range(L)]  # As[i][b] = 第 i 层的输入
    Zs: list[list[list[float]]] = [[] for _ in range(L)]  # Zs[i][b] = 第 i 层的线性输出
    for b in range(B):
        a: list[float] = list(X[b])
        for i, layer in enumerate(model.layers):
            As[i].append(a)
            z = [_sumprod(layer.W[o], a) + layer.b[o] for o in range(len(layer.W))]
            Zs[i].append(z)
            a = [math.tanh(v) for v in z] if layer.activation == "tanh" else z

    # ---- 2) 末层梯度：softmax 交叉熵就是 (p − onehot)
    D: list[list[float]] = []
    for b in range(B):
        p = _softmax(Zs[L - 1][b])
        d = p[:]
        d[Y[b]] -= 1.0
        D.append(d)

    dW = [[[0.0] * len(l.W[0]) for _ in range(len(l.W))] for l in model.layers]
    db = [[0.0] * len(l.b) for l in model.layers]

    # ---- 3) 逐层回传；`dW` / `db` 用整批转置后一次算完
    for i in range(L - 1, -1, -1):
        layer = model.layers[i]
        out_i, in_i = layer.shape
        dZ_T = [[D[b][o] for b in range(B)] for o in range(out_i)]
        A_T = [[As[i][b][j] for b in range(B)] for j in range(in_i)]
        dWi, dbi = dW[i], db[i]
        for o in range(out_i):
            row = dZ_T[o]
            dbi[o] = math.fsum(row)
            drow = dWi[o]
            for j in range(in_i):
                drow[j] = _sumprod(row, A_T[j])
        if i == 0:
            break
        # dL/dz_prev[j] = (Σ_o D[b][o]·W[i][o][j]) · tanh'(z_prev[j])
        # `As[i]` 就是第 i 层的输入，即 `tanh(z_prev)` —— 前向已经算过。
        # 这里复用它，省掉反向里 B×in 次重复的 `math.tanh`（本模型 3.4M 次/训练）。
        Aprev = As[i]
        WT = W_T[i]
        D = [
            [
                _sumprod(D[b], WT[j]) * (1.0 - Aprev[b][j] ** 2)
                for j in range(in_i)
            ]
            for b in range(B)
        ]
    return dW, db


def grad_norm(
    dW: Sequence[Sequence[Sequence[float]]],
    db: Sequence[Sequence[float]],
    *,
    inv_batch: float,
) -> float:
    """梯度**全局**范数（已按 `1/B` 折算）。

    裁剪必须用全局范数（不是逐参数），否则大梯度参数会被单独削平，
    整个更新的方向就被改掉了 —— 而方向才是 SGD 唯一在用的东西。
    这个函数单独存在，是为了让 `train.py` 与对抗用例读**同一个**口径。
    """
    sq = 0.0
    for l in range(len(dW)):
        for o in range(len(dW[l])):
            for v in dW[l][o]:
                sq += v * v
    for l in range(len(db)):
        for v in db[l]:
            sq += v * v
    return math.sqrt(sq) * inv_batch
