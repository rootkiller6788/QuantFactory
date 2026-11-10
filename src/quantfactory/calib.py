"""校准：从校准集算出**全部候选共享**的中间结果。

## 这份东西只算一次，而且是本项目最重要的一处「共享」

`CalibStats` **不依赖算法、不依赖位宽、不依赖分组** —— 它只依赖校准集本身。
所以它按**校准档位**（64 / 256）算一次，然后 GPTQ 与 AWQ 的所有候选共用。

「共享」这件事不能只写在文档里：`pipeline.py` 用**对象同一性**
（`assert stats[64] is stats[64]`）把它钉住。否则某天有人把 `build_calib`
挪进候选循环，结果**看起来完全一样**，只是慢了几百倍 —— 而这种退化
没有任何断言会发现。

## 两个量

| 量 | 定义 | 谁读它 |
| --- | --- | --- |
| `hessian` | `H = (1/n)·Σ_b a_b a_bᵀ`（每层的**输入**激活二阶矩） | GPTQ |
| `act_absmean` | `(1/n)·Σ_b |a_b[j]|`（逐输入通道的平均幅度） | AWQ |

`H` 还要**阻尼**（`H[j][j] += λ·mean(diag)`）并求逆 —— 求逆是 GPTQ 唯一
昂贵的一步，也是**校准档位唯一真正影响算法的地方**：校准样本数少于输入维度时
`H` 秩亏，阻尼与条件数就决定了 GPTQ 的补偿是否稳定。这一条会被实测出来
（`diag_condition` 进产物）。

## 一条自检

`inv_residual = max|H·H⁻¹ − I|` 进产物并被门槛断言 ——
**求逆写错不会报错，只会让 GPTQ 静默退化成「带噪声的 RTN」。**
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .model import Model, forward
from .world import World

__all__ = ["CalibStats", "build_calib", "cholesky_inverse", "hessian_of"]

_sumprod = math.sumprod


@dataclass
class CalibStats:
    n: int
    act_absmean: list[list[float]]          # 每层 (in,)
    act_rms: list[list[float]]              # 每层 (in,)
    hessian: list[list[list[float]]]        # 每层 (in, in)，已阻尼
    hessian_inv: list[list[list[float]]]    # 每层 (in, in)
    diag_condition: list[float]             # 每层：对角 max/min（**对角近似**）
    inv_residual: list[float]               # 每层：max|H·H⁻¹ − I|
    damp: float
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "damp": self.damp,
            "roles": [f"layer{i}" for i in range(len(self.act_absmean))],
            "diag_condition": [round(v, 10) for v in self.diag_condition],
            "inv_residual": [round(v, 16) for v in self.inv_residual],
            "act_absmean_max": [
                round(max(v), 10) if v else 0.0 for v in self.act_absmean
            ],
            "note": self.note,
        }


def _cholesky(A: Sequence[Sequence[float]], eps: float) -> list[list[float]]:
    """`A = L·Lᵀ`，`L` 下三角。对角出现非正数时用 `eps` 兜底（阻尼已经在前面加过）。"""
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        Li = L[i]
        for j in range(i + 1):
            s = A[i][j] - _sumprod(Li[:j], L[j][:j])
            if i == j:
                Li[j] = math.sqrt(s) if s > eps else math.sqrt(eps)
            else:
                Li[j] = s / L[j][j]
    return L


def cholesky_inverse(A: Sequence[Sequence[float]], eps: float = 1e-15) -> list[list[float]]:
    """对称正定矩阵求逆（Cholesky）。`A` 必须已经阻尼过。

    `L Lᵀ = A` ⇒ `A⁻¹ = L⁻ᵀ L⁻¹`。逐列解两次三角方程即可，不必显式求 `L⁻¹`。
    内积一律走 `sumprod` —— n=64 时单次求逆 ~40 ms，六次（2 档 × 3 层）合计 0.24 s。
    """
    n = len(A)
    L = _cholesky(A, eps)
    # 列访问友好：把 L 转置一份（L_T[k][i] = L[i][k]），否则回代要按列取 L[k][i]
    LT = [[L[i][k] for i in range(n)] for k in range(n)]
    X = [[0.0] * n for _ in range(n)]
    for c in range(n):
        # L y = e_c
        y = [0.0] * n
        for i in range(n):
            rhs = 1.0 if i == c else 0.0
            y[i] = (rhs - _sumprod(L[i][:i], y[:i])) / L[i][i]
        # Lᵀ x = y  ⇒  Σ_k L[k][i] x[k] = y[i]（k ≥ i）
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            s = _sumprod(LT[i][i + 1:], x[i + 1:])
            x[i] = (y[i] - s) / L[i][i]
        for i in range(n):
            X[i][c] = x[i]
    return X


def hessian_of(
    acts_T: Sequence[Sequence[float]],
    n: int,
) -> list[list[float]]:
    """`H[j][k] = (1/n)·Σ_b A[j][b]·A[k][b]`。

    传进来的是**转置过的激活**（`acts_T[j]` 是第 j 个输入通道的整列），
    于是每个 `H[j][k]` 就是一次 `sumprod` —— 调用次数 `in²`，与校准样本数无关。
    """
    d = len(acts_T)
    H = [[0.0] * d for _ in range(d)]
    inv_n = 1.0 / n
    for j in range(d):
        Aj = acts_T[j]
        Hj = H[j]
        for k in range(j, d):
            v = _sumprod(Aj, acts_T[k]) * inv_n
            Hj[k] = v
            H[k][j] = v  # 对称，省一半内积
    return H


def _damp(H: list[list[float]], damp: float) -> float:
    """就地加阻尼，返回 `λ·mean(diag)`（进产物，用来解释条件数）。"""
    n = len(H)
    mean = math.fsum(H[i][i] for i in range(n)) / n
    add = damp * mean
    if add <= 0.0:
        add = 1e-12
    for i in range(n):
        H[i][i] += add
    return add


def _max_residual(H: Sequence[Sequence[float]], Hinv: Sequence[Sequence[float]]) -> float:
    """`max|H·H⁻¹ − I|`。**这是求逆写错的唯一探针。**"""
    n = len(H)
    worst = 0.0
    for i in range(n):
        Hi = H[i]
        for k in range(n):
            col = [Hinv[j][k] for j in range(n)]
            v = _sumprod(Hi, col) - (1.0 if i == k else 0.0)
            worst = max(worst, abs(v))
    return worst


def build_calib(
    model: Model,
    world: World,
    cfg: Mapping[str, Any],
    n: int,
) -> CalibStats:
    """从校准池的**前 n 行**算校准统计。

    ⚠️ 前 n 行而不是随机抽 —— 随机抽会让 64 档与 256 档**不是嵌套的**
    （64 档的样本不一定是 256 档的前缀），于是「更多校准数据带来什么变化」
    这个比较就混进了「换了哪些样本」这第二个变量。
    取前缀保证两个档位是**嵌套**的，比较才是单变量的。
    """
    pool = world.calib_pool
    if n <= 0:
        raise ValueError(f"校准档位必须为正，得到 {n}")
    if n > len(pool):
        raise ValueError(f"校准档位 {n} 超过校准池 {len(pool)}")

    X = pool[:n]
    n_layers = len(model.layers)

    # ---- 收集每层的输入激活（转置存：acts_T[i][j][b]）
    acts_T: list[list[list[float]]] = [
        [[] for _ in range(len(model.layers[i].W[0]))] for i in range(n_layers)
    ]
    for x in X:
        a: list[float] = list(x)
        for i, layer in enumerate(model.layers):
            Ti = acts_T[i]
            for j in range(len(a)):
                Ti[j].append(a[j])
            z = [_sumprod(layer.W[o], a) + layer.b[o] for o in range(len(layer.W))]
            a = [math.tanh(v) for v in z] if layer.activation == "tanh" else z

    damp = float(cfg["quant"]["gptq"]["damp"])
    eps = float(cfg["quant"]["eps"])

    absmean: list[list[float]] = []
    rms: list[list[float]] = []
    hessians: list[list[list[float]]] = []
    hessians_inv: list[list[list[float]]] = []
    conds: list[float] = []
    residuals: list[float] = []

    for i in range(n_layers):
        Ti = acts_T[i]
        absmean.append([math.fsum(abs(v) for v in col) / n for col in Ti])
        rms.append([math.sqrt(math.fsum(v * v for v in col) / n) for col in Ti])

        H = hessian_of(Ti, n)
        _damp(H, damp)
        Hinv = cholesky_inverse(H, eps)
        hessians.append(H)
        hessians_inv.append(Hinv)

        diag = [H[j][j] for j in range(len(H))]
        conds.append(max(diag) / min(diag) if min(diag) > 0 else float("inf"))
        residuals.append(_max_residual(H, Hinv))

    return CalibStats(
        n=n,
        act_absmean=absmean,
        act_rms=rms,
        hessian=hessians,
        hessian_inv=hessians_inv,
        diag_condition=conds,
        inv_residual=residuals,
        damp=damp,
        note=(
            "只依赖校准集本身：不依赖算法 / 位宽 / 分组 ⇒ 按档位算一次，全部候选共用。"
            "diag_condition 是对角近似的条件数（不是特征值条件数），用来解释 GPTQ 的稳定性。"
        ),
    )
