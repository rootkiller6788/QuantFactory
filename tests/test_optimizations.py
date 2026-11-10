"""优化项的回归测试：**快路径必须与慢路径逐位一致**。

这些用例保护的不是「代码能跑」，而是几处优化的核心不变量：

* `forward_logits` 只是不保留 `acts`，logits 必须与 `forward` 完全相同；
* `batch_grads` 顺带返回的 logits 必须与独立前向 `logits_of` 相同；
* `_momentum_step` 维护的 `W_T` 必须始终等于当前权重的转置；
* `_max_residual` 的转置快路径必须与「逐列重建」的教科书写法相同；
* 设计空间校验要在训练前挡住不一致的校准策略；
* 产物写出不留临时文件。

⚠️ 自包含：不依赖 conftest 夹具（理由同 `test_import.py`）。
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from quantfactory.calib import _max_residual, cholesky_inverse  # noqa: E402
from quantfactory.config import default_config, validate_config  # noqa: E402
from quantfactory.emit import write_outputs  # noqa: E402
from quantfactory.model import (  # noqa: E402
    Layer,
    Model,
    batch_grads,
    forward,
    forward_logits,
    logits_of,
    transpose_weights,
)
from quantfactory.train import _momentum_step  # noqa: E402


def _toy_model(seed: int = 0, *, d_in: int = 8, hidden: int = 8, n_classes: int = 4) -> Model:
    r = random.Random(seed)
    return Model([
        Layer(
            "layer0",
            [[r.uniform(-1.0, 1.0) for _ in range(d_in)] for _ in range(hidden)],
            [r.uniform(-0.1, 0.1) for _ in range(hidden)],
        ),
        Layer(
            "head",
            [[r.uniform(-1.0, 1.0) for _ in range(hidden)] for _ in range(n_classes)],
            [r.uniform(-0.1, 0.1) for _ in range(n_classes)],
            "none",
        ),
    ])


def test_forward_logits_matches_forward() -> None:
    model = _toy_model()
    r = random.Random(1)
    for _ in range(5):
        x = [r.uniform(-1.0, 1.0) for _ in range(8)]
        slow, acts = forward(model, x)
        fast = forward_logits(model, x)
        assert fast == slow
        assert len(acts) == len(model.layers) + 1
        assert acts[0] is not x  # 不共享调用方的列表


def test_batch_grads_logits_match_logits_of() -> None:
    model = _toy_model(seed=2)
    r = random.Random(3)
    X = [[r.uniform(-1.0, 1.0) for _ in range(8)] for _ in range(4)]
    Y = [0, 1, 2, 3]
    dW, db, logits = batch_grads(model, transpose_weights(model), X, Y)
    assert [logits[b] for b in range(len(X))] == [logits_of(model, x) for x in X]
    assert len(dW) == len(db) == len(model.layers)


def test_momentum_step_keeps_weight_transpose_in_sync() -> None:
    model = _toy_model(seed=4)
    W_T = transpose_weights(model)
    vW = [[[0.0] * len(row) for row in l.W] for l in model.layers]
    vb = [[0.0] * len(l.b) for l in model.layers]
    dW = [[[0.1 * (o + j + 1) for j in range(len(row))] for o, row in enumerate(l.W)]
          for l in model.layers]
    db = [[0.2 * (o + 1) for o in range(len(l.b))] for l in model.layers]
    _momentum_step(
        model, dW, db, vW, vb, W_T,
        norm=1.0, lr=0.1, inv_batch=1.0, clip=0.0,
        momentum=0.9, weight_decay=0.01,
    )
    assert W_T == transpose_weights(model)


def test_max_residual_matches_column_definition() -> None:
    n = 12
    r = random.Random(5)
    A = [[r.uniform(-1.0, 1.0) for _ in range(n)] for _ in range(n)]
    H = [[A[i][j] + A[j][i] if i != j else abs(A[i][i]) + n for j in range(n)]
         for i in range(n)]
    Hinv = cholesky_inverse(H)
    expected = 0.0
    for i in range(n):
        for k in range(n):
            col = [Hinv[j][k] for j in range(n)]
            expected = max(expected, abs(math.sumprod(H[i], col) - (1.0 if i == k else 0.0)))
    assert _max_residual(H, Hinv) == expected


def test_validate_config_rejects_inconsistent_space() -> None:
    assert validate_config(default_config()) == []

    cfg = default_config()
    cfg["space"]["algorithms"].append("gptqq")
    cfg["space"]["calib_sizes"]["rtn"] = [64]
    cfg["space"]["calib_sizes"]["gptq"] = [0]
    cfg["space"]["groups"].append(cfg["space"]["groups"][0])
    problems = "\n".join(validate_config(cfg))
    assert "未知算法 'gptqq'" in problems
    assert "不读激活统计" in problems
    assert "需要正校准档位" in problems
    assert "有重复取值" in problems


def test_write_outputs_is_atomic(tmp_path: Path) -> None:
    written = write_outputs(tmp_path, {
        "a.json": {"k": 1},
        "sub/b.csv": "x\n",
    })
    assert written == ["a.json", "sub/b.csv"]
    assert (tmp_path / "a.json").read_text(encoding="utf-8") == '{\n  "k": 1\n}\n'
    assert (tmp_path / "sub" / "b.csv").read_text(encoding="utf-8") == "x\n"
    assert list(tmp_path.rglob("*.tmp")) == []
