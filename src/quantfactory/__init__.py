"""QuantFactory —— 精度设计空间工厂。

`Model → Profile → Calibration → Quantize(×算法×位宽×分组×校准) → Export
 → Benchmark → Quality / Latency / Memory / Size → Pareto Frontier`

**零第三方依赖**：核心链路只用标准库（`math.sumprod` 是唯一硬理由，故 `>=3.12`）。
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
