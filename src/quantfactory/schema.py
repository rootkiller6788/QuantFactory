"""数据结构。**全部 dataclass，`to_dict()` 是唯一的序列化口径。**

`rng.canonical_json` 的 `_fallback` 会调用 `to_dict()`，所以「进指纹的东西」
和「落盘的东西」用的是同一份实现 —— 两份实现漂移的后果是静默的
（指纹说没变，产物却变了）。

## 三个层次的命名要分清

| 名字 | 是什么 | 在哪 |
| --- | --- | --- |
| `spec` | 一个**设计点**（算法 × 精度 × 分组 × 校准） | `QuantSpec` |
| `reading` | 把这个设计点**真的跑一遍**之后的四维读数 | `Reading` |
| `row` | 矩阵里的一行（`reading` 的扁平化视图） | `matrix.py` |

**「名义位宽」与「有效位宽」是两个不同的字段，永远不要合并。**
`nominal_bits` 是配置里写的那个数（4），`effective_bits` 是算上元数据之后
真正要存的位数（4.25 或 6.0）。报告里凡是比较体积的地方都必须用后者 ——
本项目的一条核心结论就是「同一个 INT4 在不同算法下的实际体积差 30%」。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

__all__ = [
    "QuantSpec",
    "Reading",
    "Selection",
    "OBJECTIVES",
    "OBJECTIVE_LABELS",
]


# 四维目标。**全部按「最大化」口径** —— 最小化的那三维在这里就是负号，
# 于是 `pareto.py` 里不会出现「方向」这个概念，也就不会写反。
# 顺序就是 `vector()` 的顺序，**改了它等于改了所有历史产物**。
OBJECTIVES: tuple[str, ...] = ("quality", "neg_latency", "neg_memory", "neg_size")

OBJECTIVE_LABELS: dict[str, str] = {
    "quality": "Quality（测试集准确率，↑）",
    "neg_latency": "Latency（单 token 毫秒，↓）",
    "neg_memory": "Memory（峰值显存 GB，↓）",
    "neg_size": "Size（磁盘体积 GB，↓）",
}


@dataclass(frozen=True)
class QuantSpec:
    """一个设计点。`spec_id` 是它在整个空间里的**唯一名字**。

    字段顺序就是 `spec_id` 的成分顺序 —— 改顺序会让所有历史 `spec_id` 失效，
    所以它被钉在这里，而不是在 `spec_id` 的实现里。
    """

    algorithm: str
    precision: str
    group: int
    calib: int  # 0 = 不需要校准

    @property
    def calib_label(self) -> str:
        return "none" if self.calib <= 0 else f"c{self.calib}"

    @property
    def group_label(self) -> str:
        return f"g{self.group}"

    @property
    def spec_id(self) -> str:
        return f"{self.algorithm}|{self.precision}|{self.group_label}|{self.calib_label}"

    @property
    def is_quantized(self) -> bool:
        return True

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "QuantSpec":
        """`to_dict()` 的逆。**只读命令要从 `readings.json` 重建对象时用它。**
        少了它，只读路径就只能去解析 CSV —— 那是两条会漂移的实现。"""
        return cls(
            algorithm=str(d["algorithm"]),
            precision=str(d["precision"]),
            group=int(d["group"]),
            calib=int(d["calib"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "precision": self.precision,
            "group": self.group,
            "calib": self.calib,
            "spec_id": self.spec_id,
        }


@dataclass
class Reading:
    """把 `spec` 真的跑一遍之后得到的东西。**四维读数 + 诊断量。**

    诊断量（`weight_rel_error` 等）不进目标向量 —— 它们是**解释**，
    不是**被优化的东西**。混进去会让「前沿」这个词失去意义。
    """

    spec: QuantSpec
    # ---- 四维目标（原始方向；负号在 `vector()` 里加）
    quality: float
    latency_ms: float
    memory_gb: float
    size_gb: float
    # ---- 位宽口径
    nominal_bits: float
    effective_bits: float
    metadata_bits_per_weight: float
    # ---- 块结构
    n_blocks: int
    n_params: int
    # ---- 诊断（不进目标）
    quality_loss: float
    weight_rel_error: float
    weight_max_abs_error: float
    layer_bits: dict[str, float] = field(default_factory=dict)
    # ⚠️ **这个字段故意不进 `to_dict()`。**
    # 它是墙钟时间（量化 + 一次全测试集前向），随机器变化。写进 `readings.json`
    # 之后，「同一份配置跑两次得到逐位相同的产物」这条底线当场失效 ——
    # 而失效的方式很阴险：第一份产物与第二份产物**只差几个数字**，
    # 任何 diff 都会把它当成「结果变了」。耗时只进 `benchmark.json`（那里被明确豁免）。
    candidate_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    # ---- 派生 ---------------------------------------------------------------

    @property
    def spec_id(self) -> str:
        return self.spec.spec_id

    @property
    def algorithm(self) -> str:
        return self.spec.algorithm

    @property
    def precision(self) -> str:
        return self.spec.precision

    @property
    def group(self) -> int:
        return self.spec.group

    @property
    def calib(self) -> int:
        return self.spec.calib

    def vector(self) -> tuple[float, float, float, float]:
        """目标向量（**全部最大化口径**）。顺序 == `OBJECTIVES`。"""
        return (self.quality, -self.latency_ms, -self.memory_gb, -self.size_gb)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Reading":
        """`to_dict()` 的逆。`candidate_s`（耗时）**不在产物里**，重建时为 0。"""
        return cls(
            spec=QuantSpec.from_dict(d),
            quality=float(d["quality"]),
            latency_ms=float(d["latency_ms"]),
            memory_gb=float(d["memory_gb"]),
            size_gb=float(d["size_gb"]),
            nominal_bits=float(d["nominal_bits"]),
            effective_bits=float(d["effective_bits"]),
            metadata_bits_per_weight=float(d["metadata_bits_per_weight"]),
            n_blocks=int(d["n_blocks"]),
            n_params=int(d["n_params"]),
            quality_loss=float(d["quality_loss"]),
            weight_rel_error=float(d["weight_rel_error"]),
            weight_max_abs_error=float(d["weight_max_abs_error"]),
            layer_bits={str(k): float(v) for k, v in d.get("layer_bits", {}).items()},
            notes=list(d.get("notes", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec_id": self.spec_id,
            "algorithm": self.algorithm,
            "precision": self.precision,
            "group": self.group,
            "calib": self.calib,
            "calib_label": self.spec.calib_label,
            "quality": round(self.quality, 10),
            "latency_ms": round(self.latency_ms, 10),
            "memory_gb": round(self.memory_gb, 10),
            "size_gb": round(self.size_gb, 10),
            "nominal_bits": self.nominal_bits,
            "effective_bits": round(self.effective_bits, 10),
            "metadata_bits_per_weight": round(self.metadata_bits_per_weight, 10),
            "n_blocks": self.n_blocks,
            "n_params": self.n_params,
            "quality_loss": round(self.quality_loss, 10),
            "weight_rel_error": round(self.weight_rel_error, 12),
            "weight_max_abs_error": round(self.weight_max_abs_error, 12),
            "layer_bits": {k: round(v, 10) for k, v in sorted(self.layer_bits.items())},
            "notes": list(self.notes),
        }


@dataclass
class Selection:
    """一条选型规则的结果。**规则是声明的，不是硬编码的。**"""

    rule: str
    spec_id: str | None
    quality: float | None
    latency_ms: float | None
    memory_gb: float | None
    size_gb: float | None
    effective_bits: float | None
    feasible: bool
    why: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "spec_id": self.spec_id,
            "quality": None if self.quality is None else round(self.quality, 10),
            "latency_ms": None if self.latency_ms is None else round(self.latency_ms, 10),
            "memory_gb": None if self.memory_gb is None else round(self.memory_gb, 10),
            "size_gb": None if self.size_gb is None else round(self.size_gb, 10),
            "effective_bits": (
                None if self.effective_bits is None else round(self.effective_bits, 10)
            ),
            "feasible": bool(self.feasible),
            "why": self.why,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Selection":
        """从 `selection.csv` 的字典重建。

        **这条路径存在的理由**：`visualize` 必须能从产物重建出渲染所需的一切，
        否则它就退化成「把整条流水线再跑一遍」（实测 44 s），而那样它证明的
        只是「流水线是确定的」—— 那件事已经由两次 `run` 的逐位相同证明了。
        """
        return cls(
            rule=str(d["rule"]),
            spec_id=d.get("spec_id"),
            quality=_opt_float(d.get("quality")),
            latency_ms=_opt_float(d.get("latency_ms")),
            memory_gb=_opt_float(d.get("memory_gb")),
            size_gb=_opt_float(d.get("size_gb")),
            effective_bits=_opt_float(d.get("effective_bits")),
            feasible=bool(d.get("feasible", False)),
            why=str(d.get("why", "")),
        )


def _opt_float(v: Any) -> float | None:
    return None if v is None or v == "" else float(v)
