from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class NodeInfo:
    name: str
    op_type: str
    inputs: list[str]
    outputs: list[str]
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Segment:
    id: int
    node_names: list[str]
    backend: str
    dtype: str
    reason: str = ""
    estimated_ms: float = 0.0


@dataclass
class ParityResult:
    segment_id: int
    output_name: str
    max_abs_diff: float
    mean_abs_diff: float
    rtol: float
    atol: float
    passed: bool


@dataclass
class EarlyPublishEntry:
    output_name: str
    priority: int
    producer_segment_id: int
    estimated_ready_ms: float


@dataclass
class GpuCopyPoint:
    after_segment_id: int
    from_gpu: int
    to_gpu: int
    tensor_name: str


@dataclass
class Receipt:
    model_path: str
    rules_path: str
    node_count: int
    segment_count: int
    segments: list[Segment]
    parity: list[ParityResult]
    latency_budget_ms: float
    early_publish: list[EarlyPublishEntry]
    gpu_copies: list[GpuCopyPoint]
    stitch_order: list[int]
    parity_mode: str = "synthetic"
    disclaimer: str = (
        "Third-party tool inspired by Nuro's published FTL architecture. "
        "Not affiliated with Nuro, Inc."
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
