from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from segment_receipts.models import NodeInfo


@dataclass
class SegmentRules:
    """User-defined compilation rules (FTL blog Fig 5–6 style)."""

    default_backend: str = "onnxruntime"
    default_dtype: str = "fp32"
    deny_ops: dict[str, str] = field(default_factory=dict)
    force_fp32_ops: set[str] = field(default_factory=set)
    force_fp32_nodes: set[str] = field(default_factory=set)
    break_before_nodes: set[str] = field(default_factory=set)
    break_on_op: set[str] = field(default_factory=set)
    backend_overrides: dict[str, str] = field(default_factory=dict)
    early_publish: dict[str, int] = field(default_factory=dict)
    gpu_assignments: dict[str, int] = field(default_factory=dict)
    cross_gpu_after_nodes: list[str] = field(default_factory=list)
    latency_weights: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path | str) -> SegmentRules:
        data = yaml.safe_load(Path(path).read_text()) or {}
        return cls(
            default_backend=data.get("default_backend", "onnxruntime"),
            default_dtype=data.get("default_dtype", "fp32"),
            deny_ops=dict(data.get("deny_ops", {})),
            force_fp32_ops=set(data.get("force_fp32_ops", [])),
            force_fp32_nodes=set(data.get("force_fp32_nodes", [])),
            break_before_nodes=set(data.get("break_before_nodes", [])),
            break_on_op=set(data.get("break_on_op", [])),
            backend_overrides=dict(data.get("backend_overrides", {})),
            early_publish=dict(data.get("early_publish", {})),
            gpu_assignments=dict(data.get("gpu_assignments", {})),
            cross_gpu_after_nodes=list(data.get("cross_gpu_after_nodes", [])),
            latency_weights=dict(data.get("latency_weights", {})),
        )

    def backend_for(self, node: NodeInfo) -> tuple[str, str, str]:
        """Return (backend, dtype, reason) for a node."""
        if node.name in self.backend_overrides:
            backend = self.backend_overrides[node.name]
            return backend, self._dtype_for(node), f"backend_override:{node.name}"

        if node.op_type in self.deny_ops:
            backend = self.deny_ops[node.op_type]
            return backend, self._dtype_for(node), f"deny_op:{node.op_type}"

        return self.default_backend, self._dtype_for(node), "default"

    def _dtype_for(self, node: NodeInfo) -> str:
        if node.name in self.force_fp32_nodes or node.op_type in self.force_fp32_ops:
            return "fp32"
        return self.default_dtype

    def should_break_before(self, node: NodeInfo) -> bool:
        return node.name in self.break_before_nodes

    def should_break_on_op(self, node: NodeInfo) -> bool:
        return node.op_type in self.break_on_op

    def segment_break_reason(self, node: NodeInfo) -> str | None:
        if self.should_break_before(node):
            return f"break_before:{node.name}"
        if self.should_break_on_op(node):
            return f"break_on_op:{node.op_type}"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "default_backend": self.default_backend,
            "default_dtype": self.default_dtype,
            "deny_ops": self.deny_ops,
            "force_fp32_ops": sorted(self.force_fp32_ops),
            "force_fp32_nodes": sorted(self.force_fp32_nodes),
            "break_before_nodes": sorted(self.break_before_nodes),
            "break_on_op": sorted(self.break_on_op),
            "backend_overrides": self.backend_overrides,
            "early_publish": self.early_publish,
            "gpu_assignments": self.gpu_assignments,
            "cross_gpu_after_nodes": self.cross_gpu_after_nodes,
            "latency_weights": self.latency_weights,
        }
