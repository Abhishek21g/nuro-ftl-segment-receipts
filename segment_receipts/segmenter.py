from __future__ import annotations

from segment_receipts.graph import GraphModel
from segment_receipts.models import Segment
from segment_receipts.rules import SegmentRules


def greedy_segment(graph: GraphModel, rules: SegmentRules) -> list[Segment]:
    """
    Greedy topological segmenter inspired by FTL's orchestrator segmenter.

    Walk nodes in topological order. Extend the current segment while backend/dtype
    stay compatible and no break rule fires. Start a new island on mismatch.
    """
    order = graph.topological_order()
    segments: list[Segment] = []
    current_nodes: list[str] = []
    current_backend: str | None = None
    current_dtype: str | None = None
    current_reason = "start"
    seg_id = 0

    def flush() -> None:
        nonlocal seg_id, current_nodes, current_backend, current_dtype, current_reason
        if not current_nodes:
            return
        segments.append(
            Segment(
                id=seg_id,
                node_names=list(current_nodes),
                backend=current_backend or rules.default_backend,
                dtype=current_dtype or rules.default_dtype,
                reason=current_reason,
            )
        )
        seg_id += 1
        current_nodes = []

    for name in order:
        node = graph.node_by_name[name]
        break_reason = rules.segment_break_reason(node)
        if break_reason:
            flush()
            current_reason = break_reason

        backend, dtype, reason = rules.backend_for(node)

        if current_nodes and (backend != current_backend or dtype != current_dtype):
            flush()
            current_reason = f"backend_switch:{backend}/{dtype}"

        if not current_nodes:
            current_backend = backend
            current_dtype = dtype
            if current_reason == "start":
                current_reason = reason

        current_nodes.append(name)

    flush()
    return segments


def merge_adjacent_segments(segments: list[Segment]) -> list[Segment]:
    """Merge consecutive segments with identical backend/dtype (post-pass optimization)."""
    if not segments:
        return []
    merged: list[Segment] = []
    current = segments[0]
    for seg in segments[1:]:
        if seg.backend == current.backend and seg.dtype == current.dtype:
            current = Segment(
                id=current.id,
                node_names=current.node_names + seg.node_names,
                backend=current.backend,
                dtype=current.dtype,
                reason=current.reason,
                estimated_ms=current.estimated_ms + seg.estimated_ms,
            )
        else:
            merged.append(current)
            current = seg
    merged.append(current)
    for idx, seg in enumerate(merged):
        seg.id = idx
    return merged
