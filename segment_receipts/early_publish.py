from __future__ import annotations

from segment_receipts.graph import GraphModel
from segment_receipts.models import EarlyPublishEntry, Segment
from segment_receipts.rules import SegmentRules


def build_early_publish_order(
    graph: GraphModel,
    segments: list[Segment],
    rules: SegmentRules,
) -> list[EarlyPublishEntry]:
    """
    Priority-ordered early publish schedule (FTL early output publishing).

    Maps model outputs to producer segments and sorts by user priority (lower = sooner).
    """
    output_to_segment: dict[str, int] = {}
    for seg in segments:
        for name in seg.node_names:
            node = graph.node_by_name[name]
            for out in node.outputs:
                output_to_segment[out] = seg.id

    cumulative_ms: dict[int, float] = {}
    running = 0.0
    for seg in segments:
        running += seg.estimated_ms
        cumulative_ms[seg.id] = running

    entries: list[EarlyPublishEntry] = []
    model_outputs = graph.model_outputs()

    for output in model_outputs:
        priority = rules.early_publish.get(output, 100 + len(entries))
        seg_id = output_to_segment.get(output, segments[-1].id if segments else 0)
        entries.append(
            EarlyPublishEntry(
                output_name=output,
                priority=priority,
                producer_segment_id=seg_id,
                estimated_ready_ms=cumulative_ms.get(seg_id, 0.0),
            )
        )

    entries.sort(key=lambda e: (e.priority, e.estimated_ready_ms))
    return entries
