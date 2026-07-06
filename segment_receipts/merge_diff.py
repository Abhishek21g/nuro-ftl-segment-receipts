from __future__ import annotations

from typing import Any

from segment_receipts.graph import GraphModel, load_graph
from segment_receipts.latency import apply_latency_estimates
from segment_receipts.rules import SegmentRules
from segment_receipts.segmenter import greedy_segment, merge_adjacent_segments


def build_merge_diff(
    model_path: str,
    rules_path: str,
) -> dict[str, Any]:
    """
    Compare segmentation before and after adjacent-segment merge.

    This is the worked example from the product playbook: show how greedy
    islands collapse when backends/dtypes align, and what latency changes.
    """
    graph = load_graph(model_path)
    rules = SegmentRules.from_yaml(rules_path)

    before = greedy_segment(graph, rules)
    apply_latency_estimates(before, graph, rules)
    before_ms = sum(s.estimated_ms for s in before)

    after = merge_adjacent_segments(list(before))
    apply_latency_estimates(after, graph, rules)
    after_ms = sum(s.estimated_ms for s in after)

    def _seg_view(segments) -> list[dict[str, Any]]:
        return [
            {
                "id": s.id,
                "backend": s.backend,
                "dtype": s.dtype,
                "nodes": s.node_names,
                "node_count": len(s.node_names),
                "estimated_ms": s.estimated_ms,
                "reason": s.reason,
            }
            for s in segments
        ]

    merged_pairs: list[dict[str, Any]] = []
    if len(after) < len(before):
        # Heuristic: report which before segments collapsed together
        cursor = 0
        for merged in after:
            consumed: list[int] = []
            nodes_seen = 0
            while cursor < len(before) and nodes_seen < len(merged.node_names):
                consumed.append(before[cursor].id)
                nodes_seen += len(before[cursor].node_names)
                cursor += 1
            if len(consumed) > 1:
                merged_pairs.append(
                    {
                        "into_segment": merged.id,
                        "from_segments": consumed,
                        "backend": merged.backend,
                        "dtype": merged.dtype,
                    }
                )

    savings_ms = round(before_ms - after_ms, 4)
    savings_pct = round((savings_ms / before_ms * 100) if before_ms else 0.0, 2)

    return {
        "model": model_path,
        "rules": rules_path,
        "before": {
            "segment_count": len(before),
            "latency_budget_ms": round(before_ms, 4),
            "segments": _seg_view(before),
        },
        "after": {
            "segment_count": len(after),
            "latency_budget_ms": round(after_ms, 4),
            "segments": _seg_view(after),
        },
        "merged_pairs": merged_pairs,
        "stitch_overhead_eliminated": len(before) - len(after),
        "latency_savings_ms": savings_ms,
        "latency_savings_pct": savings_pct,
        "recommendation": _recommendation(len(before), len(after), savings_pct),
    }


def _recommendation(before_count: int, after_count: int, savings_pct: float) -> str:
    if after_count >= before_count:
        return "No adjacent compatible islands to merge — segmentation is already minimal."
    if savings_pct >= 5:
        return (
            f"Merge reduces stitch points from {before_count} to {after_count} "
            f"({savings_pct:.1f}% estimated latency overhead removed)."
        )
    return (
        f"Merge reduces stitch points from {before_count} to {after_count}; "
        "latency impact is small but fewer handoffs may help executor complexity."
    )
