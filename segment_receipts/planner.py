from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from segment_receipts.early_publish import build_early_publish_order
from segment_receipts.graph import GraphModel, load_graph
from segment_receipts.gpu_split import plan_gpu_splits
from segment_receipts.latency import apply_latency_estimates
from segment_receipts.models import Receipt
from segment_receipts.parity import probe_full_model_parity
from segment_receipts.rules import SegmentRules
from segment_receipts.segmenter import greedy_segment, merge_adjacent_segments


def build_plan(model_path: Path, rules_path: Path) -> dict[str, Any]:
    graph = load_graph(model_path)
    rules = SegmentRules.from_yaml(rules_path)
    segments = greedy_segment(graph, rules)
    apply_latency_estimates(segments, graph, rules)
    early = build_early_publish_order(graph, segments, rules)
    gpu_copies = plan_gpu_splits(graph, segments, rules)

    return {
        "model": str(model_path),
        "rules": str(rules_path),
        "inputs": graph.model_inputs(),
        "outputs": graph.model_outputs(),
        "node_count": len(graph.nodes),
        "segment_count": len(segments),
        "segments": [
            {
                "id": s.id,
                "nodes": s.node_names,
                "ops": [graph.node_by_name[n].op_type for n in s.node_names],
                "backend": s.backend,
                "dtype": s.dtype,
                "reason": s.reason,
                "estimated_ms": s.estimated_ms,
            }
            for s in segments
        ],
        "early_publish_preview": [
            {"output": e.output_name, "priority": e.priority, "ready_ms": e.estimated_ready_ms}
            for e in early
        ],
        "gpu_copies_preview": [
            {
                "after_segment": c.after_segment_id,
                "from_gpu": c.from_gpu,
                "to_gpu": c.to_gpu,
                "tensor": c.tensor_name,
            }
            for c in gpu_copies
        ],
        "rules_summary": rules.to_dict(),
    }


def run_audit(
    model_path: Path,
    rules_path: Path,
    output_dir: Path,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    merge_segments: bool = False,
) -> Receipt:
    graph = load_graph(model_path)
    rules = SegmentRules.from_yaml(rules_path)
    segments = greedy_segment(graph, rules)
    if merge_segments:
        segments = merge_adjacent_segments(segments)

    total_ms = apply_latency_estimates(segments, graph, rules)
    parity = probe_full_model_parity(model_path, output_names=graph.model_outputs(), rtol=rtol, atol=atol)

    # Attribute parity to final segment for receipt display
    for p in parity:
        if p.segment_id == -1 and segments:
            p.segment_id = segments[-1].id

    early = build_early_publish_order(graph, segments, rules)
    gpu_copies = plan_gpu_splits(graph, segments, rules)
    stitch_order = [s.id for s in segments]

    receipt = Receipt(
        model_path=str(model_path),
        rules_path=str(rules_path),
        node_count=len(graph.nodes),
        segment_count=len(segments),
        segments=segments,
        parity=parity,
        latency_budget_ms=total_ms,
        early_publish=early,
        gpu_copies=gpu_copies,
        stitch_order=stitch_order,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = output_dir / "receipt.json"
    receipt_path.write_text(json.dumps(receipt.to_dict(), indent=2))
    return receipt
