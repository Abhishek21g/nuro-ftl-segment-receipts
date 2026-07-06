from __future__ import annotations

from pathlib import Path

from segment_receipts.early_publish import build_early_publish_order
from segment_receipts.graph import load_graph
from segment_receipts.gpu_split import plan_gpu_splits
from segment_receipts.latency import apply_latency_estimates
from segment_receipts.rules import SegmentRules
from segment_receipts.segmenter import greedy_segment
from segment_receipts.toy_models import multi_output_head


def test_early_publish_sorted_by_priority(tmp_path: Path) -> None:
    model = multi_output_head(tmp_path / "m.onnx")
    graph = load_graph(model)
    rules = SegmentRules.from_yaml(Path("examples/rules/multi_gpu.yaml"))
    segments = greedy_segment(graph, rules)
    apply_latency_estimates(segments, graph, rules)
    order = build_early_publish_order(graph, segments, rules)
    priorities = [e.priority for e in order]
    assert priorities == sorted(priorities)


def test_gpu_copy_after_trunk(tmp_path: Path) -> None:
    model = multi_output_head(tmp_path / "m.onnx")
    graph = load_graph(model)
    rules = SegmentRules.from_yaml(Path("examples/rules/multi_gpu.yaml"))
    segments = greedy_segment(graph, rules)
    copies = plan_gpu_splits(graph, segments, rules)
    assert len(copies) >= 1
    assert copies[0].from_gpu == 0
    assert copies[0].to_gpu == 1
