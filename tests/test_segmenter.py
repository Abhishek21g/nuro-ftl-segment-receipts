from __future__ import annotations

from pathlib import Path

import pytest

from segment_receipts.graph import load_graph
from segment_receipts.rules import SegmentRules
from segment_receipts.segmenter import greedy_segment, merge_adjacent_segments
from segment_receipts.toy_models import branched_graph, linear_chain


@pytest.fixture
def default_rules(tmp_path: Path) -> SegmentRules:
    rules_yaml = tmp_path / "rules.yaml"
    rules_yaml.write_text(
        """
default_backend: tensorrt
default_dtype: fp16
deny_ops:
  Concat: onnxruntime
break_on_op:
  - Concat
"""
    )
    return SegmentRules.from_yaml(rules_yaml)


def test_greedy_single_backend_chain(tmp_path: Path) -> None:
    model = linear_chain(tmp_path / "m.onnx", depth=2)
    graph = load_graph(model)
    rules = SegmentRules(default_backend="tensorrt", default_dtype="fp16")
    segments = greedy_segment(graph, rules)
    assert len(segments) >= 1
    assert sum(len(s.node_names) for s in segments) == len(graph.nodes)


def test_break_on_concat_creates_island(tmp_path: Path, default_rules: SegmentRules) -> None:
    model = branched_graph(tmp_path / "b.onnx")
    graph = load_graph(model)
    segments = greedy_segment(graph, default_rules)
    backends = {s.backend for s in segments}
    assert "onnxruntime" in backends


def test_break_before_node(tmp_path: Path) -> None:
    model = linear_chain(tmp_path / "m.onnx", depth=2)
    graph = load_graph(model)
    target = graph.nodes[2].name
    rules = SegmentRules(break_before_nodes={target})
    segments = greedy_segment(graph, rules)
    assert len(segments) >= 2


def test_deny_op_routes_backend(tmp_path: Path) -> None:
    model = branched_graph(tmp_path / "b.onnx")
    graph = load_graph(model)
    rules = SegmentRules(deny_ops={"Concat": "custom-kernel"})
    segments = greedy_segment(graph, rules)
    concat_seg = next(s for s in segments if any(graph.node_by_name[n].op_type == "Concat" for n in s.node_names))
    assert concat_seg.backend == "custom-kernel"


def test_fp32_breaker_dtype(tmp_path: Path) -> None:
    model = linear_chain(tmp_path / "m.onnx", depth=1)
    graph = load_graph(model)
    node = graph.nodes[0].name
    rules = SegmentRules(force_fp32_nodes={node})
    segments = greedy_segment(graph, rules)
    assert any(s.dtype == "fp32" for s in segments)


def test_merge_adjacent_compatible() -> None:
    from segment_receipts.models import Segment

    segs = [
        Segment(0, ["a"], "tensorrt", "fp16"),
        Segment(1, ["b"], "tensorrt", "fp16"),
        Segment(2, ["c"], "onnxruntime", "fp32"),
    ]
    merged = merge_adjacent_segments(segs)
    assert len(merged) == 2
    assert len(merged[0].node_names) == 2


def test_backend_switch_splits(tmp_path: Path) -> None:
    model = branched_graph(tmp_path / "b.onnx")
    graph = load_graph(model)
    rules = SegmentRules(
        default_backend="tensorrt",
        deny_ops={"Relu": "onnxruntime"},
    )
    segments = greedy_segment(graph, rules)
    assert len(segments) >= 2


def test_segment_ids_sequential(tmp_path: Path) -> None:
    model = linear_chain(tmp_path / "m.onnx", depth=3)
    graph = load_graph(model)
    rules = SegmentRules(break_on_op={"Relu"})
    segments = greedy_segment(graph, rules)
    assert [s.id for s in segments] == list(range(len(segments)))
