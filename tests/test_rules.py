from __future__ import annotations

from pathlib import Path

from segment_receipts.rules import SegmentRules


def test_load_default_rules() -> None:
    rules = SegmentRules.from_yaml(Path("examples/rules/default.yaml"))
    assert rules.default_backend == "onnxruntime"
    assert "Concat" in rules.deny_ops


def test_backend_for_deny_op() -> None:
    from segment_receipts.models import NodeInfo

    rules = SegmentRules(deny_ops={"Softmax": "custom-kernel"})
    node = NodeInfo("s", "Softmax", [], [])
    backend, dtype, reason = rules.backend_for(node)
    assert backend == "custom-kernel"
    assert dtype == "fp32"


def test_fp32_forced_op() -> None:
    from segment_receipts.models import NodeInfo

    rules = SegmentRules(force_fp32_ops={"BatchNormalization"}, default_dtype="fp16")
    node = NodeInfo("bn", "BatchNormalization", [], [])
    _, dtype, _ = rules.backend_for(node)
    assert dtype == "fp32"


def test_break_reason_before() -> None:
    from segment_receipts.models import NodeInfo

    rules = SegmentRules(break_before_nodes={"node_x"})
    node = NodeInfo("node_x", "Conv", [], [])
    assert rules.segment_break_reason(node) == "break_before:node_x"


def test_break_reason_op() -> None:
    from segment_receipts.models import NodeInfo

    rules = SegmentRules(break_on_op={"Concat"})
    node = NodeInfo("c", "Concat", [], [])
    assert rules.segment_break_reason(node) == "break_on_op:Concat"


def test_to_dict_roundtrip_keys() -> None:
    rules = SegmentRules.from_yaml(Path("examples/rules/multi_gpu.yaml"))
    d = rules.to_dict()
    assert "gpu_assignments" in d
    assert d["early_publish"]["head_boxes"] == 0
