from __future__ import annotations

from pathlib import Path

from segment_receipts.merge_diff import build_merge_diff
from segment_receipts.toy_models import linear_chain


def test_merge_diff_reduces_segments(tmp_path: Path) -> None:
    model = linear_chain(tmp_path / "chain.onnx", depth=3)
    rules = Path("examples/rules/merge_demo.yaml")
    diff = build_merge_diff(str(model), str(rules))
    assert diff["before"]["segment_count"] > diff["after"]["segment_count"]
    assert diff["after"]["segment_count"] == 1
    assert diff["stitch_overhead_eliminated"] >= 1


def test_merge_diff_recommendation(tmp_path: Path) -> None:
    model = linear_chain(tmp_path / "chain.onnx", depth=2)
    diff = build_merge_diff(str(model), "examples/rules/merge_demo.yaml")
    assert "recommendation" in diff
    assert diff["before"]["segment_count"] >= diff["after"]["segment_count"]


def test_merge_diff_merged_pairs(tmp_path: Path) -> None:
    model = linear_chain(tmp_path / "chain.onnx", depth=3)
    diff = build_merge_diff(str(model), "examples/rules/merge_demo.yaml")
    assert len(diff["merged_pairs"]) >= 1


def test_cli_diff(tmp_path: Path) -> None:
    from segment_receipts.cli import main

    model = linear_chain(tmp_path / "c.onnx", depth=2)
    out = tmp_path / "merge_diff.json"
    assert main(["diff", str(model), "-r", "examples/rules/merge_demo.yaml", "-o", str(out)]) == 0
    assert out.exists()
