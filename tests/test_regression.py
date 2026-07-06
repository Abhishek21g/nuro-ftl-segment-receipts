from __future__ import annotations

from pathlib import Path

import pytest

from segment_receipts.augment import augment_intermediate_outputs
from segment_receipts.regression import scan_regression
from segment_receipts.toy_models import branched_graph


def test_augment_adds_outputs(tmp_path: Path) -> None:
    model = branched_graph(tmp_path / "b.onnx")
    aug = augment_intermediate_outputs(model, max_tensors=10)
    assert len(aug.graph.output) > 1


def test_fp16_scan_finds_divergence(tmp_path: Path) -> None:
    model = branched_graph(tmp_path / "b.onnx")
    report = scan_regression(model, candidate="fp16_activations", max_tensors=20)
    assert report.tensors_compared > 0
    assert report.tensors_failed > 0
    assert report.first_failure is not None
    assert len(report.breaker_recommendations) >= 1


def test_breaker_has_suggested_rule(tmp_path: Path) -> None:
    model = branched_graph(tmp_path / "b.onnx")
    report = scan_regression(model, candidate="fp16_activations")
    rec = report.breaker_recommendations[0]
    assert "break_before_nodes" in rec.suggested_rule
    assert rec.node_name in rec.suggested_rule["force_fp32_nodes"]


def test_graph_layout_matches_nodes(tmp_path: Path) -> None:
    model = branched_graph(tmp_path / "b.onnx")
    report = scan_regression(model, candidate="fp16_activations")
    assert len(report.graph_layout) == 6


def test_cli_scan(tmp_path: Path) -> None:
    from segment_receipts.cli import main

    model = branched_graph(tmp_path / "b.onnx")
    out = tmp_path / "scan_out"
    assert main(["scan", str(model), "-o", str(out)]) == 0
    assert (out / "regression_report.json").exists()
    assert (out / "regression_report.html").exists()
