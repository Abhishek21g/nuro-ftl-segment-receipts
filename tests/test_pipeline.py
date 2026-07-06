from __future__ import annotations

from pathlib import Path

import pytest

from segment_receipts.doctor import diagnose
from segment_receipts.pipeline import build_plan_with_scan, execute_run
from segment_receipts.store import RunArtifacts
from segment_receipts.toy_models import branched_graph, decomposed_layernorm_trap, resnet18_mini


def test_execute_run_artifact_trail(tmp_path: Path) -> None:
    model = branched_graph(tmp_path / "b.onnx")
    out = tmp_path / "out"
    artifacts = execute_run(model, None, out)
    run_dir = artifacts.run_dir
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "regression_report.json").exists()
    assert (run_dir / "parity.json").exists()
    assert (run_dir / "receipt.json").exists()
    assert (run_dir / "merge_diff.json").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "rules.from-scan.yaml").exists()


def test_doctor_fails_on_regression(tmp_path: Path) -> None:
    model = branched_graph(tmp_path / "b.onnx")
    artifacts = execute_run(model, None, tmp_path / "out")
    findings = diagnose(artifacts)
    codes = {f.code for f in findings}
    assert "silent_regression" in codes


def test_doctor_json_status(tmp_path: Path) -> None:
    model = branched_graph(tmp_path / "b.onnx")
    artifacts = execute_run(model, None, tmp_path / "out")
    loaded = RunArtifacts.load(artifacts.run_dir)
    assert loaded.manifest.get("run_id")
    assert loaded.regression_report is not None


def test_plan_with_scan_preview(tmp_path: Path) -> None:
    model = branched_graph(tmp_path / "b.onnx")
    rules = Path("examples/rules/default.yaml")
    plan = build_plan_with_scan(model, rules)
    assert "scan_preview" in plan
    assert plan["scan_preview"]["tensors_failed"] >= 0


def test_resnet18_mini_generates(tmp_path: Path) -> None:
    path = resnet18_mini(tmp_path / "resnet18-mini.onnx")
    assert path.exists()


def test_decomposed_trap_generates(tmp_path: Path) -> None:
    path = decomposed_layernorm_trap(tmp_path / "trap.onnx")
    assert path.exists()
    import onnx

    model = onnx.load(str(path))
    ops = {n.op_type for n in model.graph.node}
    assert "ReduceMean" in ops
    assert "Sqrt" in ops


def test_cli_doctor_run(tmp_path: Path) -> None:
    from segment_receipts.cli import main

    model = branched_graph(tmp_path / "b.onnx")
    out = tmp_path / "receipts"
    assert main(["run", str(model), "-o", str(out)]) == 0
    run_dirs = list(out.iterdir())
    assert len(run_dirs) == 1
    # doctor returns 1 when regressions found (expected for fp16 scan)
    assert main(["doctor", str(run_dirs[0])]) == 1
    assert main(["doctor", str(run_dirs[0]), "--json"]) == 1


def test_cli_report_run_dir(tmp_path: Path) -> None:
    from segment_receipts.cli import main

    model = branched_graph(tmp_path / "b.onnx")
    out = tmp_path / "receipts"
    main(["run", str(model), "-o", str(out)])
    run_dir = next(out.iterdir())
    assert main(["report", str(run_dir)]) == 0
    assert (run_dir / "report.md").exists()
