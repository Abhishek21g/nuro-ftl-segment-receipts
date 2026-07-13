from __future__ import annotations

import json
from pathlib import Path

from segment_receipts.cli import main
from segment_receipts.deployproof import (
    RECEIPT_FILE,
    SIG_FILE,
    build_receipt,
    sign_run,
    verify_run,
)
from segment_receipts.pipeline import execute_run
from segment_receipts.store import RunArtifacts
from segment_receipts.toy_models import build_all_examples


def _make_run(tmp_path: Path) -> Path:
    examples = tmp_path / "ex"
    models = build_all_examples(examples)
    out = tmp_path / "receipts"
    execute_run(models["branch"], None, out, candidate="fp16_activations")
    return next(out.iterdir())


def test_build_receipt_from_run(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    artifacts = RunArtifacts.load(run_dir)
    receipt = build_receipt(artifacts)
    assert receipt["kind"] == "deployproof.behavioral_receipt"
    assert receipt["doctor_status"] == "fail"
    assert receipt["regression"]["tensors_failed"] >= 1
    assert "artifact_hashes" in receipt


def test_sign_and_verify_tamper_detected(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    sign_run(run_dir)
    assert (run_dir / RECEIPT_FILE).exists()
    assert (run_dir / SIG_FILE).exists()

    result = verify_run(run_dir)
    assert not result.approved
    assert any("doctor" in r.lower() or "drift" in r.lower() for r in result.reasons)

    # Tamper artifact → hash mismatch
    summary_path = run_dir / "summary.json"
    data = json.loads(summary_path.read_text())
    data["segment_count"] = 999
    summary_path.write_text(json.dumps(data))
    result2 = verify_run(run_dir)
    assert not result2.approved
    assert any("artifact changed" in r for r in result2.reasons)


def test_cli_sign_flash(tmp_path: Path, capsys) -> None:
    run_dir = _make_run(tmp_path)
    assert main(["sign", str(run_dir)]) == 0
    code = main(["flash", str(run_dir)])
    captured = capsys.readouterr()
    assert code == 1
    assert "FLASH BLOCKED" in captured.out


def test_approved_fixture(tmp_path: Path) -> None:
    run_dir = _make_run(tmp_path)
    summary = json.loads((run_dir / "summary.json").read_text())
    summary["status"] = "pass"
    summary["tensors_failed"] = 0
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    reg = json.loads((run_dir / "regression_report.json").read_text())
    reg["tensors_failed"] = 0
    for d in reg.get("divergences", []):
        d["passed"] = True
    (run_dir / "regression_report.json").write_text(json.dumps(reg, indent=2) + "\n")
    sign_run(run_dir)
    result = verify_run(run_dir)
    assert result.approved
