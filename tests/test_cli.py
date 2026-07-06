from __future__ import annotations

import json
from pathlib import Path

from segment_receipts.cli import main
from segment_receipts.pipeline import execute_run
from segment_receipts.planner import build_plan, run_audit
from segment_receipts.report import write_html_report
from segment_receipts.toy_models import build_all_examples


def test_build_plan_chain() -> None:
    examples = Path("examples")
    build_all_examples(examples)
    plan = build_plan(examples / "models/chain.onnx", Path("examples/rules/default.yaml"))
    assert plan["segment_count"] >= 1


def test_run_audit_writes_receipt(tmp_path: Path) -> None:
    examples = tmp_path / "ex"
    models = build_all_examples(examples)
    rules = Path("examples/rules/default.yaml")
    out = tmp_path / "out"
    receipt = run_audit(models["chain"], rules, out)
    assert (out / "receipt.json").exists()
    assert receipt.segment_count >= 1


def test_html_report(tmp_path: Path) -> None:
    examples = tmp_path / "ex"
    models = build_all_examples(examples)
    out = tmp_path / "out"
    run_audit(models["branch"], Path("examples/rules/default.yaml"), out)
    html = write_html_report(out / "receipt.json")
    assert "FTL Segment Receipt" in html.read_text()


def test_cli_plan(capsys) -> None:
    examples = Path("examples")
    build_all_examples(examples)
    code = main(["plan", str(examples / "models/chain.onnx"), "-r", "examples/rules/default.yaml"])
    captured = capsys.readouterr()
    assert code == 0
    assert "nodes" in captured.out.lower() or "scan" in captured.out.lower()


def test_cli_run(tmp_path: Path) -> None:
    examples = tmp_path / "ex"
    models = build_all_examples(examples)
    out = tmp_path / "receipts"
    code = main(
        [
            "run",
            str(models["multi_output"]),
            "-r",
            "examples/rules/multi_gpu.yaml",
            "-o",
            str(out),
        ]
    )
    assert code == 0
    run_dir = next(out.iterdir())
    assert (run_dir / "receipt.json").exists()
    assert (run_dir / "manifest.json").exists()


def test_cli_report(tmp_path: Path) -> None:
    examples = tmp_path / "ex"
    models = build_all_examples(examples)
    out = tmp_path / "receipts"
    main(["run", str(models["chain"]), "-o", str(out)])
    run_dir = next(out.iterdir())
    code = main(["report", str(run_dir)])
    assert code == 0
    assert (run_dir / "report.md").exists()
