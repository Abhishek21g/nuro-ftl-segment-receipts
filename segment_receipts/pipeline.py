from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from segment_receipts.doctor import diagnose_to_dict
from segment_receipts.planner import build_plan
from segment_receipts.regression import RegressionReport, scan_regression, write_regression_report
from segment_receipts.regression_report import write_regression_html
from segment_receipts.report import write_html_report, write_markdown_report
from segment_receipts.rules import SegmentRules
from segment_receipts.store import (
    RunArtifacts,
    new_run_id,
    write_manifest,
    write_parity,
    write_summary,
)
from segment_receipts.planner import run_audit as _run_segment_audit


def rules_from_scan(report: RegressionReport, base_rules: Path | None = None) -> SegmentRules:
    """Merge scan breaker recommendations into segment rules."""
    if base_rules and base_rules.exists():
        rules = SegmentRules.from_yaml(base_rules)
    else:
        rules = SegmentRules()

    for rec in report.breaker_recommendations[:3]:
        rules.break_before_nodes.add(rec.node_name)
        rules.force_fp32_nodes.add(rec.node_name)
        rules.backend_overrides[rec.node_name] = "onnxruntime"
    return rules


def export_rules_yaml(rules: SegmentRules, path: Path) -> Path:
    data = {
        "default_backend": rules.default_backend,
        "default_dtype": rules.default_dtype,
        "deny_ops": rules.deny_ops,
        "force_fp32_ops": sorted(rules.force_fp32_ops),
        "force_fp32_nodes": sorted(rules.force_fp32_nodes),
        "break_before_nodes": sorted(rules.break_before_nodes),
        "break_on_op": sorted(rules.break_on_op),
        "backend_overrides": rules.backend_overrides,
        "early_publish": rules.early_publish,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, sort_keys=False))
    return path


def build_plan_with_scan(
    model_path: Path,
    rules_path: Path,
    *,
    candidate: str = "fp16_activations",
    scan_preview: bool = True,
) -> dict[str, Any]:
    plan = build_plan(model_path, rules_path)
    if not scan_preview:
        return plan

    report = scan_regression(model_path, candidate=candidate, max_tensors=32)
    plan["scan_preview"] = {
        "candidate": candidate,
        "tensors_failed": report.tensors_failed,
        "tensors_compared": report.tensors_compared,
        "first_failure": report.first_failure.__dict__ if report.first_failure else None,
        "breaker_recommendations": [r.__dict__ for r in report.breaker_recommendations[:5]],
    }
    if report.first_failure:
        plan["risks"] = [
            f"Silent regression at {report.first_failure.producer_node} "
            f"(max Δ {report.first_failure.max_abs_diff:.2e}) — "
            "confirm break_before_nodes covers this node."
        ]
    else:
        plan["risks"] = []
    return plan


def execute_run(
    model_path: Path,
    rules_path: Path | None,
    output_base: Path,
    *,
    run_id: str | None = None,
    candidate: str = "fp16_activations",
    rtol: float = 1e-4,
    atol: float = 1e-5,
    merge_segments: bool = False,
    golden_npz: Path | None = None,
) -> RunArtifacts:
    """Full gold pipeline: scan → rules → segment receipt → manifest."""
    rid = run_id or new_run_id()
    run_dir = output_base / rid
    run_dir.mkdir(parents=True, exist_ok=True)

    report = scan_regression(
        model_path,
        candidate=candidate,
        golden_npz=golden_npz,
        rtol=rtol,
        atol=atol,
    )
    write_regression_report(report, run_dir)

    if rules_path is None or not rules_path.exists():
        rules = rules_from_scan(report, rules_path)
        rules_path = run_dir / "rules.from-scan.yaml"
        export_rules_yaml(rules, rules_path)
    else:
        rules_path = Path(rules_path)

    receipt = _run_segment_audit(
        model_path,
        rules_path,
        run_dir,
        rtol=rtol,
        atol=atol,
        merge_segments=merge_segments,
    )

    write_parity(run_dir, [asdict(p) for p in receipt.parity])

    doctor_preview = diagnose_to_dict(RunArtifacts.load(run_dir))
    write_summary(
        run_dir,
        {
            "run_id": rid,
            "status": doctor_preview["status"],
            "tensors_failed": report.tensors_failed,
            "tensors_compared": report.tensors_compared,
            "segment_count": receipt.segment_count,
            "parity_passed": sum(1 for p in receipt.parity if p.passed),
            "parity_total": len(receipt.parity),
            "first_failure_node": (
                report.first_failure.producer_node if report.first_failure else None
            ),
        },
    )

    write_manifest(
        run_dir,
        run_id=rid,
        model=str(model_path),
        rules=str(rules_path),
        candidate=candidate,
        extra={"doctor_status": doctor_preview["status"]},
    )

    write_markdown_report(run_dir)
    write_regression_html(run_dir / "regression_report.json")
    write_html_report(run_dir / "receipt.json", run_dir / "receipt.html")

    return RunArtifacts.load(run_dir)
