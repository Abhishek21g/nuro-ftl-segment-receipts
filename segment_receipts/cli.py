from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from segment_receipts.deployproof import sign_run, verify_run
from segment_receipts.doctor import diagnose, diagnose_to_dict
from segment_receipts.merge_diff import build_merge_diff
from segment_receipts.pipeline import build_plan_with_scan, execute_run
from segment_receipts.regression import scan_regression, write_regression_report
from segment_receipts.regression_report import write_regression_html
from segment_receipts.report import write_html_report, write_markdown_report
from segment_receipts.store import RunArtifacts


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="segment-receipts",
        description="Silent ONNX regression hunter + FTL-style segment receipts.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Preview segments + scan risks before run.")
    plan.add_argument("model", type=Path)
    plan.add_argument("-r", "--rules", type=Path, required=True)
    plan.add_argument("--no-scan", action="store_true", help="Skip regression preview")
    plan.add_argument("--candidate", default="fp16_activations")
    plan.add_argument("--json", action="store_true")
    plan.set_defaults(handler=_cmd_plan)

    run = sub.add_parser("run", help="Full pipeline: scan → segment receipt → artifact trail.")
    run.add_argument("model", type=Path)
    run.add_argument("-r", "--rules", type=Path, default=None, help="Rules YAML (auto from scan if omitted)")
    run.add_argument("-o", "--output", type=Path, default=Path("out/receipts"))
    run.add_argument("--candidate", default="fp16_activations")
    run.add_argument("--golden", type=Path, default=None)
    run.add_argument("--rtol", type=float, default=1e-4)
    run.add_argument("--atol", type=float, default=1e-5)
    run.add_argument("--merge", action="store_true")
    run.set_defaults(handler=_cmd_run)

    doctor = sub.add_parser("doctor", help="Diagnose a run directory for regressions and gaps.")
    doctor.add_argument("run", type=Path, help="Run dir under out/receipts/<id>")
    doctor.add_argument("--json", action="store_true")
    doctor.set_defaults(handler=_cmd_doctor)

    report = sub.add_parser("report", help="Render HTML + markdown from run directory.")
    report.add_argument("run", type=Path, help="Run dir or receipt.json path")
    report.set_defaults(handler=_cmd_report)

    scan = sub.add_parser("scan", help="Regression scan only (no full artifact trail).")
    scan.add_argument("model", type=Path)
    scan.add_argument("-o", "--output", type=Path, default=Path("out/receipts/scan-only"))
    scan.add_argument("--candidate", choices=["optimized", "fp16_activations", "golden"], default="fp16_activations")
    scan.add_argument("--golden", type=Path, default=None)
    scan.add_argument("--rtol", type=float, default=1e-4)
    scan.add_argument("--atol", type=float, default=1e-5)
    scan.add_argument("--max-tensors", type=int, default=64)
    scan.set_defaults(handler=_cmd_scan)

    diff = sub.add_parser("diff", help="Before/after adjacent segment merge comparison.")
    diff.add_argument("model", type=Path)
    diff.add_argument("-r", "--rules", type=Path, required=True)
    diff.add_argument("-o", "--output", type=Path, default=None)
    diff.add_argument("--json", action="store_true")
    diff.set_defaults(handler=_cmd_diff)

    sign = sub.add_parser("sign", help="Sign a run as a DeployProof behavioral receipt.")
    sign.add_argument("run", type=Path, help="Run dir under out/receipts/<id>")
    sign.add_argument("-k", "--key", type=Path, default=None, help="HMAC signing key file")
    sign.set_defaults(handler=_cmd_sign)

    verify = sub.add_parser("verify", help="Verify signed receipt + policy (exit 1 if blocked).")
    verify.add_argument("run", type=Path, help="Run dir under out/receipts/<id>")
    verify.add_argument("-k", "--key", type=Path, default=None)
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(handler=_cmd_verify)

    flash = sub.add_parser(
        "flash",
        help="Vehicle flash gate — same as verify, prints FLASH APPROVED/BLOCKED.",
    )
    flash.add_argument("run", type=Path)
    flash.add_argument("-k", "--key", type=Path, default=None)
    flash.add_argument("--json", action="store_true")
    flash.set_defaults(handler=_cmd_flash)

    return parser


def _cmd_plan(args: argparse.Namespace) -> int:
    plan = build_plan_with_scan(
        args.model,
        args.rules,
        candidate=args.candidate,
        scan_preview=not args.no_scan,
    )
    if args.json:
        print(json.dumps(plan, indent=2))
        return 0
    print(f"Model: {plan['model']}")
    print(f"Nodes: {plan['node_count']} → {plan['segment_count']} segments")
    for risk in plan.get("risks", []):
        print(f"  ⚠ {risk}")
    if sp := plan.get("scan_preview"):
        print(
            f"Scan ({sp['candidate']}): {sp['tensors_failed']}/{sp['tensors_compared']} tensors diverged"
        )
        if sp.get("first_failure"):
            ff = sp["first_failure"]
            print(f"  First failure: {ff['producer_node']} ({ff['producer_op']})")
    for seg in plan["segments"]:
        ops = ", ".join(seg["ops"][:4])
        print(
            f"  [{seg['id']}] {seg['backend']}/{seg['dtype']} "
            f"({len(seg['nodes'])} nodes) — {ops}"
        )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    artifacts = execute_run(
        args.model,
        args.rules,
        args.output,
        candidate=args.candidate,
        rtol=args.rtol,
        atol=args.atol,
        merge_segments=args.merge,
        golden_npz=args.golden,
    )
    print(f"Run: {artifacts.run_dir}")
    print(f"Doctor status: {artifacts.summary.get('status') if artifacts.summary else '?'}")
    if artifacts.summary:
        s = artifacts.summary
        print(
            f"Regression: {s.get('tensors_failed')}/{s.get('tensors_compared')} failed · "
            f"Segments: {s.get('segment_count')} · "
            f"Parity: {s.get('parity_passed')}/{s.get('parity_total')}"
        )
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    artifacts = RunArtifacts.load(args.run)
    result = diagnose_to_dict(artifacts)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"Run: {artifacts.run_dir.name} — status: {result['status'].upper()}")
        for f in diagnose(artifacts):
            icon = {"critical": "✗", "warning": "!", "info": "·"}[f.severity]
            print(f"  {icon} [{f.code}] {f.message}")
            print(f"      → {f.suggestion}")
    return 0 if result["status"] == "pass" else 1

def _cmd_report(args: argparse.Namespace) -> int:
    run = Path(args.run)
    if run.is_file():
        html = write_html_report(run)
        print(f"Wrote {html}")
        return 0
    artifacts = RunArtifacts.load(run)
    write_markdown_report(artifacts.run_dir)
    if (artifacts.run_dir / "regression_report.json").exists():
        write_regression_html(artifacts.run_dir / "regression_report.json")
    if (artifacts.run_dir / "receipt.json").exists():
        write_html_report(artifacts.run_dir / "receipt.json", artifacts.run_dir / "receipt.html")
    print(f"Wrote {artifacts.run_dir / 'report.md'}")
    return 0


def _cmd_scan(args: argparse.Namespace) -> int:
    report = scan_regression(
        args.model,
        candidate=args.candidate,
        golden_npz=args.golden,
        rtol=args.rtol,
        atol=args.atol,
        max_tensors=args.max_tensors,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    json_path = write_regression_report(report, args.output)
    html_path = write_regression_html(json_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {html_path}")
    print(f"Divergences: {report.tensors_failed}/{report.tensors_compared}")
    return 0


def _cmd_sign(args: argparse.Namespace) -> int:
    path = sign_run(args.run, key_path=args.key)
    print(f"Signed: {path}")
    print(f"Verify: segment-receipts flash {args.run}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    result = verify_run(args.run, key_path=args.key)
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        for reason in result.reasons:
            print(f"  ✗ {reason}")
        if result.approved:
            print("VERIFY: PASS")
        else:
            print("VERIFY: FAIL")
    return 0 if result.approved else 1


def _cmd_flash(args: argparse.Namespace) -> int:
    result = verify_run(args.run, key_path=args.key)
    label = "APPROVED" if result.approved else "BLOCKED"
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"FLASH {label}")
        if result.receipt:
            r = result.receipt
            print(f"  model: {r.get('model', '?')}")
            print(f"  run:   {r.get('run_id', '?')}")
            reg = r.get("regression", {})
            print(
                f"  drift: {reg.get('tensors_failed')}/{reg.get('tensors_compared')} layers · "
                f"doctor: {r.get('doctor_status')}"
            )
        for reason in result.reasons:
            print(f"  → {reason}")
    return 0 if result.approved else 1


def _cmd_diff(args: argparse.Namespace) -> int:
    result = build_merge_diff(str(args.model), str(args.rules))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print(f"Wrote {args.output}")
    if args.json or not args.output:
        print(json.dumps(result, indent=2))
    else:
        b, a = result["before"]["segment_count"], result["after"]["segment_count"]
        print(f"Before: {b} segments · After: {a} segments")
        print(result["recommendation"])
    return 0
