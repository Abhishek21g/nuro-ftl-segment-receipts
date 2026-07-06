from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from segment_receipts.merge_diff import build_merge_diff
from segment_receipts.planner import build_plan, run_audit
from segment_receipts.regression import scan_regression, write_regression_report
from segment_receipts.regression_report import write_regression_html
from segment_receipts.report import write_html_report


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
        description=(
            "Find silent ONNX compile regressions and recommend FTL-style segment breakers."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    scan = sub.add_parser(
        "scan",
        help="Locate where compile path diverges from reference (core workflow).",
    )
    scan.add_argument("model", type=Path)
    scan.add_argument("-o", "--output", type=Path, default=Path("receipts/scan"))
    scan.add_argument(
        "--candidate",
        choices=["optimized", "fp16_activations", "golden"],
        default="fp16_activations",
        help="optimized=ORT fusion drift, fp16=TensorRT/FP16 island drift, golden=npz",
    )
    scan.add_argument("--golden", type=Path, default=None, help="npz with golden tensors")
    scan.add_argument("--rtol", type=float, default=1e-4)
    scan.add_argument("--atol", type=float, default=1e-5)
    scan.add_argument("--max-tensors", type=int, default=64)
    scan.set_defaults(handler=_cmd_scan)

    plan = sub.add_parser("plan", help="Preview segment islands (secondary).")
    plan.add_argument("model", type=Path, help="Path to ONNX model")
    plan.add_argument("-r", "--rules", type=Path, required=True, help="YAML segment rules")
    plan.add_argument("--json", action="store_true", help="Print JSON plan")
    plan.set_defaults(handler=_cmd_plan)

    run = sub.add_parser("run", help="Run full audit and write receipt.json")
    run.add_argument("model", type=Path, help="Path to ONNX model")
    run.add_argument("-r", "--rules", type=Path, required=True, help="YAML segment rules")
    run.add_argument("-o", "--output", type=Path, default=Path("receipts/latest"))
    run.add_argument("--rtol", type=float, default=1e-3)
    run.add_argument("--atol", type=float, default=1e-5)
    run.add_argument("--merge", action="store_true", help="Merge adjacent compatible segments")
    run.set_defaults(handler=_cmd_run)

    report = sub.add_parser("report", help="Render HTML report from receipt.json")
    report.add_argument("receipt", type=Path, help="Path to receipt.json")
    report.add_argument("-o", "--output", type=Path, default=None)
    report.set_defaults(handler=_cmd_report)

    diff = sub.add_parser(
        "diff",
        help="Compare segmentation before/after adjacent-segment merge (stitch overhead).",
    )
    diff.add_argument("model", type=Path)
    diff.add_argument("-r", "--rules", type=Path, required=True)
    diff.add_argument("-o", "--output", type=Path, default=None, help="Write merge_diff.json")
    diff.add_argument("--json", action="store_true")
    diff.set_defaults(handler=_cmd_diff)

    return parser


def _cmd_scan(args: argparse.Namespace) -> int:
    report = scan_regression(
        args.model,
        candidate=args.candidate,
        golden_npz=args.golden,
        rtol=args.rtol,
        atol=args.atol,
        max_tensors=args.max_tensors,
    )
    json_path = write_regression_report(report, args.output)
    html_path = write_regression_html(json_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {html_path}")
    print(
        f"Divergences: {report.tensors_failed}/{report.tensors_compared} tensors above tolerance"
    )
    if report.first_failure:
        print(
            f"First failure: {report.first_failure.producer_node} "
            f"({report.first_failure.producer_op}) max Δ={report.first_failure.max_abs_diff:.2e}"
        )
    if report.breaker_recommendations:
        print(f"Recommended breakers: {len(report.breaker_recommendations)}")
        for rec in report.breaker_recommendations:
            print(f"  → {rec.node_name} ({rec.op_type})")
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    plan = build_plan(args.model, args.rules)
    if args.json:
        print(json.dumps(plan, indent=2))
    else:
        print(f"Model: {plan['model']}")
        print(f"Nodes: {plan['node_count']} → {plan['segment_count']} segments")
        for seg in plan["segments"]:
            ops = ", ".join(seg["ops"])
            print(
                f"  [{seg['id']}] {seg['backend']}/{seg['dtype']} "
                f"({len(seg['nodes'])} nodes, {seg['estimated_ms']:.3f} ms) — {ops}"
            )
        if plan["early_publish_preview"]:
            print("Early publish:")
            for e in plan["early_publish_preview"]:
                print(f"  p{e['priority']}: {e['output']} @ {e['ready_ms']:.3f} ms")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    receipt = run_audit(
        args.model,
        args.rules,
        args.output,
        rtol=args.rtol,
        atol=args.atol,
        merge_segments=args.merge,
    )
    html_path = write_html_report(args.output / "receipt.json")
    print(f"Wrote {args.output / 'receipt.json'}")
    print(f"Wrote {html_path}")
    print(
        f"Segments: {receipt.segment_count}, "
        f"budget: {receipt.latency_budget_ms:.3f} ms, "
        f"parity: {sum(1 for p in receipt.parity if p.passed)}/{len(receipt.parity)}"
    )
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    out = write_html_report(args.receipt, args.output)
    print(f"Wrote {out}")
    return 0


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
        print(f"Before: {b} segments ({result['before']['latency_budget_ms']:.3f} ms)")
        print(f"After:  {a} segments ({result['after']['latency_budget_ms']:.3f} ms)")
        print(result["recommendation"])
    return 0
