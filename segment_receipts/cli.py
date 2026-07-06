from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from segment_receipts.planner import build_plan, run_audit
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
            "ONNX segment + parity audit harness inspired by Nuro's published FTL architecture."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="Preview segment islands without running parity.")
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

    return parser


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
