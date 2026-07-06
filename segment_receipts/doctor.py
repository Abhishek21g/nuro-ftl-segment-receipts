from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from segment_receipts.store import RunArtifacts

SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


@dataclass(frozen=True)
class Finding:
    severity: str
    code: str
    message: str
    suggestion: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def diagnose(artifacts: RunArtifacts) -> list[Finding]:
    findings: list[Finding] = []
    findings.extend(_check_manifest(artifacts))
    findings.extend(_check_regression(artifacts))
    findings.extend(_check_breaker_coverage(artifacts))
    findings.extend(_check_parity(artifacts))
    findings.extend(_check_segments(artifacts))
    findings.sort(key=lambda f: SEVERITY_ORDER.get(f.severity, 9))
    return findings


def diagnose_to_dict(artifacts: RunArtifacts) -> dict[str, Any]:
    findings = diagnose(artifacts)
    return {
        "run_id": artifacts.manifest.get("run_id", artifacts.run_dir.name),
        "status": _overall_status(findings),
        "findings": [f.to_dict() for f in findings],
    }


def _overall_status(findings: list[Finding]) -> str:
    if any(f.severity == "critical" for f in findings):
        return "fail"
    if any(f.severity == "warning" for f in findings):
        return "warn"
    return "pass"


def _check_manifest(artifacts: RunArtifacts) -> list[Finding]:
    if not artifacts.manifest:
        return [
            Finding(
                "warning",
                "missing_manifest",
                "manifest.json not found — run may be incomplete.",
                "Re-run with segment-receipts run to regenerate the artifact trail.",
            )
        ]
    return []


def _check_regression(artifacts: RunArtifacts) -> list[Finding]:
    reg = artifacts.regression_report
    if reg is None:
        return [
            Finding(
                "critical",
                "missing_regression_scan",
                "No regression_report.json — silent compile drift was not scanned.",
                "Run segment-receipts run (includes scan) or scan separately first.",
            )
        ]

    failed = reg.get("tensors_failed", 0)
    compared = reg.get("tensors_compared", 0)
    findings: list[Finding] = []

    if failed > 0:
        ff = reg.get("first_failure")
        node = ff.get("producer_node", "?") if ff else "?"
        findings.append(
            Finding(
                "critical",
                "silent_regression",
                f"{failed}/{compared} intermediate tensors diverged (first topo failure: {node}).",
                "Insert FP32 segment breaker at first_failure; re-run scan until clean.",
            )
        )
    else:
        findings.append(
            Finding(
                "info",
                "regression_clean",
                f"All {compared} tensors within atol={reg.get('atol')} rtol={reg.get('rtol')}.",
                "Candidate compile path matches reference for scanned tensors.",
            )
        )
    return findings


def _check_breaker_coverage(artifacts: RunArtifacts) -> list[Finding]:
    reg = artifacts.regression_report
    if not reg or reg.get("tensors_failed", 0) == 0:
        return []

    ff = reg.get("first_failure")
    if not ff:
        return []

    first_node = ff.get("producer_node")
    rules_path = artifacts.manifest.get("rules", "")
    recs = reg.get("breaker_recommendations", [])
    rec_nodes = {r.get("node_name") for r in recs}

    findings: list[Finding] = []
    if first_node and first_node not in rec_nodes:
        findings.append(
            Finding(
                "warning",
                "breaker_gap",
                f"First failure node '{first_node}' has no breaker recommendation entry.",
                "Add break_before_nodes / force_fp32_nodes for this node in rules YAML.",
            )
        )

    if artifacts.receipt:
        covered = set()
        for seg in artifacts.receipt.get("segments", []):
            for n in seg.get("node_names", []):
                covered.add(n)
        if first_node and first_node in covered:
            findings.append(
                Finding(
                    "info",
                    "breaker_segmented",
                    f"First failure node '{first_node}' is assigned to a compiler island.",
                    "Verify island uses FP32/ORT per recommended suggested_rule.",
                )
            )
    return findings


def _check_parity(artifacts: RunArtifacts) -> list[Finding]:
    parity = artifacts.parity
    if parity is None:
        return [
            Finding(
                "warning",
                "missing_parity",
                "parity.json not found.",
                "Ensure run completed segment receipt stage.",
            )
        ]

    failed = [p for p in parity if not p.get("passed", True)]
    if failed:
        return [
            Finding(
                "critical",
                "output_parity_fail",
                f"{len(failed)}/{len(parity)} model outputs failed parity check.",
                "Tighten atol/rtol or fix export before vehicle/sim handoff.",
            )
        ]
    return [
        Finding(
            "info",
            "parity_pass",
            f"All {len(parity)} outputs passed parity ({artifacts.receipt.get('parity_mode', 'unknown') if artifacts.receipt else 'unknown'}).",
            "Output-level parity is within tolerance.",
        )
    ]


def _check_segments(artifacts: RunArtifacts) -> list[Finding]:
    receipt = artifacts.receipt
    if not receipt:
        return []
    if receipt.get("segment_count", 0) == 0:
        return [
            Finding(
                "critical",
                "no_segments",
                "Segmentation produced zero islands.",
                "Check rules YAML and ONNX graph connectivity.",
            )
        ]
    return []
