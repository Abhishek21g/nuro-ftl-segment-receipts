from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from segment_receipts.doctor import diagnose_to_dict
from segment_receipts.store import RunArtifacts

RECEIPT_FILE = "deploy_receipt.json"
SIG_FILE = "deploy_receipt.sig"
DEFAULT_DEV_KEY = b"deployproof-dev-key-nuro-demo-v1"

ARTIFACT_NAMES = (
    "manifest.json",
    "summary.json",
    "regression_report.json",
    "parity.json",
    "receipt.json",
)


def load_key(key_path: Path | None = None) -> bytes:
    if key_path is not None:
        if not key_path.exists():
            raise FileNotFoundError(f"signing key not found: {key_path}")
        return key_path.read_bytes().strip()
    env = os.environ.get("DEPLOYPROOF_KEY")
    if env:
        return env.encode()
    return DEFAULT_DEV_KEY


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def _sign_payload(payload: dict[str, Any], key: bytes) -> str:
    return hmac.new(key, _canonical(payload), hashlib.sha256).hexdigest()


@dataclass
class FlashResult:
    approved: bool
    reasons: list[str] = field(default_factory=list)
    receipt: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "flash": "APPROVED" if self.approved else "BLOCKED",
            "approved": self.approved,
            "reasons": self.reasons,
            "receipt": self.receipt,
        }


def build_receipt(artifacts: RunArtifacts) -> dict[str, Any]:
    doctor = diagnose_to_dict(artifacts)
    reg = artifacts.regression_report or {}
    summary = artifacts.summary or {}
    ff = reg.get("first_failure") or {}

    hashes: dict[str, str] = {}
    for name in ARTIFACT_NAMES:
        fp = artifacts.run_dir / name
        if fp.exists():
            hashes[name] = _sha256_file(fp)

    return {
        "version": "1",
        "kind": "deployproof.behavioral_receipt",
        "run_id": artifacts.manifest.get("run_id", artifacts.run_dir.name),
        "model": artifacts.manifest.get("model", ""),
        "candidate": artifacts.manifest.get("candidate", ""),
        "doctor_status": doctor["status"],
        "regression": {
            "tensors_failed": reg.get("tensors_failed", summary.get("tensors_failed")),
            "tensors_compared": reg.get("tensors_compared", summary.get("tensors_compared")),
            "first_failure_node": summary.get("first_failure_node") or ff.get("producer_node"),
            "atol": reg.get("atol"),
            "rtol": reg.get("rtol"),
        },
        "parity": {
            "passed": summary.get("parity_passed", 0),
            "total": summary.get("parity_total", 0),
        },
        "segment_count": summary.get("segment_count"),
        "artifact_hashes": hashes,
        "policy": {
            "require_doctor_pass": True,
            "max_tensor_failures": 0,
        },
    }


def sign_run(run_dir: Path | str, *, key_path: Path | None = None) -> Path:
    artifacts = RunArtifacts.load(run_dir)
    key = load_key(key_path)
    receipt = build_receipt(artifacts)
    receipt["signed_at"] = datetime.now(timezone.utc).isoformat()

    out = Path(run_dir)
    receipt_path = out / RECEIPT_FILE
    sig_path = out / SIG_FILE
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    sig_path.write_text(_sign_payload(receipt, key) + "\n")
    return receipt_path


def _load_signed_receipt(run_dir: Path) -> tuple[dict[str, Any], str]:
    receipt_path = run_dir / RECEIPT_FILE
    sig_path = run_dir / SIG_FILE
    if not receipt_path.exists():
        raise FileNotFoundError(f"missing {RECEIPT_FILE} — run: segment-receipts sign {run_dir}")
    if not sig_path.exists():
        raise FileNotFoundError(f"missing {SIG_FILE} — run: segment-receipts sign {run_dir}")
    receipt = json.loads(receipt_path.read_text())
    signature = sig_path.read_text().strip()
    return receipt, signature


def verify_run(run_dir: Path | str, *, key_path: Path | None = None) -> FlashResult:
    path = Path(run_dir)
    key = load_key(key_path)
    reasons: list[str] = []

    try:
        receipt, signature = _load_signed_receipt(path)
    except FileNotFoundError as exc:
        return FlashResult(approved=False, reasons=[str(exc)])

    expected_sig = _sign_payload(receipt, key)
    if not hmac.compare_digest(signature, expected_sig):
        reasons.append("invalid signature — receipt tampered or wrong key")

    for name, expected_hash in receipt.get("artifact_hashes", {}).items():
        fp = path / name
        if not fp.exists():
            reasons.append(f"missing artifact: {name}")
            continue
        if _sha256_file(fp) != expected_hash:
            reasons.append(f"artifact changed since sign: {name}")

    policy = receipt.get("policy", {})
    if policy.get("require_doctor_pass") and receipt.get("doctor_status") != "pass":
        reasons.append(f"doctor status is {receipt.get('doctor_status')} (require pass)")

    reg = receipt.get("regression", {})
    failed = reg.get("tensors_failed")
    max_fail = policy.get("max_tensor_failures", 0)
    if failed is not None and failed > max_fail:
        reasons.append(f"{failed}/{reg.get('tensors_compared')} layers drifted (max {max_fail})")

    return FlashResult(approved=not reasons, reasons=reasons, receipt=receipt)
