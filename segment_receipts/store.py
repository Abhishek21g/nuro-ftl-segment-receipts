from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


@dataclass
class RunArtifacts:
    run_dir: Path
    manifest: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] | None = None
    regression_report: dict[str, Any] | None = None
    parity: list[dict[str, Any]] | None = None
    receipt: dict[str, Any] | None = None
    merge_diff: dict[str, Any] | None = None

    @classmethod
    def load(cls, run_dir: Path | str) -> RunArtifacts:
        path = Path(run_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"run directory not found: {path}")

        def _read_json(name: str) -> dict[str, Any] | None:
            fp = path / name
            return json.loads(fp.read_text()) if fp.exists() else None

        def _read_parity() -> list[dict[str, Any]] | None:
            fp = path / "parity.json"
            if not fp.exists():
                receipt = _read_json("receipt.json")
                if receipt and "parity" in receipt:
                    return receipt["parity"]
                return None
            data = json.loads(fp.read_text())
            return data if isinstance(data, list) else data.get("results")

        return cls(
            run_dir=path,
            manifest=_read_json("manifest.json") or {},
            summary=_read_json("summary.json"),
            regression_report=_read_json("regression_report.json"),
            parity=_read_parity(),
            receipt=_read_json("receipt.json"),
            merge_diff=_read_json("merge_diff.json"),
        )


def write_manifest(
    run_dir: Path,
    *,
    run_id: str,
    model: str,
    rules: str,
    candidate: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    manifest = {
        "run_id": run_id,
        "model": model,
        "rules": rules,
        "candidate": candidate,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": [
            "manifest.json",
            "summary.json",
            "regression_report.json",
            "parity.json",
            "receipt.json",
            "merge_diff.json",
            "report.md",
        ],
        **(extra or {}),
    }
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


def write_summary(run_dir: Path, summary: dict[str, Any]) -> Path:
    path = run_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2))
    return path


def write_parity(run_dir: Path, parity: list[dict[str, Any]]) -> Path:
    path = run_dir / "parity.json"
    path.write_text(json.dumps({"results": parity}, indent=2))
    return path
