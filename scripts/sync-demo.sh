#!/usr/bin/env bash
# Bundle gold runs into site/ for GitHub Pages — multi-scenario dashboard.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
else
  PY="python3"
fi

sync_scenario() {
  local id="$1"
  local run="$2"
  local dest="$ROOT/site/data/scenarios/$id"
  mkdir -p "$dest" "$ROOT/site/demo"
  cp "$run/regression_report.json" "$dest/"
  cp "$run/summary.json" "$dest/"
  cp "$run/manifest.json" "$dest/"
  [[ -f "$run/receipt.json" ]] && cp "$run/receipt.json" "$dest/"
  [[ -f "$run/rules.from-scan.yaml" ]] && cp "$run/rules.from-scan.yaml" "$dest/"
  [[ -f "$run/regression_report.html" ]] && cp "$run/regression_report.html" "$ROOT/site/demo/${id}_regression_report.html"
  "$PY" -m segment_receipts.cli doctor "$run" --json > "$dest/doctor.json" 2>/dev/null || true
  echo "  synced $id <- $run"
}

echo "Syncing scenarios…"
mkdir -p "$ROOT/site/data"

# Prefer explicit run dirs; fall back to latest matching model_path in regression_report.json
find_run() {
  local needle="$1"
  local run
  for run in $(ls -1dt "$ROOT/out/receipts"/*/ 2>/dev/null); do
    if grep -q "\"model_path\": \"$needle\"" "$run/regression_report.json" 2>/dev/null; then
      echo "${run%/}"
      return 0
    fi
  done
  return 1
}

BRANCH_RUN="${BRANCH_RUN:-$(find_run 'examples/models/branch.graph.json' || find_run 'examples/models/branch.onnx')}"
CHAIN_RUN="${CHAIN_RUN:-$(find_run 'examples/models/chain.onnx')}"
MULTI_RUN="${MULTI_RUN:-$(find_run 'examples/models/multi_output.graph.json' || find_run 'examples/models/multi_output.onnx')}"
RESNET_RUN="${RESNET_RUN:-$(find_run 'examples/models/resnet18-mini.onnx')}"

for pair in \
  "branch:$BRANCH_RUN" \
  "chain:$CHAIN_RUN" \
  "multi_output:$MULTI_RUN" \
  "resnet18_mini:$RESNET_RUN"; do
  id="${pair%%:*}"
  run="${pair#*:}"
  if [[ -z "$run" || ! -d "$run" ]]; then
    echo "warn: missing run for $id" >&2
    continue
  fi
  sync_scenario "$id" "$run"
done

# Legacy single-demo paths (hero panel fallback)
PRIMARY="$BRANCH_RUN"
if [[ -n "$PRIMARY" && -d "$PRIMARY" ]]; then
  cp "$PRIMARY/regression_report.json" "$ROOT/site/data/regression_report.json"
  cp "$PRIMARY/summary.json" "$ROOT/site/data/summary.json"
  cp "$PRIMARY/manifest.json" "$ROOT/site/data/manifest.json"
  cp "$PRIMARY/regression_report.html" "$ROOT/site/demo/regression_report.html"
  cp "$PRIMARY/receipt.html" "$ROOT/site/demo/receipt.html" 2>/dev/null || true
  cp "$PRIMARY/report.md" "$ROOT/site/demo/report.md" 2>/dev/null || true
  cp "$PRIMARY/rules.from-scan.yaml" "$ROOT/site/demo/rules.from-scan.yaml" 2>/dev/null || true
  "$PY" -m segment_receipts.cli doctor "$PRIMARY" --json > "$ROOT/site/data/doctor.json" 2>/dev/null || true
fi

"$PY" <<'PY'
import json
from pathlib import Path

root = Path("site/data/scenarios")
catalog = []
meta = {
    "branch": {
        "title": "Branched Backbone",
        "tagline": "6 layers · 1 output",
        "description": "Fork-join CNN mimics perception branches. FP16 activation rounding drifts at the stem before either branch recovers.",
        "nodes": 6,
        "canaries": 1,
    },
    "chain": {
        "title": "Linear Chain",
        "tagline": "6 layers · sequential stack",
        "description": "Deep linear conv stack — the simplest silent regression trap. Drift compounds layer by layer.",
        "nodes": 6,
        "canaries": 1,
    },
    "multi_output": {
        "title": "Multi-Output Head",
        "tagline": "4 tensors · early-publish",
        "description": "Two detection heads from one trunk — models the FTL early-publish pattern where downstream only needs one output.",
        "nodes": 4,
        "canaries": 2,
    },
    "resnet18_mini": {
        "title": "ResNet Mini",
        "tagline": "9 layers · deeper stack",
        "description": "Blog Fig 5 class problem: export decomposition + FP16 islands on a realistic mini backbone.",
        "nodes": 9,
        "canaries": 1,
    },
}

for sid, info in meta.items():
    d = root / sid
    if not (d / "summary.json").exists():
        continue
    summary = json.loads((d / "summary.json").read_text())
    reg_path = d / "regression_report.json"
    reg = json.loads(reg_path.read_text()) if reg_path.exists() else {}
    catalog.append({
        "id": sid,
        **info,
        "model_path": reg.get("model_path", ""),
        "tensors_compared": summary.get("tensors_compared", 0),
        "tensors_failed": summary.get("tensors_failed", 0),
        "first_failure_node": summary.get("first_failure_node", ""),
        "status": summary.get("status", "unknown"),
        "segment_count": summary.get("segment_count", 0),
        "parity": f"{summary.get('parity_passed', 0)}/{summary.get('parity_total', 0)}",
        "data_prefix": f"data/scenarios/{sid}",
        "report_href": f"demo/{sid}_regression_report.html",
    })

out = {
    "product": "FTL Segment Receipts",
    "version": "2.0",
    "compile_paths": ["fp16_activations"],
    "tolerance_modes": [
        {"id": "strict", "label": "Strict", "atol": 1e-6, "hint": "Tight tolerance — catches micro-drift before it compounds."},
        {"id": "standard", "label": "Standard", "atol": 1e-5, "hint": "Default ship gate — matches segment-receipts CLI defaults."},
        {"id": "relaxed", "label": "Relaxed", "atol": 1e-4, "hint": "Looser gate — shows how many failures disappear when teams widen tolerance (dangerous)."},
    ],
    "scenarios": catalog,
    "stats": {
        "scenarios": len(catalog),
        "compile_paths": 1,
        "tolerance_modes": 3,
        "total_tensor_checks": sum(s["tensors_compared"] for s in catalog),
    },
}
Path("site/data/scenarios.json").write_text(json.dumps(out, indent=2) + "\n")
print(f"Wrote site/data/scenarios.json ({len(catalog)} scenarios)")
PY

echo "Done. Publish: bash scripts/publish-site.sh"
