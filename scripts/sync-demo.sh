#!/usr/bin/env bash
# Copy latest gold run artifacts into site/ for GitHub Pages launch.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="${1:-$(ls -1dt "$ROOT/out/receipts"/*/ 2>/dev/null | head -1)}"

if [[ -z "$RUN" || ! -d "$RUN" ]]; then
  echo "error: no run dir. Run: segment-receipts run examples/models/branch.onnx -o out/receipts" >&2
  exit 1
fi

RUN="${RUN%/}"
echo "Syncing demo from $RUN"

mkdir -p "$ROOT/site/data" "$ROOT/site/demo"
cp "$RUN/regression_report.json" "$ROOT/site/data/regression_report.json"
cp "$RUN/summary.json" "$ROOT/site/data/summary.json"
cp "$RUN/manifest.json" "$ROOT/site/data/manifest.json"
cp "$RUN/regression_report.html" "$ROOT/site/demo/regression_report.html"
cp "$RUN/receipt.html" "$ROOT/site/demo/receipt.html"
cp "$RUN/report.md" "$ROOT/site/demo/report.md"
cp "$RUN/rules.from-scan.yaml" "$ROOT/site/demo/rules.from-scan.yaml"

cd "$ROOT" && source .venv/bin/activate 2>/dev/null || true
segment-receipts doctor "$RUN" --json > "$ROOT/site/data/doctor.json" || true

echo "Done. Publish with: bash scripts/publish-site.sh"
