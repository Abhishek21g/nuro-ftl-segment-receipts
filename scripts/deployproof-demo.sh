#!/usr/bin/env bash
# One-day DeployProof demo: run → sign → flash (blocked on drift).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate 2>/dev/null || true

RUN_DIR="${1:-$(ls -1dt out/receipts/*/ 2>/dev/null | head -1)}"
RUN_DIR="${RUN_DIR%/}"

if [[ -z "$RUN_DIR" || ! -d "$RUN_DIR" ]]; then
  echo "Running fresh pipeline on branch.onnx..."
  segment-receipts run examples/models/branch.onnx -o out/receipts
  RUN_DIR="$(ls -1dt out/receipts/*/ | head -1)"
  RUN_DIR="${RUN_DIR%/}"
fi

echo "==> Sign behavioral receipt"
segment-receipts sign "$RUN_DIR"

echo ""
echo "==> Vehicle flash gate"
segment-receipts flash "$RUN_DIR" || true

echo ""
echo "Demo: model compiles but drifts → signed receipt exists → FLASH BLOCKED"
echo "Site:  https://enaguthi.com/nuro-ftl-receipts/site/#/deployproof"
