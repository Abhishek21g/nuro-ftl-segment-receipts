# FTL Segment Receipts — Silent Regression Hunter

Find **where ONNX compile paths silently diverge** and get FTL-style segment breaker rules — inspired by [Nuro's FTL blog](https://medium.com/nuro/ftl-model-compiler-framework-d6b85c670f67), not affiliated with Nuro, Inc.

> "Third party compilers are notorious for causing silent regressions." — Nuro FTL blog

## Gold workflow

```bash
pip install -e ".[dev,onnx]"

# Preview risks before run
segment-receipts plan examples/models/branch.onnx -r examples/rules/default.yaml

# Full pipeline → out/receipts/<run-id>/
segment-receipts run examples/models/resnet18-mini.onnx -o out/receipts

# Diagnose silent regressions + breaker gaps
segment-receipts doctor out/receipts/<run-id>

# Regenerate report.md + HTML
segment-receipts report out/receipts/<run-id>
```

## Artifact trail (`out/receipts/<run-id>/`)

| File | Purpose |
|------|---------|
| `manifest.json` | Run metadata |
| `summary.json` | Doctor-ready summary |
| `regression_report.json` | Per-tensor divergence scan |
| `parity.json` | Output parity results |
| `receipt.json` | Compiler island stitch receipt |
| `rules.from-scan.yaml` | Auto-generated breaker rules |
| `report.md` | Human-readable report |

## DeployProof (v1.2 — one-day build)

Signed behavioral receipt + vehicle flash gate:

```bash
segment-receipts run examples/models/branch.onnx -o out/receipts
segment-receipts sign out/receipts/<run-id>
segment-receipts flash out/receipts/<run-id>   # FLASH BLOCKED if layers drifted
```

Or: `bash scripts/deployproof-demo.sh`

Site: https://enaguthi.com/nuro-ftl-receipts/site/#/deployproof

## Demo

- **Dashboard** (Trust Me Bro–grade interactive site): https://enaguthi.com/nuro-ftl-receipts/site/
- Live Runner + scenario library + tolerance charts
- Sample regression report: https://enaguthi.com/nuro-ftl-receipts/site/demo/regression_report.html

```bash
bash scripts/sync-demo.sh   # refresh bundled scenario data
bash scripts/publish-site.sh
```

## Test

```bash
pytest tests/ -v   # 52+ tests
```

## License

MIT
