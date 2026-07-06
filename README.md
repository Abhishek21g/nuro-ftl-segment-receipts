# FTL Segment Receipts

Open ONNX segment + parity audit harness **inspired by** [Nuro's published FTL compiler architecture](https://medium.com/nuro/ftl-model-compiler-framework-d6b85c670f67) — not affiliated with Nuro, Inc.

## What's new in v0.2

- **Real ONNX + OnnxRuntime** — parity probes on actual `.onnx` models (not just JSON graphs)
- **`segment-receipts diff`** — before/after adjacent-segment merge with stitch-overhead analysis
- **Merge diff on demo site** — interactive before/after islands visualization

## Receipt covers

1. **Compiler islands** — greedy topological segmentation with YAML breaker rules
2. **Backend + dtype policy** — TensorRT / ORT / custom-kernel slots per segment
3. **Numerical parity** — ORT tolerance report (`parity_mode` in receipt JSON)
4. **Latency budget** — per-segment estimates and total stitch cost
5. **Early publish ordering** — priority table for marked outputs
6. **Multi-GPU splits** — cross-gpu-copy insertion points
7. **Merge diff** — auto-written as `merge_diff.json` on every `run`

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,onnx]"
```

## CLI

```bash
# Preview segments
segment-receipts plan examples/models/branch.onnx -r examples/rules/default.yaml

# Before/after merge comparison
segment-receipts diff examples/models/chain.onnx -r examples/rules/merge_demo.yaml

# Full audit → receipt.json + merge_diff.json + receipt.html
segment-receipts run examples/models/branch.onnx -r examples/rules/default.yaml -o receipts/demo

segment-receipts report receipts/demo/receipt.json
```

## Demo

- **Site:** [enaguthi.com/nuro-ftl-receipts/site/](https://enaguthi.com/nuro-ftl-receipts/site/)
- **Live receipt:** [demo/receipt.html](https://enaguthi.com/nuro-ftl-receipts/site/demo/receipt.html)

## Test

```bash
pytest tests/ -v
```

## License

MIT
