# FTL Segment Receipts

Open ONNX segment + parity audit harness **inspired by** [Nuro's published FTL compiler architecture](https://medium.com/nuro/ftl-model-compiler-framework-d6b85c670f67) — not affiliated with Nuro, Inc.

Takes an ONNX model and produces an audit report covering:

1. **Compiler islands** — greedy topological segmentation with YAML breaker rules
2. **Backend + dtype policy** — TensorRT / ORT / custom-kernel slots per segment
3. **Numerical parity** — tolerance report vs golden ORT execution
4. **Latency budget** — per-segment estimates and total stitch cost
5. **Early publish ordering** — priority table for marked outputs
6. **Multi-GPU splits** — cross-gpu-copy insertion points

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## CLI

```bash
# Preview segments
segment-receipts plan examples/models/branch.onnx -r examples/rules/default.yaml

# Run full audit → receipt.json + receipt.html
segment-receipts run examples/models/multi_output.onnx -r examples/rules/multi_gpu.yaml -o receipts/demo

# Render HTML from existing receipt
segment-receipts report receipts/demo/receipt.json
```

## Rules (segment breaker)

```yaml
default_backend: tensorrt
default_dtype: fp16
deny_ops:
  Concat: onnxruntime
break_on_op:
  - Concat
force_fp32_nodes:
  - sensitive_reduce
early_publish:
  head_boxes: 0
  head_classes: 1
gpu_assignments:
  trunk_conv: 0
  boxes_head: 1
cross_gpu_after_nodes:
  - trunk_relu
```

## Demo

- **Site:** [enaguthi.com/nuro-ftl-receipts/site/](https://enaguthi.com/nuro-ftl-receipts/site/)
- **Live receipt:** [enaguthi.com/nuro-ftl-receipts/site/demo/receipt.html](https://enaguthi.com/nuro-ftl-receipts/site/demo/receipt.html)

## Test

```bash
pytest tests/ -v
```

## License

MIT
