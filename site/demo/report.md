# FTL Segment Receipt — Run Report

- **Run ID:** `20260706T231316Z`
- **Model:** `examples/models/branch.onnx`
- **Rules:** `out/receipts/20260706T231316Z/rules.from-scan.yaml`
- **Candidate path:** `fp16_activations`

## Summary
- Doctor status: **fail**
- Regression: 6/6 tensors failed
- Segments: 3
- Parity: 1/1 passed
- First failure node: `stem_conv`

## Silent regression scan
FP16/TensorRT islands can drift while the rest of the graph stays FP32. This scan simulates FP16 activation rounding at every tensor boundary.

### Recommended segment breakers
- `stem_conv` (Conv): max Δ 1.65e-03
- `out_relu` (Relu): max Δ 1.94e-03
- `branch_b_conv` (Conv): max Δ 1.94e-03

## Compiler islands
- Island 0: onnxruntime/fp32 (3 nodes, 2.450 ms)
- Island 1: onnxruntime/fp32 (2 nodes, 1.350 ms)
- Island 2: onnxruntime/fp32 (1 nodes, 0.050 ms)

---
*Third-party tool inspired by Nuro's published FTL architecture. Not affiliated with Nuro, Inc.*