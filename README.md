# Silent Regression Hunter

**The problem Nuro's FTL compiler solves:** third-party compilers (TensorRT, FP16 paths) cause *silent* numerical regressions — perception drifts without crashing. Engineers insert **FP32 segment breakers** at specific nodes to isolate drift ([FTL blog Fig 6](https://medium.com/nuro/ftl-model-compiler-framework-d6b85c670f67)).

**This tool answers:** *where does your ONNX compile path first diverge, and which node should get a breaker?*

Not affiliated with Nuro, Inc.

## What it does (not a toy segment diagram)

1. **Augments** your ONNX graph to expose every intermediate activation
2. **Runs** reference path (ORT, optimizations off) vs candidate (FP16 activations, ORT fusion, or golden npz)
3. **Diffs** each tensor — finds first failure in topological order
4. **Outputs** FTL-style `break_before_nodes` / `force_fp32_nodes` recommendations

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,onnx]"
```

## Primary workflow

```bash
# Find silent regressions (default: FP16 activation drift)
segment-receipts scan model.onnx -o receipts/scan

# ORT graph-fusion drift
segment-receipts scan model.onnx --candidate optimized

# Export validation vs saved golden tensors
segment-receipts scan model.onnx --candidate golden --golden golden.npz
```

Outputs: `regression_report.json` + `regression_report.html` with graph heatmap.

## Secondary: stitch receipts

After breakers are clean, generate segment stitch receipts:

```bash
segment-receipts plan model.onnx -r rules.yaml
segment-receipts run model.onnx -r rules.yaml -o receipts/stitch
```

## Demo

- **Site:** https://enaguthi.com/nuro-ftl-receipts/site/
- **Live regression report:** https://enaguthi.com/nuro-ftl-receipts/site/demo/regression_report.html

## Test

```bash
pytest tests/ -v
```

## License

MIT
