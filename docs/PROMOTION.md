# Promotion v1.1 — Launch

## One-line pitch

Open ONNX silent-regression scanner — finds where FP16/TRT compile paths diverge, outputs FTL-style segment breakers. The pre-deploy audit Nuro's blog describes.

## Links to share

| What | URL |
|------|-----|
| Landing | https://enaguthi.com/nuro-ftl-receipts/site/ |
| Regression report | https://enaguthi.com/nuro-ftl-receipts/site/demo/regression_report.html |
| Repo | https://github.com/Abhishek21g/nuro-ftl-segment-receipts |

## Copy-paste outreach

```
Nuro's FTL blog names the problem: third-party compilers cause silent regressions.
Segment breakers (Fig 6) exist to isolate FP32 islands when numerics drift.

I built an open pre-deploy scanner that:
  • dumps every ONNX intermediate activation via ORT
  • diffs reference vs FP16 compile path
  • pinpoints first topo failure + emits break_before_nodes YAML
  • plan → run → doctor → report CLI with full artifact trail

Demo: https://enaguthi.com/nuro-ftl-receipts/site/
Repo: https://github.com/Abhishek21g/nuro-ftl-segment-receipts

Third-party tool from public FTL architecture — not affiliated with Nuro.
```

## 60s demo commands

```bash
pip install nuro-ftl-segment-receipts[onnx]
segment-receipts run examples/models/branch.onnx -o out/receipts
segment-receipts doctor out/receipts/<run-id>
```

## Channels

1. Comment on [Nuro FTL Medium post](https://medium.com/nuro/ftl-model-compiler-framework-d6b85c670f67) — technical, link regression report
2. LinkedIn — lead with silent regression problem
3. Nuro careers — ML compiler / perception infra
