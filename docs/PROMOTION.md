# Promotion v1.0 — Silent Regression Hunter

## The real problem (from Nuro's blog)

> "Third party compilers are notorious for causing silent regressions."

FTL's segment breaker (Fig 6) exists **because of this** — not as an abstract compiler feature.

## What to say (honest, high-signal)

**Don't say:** "I built a segment receipt tool inspired by FTL."

**Do say:** "I built a tool that finds *where* ONNX compile paths silently diverge and outputs the exact FP32 segment breaker Nuro's blog describes — tested on real ORT intermediate tensor dumps."

## Pitch

```
Silent regressions in TRT/FP16 compiler islands don't crash — they drift perception.

I built an open scanner that:
  • dumps every ONNX intermediate activation
  • diffs reference vs FP16/optimized compile path
  • pinpoints the FIRST failing node in topo order
  • emits FTL-style break_before_nodes / force_fp32 rules

Demo: https://enaguthi.com/nuro-ftl-receipts/site/demo/regression_report.html
Repo: https://github.com/Abhishek21g/nuro-ftl-segment-receipts

Third-party tool from Nuro's public FTL blog — not affiliated.
```

## Where this lands

| Audience | Why they care |
|----------|----------------|
| Nuro ML infra | Same problem their FTL segment breaker solves |
| Any AV perception team | ONNX→TRT handoff with silent drift |
| Compiler engineers | Intermediate tensor diff is standard debug — tool automates it |

## Channels

1. LinkedIn — lead with the **silent regression** problem, not the tool name
2. Comment on Nuro FTL Medium post — technical, link regression report
3. Nuro careers — ML compiler / perception infra roles
4. Show HN — "Finding silent ONNX compile regressions before deployment"
