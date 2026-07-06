from __future__ import annotations

from pathlib import Path

import numpy as np

from segment_receipts.latency import estimate_segment_latency_ms
from segment_receipts.parity import check_parity


def test_parity_identical_passes() -> None:
    a = {"out": np.array([1.0, 2.0], dtype=np.float32)}
    results = check_parity(a, a, segment_id=0)
    assert len(results) == 1
    assert results[0].passed


def test_parity_divergence_fails() -> None:
    a = {"out": np.array([1.0], dtype=np.float32)}
    b = {"out": np.array([2.0], dtype=np.float32)}
    results = check_parity(a, b, segment_id=0, atol=1e-6)
    assert not results[0].passed


def test_parity_shape_mismatch() -> None:
    a = {"out": np.ones((2, 2), dtype=np.float32)}
    b = {"out": np.ones((3, 3), dtype=np.float32)}
    results = check_parity(a, b, segment_id=0)
    assert not results[0].passed


def test_latency_tensorrt_faster() -> None:
    trt = estimate_segment_latency_ms(["Conv", "Relu"], "tensorrt", "fp16")
    ort = estimate_segment_latency_ms(["Conv", "Relu"], "onnxruntime", "fp32")
    assert trt < ort


def test_latency_custom_weights() -> None:
    ms = estimate_segment_latency_ms(["Conv"], "onnxruntime", "fp32", {"Conv": 5.0})
    assert ms == 5.0


def test_probe_full_model(tmp_path: Path) -> None:
    from segment_receipts.parity import probe_full_model_parity
    from segment_receipts.toy_models import linear_chain

    model = linear_chain(tmp_path / "m.onnx", depth=1)
    results = probe_full_model_parity(model)
    assert all(r.passed for r in results)
