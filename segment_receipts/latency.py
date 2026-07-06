from __future__ import annotations

DEFAULT_OP_WEIGHT_MS = {
    "Conv": 1.2,
    "Gemm": 1.0,
    "MatMul": 0.9,
    "Relu": 0.05,
    "Add": 0.08,
    "Mul": 0.08,
    "Concat": 0.15,
    "Softmax": 0.4,
    "Transpose": 0.1,
    "Reshape": 0.05,
    "Flatten": 0.05,
    "MaxPool": 0.3,
    "AveragePool": 0.3,
    "BatchNormalization": 0.2,
    "Sigmoid": 0.1,
    "Tanh": 0.1,
}

BACKEND_MULTIPLIER = {
    "tensorrt": 0.55,
    "onnxruntime": 1.0,
    "custom-kernel": 0.7,
    "inductor": 0.8,
    "openxla": 0.85,
}

DTYPE_MULTIPLIER = {
    "fp32": 1.0,
    "fp16": 0.65,
    "int8": 0.45,
}


def estimate_segment_latency_ms(
    op_types: list[str],
    backend: str,
    dtype: str,
    custom_weights: dict[str, float] | None = None,
) -> float:
    weights = {**DEFAULT_OP_WEIGHT_MS, **(custom_weights or {})}
    base = sum(weights.get(op, 0.12) for op in op_types)
    return round(base * BACKEND_MULTIPLIER.get(backend, 1.0) * DTYPE_MULTIPLIER.get(dtype, 1.0), 4)


def apply_latency_estimates(segments, graph, rules) -> float:
    """Fill per-segment estimated_ms and return total budget."""
    total = 0.0
    for seg in segments:
        ops = [graph.node_by_name[n].op_type for n in seg.node_names]
        seg.estimated_ms = estimate_segment_latency_ms(
            ops, seg.backend, seg.dtype, rules.latency_weights
        )
        total += seg.estimated_ms
    return round(total, 4)
