from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import onnxruntime as ort

from segment_receipts.augment import augment_intermediate_outputs, save_augmented
from segment_receipts.graph import load_graph


PathKind = Literal["reference", "optimized", "fp16_activations", "golden"]


@dataclass
class TensorDivergence:
    tensor_name: str
    producer_node: str
    producer_op: str
    max_abs_diff: float
    mean_abs_diff: float
    relative_l2: float
    passed: bool
    reference_path: PathKind
    candidate_path: PathKind


@dataclass
class BreakerRecommendation:
    node_name: str
    op_type: str
    tensor_name: str
    max_abs_diff: float
    reason: str
    suggested_rule: dict[str, Any]


@dataclass
class RegressionReport:
    model_path: str
    problem: str
    reference_path: PathKind
    candidate_path: PathKind
    rtol: float
    atol: float
    tensors_compared: int
    tensors_failed: int
    first_failure: TensorDivergence | None
    divergences: list[TensorDivergence]
    breaker_recommendations: list[BreakerRecommendation]
    graph_layout: list[dict[str, Any]]
    inputs_used: dict[str, list[int]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _session(model: onnx.ModelProto, optimize: bool) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    return ort.InferenceSession(
        model.SerializeToString(),
        opts,
        providers=["CPUExecutionProvider"],
    )


def _random_inputs(session: ort.InferenceSession, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    inputs: dict[str, np.ndarray] = {}
    for meta in session.get_inputs():
        shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in meta.shape]
        inputs[meta.name] = rng.standard_normal(shape).astype(np.float32)
    return inputs


def _load_npz_inputs(path: Path, session: ort.InferenceSession) -> dict[str, np.ndarray]:
    data = np.load(path)
    inputs: dict[str, np.ndarray] = {}
    for meta in session.get_inputs():
        if meta.name not in data:
            raise ValueError(f"golden npz missing input '{meta.name}'")
        inputs[meta.name] = data[meta.name].astype(np.float32)
    return inputs


def _run_session(
    session: ort.InferenceSession,
    inputs: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    names = [o.name for o in session.get_outputs()]
    values = session.run(None, inputs)
    return dict(zip(names, values))


def _fp16_activation_path(tensors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Simulate FP16 activation quantization between compiler islands."""
    degraded: dict[str, np.ndarray] = {}
    for name, arr in tensors.items():
        if arr.dtype in (np.float32, np.float64):
            degraded[name] = arr.astype(np.float16).astype(np.float32)
        else:
            degraded[name] = arr
    return degraded


def _diff_tensor(
    ref: np.ndarray,
    cand: np.ndarray,
    rtol: float,
    atol: float,
) -> tuple[float, float, float, bool]:
    if ref.shape != cand.shape:
        return float("inf"), float("inf"), float("inf"), False
    ref64 = ref.astype(np.float64)
    cand64 = cand.astype(np.float64)
    diff = np.abs(ref64 - cand64)
    max_abs = float(np.max(diff))
    mean_abs = float(np.mean(diff))
    denom = float(np.linalg.norm(ref64)) + 1e-12
    rel_l2 = float(np.linalg.norm(diff) / denom)
    passed = bool(np.allclose(ref64, cand64, rtol=rtol, atol=atol))
    return max_abs, mean_abs, rel_l2, passed


def _graph_layout(model_path: Path, divergences: list[TensorDivergence]) -> list[dict[str, Any]]:
    graph = load_graph(model_path)
    order = graph.topological_order()
    div_by_tensor = {d.tensor_name: d for d in divergences}
    div_by_node = {d.producer_node: d for d in divergences}

    layout: list[dict[str, Any]] = []
    cols = 4
    for idx, name in enumerate(order):
        node = graph.node_by_name[name]
        div = div_by_node.get(name)
        tensor_div = div_by_tensor.get(node.outputs[0]) if node.outputs else None
        score = 0.0
        if tensor_div:
            score = tensor_div.max_abs_diff
        elif div:
            score = div.max_abs_diff
        layout.append(
            {
                "node": name,
                "op": node.op_type,
                "tensor": node.outputs[0] if node.outputs else "",
                "row": idx // cols,
                "col": idx % cols,
                "max_abs_diff": score,
                "failed": bool(tensor_div and not tensor_div.passed),
            }
        )
    return layout


def _recommend_breakers(
    divergences: list[TensorDivergence],
    limit: int = 3,
    first_failure: TensorDivergence | None = None,
) -> list[BreakerRecommendation]:
    failing = [d for d in divergences if not d.passed]
    failing.sort(key=lambda d: d.max_abs_diff, reverse=True)

    # Always recommend breaker at first topo failure
    if first_failure and first_failure not in failing:
        failing.append(first_failure)
    elif first_failure:
        failing = [first_failure] + [d for d in failing if d.producer_node != first_failure.producer_node]

    seen: set[str] = set()
    recs: list[BreakerRecommendation] = []
    for d in failing:
        if d.producer_node in seen:
            continue
        seen.add(d.producer_node)
        recs.append(
            BreakerRecommendation(
                node_name=d.producer_node,
                op_type=d.producer_op,
                tensor_name=d.tensor_name,
                max_abs_diff=d.max_abs_diff,
                reason=(
                    f"Activation '{d.tensor_name}' diverges max={d.max_abs_diff:.2e} "
                    f"between {d.reference_path} and {d.candidate_path}."
                ),
                suggested_rule={
                    "break_before_nodes": [d.producer_node],
                    "force_fp32_nodes": [d.producer_node],
                    "backend_overrides": {d.producer_node: "onnxruntime"},
                },
            )
        )
        if len(recs) >= limit:
            break
    return recs


def scan_regression(
    model_path: Path,
    *,
    candidate: PathKind = "optimized",
    golden_npz: Path | None = None,
    rtol: float = 1e-4,
    atol: float = 1e-5,
    max_tensors: int = 64,
    seed: int = 42,
    work_dir: Path | None = None,
) -> RegressionReport:
    """
    Find where an ONNX compile path first diverges from reference.

    Modes:
      reference=ORT_DISABLE_ALL, candidate=ORT_ENABLE_ALL (optimizer regressions)
      reference=ORT_DISABLE_ALL, candidate=fp16_activations (precision regressions)
      reference=ORT_DISABLE_ALL, candidate=golden (export validation vs saved tensors)
    """
    work = work_dir or Path("receipts/.scan_work")
    work.mkdir(parents=True, exist_ok=True)

    augmented = augment_intermediate_outputs(model_path, max_tensors=max_tensors)
    aug_path = save_augmented(augmented, work / "augmented.onnx")

    ref_sess = _session(augmented, optimize=False)
    inputs = _load_npz_inputs(golden_npz, ref_sess) if golden_npz else _random_inputs(ref_sess, seed)

    ref_out = _run_session(ref_sess, inputs)

    if candidate == "optimized":
        cand_sess = _session(augmented, optimize=True)
        cand_out = _run_session(cand_sess, inputs)
        ref_kind: PathKind = "reference"
        cand_kind: PathKind = "optimized"
        problem = (
            "Nuro's FTL blog: third-party compilers cause silent numerical regressions. "
            "This scan compares ORT with graph optimizations disabled (golden reference) "
            "vs enabled (simulates fused/compiled path drift)."
        )
    elif candidate == "fp16_activations":
        cand_out = _fp16_activation_path(ref_out)
        ref_kind = "reference"
        cand_kind = "fp16_activations"
        problem = (
            "FP16/TensorRT islands can drift while the rest of the graph stays FP32. "
            "This scan simulates FP16 activation rounding at every tensor boundary."
        )
    elif candidate == "golden":
        if not golden_npz:
            raise ValueError("golden candidate requires --golden npz with reference tensors")
        data = np.load(golden_npz)
        cand_out = {k: data[k] for k in ref_out if k in data}
        ref_kind = "reference"
        cand_kind = "golden"
        problem = "Export validation: ONNX runtime path vs saved framework golden tensors."
    else:
        raise ValueError(f"unknown candidate mode: {candidate}")

    graph = load_graph(model_path)
    tensor_to_node = {}
    for node in graph.nodes:
        for t in node.outputs:
            tensor_to_node[t] = node

    divergences: list[TensorDivergence] = []
    for tensor_name, ref_val in ref_out.items():
        if tensor_name not in cand_out:
            continue
        producer = tensor_to_node.get(tensor_name)
        if producer is None:
            continue
        max_d, mean_d, rel_l2, passed = _diff_tensor(ref_val, cand_out[tensor_name], rtol, atol)
        divergences.append(
            TensorDivergence(
                tensor_name=tensor_name,
                producer_node=producer.name,
                producer_op=producer.op_type,
                max_abs_diff=round(max_d, 8),
                mean_abs_diff=round(mean_d, 8),
                relative_l2=round(rel_l2, 8),
                passed=passed,
                reference_path=ref_kind,
                candidate_path=cand_kind,
            )
        )

    divergences.sort(key=lambda d: d.max_abs_diff, reverse=True)
    failed = [d for d in divergences if not d.passed]
    first_failure = failed[-1] if failed else None
    # first in topo order
    if failed:
        order = graph.topological_order()
        order_idx = {n: i for i, n in enumerate(order)}
        first_failure = min(failed, key=lambda d: order_idx.get(d.producer_node, 10**9))

    return RegressionReport(
        model_path=str(model_path),
        problem=problem,
        reference_path=ref_kind,
        candidate_path=cand_kind,
        rtol=rtol,
        atol=atol,
        tensors_compared=len(divergences),
        tensors_failed=len(failed),
        first_failure=first_failure,
        divergences=divergences,
        breaker_recommendations=_recommend_breakers(divergences, first_failure=first_failure),
        graph_layout=_graph_layout(model_path, divergences),
        inputs_used={k: list(v.shape) for k, v in inputs.items()},
    )


def write_regression_report(report: RegressionReport, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "regression_report.json"
    path.write_text(json.dumps(report.to_dict(), indent=2))
    return path
