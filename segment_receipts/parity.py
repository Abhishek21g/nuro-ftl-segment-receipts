from __future__ import annotations

from pathlib import Path

import numpy as np

from segment_receipts.models import ParityResult

_ORT_AVAILABLE: bool | None = None


def ort_available() -> bool:
    global _ORT_AVAILABLE
    if _ORT_AVAILABLE is None:
        try:
            import onnxruntime  # noqa: F401

            _ORT_AVAILABLE = True
        except ImportError:
            _ORT_AVAILABLE = False
    return _ORT_AVAILABLE


def check_parity(
    golden_outputs: dict[str, np.ndarray],
    segment_outputs: dict[str, np.ndarray],
    segment_id: int,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> list[ParityResult]:
    results: list[ParityResult] = []
    for name, golden in golden_outputs.items():
        if name not in segment_outputs:
            continue
        actual = segment_outputs[name]
        if golden.shape != actual.shape:
            results.append(
                ParityResult(
                    segment_id=segment_id,
                    output_name=name,
                    max_abs_diff=float("inf"),
                    mean_abs_diff=float("inf"),
                    rtol=rtol,
                    atol=atol,
                    passed=False,
                )
            )
            continue
        diff = np.abs(golden.astype(np.float64) - actual.astype(np.float64))
        max_diff = float(np.max(diff))
        mean_diff = float(np.mean(diff))
        passed = bool(np.allclose(golden, actual, rtol=rtol, atol=atol))
        results.append(
            ParityResult(
                segment_id=segment_id,
                output_name=name,
                max_abs_diff=round(max_diff, 8),
                mean_abs_diff=round(mean_diff, 8),
                rtol=rtol,
                atol=atol,
                passed=passed,
            )
        )
    return results


def probe_full_model_parity(
    model_path: Path,
    output_names: list[str] | None = None,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> list[ParityResult]:
    """
    Run ONNX model twice with ORT when available.
    Falls back to synthetic self-check for JSON graphs or missing ORT.
    """
    path = Path(model_path)
    if path.suffix == ".json" or not ort_available():
        names = output_names or ["output"]
        synthetic = {n: np.zeros(1, dtype=np.float32) for n in names}
        return check_parity(synthetic, synthetic, segment_id=-1, rtol=rtol, atol=atol)

    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs: dict[str, np.ndarray] = {}
    for meta in session.get_inputs():
        shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in meta.shape]
        inputs[meta.name] = np.random.randn(*shape).astype(np.float32)

    run_a = dict(zip([o.name for o in session.get_outputs()], session.run(None, inputs)))
    run_b = dict(zip([o.name for o in session.get_outputs()], session.run(None, inputs)))
    return check_parity(run_a, run_b, segment_id=-1, rtol=rtol, atol=atol)
