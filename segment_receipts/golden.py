from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort


def export_golden_npz(model_path: Path, output_path: Path, seed: int = 42) -> Path:
    """Save ORT reference tensors for export-validation scans."""
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    inputs: dict[str, np.ndarray] = {}
    for meta in sess.get_inputs():
        shape = [d if isinstance(d, int) and d > 0 else 1 for d in meta.shape]
        inputs[meta.name] = rng.standard_normal(shape).astype(np.float32)

    outputs = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, inputs)))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **inputs, **outputs)
    return output_path
