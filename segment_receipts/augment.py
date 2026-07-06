from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper


def augment_intermediate_outputs(
    model_path: Path | str,
    max_tensors: int = 64,
) -> onnx.ModelProto:
    """
    Expose intermediate node outputs as graph outputs for per-tensor comparison.

    This is how compiler teams localize silent regressions: run the same inputs
    through two execution paths and diff every activation boundary.
    """
    model = onnx.load(str(model_path))
    graph = model.graph
    existing = {o.name for o in graph.output}

    added = 0
    for node in graph.node:
        for tensor in node.output:
            if tensor in existing or not tensor:
                continue
            if added >= max_tensors:
                break
            vi = helper.make_tensor_value_info(tensor, TensorProto.FLOAT, None)
            graph.output.append(vi)
            existing.add(tensor)
            added += 1
        if added >= max_tensors:
            break

    return model


def save_augmented(model: onnx.ModelProto, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(path))
    return path
