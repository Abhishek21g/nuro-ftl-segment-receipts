from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper


def resolve_onnx_path(model_path: Path | str) -> Path:
    """Prefer .onnx on disk; accept .graph.json if sibling .onnx exists."""
    path = Path(model_path)
    if path.suffix == ".onnx":
        return path
    if path.name.endswith(".graph.json"):
        sibling = path.with_name(path.name.replace(".graph.json", ".onnx"))
        if sibling.exists():
            return sibling
    if path.suffix == ".json" and path.with_suffix(".onnx").exists():
        return path.with_suffix(".onnx")
    raise ValueError(
        f"Regression scan requires ONNX format: {path}. "
        "Generate .onnx alongside .graph.json or pass an .onnx path."
    )


def augment_intermediate_outputs(
    model_path: Path | str,
    max_tensors: int = 64,
) -> onnx.ModelProto:
    """
    Expose intermediate node outputs as graph outputs for per-tensor comparison.

    This is how compiler teams localize silent regressions: run the same inputs
    through two execution paths and diff every activation boundary.
    """
    onnx_path = resolve_onnx_path(model_path)
    model = onnx.load(str(onnx_path))
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
