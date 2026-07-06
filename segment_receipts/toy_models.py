"""Generate small graphs for tests and examples (JSON-first; ONNX when available)."""

from __future__ import annotations

import json
from pathlib import Path

from segment_receipts.graph import GraphModel


def save_graph_json(graph: GraphModel, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph.to_json_dict(), indent=2))
    return path


def _linear_chain_graph(depth: int = 3) -> GraphModel:
    nodes = []
    prev_out = "input"
    for i in range(depth):
        conv_out = f"conv{i}"
        nodes.append(
            {
                "name": f"conv_{i}",
                "op_type": "Conv",
                "inputs": [prev_out, f"w{i}"],
                "outputs": [conv_out],
                "attributes": {"kernel_shape": [1, 1]},
            }
        )
        relu_out = f"relu{i}"
        nodes.append(
            {
                "name": f"relu_{i}",
                "op_type": "Relu",
                "inputs": [conv_out],
                "outputs": [relu_out],
            }
        )
        prev_out = relu_out
    return GraphModel.from_json_dict(
        {
            "inputs": ["input"],
            "outputs": [prev_out],
            "initializers": [f"w{i}" for i in range(depth)],
            "nodes": nodes,
        }
    )


def _branched_graph() -> GraphModel:
    return GraphModel.from_json_dict(
        {
            "inputs": ["input"],
            "outputs": ["output"],
            "initializers": ["w0", "w1", "w2"],
            "nodes": [
                {
                    "name": "stem_conv",
                    "op_type": "Conv",
                    "inputs": ["input", "w0"],
                    "outputs": ["stem"],
                    "attributes": {"kernel_shape": [1, 1]},
                },
                {
                    "name": "stem_relu",
                    "op_type": "Relu",
                    "inputs": ["stem"],
                    "outputs": ["stem_relu"],
                },
                {
                    "name": "branch_a_conv",
                    "op_type": "Conv",
                    "inputs": ["stem_relu", "w1"],
                    "outputs": ["branch_a"],
                    "attributes": {"kernel_shape": [1, 1]},
                },
                {
                    "name": "branch_b_conv",
                    "op_type": "Conv",
                    "inputs": ["stem_relu", "w2"],
                    "outputs": ["branch_b"],
                    "attributes": {"kernel_shape": [1, 1]},
                },
                {
                    "name": "merge",
                    "op_type": "Concat",
                    "inputs": ["branch_a", "branch_b"],
                    "outputs": ["merged"],
                    "attributes": {"axis": 1},
                },
                {
                    "name": "out_relu",
                    "op_type": "Relu",
                    "inputs": ["merged"],
                    "outputs": ["output"],
                },
            ],
        }
    )


def _multi_output_graph() -> GraphModel:
    return GraphModel.from_json_dict(
        {
            "inputs": ["input"],
            "outputs": ["head_boxes", "head_classes"],
            "initializers": ["w0", "w1", "w2"],
            "nodes": [
                {
                    "name": "trunk_conv",
                    "op_type": "Conv",
                    "inputs": ["input", "w0"],
                    "outputs": ["trunk"],
                    "attributes": {"kernel_shape": [1, 1]},
                },
                {
                    "name": "trunk_relu",
                    "op_type": "Relu",
                    "inputs": ["trunk"],
                    "outputs": ["trunk_relu"],
                },
                {
                    "name": "boxes_head",
                    "op_type": "Conv",
                    "inputs": ["trunk_relu", "w1"],
                    "outputs": ["head_boxes"],
                    "attributes": {"kernel_shape": [1, 1]},
                },
                {
                    "name": "classes_head",
                    "op_type": "Conv",
                    "inputs": ["trunk_relu", "w2"],
                    "outputs": ["head_classes"],
                    "attributes": {"kernel_shape": [1, 1]},
                },
            ],
        }
    )


def linear_chain(path: Path, depth: int = 3) -> Path:
    graph = _linear_chain_graph(depth)
    json_path = path.with_suffix(".graph.json")
    save_graph_json(graph, json_path)
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        nodes = []
        initializers = []
        prev_out = "input"
        for i in range(depth):
            w = numpy_helper.from_array(
                np.random.randn(4, 4, 1, 1).astype(np.float32), f"w{i}"
            )
            initializers.append(w)
            conv_out = f"conv{i}"
            nodes.append(
                helper.make_node(
                    "Conv", [prev_out, f"w{i}"], [conv_out], name=f"conv_{i}", kernel_shape=[1, 1]
                )
            )
            relu_out = f"relu{i}"
            nodes.append(helper.make_node("Relu", [conv_out], [relu_out], name=f"relu_{i}"))
            prev_out = relu_out
        g = helper.make_graph(
            nodes,
            "linear_chain",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4, 8, 8])],
            [helper.make_tensor_value_info(prev_out, TensorProto.FLOAT, [1, 4, 8, 8])],
            initializer=initializers,
        )
        model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(path))
        return path
    except ImportError:
        return json_path


def branched_graph(path: Path) -> Path:
    save_graph_json(_branched_graph(), path.with_suffix(".graph.json"))
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        w0 = numpy_helper.from_array(np.random.randn(4, 3, 1, 1).astype(np.float32), "w0")
        w1 = numpy_helper.from_array(np.random.randn(4, 4, 1, 1).astype(np.float32), "w1")
        w2 = numpy_helper.from_array(np.random.randn(4, 4, 1, 1).astype(np.float32), "w2")
        nodes = [
            helper.make_node("Conv", ["input", "w0"], ["stem"], name="stem_conv", kernel_shape=[1, 1]),
            helper.make_node("Relu", ["stem"], ["stem_relu"], name="stem_relu"),
            helper.make_node("Conv", ["stem_relu", "w1"], ["branch_a"], name="branch_a_conv", kernel_shape=[1, 1]),
            helper.make_node("Conv", ["stem_relu", "w2"], ["branch_b"], name="branch_b_conv", kernel_shape=[1, 1]),
            helper.make_node("Concat", ["branch_a", "branch_b"], ["merged"], name="merge", axis=1),
            helper.make_node("Relu", ["merged"], ["output"], name="out_relu"),
        ]
        g = helper.make_graph(
            nodes,
            "branched",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 8, 8])],
            initializer=[w0, w1, w2],
        )
        model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(path))
        return path
    except ImportError:
        return path.with_suffix(".graph.json")


def multi_output_head(path: Path) -> Path:
    save_graph_json(_multi_output_graph(), path.with_suffix(".graph.json"))
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        w0 = numpy_helper.from_array(np.random.randn(8, 3, 1, 1).astype(np.float32), "w0")
        w1 = numpy_helper.from_array(np.random.randn(4, 8, 1, 1).astype(np.float32), "w1")
        w2 = numpy_helper.from_array(np.random.randn(2, 8, 1, 1).astype(np.float32), "w2")
        nodes = [
            helper.make_node("Conv", ["input", "w0"], ["trunk"], name="trunk_conv", kernel_shape=[1, 1]),
            helper.make_node("Relu", ["trunk"], ["trunk_relu"], name="trunk_relu"),
            helper.make_node("Conv", ["trunk_relu", "w1"], ["head_boxes"], name="boxes_head", kernel_shape=[1, 1]),
            helper.make_node("Conv", ["trunk_relu", "w2"], ["head_classes"], name="classes_head", kernel_shape=[1, 1]),
        ]
        g = helper.make_graph(
            nodes,
            "multi_output",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 16, 16])],
            [
                helper.make_tensor_value_info("head_boxes", TensorProto.FLOAT, [1, 4, 16, 16]),
                helper.make_tensor_value_info("head_classes", TensorProto.FLOAT, [1, 2, 16, 16]),
            ],
            initializer=[w0, w1, w2],
        )
        model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(path))
        return path
    except ImportError:
        return path.with_suffix(".graph.json")


def build_all_examples(base: Path) -> dict[str, Path]:
    models = base / "models"
    return {
        "chain": linear_chain(models / "chain.onnx"),
        "branch": branched_graph(models / "branch.onnx"),
        "multi_output": multi_output_head(models / "multi_output.onnx"),
    }
