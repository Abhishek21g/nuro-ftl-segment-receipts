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
        "resnet18_mini": resnet18_mini(models / "resnet18-mini.onnx"),
        "decomposed_trap": decomposed_layernorm_trap(models / "decomposed-trap.onnx"),
    }


def resnet18_mini(path: Path) -> Path:
    """Small ResNet-style stack for bundled demo (not full ResNet18)."""
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        nodes = []
        initializers = []
        prev = "input"
        in_ch = 3
        for i, out_ch in enumerate([16, 32, 32, 64]):
            w = numpy_helper.from_array(
                np.random.randn(out_ch, in_ch, 3, 3).astype(np.float32) * 0.1, f"w{i}"
            )
            initializers.append(w)
            conv = f"conv{i}"
            relu = f"relu{i}"
            nodes.append(
                helper.make_node(
                    "Conv",
                    [prev, f"w{i}"],
                    [conv],
                    name=f"block{i}_conv",
                    kernel_shape=[3, 3],
                    pads=[1, 1, 1, 1],
                )
            )
            nodes.append(helper.make_node("Relu", [conv], [relu], name=f"block{i}_relu"))
            prev = relu
            in_ch = out_ch

        wf = numpy_helper.from_array(
            np.random.randn(10, in_ch, 1, 1).astype(np.float32) * 0.1, "w_fc"
        )
        initializers.append(wf)
        nodes.append(
            helper.make_node(
                "Conv", [prev, "w_fc"], ["logits"], name="fc", kernel_shape=[1, 1]
            )
        )

        g = helper.make_graph(
            nodes,
            "resnet18_mini",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])],
            [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 10, 32, 32])],
            initializer=initializers,
        )
        model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(path))
        return path
    except ImportError:
        path.parent.mkdir(parents=True, exist_ok=True)
        return linear_chain(path, depth=4)


def decomposed_layernorm_trap(path: Path) -> Path:
    """Blog Fig 5 style: LayerNorm decomposed into many primitive ops."""
    try:
        import numpy as np
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        scale = numpy_helper.from_array(np.ones(8, dtype=np.float32), "scale")
        bias = numpy_helper.from_array(np.zeros(8, dtype=np.float32), "bias")
        eps = numpy_helper.from_array(np.array([1e-5], dtype=np.float32), "eps")

        nodes = [
            helper.make_node("Conv", ["input", "w0"], ["stem"], name="stem_conv", kernel_shape=[1, 1]),
            helper.make_node("ReduceMean", ["stem"], ["mean"], name="ln_mean", axes=[2, 3], keepdims=1),
            helper.make_node("Sub", ["stem", "mean"], ["centered"], name="ln_sub"),
            helper.make_node("Pow", ["centered"], ["sq"], name="ln_pow", exponent=2.0),
            helper.make_node("ReduceMean", ["sq"], ["var"], name="ln_var", axes=[2, 3], keepdims=1),
            helper.make_node("Add", ["var", "eps"], ["var_eps"], name="ln_add_eps"),
            helper.make_node("Sqrt", ["var_eps"], ["std"], name="ln_sqrt"),
            helper.make_node("Div", ["centered", "std"], ["norm"], name="ln_div"),
            helper.make_node("Mul", ["norm", "scale"], ["scaled"], name="ln_mul"),
            helper.make_node("Add", ["scaled", "bias"], ["output"], name="ln_add_bias"),
        ]
        w0 = numpy_helper.from_array(np.random.randn(8, 3, 1, 1).astype(np.float32), "w0")
        g = helper.make_graph(
            nodes,
            "decomposed_layernorm_trap",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 8, 8])],
            initializer=[w0, scale, bias, eps],
        )
        model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(path))
        return path
    except ImportError:
        return path
