from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from segment_receipts.models import NodeInfo


@dataclass
class GraphModel:
    """Compute graph — loadable from ONNX or portable JSON."""

    nodes: list[NodeInfo]
    node_by_name: dict[str, NodeInfo]
    value_producers: dict[str, str]
    input_names: list[str]
    output_names: list[str]
    source_path: str = ""

    def topological_order(self) -> list[str]:
        from collections import defaultdict, deque

        in_degree: dict[str, int] = {n.name: 0 for n in self.nodes}
        successors: dict[str, list[str]] = defaultdict(list)

        for node in self.nodes:
            deps = {
                self.value_producers[inp]
                for inp in node.inputs
                if inp in self.value_producers and self.value_producers[inp] != "__initializer__"
            }
            for dep in deps:
                if dep in in_degree:
                    in_degree[node.name] += 1
                    successors[dep].append(node.name)

        queue = deque(name for name, deg in in_degree.items() if deg == 0)
        order: list[str] = []
        while queue:
            current = queue.popleft()
            order.append(current)
            for nxt in successors[current]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(self.nodes):
            raise ValueError("Graph contains a cycle or disconnected nodes")
        return order

    def model_inputs(self) -> list[str]:
        return list(self.input_names)

    def model_outputs(self) -> list[str]:
        return list(self.output_names)


class GraphLoader(Protocol):
    @classmethod
    def from_path(cls, path: Path | str) -> GraphModel: ...


def load_graph(path: Path | str) -> GraphModel:
    path = Path(path)
    if path.suffix == ".json":
        return GraphModel.from_json(path)
    try:
        return GraphModel.from_onnx(path)
    except ImportError as exc:
        raise ImportError(
            "ONNX support requires `pip install onnx`. "
            f"Use a `.graph.json` file instead. ({exc})"
        ) from exc


# Back-compat alias
OnnxGraph = GraphModel


def _node_from_dict(data: dict[str, Any]) -> NodeInfo:
    return NodeInfo(
        name=data["name"],
        op_type=data["op_type"],
        inputs=list(data.get("inputs", [])),
        outputs=list(data.get("outputs", [])),
        attributes=dict(data.get("attributes", {})),
    )


def _build_index(nodes: list[NodeInfo], initializers: set[str]) -> tuple[dict[str, NodeInfo], dict[str, str]]:
    node_by_name = {n.name: n for n in nodes}
    value_producers: dict[str, str] = {name: "__initializer__" for name in initializers}
    for node in nodes:
        for output in node.outputs:
            value_producers[output] = node.name
    return node_by_name, value_producers


def _attach_graph_methods(cls: type) -> type:
    @classmethod
    def from_json(cls, path: Path | str) -> GraphModel:
        data = json.loads(Path(path).read_text())
        nodes = [_node_from_dict(n) for n in data["nodes"]]
        initializers = set(data.get("initializers", []))
        node_by_name, value_producers = _build_index(nodes, initializers)
        return cls(
            nodes=nodes,
            node_by_name=node_by_name,
            value_producers=value_producers,
            input_names=list(data.get("inputs", [])),
            output_names=list(data.get("outputs", [])),
            source_path=str(path),
        )

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> GraphModel:
        nodes = [_node_from_dict(n) for n in data["nodes"]]
        initializers = set(data.get("initializers", []))
        node_by_name, value_producers = _build_index(nodes, initializers)
        return cls(
            nodes=nodes,
            node_by_name=node_by_name,
            value_producers=value_producers,
            input_names=list(data.get("inputs", [])),
            output_names=list(data.get("outputs", [])),
            source_path="",
        )

    @classmethod
    def from_onnx(cls, path: Path | str) -> GraphModel:
        import onnx

        model = onnx.load(str(path))
        graph = model.graph
        initializers = {i.name for i in graph.initializer}
        nodes: list[NodeInfo] = []
        for node in graph.node:
            nodes.append(
                NodeInfo(
                    name=node.name or f"{node.op_type}_{len(nodes)}",
                    op_type=node.op_type,
                    inputs=list(node.input),
                    outputs=list(node.output),
                    attributes={a.name: _onnx_attr(a) for a in node.attribute},
                )
            )
        node_by_name, value_producers = _build_index(nodes, initializers)
        inputs = [i.name for i in graph.input if i.name not in initializers]
        outputs = [o.name for o in graph.output]
        return cls(
            nodes=nodes,
            node_by_name=node_by_name,
            value_producers=value_producers,
            input_names=inputs,
            output_names=outputs,
            source_path=str(path),
        )

    def to_json_dict(self: GraphModel) -> dict[str, Any]:
        return {
            "inputs": self.input_names,
            "outputs": self.output_names,
            "initializers": [k for k, v in self.value_producers.items() if v == "__initializer__"],
            "nodes": [
                {
                    "name": n.name,
                    "op_type": n.op_type,
                    "inputs": n.inputs,
                    "outputs": n.outputs,
                    "attributes": n.attributes,
                }
                for n in self.nodes
            ],
        }

    cls.from_json = from_json  # type: ignore[method-assign]
    cls.from_json_dict = from_json_dict  # type: ignore[method-assign]
    cls.from_onnx = from_onnx  # type: ignore[method-assign]
    cls.to_json_dict = to_json_dict  # type: ignore[method-assign]
    return cls


def _onnx_attr(attr: Any) -> object:
    import onnx

    if attr.type == onnx.AttributeProto.INT:
        return int(attr.i)
    if attr.type == onnx.AttributeProto.FLOAT:
        return float(attr.f)
    if attr.type == onnx.AttributeProto.STRING:
        return attr.s.decode("utf-8")
    if attr.type == onnx.AttributeProto.INTS:
        return list(attr.ints)
    if attr.type == onnx.AttributeProto.FLOATS:
        return list(attr.floats)
    if attr.type == onnx.AttributeProto.STRINGS:
        return [s.decode("utf-8") for s in attr.strings]
    return None


GraphModel = _attach_graph_methods(GraphModel)
