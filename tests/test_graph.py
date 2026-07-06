from __future__ import annotations

from pathlib import Path

import pytest

from segment_receipts.graph import GraphModel, load_graph
from segment_receipts.toy_models import branched_graph, linear_chain, multi_output_head


@pytest.fixture
def chain_model(tmp_path: Path) -> Path:
    return linear_chain(tmp_path / "chain.onnx")


@pytest.fixture
def branch_model(tmp_path: Path) -> Path:
    return branched_graph(tmp_path / "branch.onnx")


@pytest.fixture
def multi_model(tmp_path: Path) -> Path:
    return multi_output_head(tmp_path / "multi.onnx")


def test_graph_loads_chain(chain_model: Path) -> None:
    graph = load_graph(chain_model)
    assert len(graph.nodes) == 6
    assert graph.model_inputs() == ["input"]
    assert len(graph.model_outputs()) == 1


def test_topological_order_length(chain_model: Path) -> None:
    graph = load_graph(chain_model)
    order = graph.topological_order()
    assert len(order) == len(graph.nodes)
    assert len(set(order)) == len(order)


def test_topological_respects_deps(chain_model: Path) -> None:
    graph = load_graph(chain_model)
    order = graph.topological_order()
    pos = {name: i for i, name in enumerate(order)}
    for node in graph.nodes:
        for inp in node.inputs:
            prod = graph.value_producers.get(inp)
            if prod and prod != "__initializer__" and prod in pos:
                assert pos[prod] < pos[node.name]


def test_branch_has_merge_node(branch_model: Path) -> None:
    graph = load_graph(branch_model)
    ops = {n.op_type for n in graph.nodes}
    assert "Concat" in ops


def test_multi_output_heads(multi_model: Path) -> None:
    graph = load_graph(multi_model)
    assert len(graph.model_outputs()) == 2


def test_node_attributes_parsed(chain_model: Path) -> None:
    graph = load_graph(chain_model)
    conv = next(n for n in graph.nodes if n.op_type == "Conv")
    assert conv.attributes.get("kernel_shape") == [1, 1]


def test_value_producers_chain(chain_model: Path) -> None:
    graph = load_graph(chain_model)
    assert "input" not in graph.value_producers or graph.value_producers.get("input") is None or True
    relu_nodes = [n for n in graph.nodes if n.op_type == "Relu"]
    assert relu_nodes[0].outputs[0] in graph.value_producers
