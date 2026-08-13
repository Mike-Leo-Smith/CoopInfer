from coopinfer.frontend.full_graph import (
    annotate_synthetic_costs,
    identity_scheduling_ir,
    model_ir_from_dict,
)
from coopinfer.frontend.coopinfer_export import to_coopinfer_payload


def _sample_raw_ir():
    return {
        "metadata": {"capture": "torch.export"},
        "nodes": [
            {
                "id": "input",
                "kind": "input",
                "op": "input",
                "target": "input",
                "module_path": "",
                "placement": "free",
                "metadata": {"output_size_bytes": 16.0},
            },
            {
                "id": "a",
                "kind": "op",
                "op": "gemm",
                "target": "aten.linear.default",
                "module_path": "model.layers.0",
                "placement": "free",
                "metadata": {"output_size_bytes": 32.0},
            },
            {
                "id": "b",
                "kind": "op",
                "op": "elementwise",
                "target": "aten.add.Tensor",
                "module_path": "model.layers.0",
                "placement": "free",
                "metadata": {"output_size_bytes": 64.0},
            },
        ],
        "edges": [
            {
                "source": "input",
                "target": "a",
                "size_bytes": 16.0,
                "tensor_id": "x",
            },
            {
                "source": "a",
                "target": "b",
                "size_bytes": 32.0,
                "tensor_id": "k",
            },
            {
                "source": "a",
                "target": "b",
                "size_bytes": 64.0,
                "tensor_id": "v",
            },
        ],
    }


def test_model_ir_from_probe_json_preserves_fine_graph():
    model_ir = model_ir_from_dict(_sample_raw_ir())

    assert list(model_ir.nodes) == ["input", "a", "b"]
    assert len(model_ir.edges) == 3
    assert model_ir.nodes["a"].module_path == "model.layers.0"
    assert model_ir.edges[1].tensor_id == "k"


def test_synthetic_costs_are_separate_from_identity_conversion():
    model_ir = model_ir_from_dict(_sample_raw_ir())
    costed = annotate_synthetic_costs(
        model_ir,
        device_cost_ms=0.01,
        host_cost_ms=0.02,
    )

    assert model_ir.nodes["a"].costs_ms == {}
    assert costed.nodes["input"].costs_ms == {"device": 0.0, "host": 0.0}
    assert costed.nodes["a"].costs_ms == {"device": 0.01, "host": 0.02}
    assert costed.nodes["a"].metadata["cost_source"] == "synthetic"


def test_identity_adapter_keeps_nodes_and_aggregates_parallel_tensor_edges():
    model_ir = annotate_synthetic_costs(model_ir_from_dict(_sample_raw_ir()))
    scheduling_ir = identity_scheduling_ir(model_ir)

    assert set(scheduling_ir.nodes) == {"input", "a", "b"}
    assert all(node.members == (node.id,) for node in scheduling_ir.nodes.values())
    assert len(scheduling_ir.edges) == 2

    edge = next(
        edge
        for edge in scheduling_ir.edges
        if edge.source == "a" and edge.target == "b"
    )
    assert edge.size_bytes == 96.0
    assert edge.tensor_ids == ("k", "v")
    assert scheduling_ir.metadata["source_tensor_edge_count"] == 3
    assert scheduling_ir.metadata["backend_edge_count"] == 2


def test_identity_result_exports_to_coopinfer_schema():
    model_ir = annotate_synthetic_costs(model_ir_from_dict(_sample_raw_ir()))
    scheduling_ir = identity_scheduling_ir(model_ir)
    payload = to_coopinfer_payload(
        scheduling_ir,
        bandwidth_mb_s=1000.0,
        latency_ms=0.2,
    )

    assert payload["version"] == "1.0"
    assert len(payload["nodes"]) == 3
    assert len(payload["edges"]) == 2
    node_a = next(node for node in payload["nodes"] if node["id"] == "a")
    assert node_a["c_dev"] == 0.001
    assert node_a["c_host"] == 0.002
    edge_ab = next(
        edge for edge in payload["edges"] if edge["source"] == "a" and edge["target"] == "b"
    )
    assert edge_ab["size"] == 96.0 / (1024.0 * 1024.0)
