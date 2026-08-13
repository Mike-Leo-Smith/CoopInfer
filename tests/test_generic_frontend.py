from __future__ import annotations

import pytest

from coopinfer.cost import CostTarget, MappingCostBackend, annotate_costs
from coopinfer.frontend import (
    DependencyAwarePolicy,
    IRNode,
    ModelIR,
    TensorEdge,
    to_coopinfer_payload,
)


def _wave_like_ir() -> ModelIR:
    nodes = [
        IRNode("input", "input", kind="input"),
        IRNode("pre", "gemm"),
        IRNode("kv", "gemm"),
        IRNode("post", "attention"),
        IRNode("ae", "attention"),
        IRNode("output", "output", kind="output"),
    ]
    edges = [
        TensorEdge("input", "pre", 1024),
        TensorEdge("pre", "kv", 2048),
        TensorEdge("kv", "post", 4096, "kv_to_post"),
        TensorEdge("kv", "ae", 4096, "kv_to_ae"),
        TensorEdge("post", "output", 1024),
        TensorEdge("ae", "output", 1024),
    ]
    return ModelIR.from_parts(nodes, edges)


def _annotated_wave_ir() -> ModelIR:
    ir = _wave_like_ir()
    backend = MappingCostBackend(
        {
            ("gemm", "RTX_4090"): 0.10,
            ("attention", "RTX_4090"): 0.20,
            ("gemm", "A100_80GB"): 0.05,
            ("attention", "A100_80GB"): 0.10,
        }
    )
    return annotate_costs(
        ir,
        {
            "device": CostTarget("RTX_4090", backend),
            "host": CostTarget("A100_80GB", backend),
        },
    )


def test_cost_backend_is_decoupled_from_ir():
    ir = _annotated_wave_ir()
    assert ir.nodes["pre"].costs_ms["device"] == pytest.approx(0.10)
    assert ir.nodes["pre"].costs_ms["host"] == pytest.approx(0.05)
    assert ir.nodes["input"].costs_ms == {"device": 0.0, "host": 0.0}


def test_dependency_aware_coarsening_preserves_wave_fanout():
    schedule = DependencyAwarePolicy(max_ops_per_group=16).apply(_annotated_wave_ir())
    groups = {
        member: group_id
        for group_id, group in schedule.nodes.items()
        for member in group.members
    }

    assert groups["kv"] != groups["post"]
    assert groups["kv"] != groups["ae"]
    assert groups["post"] != groups["ae"]

    edge_pairs = {(edge.source, edge.target) for edge in schedule.edges}
    assert (groups["kv"], groups["post"]) in edge_pairs
    assert (groups["kv"], groups["ae"]) in edge_pairs


def test_simple_chain_can_be_coarsened_after_source():
    ir = ModelIR.from_parts(
        [
            IRNode("input", "input", kind="input"),
            IRNode("a", "gemm"),
            IRNode("b", "gemm"),
            IRNode("c", "gemm"),
        ],
        [
            TensorEdge("input", "a", 1),
            TensorEdge("a", "b", 1),
            TensorEdge("b", "c", 1),
        ],
    )
    backend = MappingCostBackend({}, default_ms=0.01)
    annotated = annotate_costs(
        ir,
        {
            "device": CostTarget("d", backend),
            "host": CostTarget("h", backend),
        },
    )
    schedule = DependencyAwarePolicy(max_ops_per_group=16).apply(annotated)
    assert len(schedule.nodes) == 2
    groups = [group.members for group in schedule.nodes.values()]
    assert ("input",) in groups
    assert ("a", "b", "c") in groups


def test_export_matches_existing_coopinfer_schema():
    schedule = DependencyAwarePolicy(max_ops_per_group=16).apply(_annotated_wave_ir())
    payload = to_coopinfer_payload(
        schedule,
        bandwidth_mb_s=1250.0,
        latency_ms=0.2,
    )
    assert payload["version"] == "1.0"
    assert payload["environment"]["bandwidth"] == pytest.approx(1250.0)
    assert payload["environment"]["latency"] == pytest.approx(0.2)
    assert all("c_dev" in node and "c_host" in node for node in payload["nodes"])
    assert all("size" in edge for edge in payload["edges"])


def test_torch_export_capture_smoke():
    torch = pytest.importorskip("torch")
    from coopinfer.frontend.torch_export import capture_model

    class Branch(torch.nn.Module):
        def forward(self, x):
            shared = x * 2
            left = shared + 1
            right = shared - 1
            return left + right

    ir = capture_model(Branch().eval(), args=(torch.ones(2, 4),))
    assert ir.metadata["capture"] == "torch.export"
    assert any(node.kind == "input" for node in ir.nodes.values())
    assert any(node.kind == "op" for node in ir.nodes.values())
    assert any(node.kind == "output" for node in ir.nodes.values())
    assert len(ir.edges) > 0
