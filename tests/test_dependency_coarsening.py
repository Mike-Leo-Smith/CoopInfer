from __future__ import annotations

from coopinfer.frontend import (
    DependencyAnalysisConfig,
    DependencyAwarePolicy,
    IRNode,
    ModelIR,
    TensorEdge,
    analyze_dependencies,
)


def _costed_node(node_id: str, *, device: float = 2.0, host: float = 1.0) -> IRNode:
    return IRNode(
        node_id,
        "gemm",
        costs_ms={"device": device, "host": host},
    )


def test_long_range_large_edge_scores_above_local_large_edge():
    ir = ModelIR.from_parts(
        [
            _costed_node("a"),
            _costed_node("b"),
            _costed_node("c"),
            _costed_node("d"),
            _costed_node("e"),
            _costed_node("f"),
        ],
        [
            TensorEdge("a", "b", 8_000_000, "local"),
            TensorEdge("b", "c", 1024),
            TensorEdge("c", "d", 1024),
            TensorEdge("d", "e", 1024),
            TensorEdge("e", "f", 1024),
            TensorEdge("a", "f", 8_000_000, "escape"),
        ],
    )
    analyzed = analyze_dependencies(
        ir,
        config=DependencyAnalysisConfig(
            boundary_threshold=0.45,
            long_range_span_threshold=0.20,
            communication_exposure_threshold=0.10,
        ),
    )
    deps = {
        edge.tensor_id: edge.metadata["dependency"]
        for edge in analyzed.edges
        if edge.tensor_id
    }

    assert deps["escape"]["topological_span"] > deps["local"]["topological_span"]
    assert deps["escape"]["communication_exposure"] > deps["local"]["communication_exposure"]
    assert deps["escape"]["boundary_score"] > deps["local"]["boundary_score"]
    assert deps["escape"]["preserve_boundary"] is True


def test_scored_policy_can_contract_low_value_local_fanout_edge():
    ir = ModelIR.from_parts(
        [
            _costed_node("u"),
            _costed_node("left"),
            _costed_node("right"),
            _costed_node("left_out"),
            _costed_node("right_out"),
        ],
        [
            TensorEdge("u", "left", 4096),
            TensorEdge("u", "right", 4096),
            TensorEdge("left", "left_out", 4096),
            TensorEdge("right", "right_out", 4096),
        ],
    )
    analyzed = analyze_dependencies(
        ir,
        config=DependencyAnalysisConfig(boundary_threshold=0.95),
    )
    schedule = DependencyAwarePolicy(
        max_ops_per_group=4,
        min_merge_affinity=0.0,
    ).apply(analyzed)

    groups = [set(node.members) for node in schedule.nodes.values()]
    assert any("u" in group and len(group) > 1 for group in groups)
    assert len(schedule.nodes) < len(ir.nodes)


def test_scored_policy_keeps_marked_boundary_between_groups():
    ir = ModelIR.from_parts(
        [_costed_node("a"), _costed_node("b"), _costed_node("c")],
        [TensorEdge("a", "b", 4096, "ab"), TensorEdge("b", "c", 4096, "bc")],
    )
    analyzed = analyze_dependencies(
        ir,
        config=DependencyAnalysisConfig(boundary_threshold=0.0),
    )
    schedule = DependencyAwarePolicy(max_ops_per_group=16).apply(analyzed)
    member_to_group = {
        member: group_id
        for group_id, group in schedule.nodes.items()
        for member in group.members
    }

    assert member_to_group["a"] != member_to_group["b"]
    assert member_to_group["b"] != member_to_group["c"]
