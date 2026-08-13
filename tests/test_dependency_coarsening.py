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
    source = IRNode(
        "input",
        "input",
        kind="input",
        costs_ms={"device": 0.0, "host": 0.0},
    )
    ir = ModelIR.from_parts(
        [
            source,
            _costed_node("u"),
            _costed_node("left"),
            _costed_node("right"),
            _costed_node("left_out"),
            _costed_node("right_out"),
        ],
        [
            TensorEdge("input", "u", 4096),
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
    source = IRNode(
        "input",
        "input",
        kind="input",
        costs_ms={"device": 0.0, "host": 0.0},
    )
    ir = ModelIR.from_parts(
        [source, _costed_node("a"), _costed_node("b"), _costed_node("c")],
        [
            TensorEdge("input", "a", 1024),
            TensorEdge("a", "b", 4096, "ab"),
            TensorEdge("b", "c", 4096, "bc"),
        ],
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


def test_scored_policy_rejects_contraction_that_would_cycle_quotient():
    # If {a,b,c} were contracted, the outside path a->x->c would become
    # group->x->group. The policy must stop before absorbing c.
    source = IRNode(
        "input",
        "input",
        kind="input",
        costs_ms={"device": 0.0, "host": 0.0},
    )
    ir = ModelIR.from_parts(
        [source] + [_costed_node(name) for name in ("a", "b", "x", "c")],
        [
            TensorEdge("input", "a", 1024),
            TensorEdge("a", "b", 1024),
            TensorEdge("a", "x", 1024),
            TensorEdge("b", "c", 1024),
            TensorEdge("x", "c", 1024),
        ],
    )
    analyzed = analyze_dependencies(
        ir,
        config=DependencyAnalysisConfig(boundary_threshold=0.99),
    )
    schedule = DependencyAwarePolicy(
        max_ops_per_group=4,
        min_merge_affinity=0.0,
    ).apply(analyzed)
    member_to_group = {
        member: group_id
        for group_id, group in schedule.nodes.items()
        for member in group.members
    }

    assert member_to_group["a"] != member_to_group["c"]


def test_source_like_op_remains_singleton_group():
    source_like = IRNode(
        "source_like",
        "memory",
        costs_ms={"device": 0.0, "host": 0.0},
    )
    ir = ModelIR.from_parts(
        [source_like, _costed_node("compute")],
        [TensorEdge("source_like", "compute", 4096)],
    )
    analyzed = analyze_dependencies(
        ir,
        config=DependencyAnalysisConfig(boundary_threshold=0.99),
    )
    schedule = DependencyAwarePolicy(
        max_ops_per_group=16,
        min_merge_affinity=0.0,
    ).apply(analyzed)
    member_to_group = {
        member: group_id
        for group_id, group in schedule.nodes.items()
        for member in group.members
    }

    assert member_to_group["source_like"] != member_to_group["compute"]
