from __future__ import annotations

from coopinfer.frontend import (
    IRNode,
    ModelIR,
    TensorEdge,
    detect_layer_groups,
    discover_layer_dependencies,
)


def _op(node_id: str, path: str) -> IRNode:
    return IRNode(node_id, "memory", target="aten.clone.default", module_path=path)


def test_recovers_cross_stack_dependency_through_non_layer_glue():
    ir = ModelIR.from_parts(
        [
            IRNode("input", "input", kind="input"),
            _op("a0", "model.backbone.layers.0.attn"),
            _op("a1", "model.backbone.layers.1.attn"),
            _op("clone", "model.glue"),
            _op("cat", "model.glue"),
            _op("b0", "model.expert.layers.0.attn"),
            _op("b1", "model.expert.layers.1.attn"),
            IRNode("output", "output", kind="output"),
        ],
        [
            TensorEdge("input", "a0", 1),
            TensorEdge("a0", "a1", 1),
            TensorEdge("a0", "clone", 1),
            TensorEdge("clone", "cat", 1),
            TensorEdge("cat", "b0", 1),
            TensorEdge("b0", "b1", 1),
            TensorEdge("b1", "output", 1),
        ],
    )
    grouping = detect_layer_groups(ir)
    dependencies = discover_layer_dependencies(ir, grouping)

    cross = [dependency for dependency in dependencies if dependency.cross_stack]
    assert len(cross) == 1
    assert cross[0].source_layer == 0
    assert cross[0].target_layer == 0
    assert cross[0].glue_hops >= 1


def test_does_not_skip_an_intermediate_layer():
    ir = ModelIR.from_parts(
        [
            IRNode("input", "input", kind="input"),
            _op("a0", "model.backbone.layers.0.attn"),
            _op("a1", "model.backbone.layers.1.attn"),
            _op("glue", "model.glue"),
            _op("b0", "model.expert.layers.0.attn"),
            _op("b1", "model.expert.layers.1.attn"),
            IRNode("output", "output", kind="output"),
        ],
        [
            TensorEdge("input", "a0", 1),
            TensorEdge("a0", "a1", 1),
            TensorEdge("a1", "glue", 1),
            TensorEdge("glue", "b1", 1),
            TensorEdge("b0", "b1", 1),
            TensorEdge("b1", "output", 1),
        ],
    )
    grouping = detect_layer_groups(ir)
    dependencies = discover_layer_dependencies(ir, grouping)

    pairs = {
        (dependency.source_layer, dependency.target_layer, dependency.cross_stack)
        for dependency in dependencies
    }
    assert (0, 1, True) not in pairs
    assert (1, 1, True) in pairs


def test_residual_join_prunes_transitive_same_stack_frontier():
    """Residual ancestry must not turn a sequential stack into an upper triangle.

    The glue join after layer 1 depends on both the older layer-0 residual and the
    newer layer-1 output. Because layer 1 already depends on layer 0, layer 1 is
    the latest causal frontier. Layer 0 must therefore not become a direct
    dependency of layer 2.
    """

    ir = ModelIR.from_parts(
        [
            IRNode("input", "input", kind="input"),
            _op("l0", "model.layers.0.attn"),
            _op("residual0", "model.glue"),
            _op("l1", "model.layers.1.attn"),
            _op("join1", "model.glue"),
            _op("l2", "model.layers.2.attn"),
            IRNode("output", "output", kind="output"),
        ],
        [
            TensorEdge("input", "l0", 1),
            TensorEdge("l0", "residual0", 1),
            TensorEdge("l0", "l1", 1),
            TensorEdge("residual0", "join1", 1),
            TensorEdge("l1", "join1", 1),
            TensorEdge("join1", "l2", 1),
            TensorEdge("l2", "output", 1),
        ],
    )

    grouping = detect_layer_groups(ir)
    dependencies = discover_layer_dependencies(ir, grouping)
    same_stack_pairs = {
        (dependency.source_layer, dependency.target_layer)
        for dependency in dependencies
        if not dependency.cross_stack
    }

    assert same_stack_pairs == {(0, 1), (1, 2)}
    assert (0, 2) not in same_stack_pairs
