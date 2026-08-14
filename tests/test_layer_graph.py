from __future__ import annotations

from coopinfer.frontend import IRNode, ModelIR, TensorEdge, analyze_layer_graph


def _row(shape):
    numel = 1
    for dim in shape:
        numel *= dim
    return {
        "shape": list(shape),
        "dtype": "torch.float32",
        "numel": numel,
        "nbytes": numel * 4,
    }


def _op(node_id: str, path: str, shape=(1, 8)) -> IRNode:
    row = _row(shape)
    return IRNode(
        node_id,
        "memory",
        target="aten.clone.default",
        module_path=path,
        metadata={"output_tensors": [dict(row)]},
    )


def test_layer_graph_prunes_residual_ancestry_and_uses_one_hidden_boundary():
    ir = ModelIR.from_parts(
        [
            IRNode("input", "input", kind="input"),
            _op("a0", "model.backbone.layers.0.attn"),
            _op("a1", "model.backbone.layers.1.attn"),
            _op("residual_join", "model.glue"),
            _op("a2", "model.backbone.layers.2.attn"),
            IRNode("output", "output", kind="output"),
        ],
        [
            TensorEdge("input", "a0", 32, "x"),
            TensorEdge("a0", "a1", 32, "hidden0"),
            TensorEdge("a0", "residual_join", 32, "skip"),
            TensorEdge("a1", "residual_join", 32, "hidden1"),
            TensorEdge("residual_join", "a2", 32, "joined"),
            TensorEdge("a2", "output", 32, "y"),
        ],
    )

    graph = analyze_layer_graph(ir, precision="bf16")
    pairs = {
        (dep.source_layer, dep.target_layer)
        for dep in graph.dependencies
        if not dep.cross_stack
    }
    assert pairs == {(0, 1), (1, 2)}
    assert graph.validation.non_adjacent_same_stack_dependencies == 0

    by_index = {
        group.layer_index: group.id
        for group in graph.grouping.groups.values()
        if group.kind == "layer"
    }
    payload = graph.payloads[(by_index[1], by_index[2])]
    # One [1, 8] hidden state at bf16, not both residual ancestors.
    assert payload.size_bytes == 16.0
    assert len(payload.tensor_ids) == 1
    assert graph.validation.passed


def test_layer_graph_preserves_two_distinct_cross_stack_tensors():
    ir = ModelIR.from_parts(
        [
            IRNode("input", "input", kind="input"),
            _op("a0", "model.backbone.layers.0.attn"),
            _op("a1", "model.backbone.layers.1.attn"),
            _op("k_glue", "model.cache"),
            _op("v_glue", "model.cache"),
            _op("b0", "model.expert.layers.0.attn"),
            _op("b1", "model.expert.layers.1.attn"),
            IRNode("output", "output", kind="output"),
        ],
        [
            TensorEdge("input", "a0", 32, "x"),
            TensorEdge("a0", "a1", 32, "hidden"),
            TensorEdge("a0", "k_glue", 32, "k"),
            TensorEdge("a0", "v_glue", 32, "v"),
            TensorEdge("k_glue", "b0", 32, "k_after_glue"),
            TensorEdge("v_glue", "b0", 32, "v_after_glue"),
            TensorEdge("b0", "b1", 32, "expert_hidden"),
            TensorEdge("a1", "b1", 32, "cross1"),
            TensorEdge("b1", "output", 32, "y"),
        ],
    )

    graph = analyze_layer_graph(ir, precision="bf16")
    cross = [dep for dep in graph.dependencies if dep.cross_stack]
    assert len(cross) == 2

    first = next(dep for dep in cross if dep.source_layer == 0 and dep.target_layer == 0)
    payload = graph.payloads[(first.source_group, first.target_group)]
    assert payload.size_bytes == 32.0
    assert len(payload.tensor_ids) == 2
    assert graph.validation.passed
