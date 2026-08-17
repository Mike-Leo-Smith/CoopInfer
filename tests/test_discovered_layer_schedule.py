from __future__ import annotations

import networkx as nx

from coopinfer.frontend import (
    IRNode,
    ModelIR,
    SchedulingIR,
    SchedulingNode,
    TensorEdge,
    analyze_layer_graph,
    build_layer_graph_scheduling_ir,
)
from coopinfer.solver import _cache_aware_graph, _persistent_cache_templates


def _row(shape=(1, 8)):
    numel = 1
    for dim in shape:
        numel *= dim
    return {
        "shape": list(shape),
        "dtype": "torch.float32",
        "numel": numel,
        "element_size": 4,
        "nbytes": numel * 4,
    }


def _op(node_id: str, path: str) -> IRNode:
    row = _row()
    return IRNode(
        node_id,
        "memory",
        target="aten.clone.default",
        module_path=path,
        metadata={"input_tensors": [dict(row)], "output_tensors": [dict(row)]},
    )


def _costed(layer_graph):
    return SchedulingIR(
        nodes={
            group_id: SchedulingNode(
                id=group_id,
                members=layer_graph.grouping.groups[group_id].members,
                name=layer_graph.grouping.groups[group_id].display_name,
                costs_ms={"device": 1.0, "host": 0.5},
            )
            for group_id in layer_graph.layer_ids
        },
        edges=(),
    )


def test_layer_graph_builds_costed_scheduling_ir_and_stays_dag():
    nodes = [
        IRNode("input", "input", kind="input"),
        _op("a0", "model.backbone.layers.0.attn"),
        _op("a1", "model.backbone.layers.1.attn"),
        _op("k_glue", "model.k_glue"),
        _op("v_glue", "model.v_glue"),
        _op("b0", "model.expert.layers.0.attn"),
        _op("b1", "model.expert.layers.1.attn"),
        IRNode("output", "output", kind="output"),
    ]
    edges = [
        TensorEdge("input", "a0", 32, "x"),
        TensorEdge("a0", "a1", 32, "hidden"),
        TensorEdge("a0", "k_glue", 32, "k"),
        TensorEdge("a0", "v_glue", 32, "v"),
        TensorEdge("k_glue", "b0", 32, "k_after_glue"),
        TensorEdge("v_glue", "b0", 32, "v_after_glue"),
        TensorEdge("b0", "b1", 32, "expert_hidden"),
        TensorEdge("a1", "b1", 32, "cross1"),
        TensorEdge("b1", "output", 32, "y"),
    ]
    ir = ModelIR.from_parts(nodes, edges)
    layer_graph = analyze_layer_graph(ir, precision="bf16")

    schedule = build_layer_graph_scheduling_ir(layer_graph, _costed(layer_graph))

    assert len(schedule.nodes) == 5
    layer_edges = [edge for edge in schedule.edges if not edge.source.startswith("__")]
    assert len(layer_edges) == 4

    layer_groups = [
        layer_graph.grouping.groups[group_id]
        for group_id in layer_graph.layer_ids
    ]
    by_name = {group.display_name: group.id for group in layer_groups}
    cross0 = next(
        edge
        for edge in layer_edges
        if edge.source == by_name["model.backbone.layers[0]"]
        and edge.target == by_name["model.expert.layers[0]"]
    )
    assert cross0.size_bytes == 32.0
    assert len(cross0.tensor_ids) == 2
    assert schedule.metadata["layer_dependency_edges"] == 4
    assert schedule.metadata["payload_rule"] == (
        "Fine tensor edges crossing causal layer ownership boundaries"
    )


def test_iterative_denoise_repeats_expert_but_reuses_static_kv_once():
    loop_row = _row((1, 4, 2))
    nodes = [
        IRNode(
            "x_t",
            "input",
            kind="input",
            metadata={"output_tensors": [dict(loop_row)]},
        ),
        _op("a0", "model.backbone.layers.0.attn"),
        _op("a1", "model.backbone.layers.1.attn"),
        _op("k_glue", "model.k_glue"),
        _op("v_glue", "model.v_glue"),
        _op("b0", "model.expert.layers.0.attn"),
        _op("b1", "model.expert.layers.1.attn"),
        IRNode("output", "output", kind="output"),
    ]
    edges = [
        TensorEdge("x_t", "a0", 32, "x"),
        TensorEdge("a0", "a1", 32, "hidden"),
        TensorEdge("a0", "k_glue", 32, "k"),
        TensorEdge("a0", "v_glue", 32, "v"),
        TensorEdge("k_glue", "b0", 32, "k_after_glue"),
        TensorEdge("v_glue", "b0", 32, "v_after_glue"),
        TensorEdge("b0", "b1", 32, "expert_hidden"),
        TensorEdge("a1", "b1", 32, "cross1"),
        TensorEdge("b1", "output", 32, "y"),
    ]
    ir = ModelIR.from_parts(
        nodes,
        edges,
        metadata={
            "num_inference_steps": 3,
            "captured_denoise_steps": 1,
            "chunk_size": 4,
            "max_action_dim": 2,
        },
    )
    layer_graph = analyze_layer_graph(ir, precision="bf16")
    schedule = build_layer_graph_scheduling_ir(layer_graph, _costed(layer_graph))

    assert schedule.metadata["execution_semantics"] == "iterative_denoise_v2"
    assert schedule.metadata["num_inference_steps"] == 3
    assert schedule.metadata["iterative_layer_count"] == 2
    assert schedule.metadata["kv_reuse_across_denoise_steps"] is True
    assert schedule.metadata["placement_shared_across_denoise_steps"] is False
    assert schedule.metadata["persistent_cache_policy"] == (
        "copy_once_per_resource_on_first_use"
    )
    assert schedule.metadata["static_to_iterative_edges_once"] == 2
    assert len(schedule.metadata["persistent_cache_edges"]) == 2
    assert schedule.metadata["loop_carried_state_bytes"] == 16.0

    # 2 static backbone layers + 2 expert layers x3 + synthetic source.
    assert len(schedule.nodes) == 9
    assert len(schedule.edges) == 9

    groups = layer_graph.grouping.groups
    by_name = {groups[group_id].display_name: group_id for group_id in layer_graph.layer_ids}
    expert0 = by_name["model.expert.layers[0]"]
    expert1 = by_name["model.expert.layers[1]"]
    backbone0 = by_name["model.backbone.layers[0]"]
    backbone1 = by_name["model.backbone.layers[1]"]

    assert all(f"{expert0}::step{step:03d}" in schedule.nodes for step in range(3))
    assert all(f"{expert1}::step{step:03d}" in schedule.nodes for step in range(3))

    # Static/KV dependencies are structurally emitted only for step 0. The
    # solver may add one prefetch edge if a later step first uses another
    # resource, but never one edge per repeated denoise step.
    cross_edges = [
        edge
        for edge in schedule.edges
        if edge.source in {backbone0, backbone1} and "::step" in edge.target
    ]
    assert len(cross_edges) == 2
    assert all(edge.target.endswith("::step000") for edge in cross_edges)

    internal_edges = [
        edge
        for edge in schedule.edges
        if edge.source.startswith(expert0 + "::step")
        and edge.target.startswith(expert1 + "::step")
    ]
    assert len(internal_edges) == 3

    loop_edges = [
        edge
        for edge in schedule.edges
        if edge.tensor_ids and edge.tensor_ids[0].startswith("__denoise_state_")
    ]
    assert len(loop_edges) == 2
    assert all(edge.size_bytes == 16.0 for edge in loop_edges)


def test_persistent_cache_is_copied_only_on_first_use_of_other_resource():
    graph = nx.DiGraph()
    graph.add_node("vlm0", c_dev=1.0, c_host=1.0, placement="free", x=1)
    for step in range(3):
        graph.add_node(
            f"ae0::step{step:03d}",
            c_dev=1.0,
            c_host=1.0,
            placement="free",
            x=1,
        )
    graph.add_edge("vlm0", "ae0::step000", size=1.0)
    graph.add_edge("ae0::step000", "ae0::step001", size=0.01)
    graph.add_edge("ae0::step001", "ae0::step002", size=0.01)

    templates = _persistent_cache_templates(graph)
    assert templates == [("vlm0", "ae0", "ae0::step000", 1.0)]

    # VLM Host, step0 Device, later Host: the Device copy is already created by
    # the original step0 edge; Host keeps its original copy. No second KV edge.
    warmup_device = {
        "vlm0": 1,
        "ae0::step000": 0,
        "ae0::step001": 1,
        "ae0::step002": 1,
    }
    effective = _cache_aware_graph(graph, warmup_device, templates)
    assert not effective.has_edge("vlm0", "ae0::step001")
    assert not effective.has_edge("vlm0", "ae0::step002")

    # VLM Host and step0 Host, but step1 first moves to Device. One prefetch is
    # added to step1 and is reused if step2 remains/returns Device later.
    later_device = {
        "vlm0": 1,
        "ae0::step000": 1,
        "ae0::step001": 0,
        "ae0::step002": 0,
    }
    effective = _cache_aware_graph(graph, later_device, templates)
    assert effective.has_edge("vlm0", "ae0::step001")
    assert effective.edges["vlm0", "ae0::step001"]["size"] == 1.0
    assert not effective.has_edge("vlm0", "ae0::step002")
