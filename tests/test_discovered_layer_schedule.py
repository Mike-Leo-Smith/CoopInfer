from __future__ import annotations

import networkx as nx
import pytest

from coopinfer.frontend import (
    IRNode,
    ModelIR,
    SchedulingIR,
    SchedulingNode,
    TensorEdge,
    analyze_layer_graph,
    build_layer_graph_scheduling_ir,
)
from coopinfer.solver import (
    _cache_aware_graph,
    _fold_autoregressive_graph,
    _persistent_cache_templates,
    solve,
)


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


@pytest.mark.parametrize(
    ("explicit_kv_reuse", "expected_kv_reuse", "expected_context_kind"),
    [
        (False, False, "static_conditioning"),
        (True, True, "per_layer_prefix_kv"),
    ],
)
def test_iterative_denoise_repeats_expert_but_reuses_static_kv_once(
    explicit_kv_reuse,
    expected_kv_reuse,
    expected_context_kind,
):
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
    metadata = {
        "num_inference_steps": 3,
        "captured_denoise_steps": 1,
        "chunk_size": 4,
        "max_action_dim": 2,
    }
    if explicit_kv_reuse:
        metadata.update(
            {
                "kv_reuse_across_denoise_steps": True,
                "persistent_context_kind": "per_layer_prefix_kv",
            }
        )
    ir = ModelIR.from_parts(
        nodes,
        edges,
        metadata=metadata,
    )
    layer_graph = analyze_layer_graph(ir, precision="bf16")
    schedule = build_layer_graph_scheduling_ir(layer_graph, _costed(layer_graph))

    assert schedule.metadata["execution_semantics"] == "iterative_denoise_v2"
    assert schedule.metadata["num_inference_steps"] == 3
    assert schedule.metadata["iterative_layer_count"] == 2
    assert schedule.metadata["kv_reuse_across_denoise_steps"] is expected_kv_reuse
    assert schedule.metadata["persistent_context_kind"] == expected_context_kind
    assert schedule.metadata["static_conditioning_reuse_across_steps"] is True
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


def test_autoregressive_decode_repeats_decode_stack_and_carries_growing_kv():
    nodes = [
        IRNode("token", "input", kind="input"),
        _op("p0", "prefill.model.layers.0.attn"),
        _op("p1", "prefill.model.layers.1.attn"),
        _op("d0", "decode.model.layers.0.attn"),
        _op("d1", "decode.model.layers.1.attn"),
        IRNode("output", "output", kind="output"),
    ]
    edges = [
        TensorEdge("token", "p0", 8, "input"),
        TensorEdge("p0", "p1", 32, "prefill_hidden"),
        TensorEdge("p0", "d0", 32, "kv0"),
        TensorEdge("p1", "d1", 32, "kv1"),
        TensorEdge("d0", "d1", 8, "decode_hidden"),
        TensorEdge("d1", "output", 8, "logits"),
    ]
    ir = ModelIR.from_parts(
        nodes,
        edges,
        metadata={
            "iterative_execution_kind": "autoregressive_decode",
            "num_decode_steps": 3,
            "max_action_tokens": 4,
            "captured_decode_steps": 1,
            "decode_token_bytes": 8,
            "decode_kv_increment_bytes_per_layer": 4,
        },
    )
    layer_graph = analyze_layer_graph(ir, precision="bf16")
    schedule = build_layer_graph_scheduling_ir(layer_graph, _costed(layer_graph))

    assert schedule.metadata["execution_semantics"] == "autoregressive_decode_v1"
    assert schedule.metadata["num_decode_steps"] == 3
    assert schedule.metadata["max_action_tokens"] == 4
    assert schedule.metadata["iterative_layer_count"] == 2
    assert schedule.metadata["kv_reuse_across_decode_steps"] is True
    assert schedule.metadata["dynamic_decode_cache_edges"] == 4
    assert schedule.metadata["loop_carried_state_bytes"] == 8
    assert len(schedule.nodes) == 9
    assert len(schedule.edges) == 13

    groups = layer_graph.grouping.groups
    by_name = {groups[group_id].display_name: group_id for group_id in layer_graph.layer_ids}
    decode0 = by_name["decode.model.layers[0]"]
    decode1 = by_name["decode.model.layers[1]"]
    assert all(f"{decode0}::step{step:03d}" in schedule.nodes for step in range(3))
    assert all(f"{decode1}::step{step:03d}" in schedule.nodes for step in range(3))

    cache_edges = [
        edge
        for edge in schedule.edges
        if edge.tensor_ids and edge.tensor_ids[0].startswith("__decode_kv_")
    ]
    assert sorted(edge.size_bytes for edge in cache_edges) == [20.0, 20.0, 24.0, 24.0]
    token_edges = [
        edge
        for edge in schedule.edges
        if edge.tensor_ids and edge.tensor_ids[0].startswith("__decode_state_")
    ]
    assert len(token_edges) == 2
    assert all(edge.size_bytes == 8.0 for edge in token_edges)


def test_explicit_iterative_stack_hint_supports_terminal_action_transformer():
    nodes = [
        IRNode("input", "input", kind="input"),
        IRNode(
            "action",
            "input",
            kind="input",
            metadata={"output_tensors": [_row((1, 4, 2))]},
        ),
        _op("v0", "model.vision.blocks.0.attn"),
        _op("v1", "model.vision.blocks.1.attn"),
        _op("a0", "model.action.blocks.0.attn"),
        _op("a1", "model.action.blocks.1.attn"),
        IRNode("output", "output", kind="output"),
    ]
    edges = [
        TensorEdge("input", "v0", 32, "image"),
        TensorEdge("v0", "v1", 32, "vision"),
        TensorEdge("v1", "a0", 32, "context"),
        TensorEdge("action", "a0", 16, "noisy_action"),
        TensorEdge("a0", "a1", 16, "action_hidden"),
        TensorEdge("a1", "output", 16, "prediction"),
    ]
    ir = ModelIR.from_parts(
        nodes,
        edges,
        metadata={
            "num_inference_steps": 3,
            "captured_denoise_steps": 1,
            "chunk_size": 4,
            "max_action_dim": 2,
            "iterative_stack_root_hint": "model.action.blocks",
        },
    )
    layer_graph = analyze_layer_graph(ir, precision="bf16")
    schedule = build_layer_graph_scheduling_ir(layer_graph, _costed(layer_graph))
    assert schedule.metadata["iterative_stack_root"] == "model.action.blocks"
    assert schedule.metadata["iterative_layer_count"] == 2
    assert len([node for node in schedule.nodes if "::step" in node]) == 6


def test_autoregressive_fold_sums_token_work_and_removes_loop_cycle():
    graph = nx.DiGraph()
    for step in range(3):
        graph.add_node(f"a::step{step:03d}", c_dev=1.0, c_host=2.0, placement="free", x=1)
        graph.add_node(f"b::step{step:03d}", c_dev=3.0, c_host=4.0, placement="free", x=1)
        graph.add_edge(f"a::step{step:03d}", f"b::step{step:03d}", size=0.5)
        if step < 2:
            graph.add_edge(f"b::step{step:03d}", f"a::step{step + 1:03d}", size=0.01)
            graph.add_edge(f"a::step{step:03d}", f"a::step{step + 1:03d}", size=1.0)

    folded = _fold_autoregressive_graph(graph)
    assert nx.is_directed_acyclic_graph(folded)
    assert set(folded.nodes) == {"a", "b"}
    assert folded.nodes["a"]["c_dev"] == 3.0
    assert folded.nodes["b"]["c_host"] == 12.0
    assert folded.edges["a", "b"]["size"] == 1.5


def test_autoregressive_grouped_search_is_guarded_by_exact_uniform_baseline():
    graph = nx.DiGraph()
    graph.add_node("prefix", c_dev=1.0, c_host=1.0, placement="free", x=1)
    for step in range(3):
        graph.add_node(
            f"decode::step{step:03d}",
            c_dev=4.0,
            c_host=1.0,
            placement="free",
            x=0,
        )
    graph.add_edge("prefix", "decode::step000", size=1.0, tensor_ids=("kv",))
    graph.add_edge("decode::step000", "decode::step001", size=0.01)
    graph.add_edge("decode::step001", "decode::step002", size=0.01)

    result = solve(
        graph,
        bandwidth=10.0,
        latency=5.0,
        weight_avg_latency=1.0,
        weight_max_latency=0.0,
        weight_device_utilization=0.0,
        algorithm="Random Search",
        heuristic_iterations=1,
        seed=7,
    )

    assert result.metrics.latency == 3.0
    assert set(result.assignment.values()) == {1}
    assert "baseline guard verified" in result.mode


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
    graph.add_edge("vlm0", "ae0::step000", size=1.0, tensor_ids=("context",))
    graph.add_edge("ae0::step000", "ae0::step001", size=0.01)
    graph.add_edge("ae0::step001", "ae0::step002", size=0.01)

    templates = _persistent_cache_templates(graph)
    assert len(templates) == 1
    assert templates[0].source == "vlm0"
    assert templates[0].size_mb == 1.0
    assert templates[0].consumers == (("ae0", "ae0::step000"),)

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


def test_static_conditioning_fanout_is_charged_once_per_resource():
    graph = nx.DiGraph()
    graph.add_node("vlm", c_dev=1.0, c_host=1.0, placement="free", x=1)
    for base in ("a0", "a2"):
        for step in range(2):
            graph.add_node(
                f"{base}::step{step:03d}",
                c_dev=1.0,
                c_host=1.0,
                placement="free",
                x=1,
            )
    graph.add_edge("vlm", "a0::step000", size=0.3125, tensor_ids=("ctx",))
    graph.add_edge("vlm", "a2::step000", size=0.0, tensor_ids=("ctx",))
    graph.add_edge("a0::step000", "a2::step000", size=0.0)
    graph.add_edge("a2::step000", "a0::step001", size=0.01)
    graph.add_edge("a0::step001", "a2::step001", size=0.0)

    templates = _persistent_cache_templates(graph)
    assert len(templates) == 1
    assert len(templates[0].consumers) == 2

    assignment = {node: 1 for node in graph.nodes}
    assignment["a2::step000"] = 0
    assignment["a2::step001"] = 0
    effective = _cache_aware_graph(graph, assignment, templates)
    assert effective.edges["vlm", "a2::step000"]["size"] == 0.3125
    copies = [
        (source, target)
        for source, target, attrs in effective.edges(data=True)
        if attrs.get("persistent_cache_copy")
    ]
    assert copies == [("vlm", "a2::step000")]
