from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Mapping, Sequence, Tuple

from .ir import (
    ModelIR,
    PLACEMENT_FREE,
    SchedulingEdge,
    SchedulingIR,
    SchedulingNode,
)
from .layer_dependencies import LayerDependency
from .layer_graph import (
    LayerBoundaryPayload,
    LayerGraphIR,
    discover_layer_boundary_payloads,
)
from .layerwise import LayerGrouping


# Backward-compatible type name. The semantics are now causal-ownership
# boundaries rather than source-origin frontier reachability.
LayerFrontierPayload = LayerBoundaryPayload

_PRECISION_BYTES = {
    "fp32": 4.0,
    "f32": 4.0,
    "tf32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "fp6": 0.75,
    "fp4": 0.5,
    "int4": 0.5,
    "int2": 0.25,
}

_STEP_TOKEN = "::step"


def discover_layer_frontier_payloads(
    model_ir: ModelIR,
    grouping: LayerGrouping,
    *,
    precision: str = "bf16",
) -> Dict[Tuple[str, str], LayerBoundaryPayload]:
    """Compatibility wrapper returning ownership-derived layer payloads."""

    from .layer_dependencies import discover_layer_dependencies

    dependencies = discover_layer_dependencies(model_ir, grouping)
    payloads, _ = discover_layer_boundary_payloads(
        model_ir,
        grouping,
        dependencies,
        precision=precision,
    )
    return payloads


def build_layer_graph_scheduling_ir(
    layer_graph: LayerGraphIR,
    costed_group_ir: SchedulingIR,
    *,
    synthetic_source_id: str = "__layer_source__",
) -> SchedulingIR:
    """Stage 3: convert a validated LayerGraphIR + hardware costs to SchedulingIR.

    Stage 1 captures one denoise execution structurally and records the native
    inference-step count in ModelIR metadata. Stage 3 materializes the repeated
    execution here without cloning the Fine ModelIR.

    The iterative stack is inferred from complete same-index cross-stack
    dependencies. No model-family name, Expert string, fixed layer count, or
    fixed denoise-step count is used.

    Static-to-iterative inputs (prefix KV/cache for current VLA models) are
    represented once, feeding denoise step 0. Stage 4 interprets these edges as
    persistent cache inputs: the original producer retains its local copy, and
    the payload is copied to another resource at most once when some denoise
    step first needs that resource. Denoise-step placements therefore remain
    independent search variables; step 0 can exploit VLM overlap without
    forcing later steps onto the same device.
    """

    if not layer_graph.validation.passed:
        raise ValueError(
            "LayerGraphIR validation failed: "
            f"dag={layer_graph.validation.is_dag} "
            f"missing_payloads={len(layer_graph.validation.missing_payload_dependencies)}"
        )
    base = _build_scheduling_ir(
        layer_graph.model_ir,
        layer_graph.grouping,
        layer_graph.dependencies,
        layer_graph.payloads,
        costed_group_ir,
        precision=layer_graph.precision,
        synthetic_source_id=synthetic_source_id,
    )
    return _materialize_iterative_execution(layer_graph, base)


def build_discovered_layer_scheduling_ir(
    model_ir: ModelIR,
    grouping: LayerGrouping,
    dependencies: Sequence[LayerDependency],
    costed_group_ir: SchedulingIR,
    *,
    precision: str = "bf16",
    synthetic_source_id: str = "__layer_source__",
) -> SchedulingIR:
    """Compatibility wrapper for the pre-LayerGraphIR API.

    New code should call ``analyze_layer_graph`` followed by
    ``build_layer_graph_scheduling_ir``. This compatibility path intentionally
    does not infer iterative execution because it lacks the unified LayerGraphIR
    context required for safe structural identification.
    """

    payloads, _ = discover_layer_boundary_payloads(
        model_ir,
        grouping,
        dependencies,
        precision=precision,
    )
    return _build_scheduling_ir(
        model_ir,
        grouping,
        dependencies,
        payloads,
        costed_group_ir,
        precision=precision,
        synthetic_source_id=synthetic_source_id,
    )


def _build_scheduling_ir(
    model_ir: ModelIR,
    grouping: LayerGrouping,
    dependencies: Sequence[LayerDependency],
    payloads: Mapping[Tuple[str, str], LayerBoundaryPayload],
    costed_group_ir: SchedulingIR,
    *,
    precision: str,
    synthetic_source_id: str,
) -> SchedulingIR:
    model_ir.validate()
    costed_group_ir.validate()

    payload_map = dict(payloads)
    layer_ids = {
        group_id
        for group_id, group in grouping.groups.items()
        if group.kind == "layer"
    }
    if not layer_ids:
        raise ValueError("No detected layer groups are available for scheduling")

    missing_costs = sorted(layer_ids - set(costed_group_ir.nodes))
    if missing_costs:
        raise ValueError(f"Layer groups are missing costs: {missing_costs[:8]}")

    nodes: Dict[str, SchedulingNode] = {}
    for layer_id in sorted(layer_ids):
        source_node = costed_group_ir.nodes[layer_id]
        group = grouping.groups[layer_id]
        nodes[layer_id] = SchedulingNode(
            id=layer_id,
            members=tuple(group.members),
            name=group.display_name,
            costs_ms=dict(source_node.costs_ms),
            placement=source_node.placement,
            metadata={
                **dict(source_node.metadata),
                "abstraction": "layer-graph-v2",
                "dependency_source": "causal-frontier-fine-dag",
                "payload_source": "causal-ownership-boundary",
            },
        )

    dep_pairs: Dict[Tuple[str, str], LayerDependency] = {}
    for dependency in dependencies:
        key = (dependency.source_group, dependency.target_group)
        if key[0] in layer_ids and key[1] in layer_ids:
            dep_pairs[key] = dependency

    edges = []
    for source, target in sorted(dep_pairs):
        payload = payload_map.get((source, target))
        if payload is None:
            raise RuntimeError(
                "Recovered layer dependency has no ownership boundary payload: "
                f"{source}->{target}"
            )
        edges.append(
            SchedulingEdge(
                source=source,
                target=target,
                size_bytes=float(payload.size_bytes),
                tensor_ids=payload.tensor_ids,
            )
        )

    indegree = {layer_id: 0 for layer_id in layer_ids}
    for edge in edges:
        indegree[edge.target] += 1
    roots = tuple(sorted(layer_id for layer_id, degree in indegree.items() if degree == 0))

    if synthetic_source_id in nodes:
        raise ValueError(f"Synthetic source id collides with layer id {synthetic_source_id!r}")
    nodes[synthetic_source_id] = SchedulingNode(
        id=synthetic_source_id,
        members=(),
        name="Layer Graph Source",
        costs_ms={"device": 0.0, "host": 0.0},
        placement=PLACEMENT_FREE,
        metadata={
            "abstraction": "layer-graph-v2",
            "synthetic_source": True,
            "reason": (
                "Current CoopInfer native core models indegree-zero nodes as zero-duration "
                "source events; this predecessor preserves compute cost on root layers."
            ),
        },
    )
    for root in roots:
        edges.append(
            SchedulingEdge(
                source=synthetic_source_id,
                target=root,
                size_bytes=0.0,
                tensor_ids=(),
            )
        )

    result = SchedulingIR(
        nodes=nodes,
        edges=tuple(edges),
        metadata={
            "abstraction": "layer-graph-v2",
            "source_fine_nodes": len(model_ir.nodes),
            "source_fine_edges": len(model_ir.edges),
            "detected_layer_nodes": len(layer_ids),
            "layer_dependency_edges": len(dep_pairs),
            "layer_frontier_edges": len(dep_pairs),
            "synthetic_source_edges": len(roots),
            "root_layers": list(roots),
            "dependency_rule": "causal-frontier propagation over the Fine ModelIR DAG",
            "payload_rule": "Fine tensor edges crossing causal layer ownership boundaries",
            "precision": str(precision).lower(),
        },
    )
    result.validate()
    _assert_dag(result)
    return result


def _materialize_iterative_execution(
    layer_graph: LayerGraphIR,
    base: SchedulingIR,
) -> SchedulingIR:
    metadata = layer_graph.model_ir.metadata
    total_steps = int(metadata.get("num_inference_steps", 1) or 1)
    captured_steps = int(metadata.get("captured_denoise_steps", 1) or 1)
    if total_steps <= captured_steps:
        return base
    if captured_steps != 1:
        raise ValueError(
            "Execution materialization currently requires captured_denoise_steps=1; "
            f"got {captured_steps} for num_inference_steps={total_steps}."
        )

    iterative_root = _infer_iterative_stack_root(layer_graph)
    iterative_layers = [
        group_id
        for group_id in layer_graph.layer_ids
        if layer_graph.grouping.groups[group_id].stack_root == iterative_root
    ]
    iterative_layers.sort(
        key=lambda group_id: int(
            layer_graph.grouping.groups[group_id].layer_index
            if layer_graph.grouping.groups[group_id].layer_index is not None
            else -1
        )
    )
    if not iterative_layers:
        raise ValueError(f"Iterative stack {iterative_root!r} has no detected layer nodes")

    iterative_set = set(iterative_layers)
    loop_state_bytes = _infer_loop_state_bytes(layer_graph.model_ir, layer_graph.precision)

    def step_id(base_id: str, step: int) -> str:
        return f"{base_id}{_STEP_TOKEN}{step:03d}"

    nodes: Dict[str, SchedulingNode] = {}
    for node_id, node in base.nodes.items():
        if node_id not in iterative_set:
            nodes[node_id] = SchedulingNode(
                id=node.id,
                members=tuple(node.members),
                name=node.name,
                costs_ms=dict(node.costs_ms),
                placement=node.placement,
                metadata={**dict(node.metadata), "execution_role": "static_once"},
            )
            continue
        for step in range(total_steps):
            clone_id = step_id(node_id, step)
            nodes[clone_id] = SchedulingNode(
                id=clone_id,
                members=tuple(node.members),
                name=f"{node.name} [denoise {step}]",
                costs_ms=dict(node.costs_ms),
                placement=node.placement,
                metadata={
                    **dict(node.metadata),
                    "base_layer_id": node_id,
                    "denoise_step": step,
                    "execution_role": "iterative_denoise_layer",
                    "placement_scope": "per_execution_step",
                },
            )

    edges = []
    static_to_iterative = 0
    repeated_internal = 0
    persistent_cache_edges = []
    for edge in base.edges:
        source_iter = edge.source in iterative_set
        target_iter = edge.target in iterative_set
        if not source_iter and not target_iter:
            edges.append(edge)
        elif not source_iter and target_iter:
            step0_target = step_id(edge.target, 0)
            # This edge expresses both the causal availability of the static
            # context and its original payload. Stage 4 treats it as a
            # persistent cache source. If later denoise steps use another
            # resource, one additional copy can be prefetched to that resource;
            # the cache is never charged once per denoise step.
            edges.append(
                SchedulingEdge(
                    source=edge.source,
                    target=step0_target,
                    size_bytes=edge.size_bytes,
                    tensor_ids=edge.tensor_ids,
                )
            )
            persistent_cache_edges.append(
                {
                    "source": edge.source,
                    "target_base": edge.target,
                    "step0_target": step0_target,
                    "size_bytes": float(edge.size_bytes),
                    "tensor_ids": list(edge.tensor_ids),
                }
            )
            static_to_iterative += 1
        elif source_iter and target_iter:
            for step in range(total_steps):
                edges.append(
                    SchedulingEdge(
                        source=step_id(edge.source, step),
                        target=step_id(edge.target, step),
                        size_bytes=edge.size_bytes,
                        tensor_ids=tuple(
                            f"{tensor}{_STEP_TOKEN}{step:03d}"
                            for tensor in edge.tensor_ids
                        ),
                    )
                )
            repeated_internal += 1
        else:
            edges.append(
                SchedulingEdge(
                    source=step_id(edge.source, total_steps - 1),
                    target=edge.target,
                    size_bytes=edge.size_bytes,
                    tensor_ids=edge.tensor_ids,
                )
            )

    first_layer = iterative_layers[0]
    last_layer = iterative_layers[-1]
    for step in range(total_steps - 1):
        edges.append(
            SchedulingEdge(
                source=step_id(last_layer, step),
                target=step_id(first_layer, step + 1),
                size_bytes=loop_state_bytes,
                tensor_ids=(f"__denoise_state_{step:03d}",),
            )
        )

    result = SchedulingIR(
        nodes=nodes,
        edges=tuple(edges),
        metadata={
            **dict(base.metadata),
            "execution_semantics": "iterative_denoise_v2",
            "num_inference_steps": total_steps,
            "captured_denoise_steps": captured_steps,
            "iterative_stack_root": iterative_root,
            "iterative_layer_count": len(iterative_layers),
            "static_to_iterative_edges_once": static_to_iterative,
            "persistent_cache_edges": persistent_cache_edges,
            "persistent_cache_policy": "copy_once_per_resource_on_first_use",
            "repeated_internal_edge_templates": repeated_internal,
            "kv_reuse_across_denoise_steps": True,
            "placement_shared_across_denoise_steps": False,
            "loop_carried_state_bytes": loop_state_bytes,
            "base_scheduling_nodes": len(base.nodes),
            "base_scheduling_edges": len(base.edges),
            "materialized_scheduling_nodes": len(nodes),
            "materialized_scheduling_edges": len(edges),
        },
    )
    result.validate()
    _assert_dag(result)
    return result


def _infer_iterative_stack_root(layer_graph: LayerGraphIR) -> str:
    counts: Dict[str, int] = defaultdict(int)
    for dependency in layer_graph.same_index_cross_stack_dependencies:
        target_group = layer_graph.grouping.groups.get(dependency.target_group)
        if target_group is None or not target_group.stack_root:
            continue
        counts[target_group.stack_root] += 1

    candidates = []
    for root, count in counts.items():
        layer_count = len(layer_graph.grouping.stack_layers.get(root, ()))
        if layer_count >= 2 and count == layer_count:
            candidates.append(root)
    if len(candidates) != 1:
        raise ValueError(
            "Could not uniquely infer the iterative denoise stack from complete same-index "
            "cross-stack dependencies; candidates=" + repr(sorted(candidates))
        )
    return candidates[0]


def _infer_loop_state_bytes(model_ir: ModelIR, precision: str) -> float:
    normalized = str(precision).strip().lower()
    if normalized not in _PRECISION_BYTES:
        raise ValueError(f"Unsupported communication precision {precision!r}")
    chunk_size = int(model_ir.metadata.get("chunk_size", 0) or 0)
    action_dim = int(model_ir.metadata.get("max_action_dim", 0) or 0)
    candidates = []
    if chunk_size > 0 and action_dim > 0:
        for node in model_ir.nodes.values():
            if node.kind != "input":
                continue
            for tensor in node.metadata.get("output_tensors", ()):
                shape = tensor.get("shape", ())
                if (
                    isinstance(shape, (list, tuple))
                    and len(shape) >= 2
                    and list(shape[-2:]) == [chunk_size, action_dim]
                ):
                    numel = tensor.get("numel")
                    if isinstance(numel, int) and numel > 0:
                        candidates.append(numel)
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise ValueError(
            "Could not uniquely infer loop-carried denoise state size from exported inputs; "
            f"chunk_size={chunk_size} action_dim={action_dim} candidates={unique}"
        )
    return float(unique[0]) * _PRECISION_BYTES[normalized]


def _assert_dag(scheduling_ir: SchedulingIR) -> None:
    indegree = {node_id: 0 for node_id in scheduling_ir.nodes}
    adjacency: Dict[str, list[str]] = defaultdict(list)
    for edge in scheduling_ir.edges:
        indegree[edge.target] += 1
        adjacency[edge.source].append(edge.target)
    queue = deque(sorted(node_id for node_id, value in indegree.items() if value == 0))
    visited = 0
    while queue:
        node_id = queue.popleft()
        visited += 1
        for nxt in adjacency.get(node_id, ()):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if visited != len(scheduling_ir.nodes):
        raise RuntimeError("Layer SchedulingIR contains a cycle")
