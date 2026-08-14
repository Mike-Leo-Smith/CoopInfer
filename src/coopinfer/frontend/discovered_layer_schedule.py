from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Sequence, Tuple

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
    """Stage 3: convert a validated LayerGraphIR + hardware costs to SchedulingIR."""

    if not layer_graph.validation.passed:
        raise ValueError(
            "LayerGraphIR validation failed: "
            f"dag={layer_graph.validation.is_dag} "
            f"missing_payloads={len(layer_graph.validation.missing_payload_dependencies)}"
        )
    return _build_scheduling_ir(
        layer_graph.model_ir,
        layer_graph.grouping,
        layer_graph.dependencies,
        layer_graph.payloads,
        costed_group_ir,
        precision=layer_graph.precision,
        synthetic_source_id=synthetic_source_id,
    )


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
    ``build_layer_graph_scheduling_ir``.
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
    payloads: Dict[Tuple[str, str], LayerBoundaryPayload]
    | Sequence[Tuple[Tuple[str, str], LayerBoundaryPayload]],
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
            # Preserve the old key for downstream scripts that only read counts.
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
