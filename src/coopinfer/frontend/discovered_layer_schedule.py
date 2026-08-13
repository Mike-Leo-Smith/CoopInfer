from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import prod
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from .ir import (
    ModelIR,
    PLACEMENT_FREE,
    SchedulingEdge,
    SchedulingIR,
    SchedulingNode,
    TensorEdge,
)
from .layer_dependencies import LayerDependency
from .layerwise import LayerGrouping


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


@dataclass(frozen=True)
class LayerFrontierPayload:
    source_group: str
    target_group: str
    size_bytes: float
    tensor_ids: Tuple[str, ...]


def discover_layer_frontier_payloads(
    model_ir: ModelIR,
    grouping: LayerGrouping,
    *,
    precision: str = "bf16",
) -> Dict[Tuple[str, str], LayerFrontierPayload]:
    """Recover communication payloads for discovered layer-frontier dependencies.

    For every tensor that leaves a detected source layer, trace the real Fine DAG
    through non-layer nodes until the first downstream layer frontier is reached.
    The tensor is charged once per reached target layer. This preserves distinct
    payloads such as K and V while avoiding duplicate fan-out edges for the same
    producing tensor.
    """

    model_ir.validate()
    precision = str(precision).strip().lower()
    if precision not in _PRECISION_BYTES:
        raise ValueError(f"Unsupported precision {precision!r}")

    layer_ids = {
        group_id
        for group_id, group in grouping.groups.items()
        if group.kind == "layer"
    }
    fine_adj: Dict[str, list[str]] = defaultdict(list)
    outgoing_edges: Dict[str, list[TensorEdge]] = defaultdict(list)
    for edge in model_ir.edges:
        fine_adj[edge.source].append(edge.target)
        outgoing_edges[edge.source].append(edge)

    accumulator: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)

    for source_group_id in sorted(layer_ids):
        source_group = grouping.groups[source_group_id]
        for member in source_group.members:
            for edge in outgoing_edges.get(member, ()):
                target_group_id = grouping.member_to_group[edge.target]
                if target_group_id == source_group_id:
                    continue

                frontier_targets = _first_layer_frontiers(
                    edge.target,
                    source_group_id,
                    fine_adj,
                    grouping,
                    layer_ids,
                )
                if not frontier_targets:
                    continue

                tensor_id = edge.tensor_id or edge.source
                tensor_key = f"{edge.source}:{tensor_id}"
                size_bytes = _edge_bytes_for_precision(
                    edge,
                    source_node=model_ir.nodes.get(edge.source),
                    precision=precision,
                )
                for target_layer_id in frontier_targets:
                    bucket = accumulator[(source_group_id, target_layer_id)]
                    # A producer tensor can fan out through several glue paths to
                    # the same frontier layer. Charge it only once there.
                    previous = bucket.get(tensor_key)
                    if previous is None or size_bytes > previous:
                        bucket[tensor_key] = float(size_bytes)

    return {
        key: LayerFrontierPayload(
            source_group=key[0],
            target_group=key[1],
            size_bytes=float(sum(tensors.values())),
            tensor_ids=tuple(sorted(tensors)),
        )
        for key, tensors in accumulator.items()
    }


def build_discovered_layer_scheduling_ir(
    model_ir: ModelIR,
    grouping: LayerGrouping,
    dependencies: Sequence[LayerDependency],
    costed_group_ir: SchedulingIR,
    *,
    precision: str = "bf16",
    synthetic_source_id: str = "__layer_source__",
) -> SchedulingIR:
    """Build the actual solver DAG from automatically discovered layer dependencies.

    Only detected layer groups become placement-bearing compute nodes. The edge
    set is exactly the recovered layer-frontier dependency set, with payload sizes
    traced from the real Fine DAG. A zero-cost synthetic source is added only to
    give true root layers a predecessor because the current CoopInfer native core
    treats indegree-zero nodes as source events.
    """

    model_ir.validate()
    costed_group_ir.validate()

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

    payloads = discover_layer_frontier_payloads(
        model_ir,
        grouping,
        precision=precision,
    )

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
                "abstraction": "discovered-layer-frontier-v1",
                "dependency_source": "real-fine-dag-frontier-reachability",
            },
        )

    dep_pairs: Dict[Tuple[str, str], LayerDependency] = {}
    for dependency in dependencies:
        key = (dependency.source_group, dependency.target_group)
        if key[0] not in layer_ids or key[1] not in layer_ids:
            continue
        dep_pairs[key] = dependency

    edges = []
    for (source, target), dependency in sorted(dep_pairs.items()):
        payload = payloads.get((source, target))
        if payload is None:
            raise RuntimeError(
                "Recovered layer dependency has no Fine-DAG boundary payload: "
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
            "abstraction": "discovered-layer-frontier-v1",
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
            "abstraction": "discovered-layer-frontier-v1",
            "source_fine_nodes": len(model_ir.nodes),
            "source_fine_edges": len(model_ir.edges),
            "detected_layer_nodes": len(layer_ids),
            "layer_frontier_edges": len(dep_pairs),
            "synthetic_source_edges": len(roots),
            "root_layers": list(roots),
            "dependency_rule": (
                "Each layer edge is recovered from a real Fine-DAG path that traverses "
                "only non-layer nodes before reaching the next layer frontier."
            ),
            "payload_rule": (
                "Each layer edge aggregates unique tensors leaving the source layer whose "
                "first downstream layer frontier is the target layer."
            ),
            "precision": str(precision).lower(),
        },
    )
    result.validate()
    _assert_dag(result)
    return result


def _first_layer_frontiers(
    start: str,
    source_layer_id: str,
    adjacency: Mapping[str, Sequence[str]],
    grouping: LayerGrouping,
    layer_ids: set[str],
) -> set[str]:
    queue = deque([start])
    seen = {start}
    targets: set[str] = set()
    while queue:
        current = queue.popleft()
        current_group = grouping.member_to_group[current]
        if current_group in layer_ids:
            if current_group != source_layer_id:
                targets.add(current_group)
            # Any detected layer is a frontier. Never trace through it.
            continue
        for nxt in adjacency.get(current, ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return targets


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
        raise RuntimeError(
            "Discovered layer scheduling graph contains a cycle; this indicates a "
            "frontier-recovery bug because the source Fine ModelIR is a DAG."
        )


def _edge_bytes_for_precision(
    edge: TensorEdge,
    *,
    source_node: Any,
    precision: str,
) -> float:
    rows = edge.metadata.get("tensors", ())
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        tensor_rows = [row for row in rows if isinstance(row, Mapping)]
        if tensor_rows:
            value = sum(_row_bytes(row, precision=precision) for row in tensor_rows)
            if value > 0.0:
                return float(value)

    if source_node is not None:
        rows = source_node.metadata.get("output_tensors", ())
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            tensor_rows = [row for row in rows if isinstance(row, Mapping)]
            if tensor_rows:
                value = sum(_row_bytes(row, precision=precision) for row in tensor_rows)
                if value > 0.0:
                    return float(value)

    return float(edge.size_bytes)


def _row_numel(row: Mapping[str, Any]) -> Optional[int]:
    value = row.get("numel")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    shape = row.get("shape")
    if not isinstance(shape, Sequence) or isinstance(shape, (str, bytes)):
        return None
    dims = []
    for dim in shape:
        try:
            dims.append(int(dim))
        except (TypeError, ValueError):
            return None
    return int(prod(dims)) if dims else 1


def _row_bytes(row: Mapping[str, Any], *, precision: str) -> float:
    numel = _row_numel(row)
    if numel is None:
        value = row.get("nbytes")
        return float(value) if value is not None else 0.0
    dtype = str(row.get("dtype", "")).lower()
    if "int64" in dtype or "long" in dtype:
        width = 8.0
    elif "int32" in dtype:
        width = 4.0
    elif "int16" in dtype:
        width = 2.0
    elif any(token in dtype for token in ("int8", "uint8", "bool")):
        width = 1.0
    else:
        width = _PRECISION_BYTES[precision]
    return float(numel) * width
