from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import prod
from typing import Any, Dict, Mapping, Optional, Sequence, Set, Tuple

from .ir import ModelIR, TensorEdge
from .layer_dependencies import LayerDependency, discover_layer_dependencies
from .layerwise import LayerGrouping, detect_layer_groups


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
class LayerBoundaryPayload:
    """Tensor payload crossing one immediate causal layer dependency."""

    source_group: str
    target_group: str
    size_bytes: float
    tensor_ids: Tuple[str, ...]
    crossing_edges: Tuple[Tuple[str, str, str], ...] = ()


@dataclass(frozen=True)
class LayerGraphValidation:
    """Sanity information produced together with LayerGraphIR."""

    is_dag: bool
    missing_payload_dependencies: Tuple[Tuple[str, str], ...]
    ambiguous_owner_nodes: int
    non_adjacent_same_stack_dependencies: int

    @property
    def passed(self) -> bool:
        # Multi-owner glue is informative rather than automatically invalid: a
        # real graph may merge independent branches. Missing communication
        # payloads or a cyclic layer graph are hard failures.
        return bool(self.is_dag and not self.missing_payload_dependencies)


@dataclass(frozen=True)
class LayerGraphIR:
    """Stage-2 IR: detected layers + causal dependencies + communication boundaries."""

    model_ir: ModelIR
    grouping: LayerGrouping
    dependencies: Tuple[LayerDependency, ...]
    payloads: Mapping[Tuple[str, str], LayerBoundaryPayload]
    owners: Mapping[str, Tuple[str, ...]]
    validation: LayerGraphValidation
    precision: str

    @property
    def layer_ids(self) -> Tuple[str, ...]:
        return tuple(
            group_id
            for group_id, group in self.grouping.groups.items()
            if group.kind == "layer"
        )

    @property
    def cross_stack_dependencies(self) -> Tuple[LayerDependency, ...]:
        return tuple(dep for dep in self.dependencies if dep.cross_stack)

    @property
    def same_index_cross_stack_dependencies(self) -> Tuple[LayerDependency, ...]:
        return tuple(
            dep
            for dep in self.cross_stack_dependencies
            if dep.source_layer is not None and dep.source_layer == dep.target_layer
        )


def analyze_layer_graph(
    model_ir: ModelIR,
    *,
    precision: str = "bf16",
    min_repeated_layers: int = 2,
) -> LayerGraphIR:
    """Run the complete generic Layer Graph Analysis stage.

    Stage 2 is deliberately a single public operation:

      Fine ModelIR
        -> automatic repeated-layer detection
        -> causal-frontier dependency recovery
        -> causal ownership / boundary payload recovery
        -> sanity validation
        -> LayerGraphIR

    The implementation contains no model-family, VLM/Expert, KV, layer-count, or
    adjacency rules. Architecture-specific behavior is recovered from the Fine
    DAG and torch.export module metadata.
    """

    model_ir.validate()
    normalized_precision = str(precision).strip().lower()
    if normalized_precision not in _PRECISION_BYTES:
        raise ValueError(f"Unsupported precision {precision!r}")

    grouping = detect_layer_groups(
        model_ir,
        min_repeated_layers=min_repeated_layers,
    )
    dependencies = discover_layer_dependencies(model_ir, grouping)
    payloads, owners = discover_layer_boundary_payloads(
        model_ir,
        grouping,
        dependencies,
        precision=normalized_precision,
    )

    layer_ids = {
        group_id
        for group_id, group in grouping.groups.items()
        if group.kind == "layer"
    }
    dep_pairs = {(dep.source_group, dep.target_group) for dep in dependencies}
    missing = tuple(sorted(dep_pairs - set(payloads)))
    non_adjacent = sum(
        1
        for dep in dependencies
        if not dep.cross_stack
        and dep.source_layer is not None
        and dep.target_layer is not None
        and dep.target_layer - dep.source_layer != 1
    )
    validation = LayerGraphValidation(
        is_dag=_dependency_graph_is_dag(layer_ids, dependencies),
        missing_payload_dependencies=missing,
        ambiguous_owner_nodes=sum(1 for value in owners.values() if len(value) > 1),
        non_adjacent_same_stack_dependencies=non_adjacent,
    )

    return LayerGraphIR(
        model_ir=model_ir,
        grouping=grouping,
        dependencies=tuple(dependencies),
        payloads=payloads,
        owners={node_id: tuple(sorted(value)) for node_id, value in owners.items()},
        validation=validation,
        precision=normalized_precision,
    )


def discover_layer_boundary_payloads(
    model_ir: ModelIR,
    grouping: LayerGrouping,
    dependencies: Sequence[LayerDependency],
    *,
    precision: str = "bf16",
) -> Tuple[
    Dict[Tuple[str, str], LayerBoundaryPayload],
    Dict[str, Set[str]],
]:
    """Recover communication via causal ownership boundaries.

    Every Fine node is owned by the latest causal layer frontier required to
    produce it. Layer nodes own themselves. Glue/residual/cache/reshape/connector
    nodes inherit upstream ownership, while dominated earlier owners are pruned
    using the already recovered layer DAG. Communication is then exactly the
    unique Fine tensors whose producer and consumer ownership cross one immediate
    layer dependency.

    This avoids both failure modes seen with simpler rules:
      * source-origin reachability can count residual intermediates repeatedly;
      * target-ingress/min-cut can count target-internal fan-out or choose a
        semantically invalid alternative input of a multi-input operator.
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
    dep_pairs = {(dep.source_group, dep.target_group) for dep in dependencies}

    layer_adj: Dict[str, Set[str]] = defaultdict(set)
    for dep in dependencies:
        layer_adj[dep.source_group].add(dep.target_group)

    reach_cache: Dict[Tuple[str, str], bool] = {}

    def reaches(source: str, target: str) -> bool:
        key = (source, target)
        cached = reach_cache.get(key)
        if cached is not None:
            return cached
        if source == target:
            reach_cache[key] = True
            return True
        seen = {source}
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for nxt in layer_adj.get(current, ()):
                if nxt == target:
                    reach_cache[key] = True
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        reach_cache[key] = False
        return False

    def prune_dominated(candidates: Set[str]) -> Set[str]:
        # If A -> B and a join requires both A and B, B is the latest causal
        # owner. Keeping A as well would leak residual ancestry into future
        # communication edges.
        result: Set[str] = set()
        for source in candidates:
            if any(source != other and reaches(source, other) for other in candidates):
                continue
            result.add(source)
        return result

    order, predecessors = _topological_order(model_ir)
    owners: Dict[str, Set[str]] = {}
    for node_id in order:
        group_id = grouping.member_to_group[node_id]
        if group_id in layer_ids:
            owners[node_id] = {group_id}
            continue
        candidates: Set[str] = set()
        for pred in predecessors.get(node_id, ()):
            candidates.update(owners.get(pred, set()))
        owners[node_id] = prune_dominated(candidates)

    accumulator: Dict[
        Tuple[str, str],
        Dict[str, Tuple[float, TensorEdge]],
    ] = defaultdict(dict)
    for edge in model_ir.edges:
        source_owners = owners.get(edge.source, set())
        target_owners = owners.get(edge.target, set())
        if not source_owners or not target_owners:
            continue
        for source_owner in source_owners:
            for target_owner in target_owners:
                pair = (source_owner, target_owner)
                if source_owner == target_owner or pair not in dep_pairs:
                    continue
                tensor_id = edge.tensor_id or edge.source
                tensor_key = f"{edge.source}:{tensor_id}"
                size_bytes = edge_bytes_for_precision(
                    edge,
                    source_node=model_ir.nodes.get(edge.source),
                    precision=precision,
                )
                previous = accumulator[pair].get(tensor_key)
                if previous is None or size_bytes > previous[0]:
                    accumulator[pair][tensor_key] = (float(size_bytes), edge)

    payloads: Dict[Tuple[str, str], LayerBoundaryPayload] = {}
    for pair, tensors in accumulator.items():
        payloads[pair] = LayerBoundaryPayload(
            source_group=pair[0],
            target_group=pair[1],
            size_bytes=float(sum(item[0] for item in tensors.values())),
            tensor_ids=tuple(sorted(tensors)),
            crossing_edges=tuple(
                sorted(
                    (edge.source, edge.target, edge.tensor_id or edge.source)
                    for _, edge in tensors.values()
                )
            ),
        )
    return payloads, owners


def layer_graph_to_dict(layer_graph: LayerGraphIR) -> Dict[str, object]:
    """Serialize the compact Stage-2 result without duplicating the Fine ModelIR."""

    grouping = layer_graph.grouping
    layers = []
    for group_id in layer_graph.layer_ids:
        group = grouping.groups[group_id]
        layers.append(
            {
                "id": group.id,
                "name": group.display_name,
                "stack_root": group.stack_root,
                "layer_index": group.layer_index,
                "fine_node_count": len(group.members),
            }
        )

    dependency_rows = []
    for dep in layer_graph.dependencies:
        payload = layer_graph.payloads.get((dep.source_group, dep.target_group))
        dependency_rows.append(
            {
                "source_group": dep.source_group,
                "target_group": dep.target_group,
                "source_stack": dep.source_stack,
                "target_stack": dep.target_stack,
                "source_layer": dep.source_layer,
                "target_layer": dep.target_layer,
                "cross_stack": dep.cross_stack,
                "payload_bytes": None if payload is None else payload.size_bytes,
                "tensor_ids": [] if payload is None else list(payload.tensor_ids),
            }
        )

    return {
        "metadata": {
            "abstraction": "layer-graph-v2",
            "precision": layer_graph.precision,
            "fine_nodes": len(layer_graph.model_ir.nodes),
            "fine_edges": len(layer_graph.model_ir.edges),
            "layer_stacks": len(grouping.stack_layers),
            "layer_nodes": len(layer_graph.layer_ids),
            "dependencies": len(layer_graph.dependencies),
            "cross_stack_dependencies": len(layer_graph.cross_stack_dependencies),
            "same_index_cross_stack_dependencies": len(
                layer_graph.same_index_cross_stack_dependencies
            ),
        },
        "stacks": [
            {"root": root, "layers": list(indices), "count": len(indices)}
            for root, indices in grouping.stack_layers.items()
        ],
        "layers": layers,
        "dependencies": dependency_rows,
        "validation": {
            "is_dag": layer_graph.validation.is_dag,
            "missing_payload_dependencies": [
                list(pair)
                for pair in layer_graph.validation.missing_payload_dependencies
            ],
            "ambiguous_owner_nodes": layer_graph.validation.ambiguous_owner_nodes,
            "non_adjacent_same_stack_dependencies": (
                layer_graph.validation.non_adjacent_same_stack_dependencies
            ),
            "passed": layer_graph.validation.passed,
        },
    }


def edge_bytes_for_precision(
    edge: TensorEdge,
    *,
    source_node: Any,
    precision: str,
) -> float:
    """Return communication bytes under the requested floating-point precision."""

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


def _topological_order(
    model_ir: ModelIR,
) -> Tuple[Tuple[str, ...], Dict[str, Tuple[str, ...]]]:
    indegree = {node_id: 0 for node_id in model_ir.nodes}
    adjacency: Dict[str, list[str]] = defaultdict(list)
    predecessors_lists: Dict[str, list[str]] = defaultdict(list)
    for edge in model_ir.edges:
        indegree[edge.target] += 1
        adjacency[edge.source].append(edge.target)
        predecessors_lists[edge.target].append(edge.source)

    queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    order = []
    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for nxt in adjacency.get(node_id, ()):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    if len(order) != len(model_ir.nodes):
        raise RuntimeError("Fine ModelIR is not a DAG")
    return tuple(order), {
        node_id: tuple(values) for node_id, values in predecessors_lists.items()
    }


def _dependency_graph_is_dag(
    layer_ids: Set[str],
    dependencies: Sequence[LayerDependency],
) -> bool:
    indegree = {layer_id: 0 for layer_id in layer_ids}
    adjacency: Dict[str, list[str]] = defaultdict(list)
    for dep in dependencies:
        if dep.source_group not in layer_ids or dep.target_group not in layer_ids:
            continue
        indegree[dep.target_group] += 1
        adjacency[dep.source_group].append(dep.target_group)
    queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for nxt in adjacency.get(current, ()):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)
    return visited == len(indegree)


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
