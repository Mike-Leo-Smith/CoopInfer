from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

from .ir import ModelIR, TensorEdge


def _percentile(values: Iterable[float], q: float) -> float:
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    q = min(1.0, max(0.0, float(q)))
    pos = q * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _robust_unit(value: float, scale: float) -> float:
    if value <= 0.0 or scale <= 0.0:
        return 0.0
    return min(1.0, float(value) / float(scale))


def _average_cost(costs: Mapping[str, float]) -> float:
    values = [float(v) for v in costs.values() if float(v) >= 0.0]
    return sum(values) / len(values) if values else 0.0


def _cost_signature(costs: Mapping[str, float], resources: Sequence[str]) -> Tuple[float, ...]:
    values = [max(0.0, float(costs.get(resource, 0.0))) for resource in resources]
    total = sum(values)
    if total <= 0.0:
        return tuple(0.0 for _ in resources)
    return tuple(value / total for value in values)


def _signature_distance(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    # Total-variation distance lies in [0, 1].
    return 0.5 * sum(abs(float(a) - float(b)) for a, b in zip(left, right))


@dataclass(frozen=True)
class DependencyAnalysisConfig:
    """Generic scoring knobs for cost-aware DAG coarsening.

    The analysis deliberately uses graph structure, tensor sizes, and already
    attached per-resource costs only. It contains no model-, layer-, KV-, VLM-,
    or Action-Expert-specific rules.
    """

    boundary_threshold: float = 0.55
    long_range_span_threshold: float = 0.10
    communication_exposure_threshold: float = 0.18

    def __post_init__(self) -> None:
        for name, value in (
            ("boundary_threshold", self.boundary_threshold),
            ("long_range_span_threshold", self.long_range_span_threshold),
            ("communication_exposure_threshold", self.communication_exposure_threshold),
        ):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


def analyze_dependencies(
    model_ir: ModelIR,
    *,
    config: Optional[DependencyAnalysisConfig] = None,
) -> ModelIR:
    """Annotate a costed fine ModelIR with generic scheduling-boundary scores.

    The scoring combines five classical scheduling/partitioning signals:

    * topological span: long-range dependencies are harder to hide after a merge;
    * fan-out/join structure: branch and synchronization points carry choices;
    * criticality: HEFT-style longest remaining/preceding compute rank;
    * placement gradient: adjacent tasks whose hardware cost profiles differ;
    * communication exposure: large tensors crossing a long topological span.

    A complementary merge-affinity score favors local, high-volume, placement-
    similar edges, following the intuition of heavy-edge multilevel graph
    coarsening while avoiding contraction of communication-critical escape edges.
    """

    config = config or DependencyAnalysisConfig()
    model_ir.validate()

    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError(
            "Dependency analysis requires networkx. Install CoopInfer runtime "
            "dependencies before running this stage."
        ) from exc

    graph = nx.DiGraph()
    graph.add_nodes_from(model_ir.nodes)
    graph.add_edges_from((edge.source, edge.target) for edge in model_ir.edges)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Dependency analysis requires a DAG")

    # Work on clones so dependency analysis is a pure ModelIR -> ModelIR stage.
    nodes = {node_id: node.clone() for node_id, node in model_ir.nodes.items()}

    topo = list(nx.topological_sort(graph))
    topo_index = {node_id: index for index, node_id in enumerate(topo)}
    topo_denominator = max(1, len(topo) - 1)

    resources = sorted(
        {
            resource
            for node in nodes.values()
            for resource in node.costs_ms
        }
    )
    avg_cost = {
        node_id: _average_cost(nodes[node_id].costs_ms)
        for node_id in topo
    }
    cost_scale = _percentile(avg_cost.values(), 0.95)
    edge_size_scale = _percentile((edge.size_bytes for edge in model_ir.edges), 0.95)

    # HEFT-style upward/downward ranks, using mean heterogeneous compute cost.
    upward: Dict[str, float] = {}
    for node_id in reversed(topo):
        successors = list(graph.successors(node_id))
        tail = max((upward[succ] for succ in successors), default=0.0)
        upward[node_id] = avg_cost[node_id] + tail

    downward: Dict[str, float] = {}
    for node_id in topo:
        predecessors = list(graph.predecessors(node_id))
        head = max((downward[pred] for pred in predecessors), default=0.0)
        downward[node_id] = avg_cost[node_id] + head

    critical_path = max(upward.values(), default=0.0)
    signatures = {
        node_id: _cost_signature(nodes[node_id].costs_ms, resources)
        for node_id in topo
    }

    node_criticality: Dict[str, float] = {}
    for node_id in topo:
        through = downward[node_id] + upward[node_id] - avg_cost[node_id]
        slack = max(0.0, critical_path - through)
        criticality = (
            1.0 - _robust_unit(slack, critical_path)
            if critical_path > 0.0
            else 0.0
        )
        node_criticality[node_id] = criticality

        node = nodes[node_id]
        dep = dict(node.metadata.get("dependency", {}))
        dep.update(
            {
                "topo_index": topo_index[node_id],
                "in_degree": int(graph.in_degree(node_id)),
                "out_degree": int(graph.out_degree(node_id)),
                "upward_rank_ms": upward[node_id],
                "downward_rank_ms": downward[node_id],
                "criticality": criticality,
                "average_cost_ms": avg_cost[node_id],
                "compute_importance": _robust_unit(avg_cost[node_id], cost_scale),
                "cost_signature": list(signatures[node_id]),
                "cost_signature_resources": list(resources),
            }
        )
        node.metadata["dependency"] = dep

    annotated_edges = []
    preserved = 0
    for edge in model_ir.edges:
        src = edge.source
        dst = edge.target
        span = max(0.0, (topo_index[dst] - topo_index[src]) / topo_denominator)
        size_norm = _robust_unit(
            math.log1p(float(edge.size_bytes)),
            math.log1p(edge_size_scale),
        )
        fanout = min(1.0, max(0, graph.out_degree(src) - 1) / 3.0)
        join = min(1.0, max(0, graph.in_degree(dst) - 1) / 3.0)
        criticality = min(node_criticality[src], node_criticality[dst])
        placement_gradient = _signature_distance(signatures[src], signatures[dst])
        compute_importance = max(
            _robust_unit(avg_cost[src], cost_scale),
            _robust_unit(avg_cost[dst], cost_scale),
        )

        # A large local edge is a good contraction target (heavy-edge intuition),
        # while a large edge that escapes far in topological order is a valuable
        # communication/scheduling boundary.
        communication_exposure = size_norm * span
        local_heavy_affinity = size_norm * (1.0 - span)

        boundary_score = min(
            1.0,
            0.25 * span
            + 0.12 * fanout
            + 0.10 * join
            + 0.18 * criticality
            + 0.15 * placement_gradient
            + 0.10 * compute_importance
            + 0.10 * communication_exposure,
        )
        placement_similarity = 1.0 - placement_gradient
        merge_affinity = min(
            1.0,
            0.45 * (1.0 - boundary_score)
            + 0.35 * local_heavy_affinity
            + 0.20 * placement_similarity,
        )

        long_range_exposed = (
            span >= config.long_range_span_threshold
            and communication_exposure >= config.communication_exposure_threshold
        )
        preserve_boundary = bool(
            boundary_score >= config.boundary_threshold or long_range_exposed
        )
        if preserve_boundary:
            preserved += 1

        metadata = dict(edge.metadata)
        dep = dict(metadata.get("dependency", {}))
        dep.update(
            {
                "topological_span": span,
                "tensor_size_normalized": size_norm,
                "fanout_score": fanout,
                "join_score": join,
                "criticality": criticality,
                "placement_gradient": placement_gradient,
                "compute_importance": compute_importance,
                "communication_exposure": communication_exposure,
                "local_heavy_affinity": local_heavy_affinity,
                "boundary_score": boundary_score,
                "merge_affinity": merge_affinity,
                "preserve_boundary": preserve_boundary,
            }
        )
        metadata["dependency"] = dep
        annotated_edges.append(
            TensorEdge(
                source=edge.source,
                target=edge.target,
                size_bytes=edge.size_bytes,
                tensor_id=edge.tensor_id,
                metadata=metadata,
            )
        )

    result = ModelIR.from_parts(
        nodes.values(),
        annotated_edges,
        metadata=dict(model_ir.metadata),
    )
    result.metadata["dependency_analysis"] = {
        "method": "cost-aware-multilevel-dag",
        "critical_path_mean_cost_ms": critical_path,
        "resources": resources,
        "boundary_threshold": config.boundary_threshold,
        "long_range_span_threshold": config.long_range_span_threshold,
        "communication_exposure_threshold": config.communication_exposure_threshold,
        "preserved_edges": preserved,
        "edge_count": len(model_ir.edges),
    }
    return result
