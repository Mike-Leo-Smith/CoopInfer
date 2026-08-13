from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .ir import (
    ModelIR,
    PLACEMENT_FREE,
    SchedulingEdge,
    SchedulingIR,
    SchedulingNode,
)


class CoarseningPolicy:
    def apply(self, model_ir: ModelIR) -> SchedulingIR:
        raise NotImplementedError


@dataclass(frozen=True)
class DependencyAwarePolicy(CoarseningPolicy):
    """Greedy coarsening that never hides graph fan-out/join opportunities.

    Capture stays fine-grained. Search granularity is reduced by merging only
    simple one-producer/one-consumer chains. Fan-out, joins, model I/O, explicit
    hard boundaries, placement conflicts, and optional module-scope changes are
    preserved as scheduling boundaries.

    networkx is imported lazily inside ``apply`` so graph capture and cost-only
    workflows do not need scheduler/coarsening dependencies installed.
    """

    max_ops_per_group: int = 16
    max_group_cost_ms: Optional[float] = None
    reference_resource: Optional[str] = None
    preserve_module_boundaries: bool = False

    def __post_init__(self) -> None:
        if self.max_ops_per_group < 1:
            raise ValueError("max_ops_per_group must be >= 1")
        if self.max_group_cost_ms is not None and self.max_group_cost_ms <= 0:
            raise ValueError("max_group_cost_ms must be positive")

    def apply(self, model_ir: ModelIR) -> SchedulingIR:
        try:
            import networkx as nx
        except ImportError as exc:
            raise RuntimeError(
                "Dependency-aware coarsening requires networkx. Install CoopInfer "
                "runtime dependencies before running the coarsening stage."
            ) from exc

        model_ir.validate()
        graph = nx.DiGraph()
        graph.add_nodes_from(model_ir.nodes)
        graph.add_edges_from((edge.source, edge.target) for edge in model_ir.edges)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("DependencyAwarePolicy requires a DAG")

        groups: List[Tuple[str, ...]] = []
        assigned: Set[str] = set()

        for start in nx.topological_sort(graph):
            if start in assigned:
                continue
            members = [start]
            assigned.add(start)
            current = start

            while len(members) < self.max_ops_per_group:
                successors = list(graph.successors(current))
                if len(successors) != 1:
                    break
                nxt = successors[0]
                if nxt in assigned or graph.in_degree(nxt) != 1:
                    break
                if not self._can_merge(model_ir, graph, current, nxt, members):
                    break
                members.append(nxt)
                assigned.add(nxt)
                current = nxt

            groups.append(tuple(members))

        member_to_group: Dict[str, str] = {}
        schedule_nodes: Dict[str, SchedulingNode] = {}
        for index, members in enumerate(groups):
            group_id = f"g{index:04d}_{members[0]}"
            for member in members:
                member_to_group[member] = group_id
            schedule_nodes[group_id] = self._build_group(model_ir, group_id, members)

        edge_accumulator: Dict[Tuple[str, str], Dict[str, object]] = {}
        for edge in model_ir.edges:
            source = member_to_group[edge.source]
            target = member_to_group[edge.target]
            if source == target:
                continue
            key = (source, target)
            bucket = edge_accumulator.setdefault(
                key,
                {"size_bytes": 0.0, "tensor_ids": []},
            )
            bucket["size_bytes"] = float(bucket["size_bytes"]) + float(edge.size_bytes)
            tensor_id = edge.tensor_id or f"{edge.source}->{edge.target}"
            cast_ids = bucket["tensor_ids"]
            assert isinstance(cast_ids, list)
            cast_ids.append(tensor_id)

        schedule_edges = tuple(
            SchedulingEdge(
                source=source,
                target=target,
                size_bytes=float(values["size_bytes"]),
                tensor_ids=tuple(values["tensor_ids"]),
            )
            for (source, target), values in edge_accumulator.items()
        )
        result = SchedulingIR(
            nodes=schedule_nodes,
            edges=schedule_edges,
            metadata={
                "coarsening_policy": "dependency-aware",
                "source_node_count": len(model_ir.nodes),
                "source_edge_count": len(model_ir.edges),
            },
        )
        result.validate()
        return result

    def _can_merge(
        self,
        model_ir: ModelIR,
        graph: Any,
        current: str,
        nxt: str,
        members: Sequence[str],
    ) -> bool:
        current_node = model_ir.nodes[current]
        next_node = model_ir.nodes[nxt]

        if current_node.kind in {"input", "output"}:
            return False
        if next_node.kind in {"input", "output"}:
            return False
        if current_node.metadata.get("hard_boundary_after", False):
            return False
        if next_node.metadata.get("hard_boundary_before", False):
            return False

        if graph.out_degree(current) != 1 or graph.in_degree(nxt) != 1:
            return False

        if self.preserve_module_boundaries:
            if current_node.module_path != next_node.module_path:
                return False

        fixed = {
            model_ir.nodes[node_id].placement
            for node_id in list(members) + [nxt]
            if model_ir.nodes[node_id].placement != PLACEMENT_FREE
        }
        if len(fixed) > 1:
            return False

        if self.max_group_cost_ms is not None:
            total = sum(
                self._reference_cost(model_ir.nodes[node_id])
                for node_id in list(members) + [nxt]
            )
            if total > self.max_group_cost_ms:
                return False

        return True

    def _reference_cost(self, node) -> float:
        if not node.costs_ms:
            return 0.0
        if self.reference_resource is not None:
            if self.reference_resource not in node.costs_ms:
                raise ValueError(
                    f"Node {node.id!r} is missing reference resource "
                    f"{self.reference_resource!r}"
                )
            return float(node.costs_ms[self.reference_resource])
        return max(float(value) for value in node.costs_ms.values())

    @staticmethod
    def _build_group(
        model_ir: ModelIR,
        group_id: str,
        members: Tuple[str, ...],
    ) -> SchedulingNode:
        resources: Set[str] = set()
        for member in members:
            resources.update(model_ir.nodes[member].costs_ms)
        costs = {
            resource: sum(
                float(model_ir.nodes[member].costs_ms.get(resource, 0.0))
                for member in members
            )
            for resource in sorted(resources)
        }
        fixed = {
            model_ir.nodes[member].placement
            for member in members
            if model_ir.nodes[member].placement != PLACEMENT_FREE
        }
        if len(fixed) > 1:
            raise ValueError(f"Group {group_id} contains conflicting placements: {fixed}")
        placement = next(iter(fixed)) if fixed else PLACEMENT_FREE
        name = members[0] if len(members) == 1 else f"{members[0]}..{members[-1]}"
        return SchedulingNode(
            id=group_id,
            members=members,
            name=name,
            costs_ms=costs,
            placement=placement,
            metadata={"member_count": len(members)},
        )
