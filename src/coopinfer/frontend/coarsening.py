from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

from .ir import (
    ModelIR,
    PLACEMENT_FREE,
    SchedulingEdge,
    SchedulingIR,
    SchedulingNode,
)


EdgeScore = Dict[str, Union[float, bool]]


class CoarseningPolicy:
    def apply(self, model_ir: ModelIR) -> SchedulingIR:
        raise NotImplementedError


@dataclass(frozen=True)
class DependencyAwarePolicy(CoarseningPolicy):
    """Greedy path contraction guided by generic dependency scores.

    With an unannotated ModelIR this policy keeps the earlier conservative
    behavior and merges only strict 1->1 chains. After ``analyze_dependencies``
    it may also contract low-value local fan-out/join edges, but it preserves
    edges marked as scheduling boundaries and rejects contractions that would
    introduce a cycle in the quotient DAG.

    The policy is model agnostic: it consumes graph/cost annotations only.
    """

    max_ops_per_group: int = 16
    max_group_cost_ms: Optional[float] = None
    reference_resource: Optional[str] = None
    preserve_module_boundaries: bool = False
    min_merge_affinity: float = 0.45

    def __post_init__(self) -> None:
        if self.max_ops_per_group < 1:
            raise ValueError("max_ops_per_group must be >= 1")
        if self.max_group_cost_ms is not None and self.max_group_cost_ms <= 0:
            raise ValueError("max_group_cost_ms must be positive")
        if not 0.0 <= self.min_merge_affinity <= 1.0:
            raise ValueError("min_merge_affinity must be in [0, 1]")

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

        dependency_scored = any(
            isinstance(edge.metadata.get("dependency"), dict)
            for edge in model_ir.edges
        )
        edge_scores = self._build_edge_scores(model_ir)

        groups: List[Tuple[str, ...]] = []
        assigned: Set[str] = set()

        for start in nx.topological_sort(graph):
            if start in assigned:
                continue
            members = [start]
            assigned.add(start)
            current = start

            while len(members) < self.max_ops_per_group:
                candidates = []
                for nxt in graph.successors(current):
                    if nxt in assigned:
                        continue
                    if not self._can_merge(
                        model_ir,
                        graph,
                        current,
                        nxt,
                        members,
                        edge_scores,
                        dependency_scored,
                    ):
                        continue
                    score = edge_scores.get((current, nxt), {}).get(
                        "merge_affinity", 1.0
                    )
                    candidates.append((float(score), str(nxt)))

                if not candidates:
                    break

                # Prefer the strongest local affinity. Stable node-id tie-break
                # keeps coarsening deterministic.
                _, nxt = max(candidates, key=lambda item: (item[0], item[1]))
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

        quotient = nx.DiGraph()
        quotient.add_nodes_from(schedule_nodes)
        quotient.add_edges_from((edge.source, edge.target) for edge in schedule_edges)
        if not nx.is_directed_acyclic_graph(quotient):
            raise RuntimeError("Coarsening produced a cyclic quotient graph")

        group_sizes = [len(members) for members in groups]
        result = SchedulingIR(
            nodes=schedule_nodes,
            edges=schedule_edges,
            metadata={
                "coarsening_policy": "dependency-aware-scored"
                if dependency_scored
                else "dependency-aware-conservative",
                "source_node_count": len(model_ir.nodes),
                "source_edge_count": len(model_ir.edges),
                "coarse_node_count": len(schedule_nodes),
                "coarse_edge_count": len(schedule_edges),
                "dependency_scored": dependency_scored,
                "max_ops_per_group": self.max_ops_per_group,
                "min_merge_affinity": self.min_merge_affinity,
                "largest_group": max(group_sizes, default=0),
                "average_group_size": (
                    sum(group_sizes) / len(group_sizes) if group_sizes else 0.0
                ),
            },
        )
        result.validate()
        return result

    @staticmethod
    def _build_edge_scores(model_ir: ModelIR) -> Dict[Tuple[str, str], EdgeScore]:
        scores: Dict[Tuple[str, str], EdgeScore] = {}
        for edge in model_ir.edges:
            dep = edge.metadata.get("dependency", {})
            if not isinstance(dep, dict):
                continue
            key = (edge.source, edge.target)
            bucket = scores.setdefault(
                key,
                {
                    "preserve_boundary": False,
                    "boundary_score": 0.0,
                    "merge_affinity": 1.0,
                },
            )
            bucket["preserve_boundary"] = bool(bucket["preserve_boundary"]) or bool(
                dep.get("preserve_boundary", False)
            )
            bucket["boundary_score"] = max(
                float(bucket["boundary_score"]),
                float(dep.get("boundary_score", 0.0)),
            )
            bucket["merge_affinity"] = min(
                float(bucket["merge_affinity"]),
                float(dep.get("merge_affinity", 1.0)),
            )
        return scores

    def _can_merge(
        self,
        model_ir: ModelIR,
        graph: Any,
        current: str,
        nxt: str,
        members: Sequence[str],
        edge_scores: Dict[Tuple[str, str], EdgeScore],
        dependency_scored: bool,
    ) -> bool:
        current_node = model_ir.nodes[current]
        next_node = model_ir.nodes[nxt]

        if current_node.kind in {"input", "output"}:
            return False
        if next_node.kind in {"input", "output"}:
            return False

        # CoopInfer's current evaluator treats every indegree-0 graph node as a
        # zero-duration source event. Never absorb real downstream compute into
        # such a source-like node or that compute cost would disappear.
        if any(graph.in_degree(node_id) == 0 for node_id in members):
            return False

        if current_node.metadata.get("hard_boundary_after", False):
            return False
        if next_node.metadata.get("hard_boundary_before", False):
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

        if not dependency_scored:
            # Backward-compatible conservative mode.
            return graph.out_degree(current) == 1 and graph.in_degree(nxt) == 1

        dep = edge_scores.get((current, nxt), {})
        if bool(dep.get("preserve_boundary", False)):
            return False
        if float(dep.get("merge_affinity", 0.0)) < self.min_merge_affinity:
            return False

        if self._would_create_quotient_cycle(graph, members, nxt):
            return False

        return True

    @staticmethod
    def _would_create_quotient_cycle(graph: Any, members: Sequence[str], nxt: str) -> bool:
        """Return True if contracting ``members + [nxt]`` would create a DAG cycle.

        In a DAG this happens when an earlier group member also reaches ``nxt``
        through nodes outside the proposed group. After contraction that outside
        path would become group -> ... -> group.
        """

        import networkx as nx

        group = set(members)
        candidate = group | {nxt}
        external_predecessors = [
            pred for pred in graph.predecessors(nxt) if pred not in candidate
        ]
        if not external_predecessors:
            return False

        for member in group:
            for pred in external_predecessors:
                if nx.has_path(graph, member, pred):
                    return True
        return False

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
        criticality = max(
            (
                float(
                    model_ir.nodes[member]
                    .metadata.get("dependency", {})
                    .get("criticality", 0.0)
                )
                for member in members
            ),
            default=0.0,
        )
        return SchedulingNode(
            id=group_id,
            members=members,
            name=name,
            costs_ms=costs,
            placement=placement,
            metadata={
                "member_count": len(members),
                "max_member_criticality": criticality,
            },
        )
