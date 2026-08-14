from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .ir import ModelIR
from .layerwise import LayerGrouping


@dataclass(frozen=True)
class LayerDependency:
    """A layer-to-layer dependency recovered from the real Fine ModelIR DAG."""

    source_group: str
    target_group: str
    path_groups: Tuple[str, ...]
    path_fine_nodes: Tuple[str, ...]
    source_stack: str
    target_stack: str
    source_layer: Optional[int]
    target_layer: Optional[int]

    @property
    def cross_stack(self) -> bool:
        return bool(
            self.source_stack
            and self.target_stack
            and self.source_stack != self.target_stack
        )

    @property
    def glue_hops(self) -> int:
        return max(0, len(self.path_groups) - 2)

    def to_dict(self) -> Dict[str, object]:
        return {
            "source_group": self.source_group,
            "target_group": self.target_group,
            "source_stack": self.source_stack,
            "target_stack": self.target_stack,
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "cross_stack": self.cross_stack,
            "glue_hops": self.glue_hops,
            "path_groups": list(self.path_groups),
            "path_fine_nodes": list(self.path_fine_nodes),
            "derivation": (
                "causal-frontier propagation over the real Fine ModelIR DAG; at joins, "
                "earlier layer frontiers dominated by later required frontiers are pruned"
            ),
        }


def discover_layer_dependencies(
    model_ir: ModelIR,
    grouping: LayerGrouping,
) -> Tuple[LayerDependency, ...]:
    """Recover immediate causal layer dependencies without model-specific rules.

    Exported graphs often place residual adds, clones, casts, cache plumbing, and
    other glue operations outside repeated layer module paths. A plain reachability
    search can therefore create transitive false dependencies such as L0->L2 when
    the actual causal chain is L0->L1->L2.

    This routine propagates the *latest causal layer frontiers* through the Fine DAG:

    - Fine nodes are processed in topological order.
    - Values produced inside a detected layer are owned by that layer frontier.
    - Non-layer/glue nodes inherit the frontiers required by their inputs.
    - At joins, a frontier is removed when another candidate frontier is already
      known to depend on it. This keeps the latest required frontier and prevents
      residual ancestry from leaking into fictitious long-range layer edges.
    - Entering a different detected layer records dependencies from the current
      maximal frontiers to that layer, then resets the produced value to the new
      layer frontier.

    No layer count, stack role, adjacency assumption, KV rule, or model family is
    encoded. Every reported dependency still has an explicit witness path in the
    original Fine ModelIR DAG.
    """

    model_ir.validate()

    layer_ids = {
        group_id
        for group_id, group in grouping.groups.items()
        if group.kind == "layer"
    }
    if not layer_ids:
        return ()

    order, predecessors = _topological_order(model_ir)

    # For each fine node, map every latest causal layer frontier to one shortest
    # witness path from a member of that layer to this fine node.
    frontiers: Dict[str, Dict[str, Tuple[str, ...]]] = {}

    # Layer precedence discovered so far. This relation is used only to remove
    # dominated frontiers at glue joins; it is derived from the Fine DAG itself.
    layer_adj: Dict[str, set[str]] = defaultdict(set)
    recovered: Dict[Tuple[str, str], LayerDependency] = {}

    def reaches(source: str, target: str) -> bool:
        if source == target:
            return True
        seen = {source}
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for nxt in layer_adj.get(current, ()):
                if nxt == target:
                    return True
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        return False

    def merge_candidate(
        bucket: Dict[str, Tuple[str, ...]],
        layer_id: str,
        path: Tuple[str, ...],
    ) -> None:
        previous = bucket.get(layer_id)
        if previous is None or len(path) < len(previous):
            bucket[layer_id] = path

    def prune_dominated(
        candidates: Mapping[str, Tuple[str, ...]],
    ) -> Dict[str, Tuple[str, ...]]:
        keys = tuple(candidates)
        result: Dict[str, Tuple[str, ...]] = {}
        for source in keys:
            dominated = any(
                source != other and reaches(source, other)
                for other in keys
            )
            if not dominated:
                result[source] = candidates[source]
        return result

    for node_id in order:
        candidates: Dict[str, Tuple[str, ...]] = {}
        for pred in predecessors.get(node_id, ()):
            for layer_id, path in frontiers.get(pred, {}).items():
                merge_candidate(candidates, layer_id, (*path, node_id))

        candidates = prune_dominated(candidates)
        group_id = grouping.member_to_group[node_id]

        if group_id in layer_ids:
            target = grouping.groups[group_id]
            for source_group_id, path in candidates.items():
                if source_group_id == group_id:
                    continue
                source = grouping.groups[source_group_id]
                key = (source_group_id, group_id)
                group_path = _compress_group_path(path, grouping)
                candidate = LayerDependency(
                    source_group=source_group_id,
                    target_group=group_id,
                    path_groups=group_path,
                    path_fine_nodes=path,
                    source_stack=source.stack_root,
                    target_stack=target.stack_root,
                    source_layer=source.layer_index,
                    target_layer=target.layer_index,
                )
                previous = recovered.get(key)
                if previous is None or len(candidate.path_fine_nodes) < len(
                    previous.path_fine_nodes
                ):
                    recovered[key] = candidate

                if reaches(group_id, source_group_id):
                    raise RuntimeError(
                        "Layer causal-frontier propagation would create a cycle: "
                        f"{source_group_id}->{group_id}"
                    )
                layer_adj[source_group_id].add(group_id)

            # Any fine value produced inside this layer is now owned by this layer
            # frontier. Resetting here is what prevents old residual ancestry from
            # producing transitive false layer dependencies.
            frontiers[node_id] = {group_id: (node_id,)}
        else:
            frontiers[node_id] = candidates

    return tuple(
        recovered[key]
        for key in sorted(
            recovered,
            key=lambda item: (
                grouping.groups[item[0]].stack_root,
                grouping.groups[item[0]].layer_index
                if grouping.groups[item[0]].layer_index is not None
                else -1,
                grouping.groups[item[1]].stack_root,
                grouping.groups[item[1]].layer_index
                if grouping.groups[item[1]].layer_index is not None
                else -1,
            ),
        )
    )


def layer_dependency_summary(
    dependencies: Sequence[LayerDependency],
) -> Dict[str, object]:
    cross = [dependency for dependency in dependencies if dependency.cross_stack]
    same_index_cross = [
        dependency
        for dependency in cross
        if dependency.source_layer is not None
        and dependency.source_layer == dependency.target_layer
    ]
    return {
        "layer_frontier_dependencies": len(dependencies),
        "cross_stack_dependencies": len(cross),
        "same_index_cross_stack_dependencies": len(same_index_cross),
        "dependencies": [dependency.to_dict() for dependency in dependencies],
    }


def _topological_order(
    model_ir: ModelIR,
) -> Tuple[Tuple[str, ...], Dict[str, Tuple[str, ...]]]:
    indegree = {node_id: 0 for node_id in model_ir.nodes}
    adjacency: Dict[str, List[str]] = defaultdict(list)
    predecessors_list: Dict[str, List[str]] = defaultdict(list)
    for edge in model_ir.edges:
        indegree[edge.target] += 1
        adjacency[edge.source].append(edge.target)
        predecessors_list[edge.target].append(edge.source)

    queue = deque(sorted(node_id for node_id, degree in indegree.items() if degree == 0))
    order: List[str] = []
    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for nxt in adjacency.get(node_id, ()):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if len(order) != len(model_ir.nodes):
        raise RuntimeError("Fine ModelIR is not a DAG")

    predecessors = {
        node_id: tuple(values)
        for node_id, values in predecessors_list.items()
    }
    return tuple(order), predecessors


def _compress_group_path(
    fine_path: Sequence[str],
    grouping: LayerGrouping,
) -> Tuple[str, ...]:
    groups: List[str] = []
    for node_id in fine_path:
        group_id = grouping.member_to_group[node_id]
        if not groups or groups[-1] != group_id:
            groups.append(group_id)
    return tuple(groups)
