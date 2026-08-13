from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
                "shortest real Fine ModelIR directed path from source layer to the "
                "next layer frontier, traversing non-layer fine nodes only"
            ),
        }


def discover_layer_dependencies(
    model_ir: ModelIR,
    grouping: LayerGrouping,
) -> Tuple[LayerDependency, ...]:
    """Recover layer-frontier dependencies without inventing model-specific edges.

    Exported graphs often place glue operations (cat/clone/reshape/cast/etc.)
    between repeated transformer layers, so requiring a direct layer->layer Fine
    edge misses real dependencies. This routine searches the original Fine DAG.

    Semantics:
    - Every reported result has an explicit real Fine-node directed path.
    - The search may traverse nodes in the source layer and non-layer nodes.
    - The search stops as soon as it reaches any *other* detected layer group;
      therefore no dependency is inferred through an intermediate layer.
    - The SchedulingIR is not modified. This is a dependency-discovery overlay,
      so auxiliary compute/communication remains explicit for exact evaluation.
    """

    model_ir.validate()

    fine_adj: Dict[str, List[str]] = defaultdict(list)
    for edge in model_ir.edges:
        fine_adj[edge.source].append(edge.target)

    layer_ids = {
        group_id
        for group_id, group in grouping.groups.items()
        if group.kind == "layer"
    }

    recovered: Dict[Tuple[str, str], LayerDependency] = {}

    for source_id in sorted(layer_ids):
        source = grouping.groups[source_id]

        # Multi-source BFS: every fine op inside the source layer is a legitimate
        # producer. Starting from all members avoids imposing an artificial
        # intra-layer ordering on dependency discovery.
        queue = deque((member, (member,)) for member in source.members)
        best_depth: Dict[str, int] = {member: 0 for member in source.members}

        while queue:
            current, fine_path = queue.popleft()

            for nxt in fine_adj.get(current, ()):
                target_group_id = grouping.member_to_group[nxt]

                if target_group_id != source_id and target_group_id in layer_ids:
                    target = grouping.groups[target_group_id]
                    candidate_path = (*fine_path, nxt)
                    group_path = _compress_group_path(candidate_path, grouping)
                    key = (source_id, target_group_id)
                    candidate = LayerDependency(
                        source_group=source_id,
                        target_group=target_group_id,
                        path_groups=group_path,
                        path_fine_nodes=candidate_path,
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
                    # Frontier semantics: do not traverse through another layer.
                    continue

                next_depth = len(fine_path)
                if best_depth.get(nxt, 1 << 30) <= next_depth:
                    continue
                best_depth[nxt] = next_depth
                queue.append((nxt, (*fine_path, nxt)))

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
