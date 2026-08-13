from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .ir import ModelIR
from .layerwise import LayerGrouping


@dataclass(frozen=True)
class LayerDependency:
    """A layer-to-layer dependency recovered from real Fine ModelIR reachability.

    The dependency is not a synthetic scheduling edge. It is a discovery overlay:
    starting from one detected layer group, traverse only non-layer groups until
    the next layer frontier is reached. Therefore every reported dependency is
    backed by an actual path in the quotient graph, while no intermediate layer
    is skipped.
    """

    source_group: str
    target_group: str
    path_groups: Tuple[str, ...]
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
            "derivation": (
                "shortest real quotient-graph path from source layer to the next "
                "layer frontier, traversing non-layer groups only"
            ),
        }


def discover_layer_dependencies(
    model_ir: ModelIR,
    grouping: LayerGrouping,
) -> Tuple[LayerDependency, ...]:
    """Recover layer-frontier dependencies without inventing model-specific edges.

    A direct Fine edge between layer groups is only one special case. More often,
    exported graphs contain glue operations (cat/clone/reshape/cast/etc.) between
    repeated transformer layers. This routine removes that representational
    accident for *dependency discovery only* by tracing through non-layer groups.

    Important semantics:
    - Every result corresponds to a real directed path in the Fine/quotient DAG.
    - Traversal stops at the first encountered layer group on each path, so the
      routine never infers a dependency through another layer.
    - The SchedulingIR itself is not modified; auxiliary compute/communication
      remains explicit for the exact CoopInfer evaluator.
    """

    model_ir.validate()

    group_adj: Dict[str, List[str]] = defaultdict(list)
    seen_group_edges = set()
    for edge in model_ir.edges:
        src = grouping.member_to_group[edge.source]
        dst = grouping.member_to_group[edge.target]
        if src == dst or (src, dst) in seen_group_edges:
            continue
        seen_group_edges.add((src, dst))
        group_adj[src].append(dst)

    layer_ids = {
        group_id
        for group_id, group in grouping.groups.items()
        if group.kind == "layer"
    }

    recovered: Dict[Tuple[str, str], LayerDependency] = {}
    for source_id in sorted(layer_ids):
        source = grouping.groups[source_id]
        queue = deque()
        best_depth: Dict[str, int] = {}

        for nxt in group_adj.get(source_id, ()):
            queue.append((nxt, (source_id, nxt)))
            best_depth[nxt] = 1

        while queue:
            current, path = queue.popleft()

            if current in layer_ids:
                if current != source_id:
                    target = grouping.groups[current]
                    key = (source_id, current)
                    candidate = LayerDependency(
                        source_group=source_id,
                        target_group=current,
                        path_groups=path,
                        source_stack=source.stack_root,
                        target_stack=target.stack_root,
                        source_layer=source.layer_index,
                        target_layer=target.layer_index,
                    )
                    previous = recovered.get(key)
                    if previous is None or len(candidate.path_groups) < len(previous.path_groups):
                        recovered[key] = candidate
                # Do not traverse through another detected layer. This makes the
                # result a layer-frontier dependency, not arbitrary transitive closure.
                continue

            for nxt in group_adj.get(current, ()):
                depth = len(path)
                if best_depth.get(nxt, 1 << 30) <= depth:
                    continue
                best_depth[nxt] = depth
                queue.append((nxt, (*path, nxt)))

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
