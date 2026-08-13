from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from .ir import ModelIR, PLACEMENT_FREE, SchedulingEdge, SchedulingIR, SchedulingNode


_LAYER_TOKEN_RE = re.compile(
    r"(?P<prefix>.*?)(?P<collection>layers?|blocks?|block|h)[._](?P<index>\d+)(?:[._]|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class LayerGroup:
    id: str
    kind: str
    members: Tuple[str, ...]
    display_name: str
    stack_root: str = ""
    layer_index: Optional[int] = None


@dataclass(frozen=True)
class LayerGrouping:
    groups: Mapping[str, LayerGroup]
    member_to_group: Mapping[str, str]
    stack_layers: Mapping[str, Tuple[int, ...]]


def detect_layer_groups(
    model_ir: ModelIR,
    *,
    min_repeated_layers: int = 2,
) -> LayerGrouping:
    """Detect repeated layer stacks and build a lossless node partition.

    The detector is intentionally architecture-agnostic. It recognizes common
    repeated module containers (layers/layer/blocks/block/h) from torch.export's
    ``module_path`` metadata. A container is treated as a layer stack only when
    at least ``min_repeated_layers`` distinct indices are visible.

    Every fine node belongs to exactly one group. Nodes not assigned to a
    repeated layer are retained in connected auxiliary regions; input/output and
    source-like nodes stay singleton so current CoopInfer source-task semantics
    remain correct.
    """

    if min_repeated_layers < 1:
        raise ValueError("min_repeated_layers must be >= 1")
    model_ir.validate()

    incoming: Dict[str, int] = {node_id: 0 for node_id in model_ir.nodes}
    outgoing: Dict[str, List[str]] = defaultdict(list)
    undirected: Dict[str, Set[str]] = defaultdict(set)
    for edge in model_ir.edges:
        incoming[edge.target] += 1
        outgoing[edge.source].append(edge.target)
        undirected[edge.source].add(edge.target)
        undirected[edge.target].add(edge.source)

    candidates: Dict[str, Tuple[str, int]] = {}
    indices_by_root: Dict[str, Set[int]] = defaultdict(set)
    for node_id, node in model_ir.nodes.items():
        found = _layer_token(node.module_path)
        if found is None:
            continue
        root, index = found
        candidates[node_id] = (root, index)
        indices_by_root[root].add(index)

    valid_roots = {
        root
        for root, indices in indices_by_root.items()
        if len(indices) >= min_repeated_layers
    }

    members_by_key: Dict[Tuple[str, int], List[str]] = defaultdict(list)
    assigned: Set[str] = set()
    for node_id, (root, index) in candidates.items():
        if root not in valid_roots:
            continue
        node = model_ir.nodes[node_id]
        if node.kind in {"input", "output"}:
            continue
        members_by_key[(root, index)].append(node_id)
        assigned.add(node_id)

    groups: Dict[str, LayerGroup] = {}
    member_to_group: Dict[str, str] = {}

    root_order = {root: i for i, root in enumerate(sorted(valid_roots))}
    for (root, index), members in sorted(
        members_by_key.items(), key=lambda item: (root_order[item[0][0]], item[0][1])
    ):
        group_id = f"layer_s{root_order[root]:02d}_l{index:03d}"
        ordered_members = tuple(members)
        group = LayerGroup(
            id=group_id,
            kind="layer",
            members=ordered_members,
            display_name=f"{_short_stack_name(root)}[{index}]",
            stack_root=root,
            layer_index=index,
        )
        groups[group_id] = group
        for member in ordered_members:
            member_to_group[member] = group_id

    # Inputs/outputs and source-like ops must be singleton groups. CoopInfer's
    # current scheduler interprets indegree-0 nodes as zero-duration source tasks.
    singleton_ids: Set[str] = set()
    for node_id, node in model_ir.nodes.items():
        if node_id in assigned:
            continue
        if node.kind in {"input", "output"} or (node.kind == "op" and incoming[node_id] == 0):
            singleton_ids.add(node_id)

    singleton_counter = 0
    for node_id in sorted(singleton_ids):
        node = model_ir.nodes[node_id]
        group_id = f"singleton_{singleton_counter:03d}_{_safe_id(node_id)}"
        singleton_counter += 1
        group = LayerGroup(
            id=group_id,
            kind=node.kind if node.kind in {"input", "output"} else "source_like",
            members=(node_id,),
            display_name=node_id,
        )
        groups[group_id] = group
        member_to_group[node_id] = group_id
        assigned.add(node_id)

    # Remaining non-layer nodes are retained as connected auxiliary regions.
    # This is deliberately conservative: no fine node disappears just because
    # automatic layer detection could not classify it.
    remaining = set(model_ir.nodes) - assigned
    region_counter = 0
    while remaining:
        start = min(remaining)
        queue = deque([start])
        component: List[str] = []
        remaining.remove(start)
        while queue:
            current = queue.popleft()
            component.append(current)
            for nxt in undirected.get(current, ()):
                if nxt in remaining:
                    remaining.remove(nxt)
                    queue.append(nxt)
        group_id = f"region_{region_counter:03d}"
        region_counter += 1
        ordered_members = tuple(component)
        group = LayerGroup(
            id=group_id,
            kind="aux_region",
            members=ordered_members,
            display_name=f"aux_region_{region_counter - 1}",
        )
        groups[group_id] = group
        for member in ordered_members:
            member_to_group[member] = group_id

    if set(member_to_group) != set(model_ir.nodes):
        missing = sorted(set(model_ir.nodes) - set(member_to_group))
        extra = sorted(set(member_to_group) - set(model_ir.nodes))
        raise RuntimeError(
            f"Layer grouping coverage failure: missing={missing[:8]} extra={extra[:8]}"
        )

    stack_layers = {
        root: tuple(sorted(indices_by_root[root]))
        for root in sorted(valid_roots)
    }
    return LayerGrouping(
        groups=groups,
        member_to_group=member_to_group,
        stack_layers=stack_layers,
    )


def build_layer_scheduling_ir(
    model_ir: ModelIR,
    grouping: LayerGrouping,
    *,
    group_costs: Optional[Mapping[str, Mapping[str, float]]] = None,
    group_metadata: Optional[Mapping[str, Mapping[str, object]]] = None,
) -> SchedulingIR:
    """Project a Fine ModelIR to a quotient SchedulingIR without inventing edges."""

    model_ir.validate()
    group_costs = group_costs or {}
    group_metadata = group_metadata or {}

    internal_edge_count: Dict[str, int] = defaultdict(int)
    edge_acc: Dict[Tuple[str, str], Dict[str, object]] = {}
    cross_fine_edge_count = 0
    for edge in model_ir.edges:
        source_group = grouping.member_to_group[edge.source]
        target_group = grouping.member_to_group[edge.target]
        if source_group == target_group:
            internal_edge_count[source_group] += 1
            continue
        cross_fine_edge_count += 1
        key = (source_group, target_group)
        bucket = edge_acc.setdefault(
            key,
            {"size_bytes": 0.0, "tensor_ids": [], "fine_edges": []},
        )
        bucket["size_bytes"] = float(bucket["size_bytes"]) + float(edge.size_bytes)
        tensor_id = edge.tensor_id or f"{edge.source}->{edge.target}"
        tensor_ids = bucket["tensor_ids"]
        fine_edges = bucket["fine_edges"]
        assert isinstance(tensor_ids, list)
        assert isinstance(fine_edges, list)
        tensor_ids.append(tensor_id)
        fine_edges.append((edge.source, edge.target, tensor_id))

    nodes: Dict[str, SchedulingNode] = {}
    for group_id, group in grouping.groups.items():
        fixed = {
            model_ir.nodes[member].placement
            for member in group.members
            if model_ir.nodes[member].placement != PLACEMENT_FREE
        }
        if len(fixed) > 1:
            raise ValueError(f"Layer group {group_id} contains conflicting placements: {fixed}")
        placement = next(iter(fixed)) if fixed else PLACEMENT_FREE
        metadata = {
            "abstraction": "layer-quotient-v1",
            "group_kind": group.kind,
            "member_count": len(group.members),
            "internal_fine_edges": internal_edge_count.get(group_id, 0),
            "stack_root": group.stack_root,
            "layer_index": group.layer_index,
            "members": list(group.members),
        }
        metadata.update(dict(group_metadata.get(group_id, {})))
        nodes[group_id] = SchedulingNode(
            id=group_id,
            members=group.members,
            name=group.display_name,
            costs_ms={
                str(resource): float(value)
                for resource, value in group_costs.get(group_id, {}).items()
            },
            placement=placement,
            metadata=metadata,
        )

    edges = tuple(
        SchedulingEdge(
            source=source,
            target=target,
            size_bytes=float(values["size_bytes"]),
            tensor_ids=tuple(str(item) for item in values["tensor_ids"]),
        )
        for (source, target), values in edge_acc.items()
    )

    result = SchedulingIR(
        nodes=nodes,
        edges=edges,
        metadata={
            "abstraction": "layer-quotient-v1",
            "source_fine_nodes": len(model_ir.nodes),
            "source_fine_edges": len(model_ir.edges),
            "member_coverage": len(grouping.member_to_group),
            "cross_fine_edges": cross_fine_edge_count,
            "quotient_edges": len(edges),
            "detected_layer_stacks": len(grouping.stack_layers),
            "detected_layer_groups": sum(
                1 for group in grouping.groups.values() if group.kind == "layer"
            ),
            "aux_region_groups": sum(
                1 for group in grouping.groups.values() if group.kind == "aux_region"
            ),
        },
    )
    result.validate()
    return result


def layer_mapping_dict(grouping: LayerGrouping) -> Dict[str, object]:
    return {
        "stack_layers": {
            root: list(indices) for root, indices in grouping.stack_layers.items()
        },
        "groups": [
            {
                "id": group.id,
                "kind": group.kind,
                "display_name": group.display_name,
                "stack_root": group.stack_root,
                "layer_index": group.layer_index,
                "members": list(group.members),
            }
            for group in grouping.groups.values()
        ],
        "member_to_group": dict(grouping.member_to_group),
    }


def _layer_token(module_path: str) -> Optional[Tuple[str, int]]:
    text = _normalize_module_path(module_path)
    matches = list(_LAYER_TOKEN_RE.finditer(text))
    if not matches:
        return None
    # Use the deepest repeated container when nested stacks are visible.
    match = matches[-1]
    prefix = match.group("prefix").rstrip("._")
    collection = match.group("collection").lower()
    root = f"{prefix}.{collection}" if prefix else collection
    return root, int(match.group("index"))


def _normalize_module_path(path: str) -> str:
    text = str(path)
    for token in ("[", "]", "'", '"'):
        text = text.replace(token, ".")
    text = re.sub(r"\.+", ".", text)
    return text.strip(".")


def _short_stack_name(root: str) -> str:
    parts = [part for part in root.split(".") if part]
    return ".".join(parts[-4:]) if parts else "layer"


def _safe_id(value: str) -> str:
    text = re.sub(r"[^0-9a-zA-Z_]+", "_", str(value)).strip("_")
    return text[:48] or "node"
