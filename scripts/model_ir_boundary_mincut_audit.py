from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path
import sys
from typing import Dict, Iterable, Mapping, Sequence

import networkx as nx


def _add_repo_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _short_root(root: str) -> str:
    parts = [part for part in root.split(".") if part]
    if len(parts) <= 4:
        return root
    return ".".join(parts[-4:])


def _reachable(starts: Iterable[str], adjacency: Mapping[str, Sequence[str]], allowed: set[str]) -> set[str]:
    queue = deque(node for node in starts if node in allowed)
    seen = set(queue)
    while queue:
        current = queue.popleft()
        for nxt in adjacency.get(current, ()):
            if nxt in allowed and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only weighted minimum-tensor-cut audit for discovered layer dependencies. "
            "For each source->target layer dependency it searches only the real Fine-DAG "
            "corridor after leaving the source layer, through non-layer glue, until entering "
            "the target layer. The minimum weighted vertex cut is the smallest set of live "
            "tensor values whose transfer can bridge that placement boundary."
        )
    )
    parser.add_argument("model_ir", type=Path)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--cross-only", action="store_true")
    parser.add_argument("--details-limit", type=int, default=8)
    args = parser.parse_args()

    _add_repo_src_to_path()
    from coopinfer.frontend import (
        detect_layer_groups,
        discover_layer_dependencies,
        load_model_ir_json,
    )
    from coopinfer.frontend.discovered_layer_schedule import _row_bytes

    model_ir = load_model_ir_json(args.model_ir)
    grouping = detect_layer_groups(model_ir)
    dependencies = discover_layer_dependencies(model_ir, grouping)

    adjacency: Dict[str, list[str]] = defaultdict(list)
    reverse: Dict[str, list[str]] = defaultdict(list)
    edges_by_source: Dict[str, list[object]] = defaultdict(list)
    for edge in model_ir.edges:
        adjacency[edge.source].append(edge.target)
        reverse[edge.target].append(edge.source)
        edges_by_source[edge.source].append(edge)

    layer_ids = {
        group_id for group_id, group in grouping.groups.items() if group.kind == "layer"
    }

    def tensor_bytes(node_id: str) -> float:
        node = model_ir.nodes[node_id]
        rows = node.metadata.get("output_tensors", ())
        if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
            value = sum(
                _row_bytes(row, precision=args.precision)
                for row in rows
                if isinstance(row, Mapping)
            )
            if value > 0.0:
                return float(value)
        return float(node.metadata.get("output_size_bytes", 0.0) or 0.0)

    def cut_for_dependency(dependency):
        source_id = dependency.source_group
        target_id = dependency.target_group
        source_group = grouping.groups[source_id]
        target_group = grouping.groups[target_id]
        source_members = set(source_group.members)
        target_members = set(target_group.members)

        # The corridor starts only after a value leaves the source layer. Other
        # detected layers are hard barriers: a dependency is not allowed to route
        # through an intermediate scheduling layer.
        boundary_starts: set[str] = set()
        boundary_next: set[str] = set()
        for member in source_members:
            for edge in edges_by_source.get(member, ()):
                target_group_id = grouping.member_to_group[edge.target]
                if target_group_id == source_id:
                    continue
                if target_group_id in layer_ids and target_group_id != target_id:
                    continue
                boundary_starts.add(member)
                boundary_next.add(edge.target)

        allowed = set(target_members)
        allowed.update(boundary_starts)
        for node_id in model_ir.nodes:
            group_id = grouping.member_to_group[node_id]
            if group_id not in layer_ids:
                allowed.add(node_id)

        # Keep only nodes lying on a real path from one source boundary value to
        # the target layer. This makes the cut pair-specific and excludes unrelated
        # glue/cache branches.
        forward = _reachable(boundary_next, adjacency, allowed)
        backward = _reachable(target_members, reverse, allowed)
        middle = forward & backward
        corridor = set(middle) | target_members

        # Re-add only source producer values that actually feed this corridor.
        starts: set[str] = set()
        for member in boundary_starts:
            if any(nxt in middle or nxt in target_members for nxt in adjacency.get(member, ())):
                starts.add(member)
                corridor.add(member)

        if not starts or not (corridor & target_members):
            return 0.0, (), 0

        # Weighted vertex cut via node splitting. A compute node itself is not
        # transferred; its produced tensor value is. Target-layer members are sink
        # terminals, so the cut must happen before entering target compute.
        capacities = {node_id: tensor_bytes(node_id) for node_id in corridor}
        finite_total = sum(value for value in capacities.values() if value >= 0.0)
        inf = max(1.0, finite_total + 1.0) * 1000.0

        graph = nx.DiGraph()
        super_source = "__cut_source__"
        super_sink = "__cut_sink__"
        graph.add_node(super_source)
        graph.add_node(super_sink)

        for node_id in corridor:
            node_in = ("in", node_id)
            node_out = ("out", node_id)
            graph.add_node(node_in)
            graph.add_node(node_out)
            if node_id in target_members:
                cap = inf
            else:
                cap = max(0.0, capacities[node_id])
            graph.add_edge(node_in, node_out, capacity=cap)

        for source in corridor:
            for target in adjacency.get(source, ()):
                if target in corridor:
                    graph.add_edge(("out", source), ("in", target), capacity=inf)

        for node_id in starts:
            graph.add_edge(super_source, ("in", node_id), capacity=inf)
        for node_id in target_members & corridor:
            # Reaching target input is enough; target compute itself stays remote.
            graph.add_edge(("in", node_id), super_sink, capacity=inf)

        cut_value, (left, right) = nx.minimum_cut(
            graph, super_source, super_sink, capacity="capacity"
        )

        cut_nodes = []
        for node_id in corridor:
            if node_id in target_members:
                continue
            if ("in", node_id) in left and ("out", node_id) in right:
                cut_nodes.append(node_id)
        cut_nodes.sort()
        return float(cut_value), tuple(cut_nodes), len(corridor)

    print("===== minimum tensor cut boundary audit =====")
    print(f"model_ir={args.model_ir}")
    print(f"precision={args.precision}")
    print(f"dependencies={len(dependencies)}")

    same_values = []
    cross_values = []
    for dependency in dependencies:
        if args.cross_only and not dependency.cross_stack:
            continue
        source = grouping.groups[dependency.source_group]
        target = grouping.groups[dependency.target_group]
        cut_bytes, cut_nodes, corridor_nodes = cut_for_dependency(dependency)
        relation = "cross" if dependency.cross_stack else "same"
        if dependency.cross_stack:
            cross_values.append(cut_bytes)
        else:
            same_values.append(cut_bytes)
        print(
            f"[{relation}] {_short_root(source.stack_root)}[{source.layer_index}] -> "
            f"{_short_root(target.stack_root)}[{target.layer_index}] "
            f"mincut_bytes={cut_bytes:.0f} MiB={cut_bytes / (1024 ** 2):.6f} "
            f"tensors={len(cut_nodes)} corridor_nodes={corridor_nodes}"
        )
        for node_id in cut_nodes[: args.details_limit]:
            node = model_ir.nodes[node_id]
            print(
                f"  tensor={node_id} op={node.op} target={node.target} "
                f"module_path={node.module_path} bytes={tensor_bytes(node_id):.0f}"
            )
            rows = node.metadata.get("output_tensors", ())
            for row in rows:
                if isinstance(row, Mapping):
                    print(
                        f"    shape={row.get('shape')} dtype={row.get('dtype')} "
                        f"numel={row.get('numel')}"
                    )
        remaining = len(cut_nodes) - args.details_limit
        if remaining > 0:
            print(f"  ... {remaining} more tensors")

    if same_values:
        print("\n===== same-stack mincut summary =====")
        print(f"count={len(same_values)} min={min(same_values):.0f} max={max(same_values):.0f}")
    if cross_values:
        print("\n===== cross-stack mincut summary =====")
        print(f"count={len(cross_values)} min={min(cross_values):.0f} max={max(cross_values):.0f}")


if __name__ == "__main__":
    main()
