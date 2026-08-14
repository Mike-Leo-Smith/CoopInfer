from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from pathlib import Path
import sys
from typing import Dict, Mapping, Sequence, Set, Tuple


def _add_repo_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _topological_order(model_ir):
    indegree = {node_id: 0 for node_id in model_ir.nodes}
    adjacency = defaultdict(list)
    predecessors = defaultdict(list)
    for edge in model_ir.edges:
        indegree[edge.target] += 1
        adjacency[edge.source].append(edge.target)
        predecessors[edge.target].append(edge.source)
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
    return order, predecessors


def _short_root(root: str) -> str:
    parts = [part for part in root.split(".") if part]
    return root if len(parts) <= 4 else ".".join(parts[-4:])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit of communication payloads derived from causal ownership. "
            "Each non-layer Fine node is assigned to the latest causal layer frontier(s); "
            "communication is the set of Fine tensor edges that cross from one layer owner "
            "to another recovered immediate layer dependency."
        )
    )
    parser.add_argument("model_ir", type=Path)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--details-limit", type=int, default=4)
    args = parser.parse_args()

    _add_repo_src_to_path()
    from coopinfer.frontend import (
        detect_layer_groups,
        discover_layer_dependencies,
        load_model_ir_json,
    )
    from coopinfer.frontend.discovered_layer_schedule import _edge_bytes_for_precision

    model_ir = load_model_ir_json(args.model_ir)
    grouping = detect_layer_groups(model_ir)
    dependencies = discover_layer_dependencies(model_ir, grouping)
    layer_ids = {
        group_id for group_id, group in grouping.groups.items() if group.kind == "layer"
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
        # Keep only maximal layer frontiers. If A -> B in the recovered causal
        # layer DAG and a glue node requires both, B is the later owner.
        result = set()
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

    # Aggregate the unique producer tensors that cross from one causal owner to
    # another owner connected by an immediate recovered layer dependency.
    accum: Dict[Tuple[str, str], Dict[str, Tuple[float, object]]] = defaultdict(dict)
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
                size_bytes = _edge_bytes_for_precision(
                    edge,
                    source_node=model_ir.nodes.get(edge.source),
                    precision=str(args.precision).lower(),
                )
                previous = accum[pair].get(tensor_key)
                if previous is None or size_bytes > previous[0]:
                    accum[pair][tensor_key] = (float(size_bytes), edge)

    print("===== causal ownership boundary audit =====")
    print(f"model_ir={args.model_ir}")
    print(f"precision={args.precision}")
    print(f"fine_nodes={len(model_ir.nodes)}")
    print(f"fine_edges={len(model_ir.edges)}")
    print(f"dependencies={len(dependencies)}")

    missing = []
    rows = []
    for dep in dependencies:
        pair = (dep.source_group, dep.target_group)
        tensors = accum.get(pair, {})
        if not tensors:
            missing.append(pair)
            continue
        total = sum(value[0] for value in tensors.values())
        rows.append((dep, total, tensors))

    print(f"missing_payload_dependencies={len(missing)}")
    for source, target in missing[:20]:
        print(f"  MISSING {source} -> {target}")

    same_sizes = []
    cross_sizes = []
    pattern_counter = Counter()
    for dep, total, tensors in rows:
        source = grouping.groups[dep.source_group]
        target = grouping.groups[dep.target_group]
        relation = "cross" if dep.cross_stack else "same"
        print(
            f"[{relation}] {_short_root(source.stack_root)}[{source.layer_index}] -> "
            f"{_short_root(target.stack_root)}[{target.layer_index}] "
            f"ownership_bytes={total:.0f} MiB={total / (1024 ** 2):.6f} "
            f"tensors={len(tensors)}"
        )
        if dep.cross_stack:
            cross_sizes.append(total)
        else:
            same_sizes.append(total)
        pattern_counter[(source.stack_root, target.stack_root, round(total, 6), len(tensors))] += 1

        if args.details_limit > 0:
            for tensor_key, (size_bytes, edge) in list(sorted(tensors.items()))[: args.details_limit]:
                node = model_ir.nodes.get(edge.source)
                print(
                    f"  tensor={tensor_key} producer={edge.source} -> {edge.target} "
                    f"bytes={size_bytes:.0f}"
                )
                if node is not None:
                    print(
                        f"    op={node.op} target={node.target} module_path={node.module_path}"
                    )
                    for row in node.metadata.get("output_tensors", ()):
                        if isinstance(row, Mapping):
                            print(
                                f"    shape={row.get('shape')} dtype={row.get('dtype')} "
                                f"numel={row.get('numel')}"
                            )
            remaining = len(tensors) - args.details_limit
            if remaining > 0:
                print(f"  ... {remaining} more tensors")

    print("\n===== ownership payload patterns =====")
    for (source_root, target_root, size_bytes, tensor_count), count in sorted(
        pattern_counter.items(), key=lambda item: (-item[1], item[0])
    ):
        relation = "same" if source_root == target_root else "cross"
        print(
            f"  count={count:3d} [{relation}] bytes={size_bytes:.0f} "
            f"MiB={size_bytes / (1024 ** 2):.6f} tensors={tensor_count} "
            f"{_short_root(source_root)} -> {_short_root(target_root)}"
        )

    if same_sizes:
        print(
            f"\nsame_stack: count={len(same_sizes)} min={min(same_sizes):.0f} "
            f"max={max(same_sizes):.0f}"
        )
    if cross_sizes:
        print(
            f"cross_stack: count={len(cross_sizes)} min={min(cross_sizes):.0f} "
            f"max={max(cross_sizes):.0f}"
        )

    ambiguous_nodes = sum(1 for value in owners.values() if len(value) > 1)
    print(f"ambiguous_multi_owner_nodes={ambiguous_nodes}")
    if not missing:
        print("OWNERSHIP_AUDIT_STATUS=PASS_ALL_DEPENDENCIES_HAVE_CROSSING_TENSORS")
    else:
        print("OWNERSHIP_AUDIT_STATUS=CHECK_MISSING_DEPENDENCIES")


if __name__ == "__main__":
    main()
