from __future__ import annotations

import argparse
from collections import defaultdict, deque
from pathlib import Path
import sys
from typing import Dict, Mapping, Tuple


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit layer dependencies with causal-frontier propagation. This is read-only: "
            "it does not modify the input ModelIR or the production dependency discovery path."
        )
    )
    parser.add_argument("model_ir", type=Path)
    args = parser.parse_args()

    _add_repo_src_to_path()
    from coopinfer.frontend import detect_layer_groups, load_model_ir_json

    model_ir = load_model_ir_json(args.model_ir)
    grouping = detect_layer_groups(model_ir)
    layer_ids = {
        group_id
        for group_id, group in grouping.groups.items()
        if group.kind == "layer"
    }

    order, predecessors = _topological_order(model_ir)

    # Each fine-node output carries the set of latest causal layer frontiers that
    # are required to produce it. The witness path starts at a member of that layer.
    frontiers: Dict[str, Dict[str, Tuple[str, ...]]] = {}

    # Layer precedence discovered so far. Because fine nodes are processed in
    # topological order, these edges only point from earlier to later frontiers.
    layer_adj: Dict[str, set[str]] = defaultdict(set)
    dependencies: Dict[Tuple[str, str], Tuple[str, ...]] = {}

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

    def prune_dominated(
        candidates: Mapping[str, Tuple[str, ...]],
    ) -> Dict[str, Tuple[str, ...]]:
        # If H already precedes K, and a join depends on both H and K, K is the
        # later causal frontier. H must complete anyway before K, so the join's
        # newly produced value is attributed to K rather than propagated as a
        # fictitious direct H -> every-future-layer dependency.
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

    def merge_candidate(
        bucket: Dict[str, Tuple[str, ...]],
        layer_id: str,
        path: Tuple[str, ...],
    ) -> None:
        previous = bucket.get(layer_id)
        if previous is None or len(path) < len(previous):
            bucket[layer_id] = path

    for node_id in order:
        candidates: Dict[str, Tuple[str, ...]] = {}
        for pred in predecessors.get(node_id, ()):
            for layer_id, path in frontiers.get(pred, {}).items():
                merge_candidate(candidates, layer_id, (*path, node_id))

        candidates = prune_dominated(candidates)
        group_id = grouping.member_to_group[node_id]

        if group_id in layer_ids:
            # Entering/continuing a detected layer establishes precedence from
            # every latest external frontier feeding this layer.
            for source_group, path in candidates.items():
                if source_group == group_id:
                    continue
                key = (source_group, group_id)
                previous = dependencies.get(key)
                if previous is None or len(path) < len(previous):
                    dependencies[key] = path
                if reaches(group_id, source_group):
                    raise RuntimeError(
                        f"Layer frontier cycle would be created: {source_group}->{group_id}"
                    )
                layer_adj[source_group].add(group_id)

            # Any value produced by a fine op inside the layer is now owned by
            # this layer frontier. Resetting the witness prevents older residual
            # ancestry from leaking through the layer as a direct dependency.
            frontiers[node_id] = {group_id: (node_id,)}
        else:
            # Glue/transform/join operations inherit the latest causal frontiers.
            # At joins, prune_dominated() keeps only maximal frontiers.
            frontiers[node_id] = candidates

    def dep_sort(item):
        source, target = item[0]
        s = grouping.groups[source]
        t = grouping.groups[target]
        return (
            s.stack_root,
            -1 if s.layer_index is None else s.layer_index,
            t.stack_root,
            -1 if t.layer_index is None else t.layer_index,
        )

    deps = sorted(dependencies.items(), key=dep_sort)
    same = []
    cross = []
    pair_counts = defaultdict(int)
    delta_counts = defaultdict(lambda: defaultdict(int))
    for (source, target), path in deps:
        s = grouping.groups[source]
        t = grouping.groups[target]
        pair_counts[(s.stack_root, t.stack_root)] += 1
        if s.stack_root == t.stack_root:
            same.append((s, t, path))
            if s.layer_index is not None and t.layer_index is not None:
                delta_counts[s.stack_root][t.layer_index - s.layer_index] += 1
        else:
            cross.append((s, t, path))

    same_index_cross = [
        row
        for row in cross
        if row[0].layer_index is not None
        and row[0].layer_index == row[1].layer_index
    ]

    print("===== causal frontier audit =====")
    print(f"model_ir={args.model_ir}")
    print(f"fine_nodes={len(model_ir.nodes)}")
    print(f"fine_edges={len(model_ir.edges)}")
    print(f"layer_stacks={len(grouping.stack_layers)}")
    for index, (root, layers) in enumerate(grouping.stack_layers.items()):
        print(f"  stack[{index}] root={root} count={len(layers)} layers={list(layers)}")
    print(f"causal_layer_dependencies={len(deps)}")
    print(f"cross_stack_dependencies={len(cross)}")
    print(f"same_index_cross_stack_dependencies={len(same_index_cross)}")

    print("\n===== stack-pair counts =====")
    for (source_root, target_root), count in sorted(
        pair_counts.items(), key=lambda item: (-item[1], item[0])
    ):
        relation = "same" if source_root == target_root else "cross"
        print(f"{count:4d}  [{relation}]  {source_root} -> {target_root}")

    print("\n===== same-stack layer deltas =====")
    for root in sorted(delta_counts):
        print(root)
        for delta, count in sorted(delta_counts[root].items()):
            print(f"  delta={delta}: {count}")

    print("\n===== cross-stack dependencies =====")
    for s, t, path in cross:
        print(
            f"  {s.stack_root}[{s.layer_index}] -> "
            f"{t.stack_root}[{t.layer_index}] path_len={len(path)}"
        )

    non_adjacent = [
        (s, t, path)
        for s, t, path in same
        if s.layer_index is not None
        and t.layer_index is not None
        and t.layer_index - s.layer_index != 1
    ]
    print("\n===== non-adjacent same-stack dependencies =====")
    print(f"count={len(non_adjacent)}")
    for s, t, path in non_adjacent[:20]:
        print(
            f"  {s.stack_root}[{s.layer_index}] -> "
            f"{t.stack_root}[{t.layer_index}] "
            f"delta={t.layer_index - s.layer_index} path_len={len(path)}"
        )

    if not non_adjacent:
        print("CAUSAL_AUDIT_STATUS=PASS_NO_NON_ADJACENT_SAME_STACK")
    else:
        print("CAUSAL_AUDIT_STATUS=CHECK_REMAINING_NON_ADJACENT")


if __name__ == "__main__":
    main()
