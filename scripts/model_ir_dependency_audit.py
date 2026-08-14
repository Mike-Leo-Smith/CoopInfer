from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path


def _short(root: str) -> str:
    return root.split(".")[-3:] and ".".join(root.split(".")[-3:])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit automatically discovered layer dependencies from a saved Fine ModelIR."
    )
    parser.add_argument("model_ir", type=Path)
    parser.add_argument("--max-examples", type=int, default=20)
    args = parser.parse_args()

    from coopinfer.frontend import (
        detect_layer_groups,
        discover_layer_dependencies,
        load_model_ir_json,
    )

    ir = load_model_ir_json(args.model_ir)
    grouping = detect_layer_groups(ir)
    deps = discover_layer_dependencies(ir, grouping)

    print("===== dependency audit =====")
    print(f"model_ir={args.model_ir}")
    print(f"fine_nodes={len(ir.nodes)}")
    print(f"fine_edges={len(ir.edges)}")
    print(f"layer_stacks={len(grouping.stack_layers)}")
    for i, (root, layers) in enumerate(grouping.stack_layers.items()):
        print(f"  stack[{i}] root={root} count={len(layers)} layers={list(layers)}")
    print(f"layer_frontier_dependencies={len(deps)}")

    pair_counts: Counter[tuple[str, str]] = Counter()
    same_stack_delta_counts: dict[str, Counter[int | str]] = defaultdict(Counter)
    cross_rows = []
    suspicious_same_stack = []

    for d in deps:
        pair_counts[(d.source_stack, d.target_stack)] += 1
        if d.source_stack == d.target_stack:
            if d.source_layer is None or d.target_layer is None:
                delta: int | str = "NA"
            else:
                delta = d.target_layer - d.source_layer
            same_stack_delta_counts[d.source_stack][delta] += 1
            if delta != 1:
                suspicious_same_stack.append(d)
        else:
            cross_rows.append(d)

    print("\n===== stack-pair counts =====")
    for (src, dst), count in sorted(pair_counts.items(), key=lambda item: (-item[1], item[0])):
        relation = "same" if src == dst else "cross"
        print(f"{count:4d}  [{relation}]  {src} -> {dst}")

    print("\n===== same-stack layer deltas =====")
    for root, counts in sorted(same_stack_delta_counts.items()):
        print(f"{root}")
        for delta, count in sorted(counts.items(), key=lambda item: str(item[0])):
            print(f"  delta={delta}: {count}")

    print("\n===== cross-stack dependencies =====")
    print(f"count={len(cross_rows)}")
    for d in cross_rows[: args.max_examples]:
        print(
            f"  {d.source_stack}[{d.source_layer}] -> "
            f"{d.target_stack}[{d.target_layer}] glue_hops={d.glue_hops}"
        )
    if len(cross_rows) > args.max_examples:
        print(f"  ... {len(cross_rows) - args.max_examples} more")

    print("\n===== suspicious same-stack non-adjacent dependencies =====")
    print(f"count={len(suspicious_same_stack)}")
    for d in suspicious_same_stack[: args.max_examples]:
        delta = (
            d.target_layer - d.source_layer
            if d.source_layer is not None and d.target_layer is not None
            else "NA"
        )
        print(
            f"  {d.source_stack}[{d.source_layer}] -> "
            f"{d.target_stack}[{d.target_layer}] delta={delta} glue_hops={d.glue_hops}"
        )
        print(f"    fine_path={' -> '.join(d.path_fine_nodes[:8])}")
        if len(d.path_fine_nodes) > 8:
            print(f"    ... path_len={len(d.path_fine_nodes)}")

    if suspicious_same_stack:
        print("\nAUDIT_STATUS=CHECK_NON_ADJACENT_SAME_STACK")
    else:
        print("\nAUDIT_STATUS=PASS_ADJACENT_SAME_STACK")


if __name__ == "__main__":
    main()
