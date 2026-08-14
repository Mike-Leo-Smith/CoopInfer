from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


def _add_repo_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _short_root(root: str) -> str:
    parts = [part for part in root.split(".") if part]
    return root if len(parts) <= 4 else ".".join(parts[-4:])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2: Fine ModelIR -> LayerGraphIR. Runs automatic layer detection, "
            "causal dependency recovery, causal ownership boundary recovery, and sanity validation."
        )
    )
    parser.add_argument("input", type=Path, help="Fine ModelIR JSON")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--min-repeated-layers", type=int, default=2)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print every recovered layer dependency in addition to the compact summary.",
    )
    args = parser.parse_args()

    _add_repo_src_to_path()
    from coopinfer.frontend import analyze_layer_graph, layer_graph_to_dict, load_model_ir_json

    model_ir = load_model_ir_json(args.input)
    layer_graph = analyze_layer_graph(
        model_ir,
        precision=args.precision,
        min_repeated_layers=args.min_repeated_layers,
    )

    grouping = layer_graph.grouping
    print("===== Stage 2: Layer Graph Analysis =====")
    print(f"input={args.input}")
    print(f"precision={layer_graph.precision}")
    print(f"fine_nodes={len(model_ir.nodes)}")
    print(f"fine_edges={len(model_ir.edges)}")
    print(f"layer_stacks={len(grouping.stack_layers)}")
    for index, (root, layers) in enumerate(grouping.stack_layers.items()):
        print(f"  stack[{index}] root={root} count={len(layers)} layers={list(layers)}")
    print(f"layer_nodes={len(layer_graph.layer_ids)}")
    print(f"dependencies={len(layer_graph.dependencies)}")
    print(f"cross_stack_dependencies={len(layer_graph.cross_stack_dependencies)}")
    print(
        "same_index_cross_stack_dependencies="
        f"{len(layer_graph.same_index_cross_stack_dependencies)}"
    )

    patterns = Counter()
    for dep in layer_graph.dependencies:
        payload = layer_graph.payloads.get((dep.source_group, dep.target_group))
        if payload is None:
            continue
        relation = "cross" if dep.cross_stack else "same"
        patterns[
            (
                relation,
                dep.source_stack,
                dep.target_stack,
                round(float(payload.size_bytes), 6),
                len(payload.tensor_ids),
            )
        ] += 1

    print("\n===== boundary patterns =====")
    for (relation, source_root, target_root, size_bytes, tensor_count), count in sorted(
        patterns.items(), key=lambda item: (-item[1], item[0])
    ):
        print(
            f"  count={count:3d} [{relation}] bytes={size_bytes:.0f} "
            f"MiB={size_bytes / (1024 ** 2):.6f} tensors={tensor_count} "
            f"{_short_root(source_root)} -> {_short_root(target_root)}"
        )

    if args.details:
        print("\n===== dependencies =====")
        for dep in layer_graph.dependencies:
            payload = layer_graph.payloads.get((dep.source_group, dep.target_group))
            source = grouping.groups[dep.source_group]
            target = grouping.groups[dep.target_group]
            relation = "cross" if dep.cross_stack else "same"
            if payload is None:
                payload_text = "payload=MISSING"
            else:
                payload_text = (
                    f"bytes={payload.size_bytes:.0f} "
                    f"MiB={payload.size_bytes / (1024 ** 2):.6f} "
                    f"tensors={len(payload.tensor_ids)}"
                )
            print(
                f"[{relation}] {_short_root(source.stack_root)}[{source.layer_index}] -> "
                f"{_short_root(target.stack_root)}[{target.layer_index}] {payload_text}"
            )

    validation = layer_graph.validation
    print("\n===== validation =====")
    print(f"layer_graph_dag={validation.is_dag}")
    print(f"missing_payload_dependencies={len(validation.missing_payload_dependencies)}")
    print(f"ambiguous_owner_nodes={validation.ambiguous_owner_nodes}")
    print(
        "non_adjacent_same_stack_dependencies="
        f"{validation.non_adjacent_same_stack_dependencies}"
    )
    status = "PASS" if validation.passed else "FAIL"
    print(f"LAYER_GRAPH_STATUS={status}")

    output = args.output
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(layer_graph_to_dict(layer_graph), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"layer_graph={output}")

    if not validation.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
