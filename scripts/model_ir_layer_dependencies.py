from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _add_repo_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recover layer-to-layer frontier dependencies from a real Fine ModelIR "
            "by tracing through non-layer glue groups only. No synthetic dependency "
            "is added and no evaluator/solver is run."
        )
    )
    parser.add_argument("input", type=Path, help="Real Fine ModelIR JSON")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-repeated-layers", type=int, default=2)
    args = parser.parse_args()

    _add_repo_src_to_path()
    from coopinfer.frontend import (
        detect_layer_groups,
        discover_layer_dependencies,
        layer_dependency_summary,
        load_model_ir_json,
    )

    model_ir = load_model_ir_json(args.input)
    grouping = detect_layer_groups(
        model_ir,
        min_repeated_layers=args.min_repeated_layers,
    )
    dependencies = discover_layer_dependencies(model_ir, grouping)
    summary = layer_dependency_summary(dependencies)

    output = args.output or args.input.with_name(
        f"{args.input.stem}_layer_dependencies.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    cross = [dependency for dependency in dependencies if dependency.cross_stack]
    same_index = [
        dependency
        for dependency in cross
        if dependency.source_layer is not None
        and dependency.source_layer == dependency.target_layer
    ]

    print("===== automatic layer dependency recovery =====")
    print(f"input={args.input}")
    print(f"fine_nodes={len(model_ir.nodes)}")
    print(f"fine_edges={len(model_ir.edges)}")
    print(f"detected_layer_stacks={len(grouping.stack_layers)}")
    print(f"layer_frontier_dependencies={len(dependencies)}")
    print(f"cross_stack_dependencies={len(cross)}")
    print(f"same_index_cross_stack_dependencies={len(same_index)}")
    print("dependency_rule=real reachability through non-layer groups only")
    print("scheduling_graph_modified=False")
    print(f"output={output}")

    for dependency in cross:
        source_name = grouping.groups[dependency.source_group].display_name
        target_name = grouping.groups[dependency.target_group].display_name
        print(
            f"  {dependency.source_layer} -> {dependency.target_layer} "
            f"glue_hops={dependency.glue_hops} "
            f"{source_name} -> {target_name}"
        )

    print("evaluation=not_run")
    print("solver=not_run")


if __name__ == "__main__":
    main()
