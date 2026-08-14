from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict


def _add_repo_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _add_genz_root(root: Path | None) -> None:
    if root is None:
        return
    root = root.resolve()
    candidates = (root, root / "genz", root / "vla-perf")
    for candidate in reversed(candidates):
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _scheduling_ir_dict(scheduling_ir) -> Dict[str, Any]:
    return {
        "metadata": dict(scheduling_ir.metadata),
        "nodes": [
            {
                "id": node.id,
                "name": node.name,
                "members": list(node.members),
                "costs_ms": dict(node.costs_ms),
                "placement": node.placement,
                "metadata": dict(node.metadata),
            }
            for node in scheduling_ir.nodes.values()
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "size_bytes": float(edge.size_bytes),
                "tensor_ids": list(edge.tensor_ids),
            }
            for edge in scheduling_ir.edges
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the actual DAG-safe Layer SchedulingIR from automatically detected "
            "repeated layers and automatically recovered Fine-DAG frontier dependencies, "
            "cost layers with GenZ, and export CoopInfer JSON. No solver is run."
        )
    )
    parser.add_argument("input", type=Path, help="Real Fine ModelIR JSON")
    parser.add_argument("--genz-root", type=Path, default=None)
    parser.add_argument("--device", default="RTX_4090")
    parser.add_argument("--host", default="A100_80GB")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--compute-efficiency", type=float, default=1.0)
    parser.add_argument("--memory-efficiency", type=float, default=1.0)
    parser.add_argument("--min-repeated-layers", type=int, default=2)
    parser.add_argument("--bandwidth-mb-s", type=float, default=1250.0)
    parser.add_argument("--latency-ms", type=float, default=0.2)
    parser.add_argument("--layer-ir-output", type=Path, default=None)
    parser.add_argument("--coopinfer-output", type=Path, default=None)
    args = parser.parse_args()

    _add_repo_src_to_path()
    _add_genz_root(args.genz_root)

    from coopinfer.cost import annotate_layer_genz_costs
    from coopinfer.frontend import (
        build_discovered_layer_scheduling_ir,
        detect_layer_groups,
        discover_layer_dependencies,
        load_model_ir_json,
        to_coopinfer_payload,
    )

    model_ir = load_model_ir_json(args.input)
    grouping = detect_layer_groups(
        model_ir,
        min_repeated_layers=args.min_repeated_layers,
    )
    dependencies = discover_layer_dependencies(model_ir, grouping)

    # Cost table only: deliberately do NOT build the old aux-region quotient
    # topology here. The discovered-layer pipeline owns its own DAG topology.
    costed_groups = annotate_layer_genz_costs(
        model_ir,
        grouping,
        {"device": args.device, "host": args.host},
        precision=args.precision,
        compute_efficiency=args.compute_efficiency,
        memory_efficiency=args.memory_efficiency,
        cost_only=True,
    )

    layer_ir = build_discovered_layer_scheduling_ir(
        model_ir,
        grouping,
        dependencies,
        costed_groups,
        precision=args.precision,
    )

    layer_ir_output = args.layer_ir_output or args.input.with_name(
        f"{args.input.stem}_discovered_layer_ir.json"
    )
    coopinfer_output = args.coopinfer_output or args.input.with_name(
        f"{args.input.stem}_discovered_layer_coopinfer.json"
    )
    layer_ir_output.parent.mkdir(parents=True, exist_ok=True)
    coopinfer_output.parent.mkdir(parents=True, exist_ok=True)

    layer_ir_output.write_text(
        json.dumps(_scheduling_ir_dict(layer_ir), indent=2) + "\n",
        encoding="utf-8",
    )
    payload = to_coopinfer_payload(
        layer_ir,
        bandwidth_mb_s=args.bandwidth_mb_s,
        latency_ms=args.latency_ms,
    )
    coopinfer_output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )

    layer_ids = {
        group_id
        for group_id, group in grouping.groups.items()
        if group.kind == "layer"
    }
    layer_only_totals = {
        resource: sum(
            float(costed_groups.nodes[group_id].costs_ms.get(resource, 0.0))
            for group_id in layer_ids
        )
        for resource in ("device", "host")
    }

    cross = [dependency for dependency in dependencies if dependency.cross_stack]
    same_index = [
        dependency
        for dependency in cross
        if dependency.source_layer is not None
        and dependency.source_layer == dependency.target_layer
    ]
    edge_by_pair = {
        (edge.source, edge.target): edge
        for edge in layer_ir.edges
        if edge.source in layer_ids and edge.target in layer_ids
    }

    print("===== discovered Layer SchedulingIR =====")
    print(f"input={args.input}")
    print(f"fine_nodes={len(model_ir.nodes)}")
    print(f"fine_edges={len(model_ir.edges)}")
    print(f"detected_layer_stacks={len(grouping.stack_layers)}")
    for i, (root, indices) in enumerate(grouping.stack_layers.items()):
        print(f"  stack[{i}] root={root} layers={list(indices)} count={len(indices)}")
    print(f"detected_layer_nodes={len(layer_ids)}")
    print(f"layer_frontier_dependencies={len(dependencies)}")
    print(f"cross_stack_dependencies={len(cross)}")
    print(f"same_index_cross_stack_dependencies={len(same_index)}")
    print(f"solver_graph_nodes={len(layer_ir.nodes)}")
    print(f"solver_graph_edges={len(layer_ir.edges)}")
    print("solver_graph_dag=True")
    print("dependency_edges_written_into_solver_graph=True")
    print("old_aux_region_quotient_graph_used_by_solver=False")
    print("old_aux_region_quotient_graph_used_by_costing=False")
    print(f"layer_cost_total_device_ms={layer_only_totals['device']:.6f}")
    print(f"layer_cost_total_host_ms={layer_only_totals['host']:.6f}")
    print(f"network={args.bandwidth_mb_s} MB/s + {args.latency_ms} ms/transfer")

    print("cross_stack_edges:")
    for dependency in cross:
        edge = edge_by_pair[(dependency.source_group, dependency.target_group)]
        source_name = grouping.groups[dependency.source_group].display_name
        target_name = grouping.groups[dependency.target_group].display_name
        print(
            f"  {dependency.source_layer} -> {dependency.target_layer} "
            f"size={edge.size_bytes / 1024.0 / 1024.0:.4f} MiB "
            f"tensors={len(edge.tensor_ids)} "
            f"{source_name} -> {target_name}"
        )

    print(f"layer_ir={layer_ir_output}")
    print(f"coopinfer={coopinfer_output}")
    print("evaluation=not_run")
    print("solver=not_run")


if __name__ == "__main__":
    main()
