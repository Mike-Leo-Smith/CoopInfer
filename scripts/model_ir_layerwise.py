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


def _default_path(input_path: Path, suffix: str) -> Path:
    return input_path.with_name(f"{input_path.stem}_{suffix}.json")


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


def _mapping_with_edge_provenance(model_ir, grouping) -> Dict[str, Any]:
    from coopinfer.frontend import layer_mapping_dict

    data = layer_mapping_dict(grouping)
    quotient: Dict[tuple[str, str], Dict[str, Any]] = {}
    internal = 0
    for edge in model_ir.edges:
        source_group = grouping.member_to_group[edge.source]
        target_group = grouping.member_to_group[edge.target]
        tensor_id = edge.tensor_id or f"{edge.source}->{edge.target}"
        if source_group == target_group:
            internal += 1
            continue
        key = (source_group, target_group)
        bucket = quotient.setdefault(
            key,
            {
                "source": source_group,
                "target": target_group,
                "size_bytes": 0.0,
                "fine_edges": [],
            },
        )
        bucket["size_bytes"] += float(edge.size_bytes)
        bucket["fine_edges"].append(
            {
                "source": edge.source,
                "target": edge.target,
                "tensor_id": tensor_id,
                "size_bytes": float(edge.size_bytes),
            }
        )
    data["provenance"] = {
        "fine_node_count": len(model_ir.nodes),
        "covered_fine_nodes": len(grouping.member_to_group),
        "coverage_complete": set(grouping.member_to_group) == set(model_ir.nodes),
        "fine_edge_count": len(model_ir.edges),
        "internal_fine_edges": internal,
        "cross_fine_edges": sum(len(item["fine_edges"]) for item in quotient.values()),
        "quotient_edges": list(quotient.values()),
        "dependency_rule": (
            "A quotient edge exists iff at least one real Fine ModelIR edge crosses "
            "the corresponding groups; no layer dependency is manually invented."
        ),
    }
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Automatically detect repeated layer stacks from a real Fine ModelIR, "
            "derive a reversible layer quotient graph, cost each layer workload "
            "with GenZ hardware systems, and export CoopInfer JSON. No evaluation "
            "or solver is run."
        )
    )
    parser.add_argument("input", type=Path, help="Fine ModelIR JSON (costed or uncosted)")
    parser.add_argument("--layer-ir-output", type=Path, default=None)
    parser.add_argument("--mapping-output", type=Path, default=None)
    parser.add_argument("--coopinfer-output", type=Path, default=None)
    parser.add_argument("--genz-root", type=Path, default=None)
    parser.add_argument("--device", default="RTX_4090")
    parser.add_argument("--host", default="A100_80GB")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--compute-efficiency", type=float, default=1.0)
    parser.add_argument("--memory-efficiency", type=float, default=1.0)
    parser.add_argument("--min-repeated-layers", type=int, default=2)
    parser.add_argument("--bandwidth-mb-s", type=float, default=1250.0)
    parser.add_argument("--latency-ms", type=float, default=0.2)
    args = parser.parse_args()

    _add_repo_src_to_path()
    _add_genz_root(args.genz_root)

    from coopinfer.cost import annotate_layer_genz_costs
    from coopinfer.frontend import (
        detect_layer_groups,
        load_model_ir_json,
        to_coopinfer_payload,
    )

    model_ir = load_model_ir_json(args.input)
    original_node_ids = tuple(model_ir.nodes)
    original_edges = tuple(
        (edge.source, edge.target, edge.tensor_id, float(edge.size_bytes))
        for edge in model_ir.edges
    )

    grouping = detect_layer_groups(
        model_ir,
        min_repeated_layers=args.min_repeated_layers,
    )
    scheduling_ir = annotate_layer_genz_costs(
        model_ir,
        grouping,
        {"device": args.device, "host": args.host},
        precision=args.precision,
        compute_efficiency=args.compute_efficiency,
        memory_efficiency=args.memory_efficiency,
    )

    # Explicit non-mutation check: layer detection/costing must not rewrite the
    # source Fine ModelIR or its dependency list.
    fine_ir_unchanged = (
        tuple(model_ir.nodes) == original_node_ids
        and tuple(
            (edge.source, edge.target, edge.tensor_id, float(edge.size_bytes))
            for edge in model_ir.edges
        )
        == original_edges
    )
    if not fine_ir_unchanged:
        raise RuntimeError("Layerwise pipeline mutated the source Fine ModelIR")

    layer_ir_path = args.layer_ir_output or _default_path(args.input, "layer_ir")
    mapping_path = args.mapping_output or _default_path(args.input, "layer_mapping")
    coopinfer_path = args.coopinfer_output or _default_path(
        args.input, "layer_genz_coopinfer"
    )
    for path in (layer_ir_path, mapping_path, coopinfer_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    layer_ir_path.write_text(
        json.dumps(_scheduling_ir_dict(scheduling_ir), indent=2) + "\n",
        encoding="utf-8",
    )
    mapping_path.write_text(
        json.dumps(_mapping_with_edge_provenance(model_ir, grouping), indent=2) + "\n",
        encoding="utf-8",
    )
    payload = to_coopinfer_payload(
        scheduling_ir,
        bandwidth_mb_s=args.bandwidth_mb_s,
        latency_ms=args.latency_ms,
    )
    coopinfer_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    layer_groups = [
        group for group in grouping.groups.values() if group.kind == "layer"
    ]
    aux_groups = [
        group for group in grouping.groups.values() if group.kind == "aux_region"
    ]
    singleton_groups = [
        group
        for group in grouping.groups.values()
        if group.kind in {"input", "output", "source_like"}
    ]
    total_costs = {
        resource: sum(float(node.costs_ms.get(resource, 0.0)) for node in scheduling_ir.nodes.values())
        for resource in ("device", "host")
    }

    print("===== automatic layerwise ModelIR =====")
    print(f"input={args.input}")
    print(f"fine_nodes={len(model_ir.nodes)}")
    print(f"fine_edges={len(model_ir.edges)}")
    print(f"detected_layer_stacks={len(grouping.stack_layers)}")
    for i, (root, indices) in enumerate(grouping.stack_layers.items()):
        print(
            f"  stack[{i}] root={root} layers={list(indices)} count={len(indices)}"
        )
    print(f"layer_groups={len(layer_groups)}")
    print(f"aux_region_groups={len(aux_groups)}")
    print(f"singleton_groups={len(singleton_groups)}")
    print(f"layer_graph_nodes={len(scheduling_ir.nodes)}")
    print(f"layer_graph_edges={len(scheduling_ir.edges)}")
    print(f"member_coverage={len(grouping.member_to_group)}/{len(model_ir.nodes)}")
    print(f"fine_ir_unchanged={fine_ir_unchanged}")
    print("dependency_edges=derived_only_from_real_fine_edges")
    print(
        f"layer_cost_source={scheduling_ir.metadata.get('cost_source')} "
        f"precision={args.precision}"
    )
    print(f"layer_cost_total_device_ms={total_costs['device']:.6f}")
    print(f"layer_cost_total_host_ms={total_costs['host']:.6f}")
    print(f"layer_ir={layer_ir_path}")
    print(f"mapping={mapping_path}")
    print(f"coopinfer={coopinfer_path}")
    print("evaluation=not_run")
    print("solver=not_run")


if __name__ == "__main__":
    main()
