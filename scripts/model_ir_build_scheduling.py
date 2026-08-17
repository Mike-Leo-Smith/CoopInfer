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
            "Stage 3: analyze the LayerGraph, estimate per-layer hardware costs with GenZ, "
            "materialize iterative execution semantics when exported metadata requests it, "
            "attach network payloads, and build a costed SchedulingIR. The CoopInfer solver is not run."
        )
    )
    parser.add_argument("input", type=Path, help="Fine ModelIR JSON from Stage 1")
    parser.add_argument("--genz-root", type=Path, default=None)
    parser.add_argument("--device", default="RTX_4090")
    parser.add_argument("--host", default="A100_80GB")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--compute-efficiency", type=float, default=1.0)
    parser.add_argument("--memory-efficiency", type=float, default=1.0)
    parser.add_argument("--min-repeated-layers", type=int, default=2)
    parser.add_argument("--bandwidth-mb-s", type=float, default=1250.0)
    parser.add_argument("--latency-ms", type=float, default=0.2)
    parser.add_argument("--scheduling-output", type=Path, default=None)
    parser.add_argument("--coopinfer-output", type=Path, default=None)
    args = parser.parse_args()

    _add_repo_src_to_path()
    _add_genz_root(args.genz_root)

    from coopinfer.cost import annotate_layer_genz_costs
    from coopinfer.frontend import (
        analyze_layer_graph,
        build_layer_graph_scheduling_ir,
        load_model_ir_json,
        to_coopinfer_payload,
    )

    model_ir = load_model_ir_json(args.input)
    layer_graph = analyze_layer_graph(
        model_ir,
        precision=args.precision,
        min_repeated_layers=args.min_repeated_layers,
    )
    if not layer_graph.validation.passed:
        raise RuntimeError("Stage-2 LayerGraph validation failed; refusing to build SchedulingIR")

    costed_groups = annotate_layer_genz_costs(
        model_ir,
        layer_graph.grouping,
        {"device": args.device, "host": args.host},
        precision=args.precision,
        compute_efficiency=args.compute_efficiency,
        memory_efficiency=args.memory_efficiency,
        cost_only=True,
    )

    scheduling_ir = build_layer_graph_scheduling_ir(
        layer_graph,
        costed_groups,
    )

    scheduling_output = args.scheduling_output or args.input.with_name(
        f"{args.input.stem}_scheduling_ir.json"
    )
    coopinfer_output = args.coopinfer_output or args.input.with_name(
        f"{args.input.stem}_coopinfer.json"
    )
    scheduling_output.parent.mkdir(parents=True, exist_ok=True)
    coopinfer_output.parent.mkdir(parents=True, exist_ok=True)

    scheduling_output.write_text(
        json.dumps(_scheduling_ir_dict(scheduling_ir), indent=2) + "\n",
        encoding="utf-8",
    )
    coopinfer_payload = to_coopinfer_payload(
        scheduling_ir,
        bandwidth_mb_s=args.bandwidth_mb_s,
        latency_ms=args.latency_ms,
    )
    coopinfer_output.write_text(
        json.dumps(coopinfer_payload, indent=2) + "\n",
        encoding="utf-8",
    )

    structural_totals = {
        resource: sum(
            float(costed_groups.nodes[group_id].costs_ms.get(resource, 0.0))
            for group_id in layer_graph.layer_ids
        )
        for resource in ("device", "host")
    }
    execution_totals = {
        resource: sum(
            float(node.costs_ms.get(resource, 0.0))
            for node in scheduling_ir.nodes.values()
        )
        for resource in ("device", "host")
    }

    print("===== Stage 3: Cost + SchedulingIR =====")
    print(f"input={args.input}")
    print(f"layer_nodes={len(layer_graph.layer_ids)}")
    print(f"layer_dependencies={len(layer_graph.dependencies)}")
    print(f"scheduling_nodes={len(scheduling_ir.nodes)}")
    print(f"scheduling_edges={len(scheduling_ir.edges)}")
    print("scheduling_dag=True")
    print(f"device={args.device}")
    print(f"host={args.host}")
    print(f"precision={args.precision}")
    print(f"structural_layer_cost_total_device_ms={structural_totals['device']:.6f}")
    print(f"structural_layer_cost_total_host_ms={structural_totals['host']:.6f}")
    print(f"execution_compute_total_device_ms={execution_totals['device']:.6f}")
    print(f"execution_compute_total_host_ms={execution_totals['host']:.6f}")
    execution_semantics = scheduling_ir.metadata.get("execution_semantics", "single_pass")
    print(f"execution_semantics={execution_semantics}")
    if execution_semantics == "iterative_denoise_v1":
        print(f"num_inference_steps={scheduling_ir.metadata['num_inference_steps']}")
        print(f"iterative_layer_count={scheduling_ir.metadata['iterative_layer_count']}")
        print(f"kv_reuse_across_denoise_steps={scheduling_ir.metadata['kv_reuse_across_denoise_steps']}")
        print(f"static_to_iterative_edges_once={scheduling_ir.metadata['static_to_iterative_edges_once']}")
        print(f"loop_carried_state_bytes={scheduling_ir.metadata['loop_carried_state_bytes']:.0f}")
    print(f"network={args.bandwidth_mb_s} MB/s + {args.latency_ms} ms/transfer")
    print(f"scheduling_ir={scheduling_output}")
    print(f"coopinfer={coopinfer_output}")
    print("solver=not_run")


if __name__ == "__main__":
    main()
