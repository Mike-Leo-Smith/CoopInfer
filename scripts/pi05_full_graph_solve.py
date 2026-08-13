from __future__ import annotations

import argparse
import json
from pathlib import Path


def _assignment(node_ids, value: int):
    return {str(node_id): int(value) for node_id in node_ids}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a fine-grained pi0.5 ModelIR into CoopInfer's backend schema "
            "without coarsening, inject synthetic costs, and run backend smoke checks."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/pi05_export_probe/prefix_ae_ir.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/pi05_export_probe/prefix_ae_coopinfer_synthetic.json"),
    )
    parser.add_argument("--device-cost-ms", type=float, default=0.001)
    parser.add_argument("--host-cost-ms", type=float, default=0.002)
    parser.add_argument("--bandwidth-mb-s", type=float, default=1250.0)
    parser.add_argument("--latency-ms", type=float, default=0.2)
    parser.add_argument(
        "--algorithm",
        choices=("Auto", "Enumerate", "Random Search", "Simulated Annealing"),
        default="Random Search",
    )
    parser.add_argument("--heuristic-iterations", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--solver-threads", type=int, default=0)
    parser.add_argument(
        "--skip-solve",
        action="store_true",
        help="Only convert, validate, and evaluate all-Device/all-Host baselines",
    )
    args = parser.parse_args()

    from coopinfer.evaluator import evaluate
    from coopinfer.frontend import (
        annotate_synthetic_costs,
        identity_scheduling_ir,
        load_model_ir_json,
        to_coopinfer_payload,
    )
    from coopinfer.model import load_from_json, validate_graph
    from coopinfer.solver import solve

    model_ir = load_model_ir_json(args.input)
    costed_ir = annotate_synthetic_costs(
        model_ir,
        device_cost_ms=args.device_cost_ms,
        host_cost_ms=args.host_cost_ms,
    )
    scheduling_ir = identity_scheduling_ir(costed_ir)
    payload = to_coopinfer_payload(
        scheduling_ir,
        bandwidth_mb_s=args.bandwidth_mb_s,
        latency_ms=args.latency_ms,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    state = load_from_json(args.output)
    validate_graph(state.graph, require_dag=True)

    print("===== full graph conversion =====")
    print(f"input={args.input}")
    print(f"output={args.output}")
    print(f"model_ir_nodes={len(model_ir.nodes)}")
    print(f"model_ir_tensor_edges={len(model_ir.edges)}")
    print(f"backend_nodes={state.graph.number_of_nodes()}")
    print(f"backend_edges={state.graph.number_of_edges()}")
    print(f"cost_source={scheduling_ir.metadata.get('cost_source')}")
    print(
        "synthetic_costs_ms="
        f"device:{args.device_cost_ms} host:{args.host_cost_ms}"
    )
    print(
        "network="
        f"{args.bandwidth_mb_s} MB/s + {args.latency_ms} ms/transfer"
    )

    all_device = evaluate(
        state.graph,
        _assignment(state.graph.nodes, 0),
        bandwidth=args.bandwidth_mb_s,
        latency=args.latency_ms,
        weight_avg_latency=1.0,
        weight_max_latency=0.0,
        weight_device_utilization=0.0,
    )
    all_host = evaluate(
        state.graph,
        _assignment(state.graph.nodes, 1),
        bandwidth=args.bandwidth_mb_s,
        latency=args.latency_ms,
        weight_avg_latency=1.0,
        weight_max_latency=0.0,
        weight_device_utilization=0.0,
    )

    print("\n===== backend baselines =====")
    print(f"all_device_latency_ms={all_device.latency:.6f}")
    print(f"all_host_latency_ms={all_host.latency:.6f}")
    print(f"all_device_transfers={len(all_device.transfer_records)}")
    print(f"all_host_transfers={len(all_host.transfer_records)}")

    if args.skip_solve:
        print("\nsolver=skipped")
        return

    result = solve(
        state.graph,
        bandwidth=args.bandwidth_mb_s,
        latency=args.latency_ms,
        weight_avg_latency=1.0,
        weight_max_latency=0.0,
        weight_device_utilization=0.0,
        algorithm=args.algorithm,
        heuristic_iterations=args.heuristic_iterations,
        seed=args.seed,
        solver_threads=args.solver_threads,
    )
    device_nodes = sum(1 for value in result.assignment.values() if value == 0)
    host_nodes = sum(1 for value in result.assignment.values() if value == 1)

    print("\n===== solver smoke =====")
    print(f"mode={result.mode}")
    print(f"iterations={result.iterations}")
    print(f"latency_ms={result.metrics.latency:.6f}")
    print(f"device_nodes={device_nodes}")
    print(f"host_nodes={host_nodes}")
    print(f"transfers={len(result.metrics.transfer_records)}")
    print("NOTE: latency/placement are plumbing-only because costs are synthetic.")


if __name__ == "__main__":
    main()
