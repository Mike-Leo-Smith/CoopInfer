from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


def _add_repo_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _assignment(graph, value: int):
    result = {}
    for node_id, attrs in graph.nodes(data=True):
        placement = str(attrs.get("placement", "free")).lower()
        if placement == "device":
            result[str(node_id)] = 0
        elif placement == "host":
            result[str(node_id)] = 1
        else:
            result[str(node_id)] = int(value)
    return result


def _baseline_summary(result) -> dict:
    return {
        "latency_ms": float(result.latency),
        "max_frame_latency_ms": float(result.max_frame_latency),
        "initiation_interval_ms": float(result.initiation_interval),
        "device_utilization": float(result.device_utilization),
        "host_utilization": float(result.host_utilization),
        "network_utilization": float(result.network_utilization),
        "num_transfers": len(result.transfer_records),
    }


def _node_results(graph, assignment) -> dict:
    nodes = {}
    for node_id, attrs in graph.nodes(data=True):
        key = str(node_id)
        value = int(assignment[key] if key in assignment else assignment[node_id])
        nodes[key] = {
            "name": str(attrs.get("name", key)),
            "assignment": value,
            "resource": "Device" if value == 0 else "Host",
            "placement_constraint": str(attrs.get("placement", "free")),
            "c_dev_ms": float(attrs.get("c_dev", 0.0)),
            "c_host_ms": float(attrs.get("c_host", 0.0)),
        }
    return nodes


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 4: run the CoopInfer solver on the solver-ready JSON produced by Stage 3."
        )
    )
    parser.add_argument("input", type=Path, help="*_coopinfer.json from Stage 3")
    parser.add_argument(
        "--algorithm",
        choices=("Auto", "Enumerate", "Random Search", "Simulated Annealing"),
        default="Random Search",
    )
    parser.add_argument("--heuristic-iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--solver-threads", type=int, default=None)
    parser.add_argument("--bandwidth-mb-s", type=float, default=None)
    parser.add_argument("--latency-ms", type=float, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional solve-result JSON. Stores solver/environment metadata, baselines, "
            "best assignment, per-node placement/costs, and detailed schedule metrics."
        ),
    )
    args = parser.parse_args()

    _add_repo_src_to_path()
    from coopinfer.evaluator import evaluate
    from coopinfer.model import load_from_json, validate_graph
    from coopinfer.solver import solve

    state = load_from_json(args.input)
    validate_graph(state.graph, require_dag=True)
    env = state.environment

    bandwidth = env.bandwidth if args.bandwidth_mb_s is None else args.bandwidth_mb_s
    latency = env.latency if args.latency_ms is None else args.latency_ms
    solver_threads = env.solver_threads if args.solver_threads is None else args.solver_threads

    evaluation_common = {
        "bandwidth": bandwidth,
        "latency": latency,
        "weight_avg_latency": env.weight_avg_latency,
        "weight_max_latency": env.weight_max_latency,
        "weight_device_utilization": env.weight_device_utilization,
        "batch_transfers": env.batch_transfers,
        "pipeline_unroll": env.pipeline_unroll,
    }

    all_device = evaluate(
        state.graph,
        _assignment(state.graph, 0),
        **evaluation_common,
    )
    all_host = evaluate(
        state.graph,
        _assignment(state.graph, 1),
        **evaluation_common,
    )

    print("===== Stage 4: CoopInfer Solver =====")
    print(f"input={args.input}")
    print(f"nodes={state.graph.number_of_nodes()}")
    print(f"edges={state.graph.number_of_edges()}")
    print(f"network={bandwidth} MB/s + {latency} ms/transfer")
    print(f"all_device_latency_ms={all_device.latency:.6f}")
    print(f"all_host_latency_ms={all_host.latency:.6f}")

    result = solve(
        state.graph,
        bandwidth=bandwidth,
        latency=latency,
        weight_avg_latency=env.weight_avg_latency,
        weight_max_latency=env.weight_max_latency,
        weight_device_utilization=env.weight_device_utilization,
        algorithm=args.algorithm,
        heuristic_iterations=args.heuristic_iterations,
        seed=args.seed,
        latency_limit=env.latency_limit,
        batch_transfers=env.batch_transfers,
        pipeline_unroll=env.pipeline_unroll,
        max_frame_latency_limit=env.max_frame_latency_limit,
        solver_threads=solver_threads,
        anneal_initial_temp=env.anneal_initial_temp,
        anneal_final_temp=env.anneal_final_temp,
    )

    assignment = {str(node_id): int(value) for node_id, value in result.assignment.items()}
    device_nodes = sum(1 for value in assignment.values() if value == 0)
    host_nodes = sum(1 for value in assignment.values() if value == 1)
    print("\n===== best schedule =====")
    print(f"mode={result.mode}")
    print(f"iterations={result.iterations}")
    print(f"latency_ms={result.metrics.latency:.6f}")
    print(f"device_nodes={device_nodes}")
    print(f"host_nodes={host_nodes}")
    print(f"transfers={len(result.metrics.transfer_records)}")

    if args.output is not None:
        output = args.output
        output.parent.mkdir(parents=True, exist_ok=True)
        metrics = result.metrics
        payload = {
            "version": "1.0",
            "input_config": str(args.input),
            "solver": {
                "algorithm": args.algorithm,
                "iterations_requested": int(args.heuristic_iterations),
                "iterations_reported": int(result.iterations),
                "seed": int(args.seed),
                "mode": str(result.mode),
            },
            "environment": {
                "bandwidth_mb_s": float(bandwidth),
                "latency_ms": float(latency),
                "weight_avg_latency": float(env.weight_avg_latency),
                "weight_max_latency": float(env.weight_max_latency),
                "weight_device_utilization": float(env.weight_device_utilization),
                "latency_limit": float(env.latency_limit),
                "batch_transfers": bool(env.batch_transfers),
                "pipeline_unroll": int(env.pipeline_unroll),
                "max_frame_latency_limit": float(env.max_frame_latency_limit),
                "solver_threads": int(solver_threads),
                "anneal_initial_temp": float(env.anneal_initial_temp),
                "anneal_final_temp": float(env.anneal_final_temp),
            },
            "baselines": {
                "all_device": _baseline_summary(all_device),
                "all_host": _baseline_summary(all_host),
            },
            "summary": {
                "latency_ms": float(metrics.latency),
                "max_frame_latency_ms": float(metrics.max_frame_latency),
                "initiation_interval_ms": float(metrics.initiation_interval),
                "loss": float(metrics.loss),
                "device_utilization": float(metrics.device_utilization),
                "host_utilization": float(metrics.host_utilization),
                "network_utilization": float(metrics.network_utilization),
                "pipeline_unroll": int(metrics.pipeline_unroll),
                "num_nodes": state.graph.number_of_nodes(),
                "num_edges": state.graph.number_of_edges(),
                "device_nodes": device_nodes,
                "host_nodes": host_nodes,
                "num_transfers": len(metrics.transfer_records),
            },
            "assignment": assignment,
            "nodes": _node_results(state.graph, assignment),
            "metrics": asdict(metrics),
        }
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"solve_result={output}")


if __name__ == "__main__":
    main()
