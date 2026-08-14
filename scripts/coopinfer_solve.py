from __future__ import annotations

import argparse
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

    common = {
        "bandwidth": bandwidth,
        "latency": latency,
        "weight_avg_latency": env.weight_avg_latency,
        "weight_max_latency": env.weight_max_latency,
        "weight_device_utilization": env.weight_device_utilization,
    }

    all_device = evaluate(state.graph, _assignment(state.graph, 0), **common)
    all_host = evaluate(state.graph, _assignment(state.graph, 1), **common)

    print("===== Stage 4: CoopInfer Solver =====")
    print(f"input={args.input}")
    print(f"nodes={state.graph.number_of_nodes()}")
    print(f"edges={state.graph.number_of_edges()}")
    print(f"network={bandwidth} MB/s + {latency} ms/transfer")
    print(f"all_device_latency_ms={all_device.latency:.6f}")
    print(f"all_host_latency_ms={all_host.latency:.6f}")

    result = solve(
        state.graph,
        algorithm=args.algorithm,
        heuristic_iterations=args.heuristic_iterations,
        seed=args.seed,
        solver_threads=solver_threads,
        anneal_initial_temp=env.anneal_initial_temp,
        anneal_final_temp=env.anneal_final_temp,
        **common,
    )

    device_nodes = sum(1 for value in result.assignment.values() if value == 0)
    host_nodes = sum(1 for value in result.assignment.values() if value == 1)
    print("\n===== best schedule =====")
    print(f"mode={result.mode}")
    print(f"iterations={result.iterations}")
    print(f"latency_ms={result.metrics.latency:.6f}")
    print(f"device_nodes={device_nodes}")
    print(f"host_nodes={host_nodes}")
    print(f"transfers={len(result.metrics.transfer_records)}")


if __name__ == "__main__":
    main()
