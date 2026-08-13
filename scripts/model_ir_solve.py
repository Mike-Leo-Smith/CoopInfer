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


def _assignment(graph, value: int) -> dict[str, int]:
    result: dict[str, int] = {}
    for node_id, attrs in graph.nodes(data=True):
        placement = str(attrs.get("placement", "free")).lower()
        if placement == "device":
            result[str(node_id)] = 0
        elif placement == "host":
            result[str(node_id)] = 1
        else:
            result[str(node_id)] = int(value)
    return result


def _default_output(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_coopinfer.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Send an already-costed fine ModelIR into CoopInfer without coarsening. "
            "This is the generic Full Graph reference path."
        )
    )
    parser.add_argument("input", type=Path, help="Costed ModelIR JSON")
    parser.add_argument("--output", type=Path, default=None)
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
    parser.add_argument("--skip-solve", action="store_true")
    args = parser.parse_args()

    _add_repo_src_to_path()

    from coopinfer.evaluator import evaluate
    from coopinfer.frontend import (
        identity_scheduling_ir,
        load_model_ir_json,
        to_coopinfer_payload,
    )
    from coopinfer.model import load_from_json, validate_graph
    from coopinfer.solver import solve

    model_ir = load_model_ir_json(args.input)
    scheduling_ir = identity_scheduling_ir(model_ir)
    payload = to_coopinfer_payload(
        scheduling_ir,
        bandwidth_mb_s=args.bandwidth_mb_s,
        latency_ms=args.latency_ms,
    )

    output = args.output or _default_output(args.input)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    state = load_from_json(output)
    validate_graph(state.graph, require_dag=True)

    print("===== generic Full Graph path =====")
    print(f"input={args.input}")
    print(f"output={output}")
    print(f"model_ir_nodes={len(model_ir.nodes)}")
    print(f"model_ir_edges={len(model_ir.edges)}")
    print(f"backend_nodes={state.graph.number_of_nodes()}")
    print(f"backend_edges={state.graph.number_of_edges()}")
    print(f"cost_source={model_ir.metadata.get('cost_source', 'unknown')}")
    print(
        "network="
        f"{args.bandwidth_mb_s} MB/s + {args.latency_ms} ms/transfer"
    )

    common = {
        "bandwidth": args.bandwidth_mb_s,
        "latency": args.latency_ms,
        "weight_avg_latency": 1.0,
        "weight_max_latency": 0.0,
        "weight_device_utilization": 0.0,
    }
    all_device = evaluate(
        state.graph,
        _assignment(state.graph, 0),
        **common,
    )
    all_host = evaluate(
        state.graph,
        _assignment(state.graph, 1),
        **common,
    )

    print("\n===== baselines =====")
    print(f"all_device_latency_ms={all_device.latency:.6f}")
    print(f"all_host_latency_ms={all_host.latency:.6f}")
    print(f"all_device_transfers={len(all_device.transfer_records)}")
    print(f"all_host_transfers={len(all_host.transfer_records)}")

    if args.skip_solve:
        print("\nsolver=skipped")
        return

    result = solve(
        state.graph,
        algorithm=args.algorithm,
        heuristic_iterations=args.heuristic_iterations,
        seed=args.seed,
        solver_threads=args.solver_threads,
        **common,
    )
    device_nodes = sum(1 for value in result.assignment.values() if value == 0)
    host_nodes = sum(1 for value in result.assignment.values() if value == 1)

    print("\n===== solver =====")
    print(f"mode={result.mode}")
    print(f"iterations={result.iterations}")
    print(f"latency_ms={result.metrics.latency:.6f}")
    print(f"device_nodes={device_nodes}")
    print(f"host_nodes={host_nodes}")
    print(f"transfers={len(result.metrics.transfer_records)}")


if __name__ == "__main__":
    main()
