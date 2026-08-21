from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys


DEFAULT_MODELS = (
    "smolvla",
    "pi0",
    "pi05",
    "pi0_fast",
    "xvla",
    "groot_n17",
)


def _add_repo_src_to_path(repo_root: Path) -> None:
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _assignment(graph, value: int) -> dict[str, int]:
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
    return {
        str(node_id): {
            "name": str(attrs.get("name", node_id)),
            "assignment": int(assignment[str(node_id)]),
            "resource": "GPU1/Device" if int(assignment[str(node_id)]) == 0 else "GPU0/Host",
            "placement_constraint": str(attrs.get("placement", "free")),
            "c_dev_ms": float(attrs.get("c_dev", 0.0)),
            "c_host_ms": float(attrs.get("c_host", 0.0)),
        }
        for node_id, attrs in graph.nodes(data=True)
    }


def _write_result(
    *,
    output: Path,
    input_config: Path,
    state,
    assignment: dict[str, int],
    metrics,
    mode: str,
    algorithm: str,
    iterations_requested: int,
    iterations_reported: int,
    all_device,
    all_host,
) -> None:
    env = state.environment
    device_nodes = sum(value == 0 for value in assignment.values())
    host_nodes = sum(value == 1 for value in assignment.values())
    payload = {
        "version": "1.0",
        "experiment": "dual_rtx4090_pipeline_matrix_v1",
        "input_config": str(input_config),
        "solver": {
            "algorithm": algorithm,
            "iterations_requested": int(iterations_requested),
            "iterations_reported": int(iterations_reported),
            "seed": 7,
            "mode": mode,
        },
        "environment": {
            "gpu0": "RTX_4090 (Host label)",
            "gpu1": "RTX_4090 (Device label)",
            "bandwidth_mb_s": float(env.bandwidth),
            "latency_ms": float(env.latency),
            "weight_avg_latency": float(env.weight_avg_latency),
            "weight_max_latency": float(env.weight_max_latency),
            "weight_device_utilization": float(env.weight_device_utilization),
            "pipeline_unroll": int(env.pipeline_unroll),
            "source_release_policy": "saturated_zero_period",
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
            "device_nodes": int(device_nodes),
            "host_nodes": int(host_nodes),
            "num_transfers": len(metrics.transfer_records),
        },
        "assignment": assignment,
        "nodes": _node_results(state.graph, assignment),
        "metrics": asdict(metrics),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the dual-RTX-4090 pipeline matrix.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--unrolls", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--modes", nargs="+", choices=("free", "split"), default=["free", "split"])
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    _add_repo_src_to_path(repo_root)
    from coopinfer.evaluator import evaluate
    from coopinfer.model import load_from_json
    from coopinfer.solver import solve

    completed = 0
    skipped = 0
    for model in args.models:
        model_rows = []
        for unroll in args.unrolls:
            for placement_mode in args.modes:
                config = (
                    repo_root
                    / "examples"
                    / "models"
                    / model
                    / "dual_rtx4090"
                    / f"{model}_dual4090_{placement_mode}_u{unroll}_coopinfer.json"
                )
                output = (
                    repo_root
                    / "results"
                    / "dual_rtx4090"
                    / model
                    / f"{model}_dual4090_{placement_mode}_u{unroll}_solve.json"
                )
                if output.exists() and not args.force:
                    payload = json.loads(output.read_text(encoding="utf-8"))
                    model_rows.append(payload["summary"] | {"mode": placement_mode})
                    skipped += 1
                    continue

                state = load_from_json(config)
                env = state.environment
                common = {
                    "bandwidth": env.bandwidth,
                    "latency": env.latency,
                    "weight_avg_latency": env.weight_avg_latency,
                    "weight_max_latency": env.weight_max_latency,
                    "weight_device_utilization": env.weight_device_utilization,
                    "batch_transfers": env.batch_transfers,
                    "pipeline_unroll": env.pipeline_unroll,
                }
                if placement_mode == "split":
                    assignment = _assignment(state.graph, 1)
                    metrics = evaluate(state.graph, assignment, **common)
                    # Every node is fixed in split configs, so constrained
                    # all-device/all-host baselines are identical to the one
                    # legal assignment. Avoid evaluating the 4.6k-node
                    # autoregressive graph three times.
                    all_device = metrics
                    all_host = metrics
                    algorithm = "Fixed Assignment"
                    mode = "Fixed VLM/prefill@GPU0 + iterative action@GPU1"
                    iterations_reported = 0
                else:
                    all_device = evaluate(state.graph, _assignment(state.graph, 0), **common)
                    all_host = evaluate(state.graph, _assignment(state.graph, 1), **common)
                    solved = solve(
                        state.graph,
                        algorithm="Random Search",
                        heuristic_iterations=args.iterations,
                        seed=7,
                        **common,
                    )
                    assignment = {str(key): int(value) for key, value in solved.assignment.items()}
                    metrics = solved.metrics
                    algorithm = "Random Search"
                    mode = solved.mode
                    iterations_reported = solved.iterations

                _write_result(
                    output=output,
                    input_config=config.relative_to(repo_root),
                    state=state,
                    assignment=assignment,
                    metrics=metrics,
                    mode=mode,
                    algorithm=algorithm,
                    iterations_requested=args.iterations if placement_mode == "free" else 0,
                    iterations_reported=iterations_reported,
                    all_device=all_device,
                    all_host=all_host,
                )
                row = {
                    "mode": placement_mode,
                    "pipeline_unroll": unroll,
                    "latency_ms": metrics.latency,
                    "max_frame_latency_ms": metrics.max_frame_latency,
                    "initiation_interval_ms": metrics.initiation_interval,
                    "device_utilization": metrics.device_utilization,
                    "host_utilization": metrics.host_utilization,
                    "network_utilization": metrics.network_utilization,
                    "num_transfers": len(metrics.transfer_records),
                    "device_nodes": sum(value == 0 for value in assignment.values()),
                    "host_nodes": sum(value == 1 for value in assignment.values()),
                }
                model_rows.append(row)
                completed += 1
                print(
                    f"{model} {placement_mode} u{unroll}: "
                    f"lat={metrics.latency:.6f} ms, "
                    f"II={metrics.initiation_interval:.6f} ms, "
                    f"transfers={len(metrics.transfer_records)}",
                    flush=True,
                )

        summary_path = repo_root / "results" / "dual_rtx4090" / model / "summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps(
                {
                    "model": model,
                    "experiment": "dual_rtx4090_pipeline_matrix_v1",
                    "rows": sorted(model_rows, key=lambda row: (row["pipeline_unroll"], row["mode"])),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print(f"completed={completed}")
    print(f"skipped={skipped}")


if __name__ == "__main__":
    main()
