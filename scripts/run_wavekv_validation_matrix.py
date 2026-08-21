from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import sys


MODELS = ("molmoact2", "eo1", "pi0", "pi05", "smolvla")
HARDWARE = {
    "dual_rtx4090": {
        "host": "RTX_4090",
        "device": "RTX_4090",
        "gpu0": "RTX 4090 (Host label; Backbone/VLM in fix mode)",
        "gpu1": "RTX 4090 (Device label; Action/denoise in fix mode)",
    },
    "a100_rtx4090": {
        "host": "A100_80GB",
        "device": "RTX_4090",
        "gpu0": "A100 80GB (Host label; Backbone/VLM in fix mode)",
        "gpu1": "RTX 4090 (Device label; Action/denoise in fix mode)",
    },
}
NETWORKS = {
    "normal": {"bandwidth_mb_s": 1250.0, "latency_ms": 0.2},
    "fast": {"bandwidth_mb_s": 20000.0, "latency_ms": 0.01},
}


def _add_src(repo_root: Path) -> None:
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _assignment(graph, free_value: int) -> dict[str, int]:
    result = {}
    for node_id, attrs in graph.nodes(data=True):
        placement = str(attrs.get("placement", "free")).lower()
        if placement == "device":
            result[str(node_id)] = 0
        elif placement == "host":
            result[str(node_id)] = 1
        else:
            result[str(node_id)] = int(free_value)
    return result


def _summary(metrics, assignment: dict[str, int], num_nodes: int, num_edges: int) -> dict:
    return {
        "latency_ms": float(metrics.latency),
        "max_frame_latency_ms": float(metrics.max_frame_latency),
        "initiation_interval_ms": float(metrics.initiation_interval),
        "loss": float(metrics.loss),
        "device_utilization": float(metrics.device_utilization),
        "host_utilization": float(metrics.host_utilization),
        "network_utilization": float(metrics.network_utilization),
        "pipeline_unroll": int(metrics.pipeline_unroll),
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "device_nodes": sum(value == 0 for value in assignment.values()),
        "host_nodes": sum(value == 1 for value in assignment.values()),
        "num_transfers": len(metrics.transfer_records),
    }


def _baseline(metrics) -> dict:
    return {
        "latency_ms": float(metrics.latency),
        "max_frame_latency_ms": float(metrics.max_frame_latency),
        "initiation_interval_ms": float(metrics.initiation_interval),
        "device_utilization": float(metrics.device_utilization),
        "host_utilization": float(metrics.host_utilization),
        "network_utilization": float(metrics.network_utilization),
        "num_transfers": len(metrics.transfer_records),
    }


def _single_baseline_fields(baselines: dict, latency_ms: float) -> dict:
    device_ms = float(baselines["all_device"]["latency_ms"])
    host_ms = float(baselines["all_host"]["latency_ms"])
    if host_ms < device_ms:
        best_ms = host_ms
        resource = "GPU0/Host"
    else:
        best_ms = device_ms
        resource = "GPU1/Device"
    speedup_pct = (best_ms - float(latency_ms)) / best_ms * 100.0 if best_ms > 0 else 0.0
    return {
        "single_gpu0_host_latency_ms": host_ms,
        "single_gpu1_device_latency_ms": device_ms,
        "best_single_latency_ms": best_ms,
        "best_single_resource": resource,
        "speedup_vs_best_single_pct": speedup_pct,
    }


def _variant(base: dict, *, model: str, hardware: str, network: str, mode: str, unroll: int) -> dict:
    payload = deepcopy(base)
    net = NETWORKS[network]
    payload["environment"]["bandwidth"] = net["bandwidth_mb_s"]
    payload["environment"]["latency"] = net["latency_ms"]
    payload["environment"]["pipeline_unroll"] = int(unroll)
    payload["environment"]["weight_avg_latency"] = 1.0
    payload["environment"]["weight_max_latency"] = 0.0
    payload["environment"]["weight_device_utilization"] = 0.0
    for node in payload["nodes"]:
        if mode == "free":
            node["placement"] = "free"
            node["x"] = 1
        else:
            iterative = "::step" in str(node["id"])
            node["placement"] = "device" if iterative else "host"
            node["x"] = 0 if iterative else 1
    payload["experiment"] = {
        "name": "wavekv_validation_matrix_v1",
        "model": model,
        "hardware": hardware,
        "host_hardware": HARDWARE[hardware]["host"],
        "device_hardware": HARDWARE[hardware]["device"],
        "gpu0": HARDWARE[hardware]["gpu0"],
        "gpu1": HARDWARE[hardware]["gpu1"],
        "network": network,
        "placement_mode": mode,
        "fix_definition": "Backbone/VLM@GPU0 + Action/denoise@GPU1",
        "pipeline_unroll": int(unroll),
        "bandwidth_mb_s": net["bandwidth_mb_s"],
        "latency_ms": net["latency_ms"],
        "source_release_policy": "saturated_zero_period",
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate and run the MolmoAct2/EO-1 Wave-KV matrix.")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--hardware", nargs="+", choices=tuple(HARDWARE), default=list(HARDWARE))
    parser.add_argument("--networks", nargs="+", choices=tuple(NETWORKS), default=list(NETWORKS))
    parser.add_argument("--modes", nargs="+", choices=("free", "fix"), default=["free", "fix"])
    parser.add_argument("--unrolls", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    if any(value not in (1, 2) for value in args.unrolls):
        raise ValueError("This validation is intentionally limited to unroll 1 and 2")

    repo_root = args.repo_root.resolve()
    _add_src(repo_root)
    from coopinfer.evaluator import evaluate
    from coopinfer.model import load_from_json
    from coopinfer.solver import solve

    rows = []
    configs = []
    for model in args.models:
        for hardware in args.hardware:
            base_path = (
                repo_root / "examples" / "models" / model / hardware
                / f"{model}_{hardware}_base_coopinfer.json"
            )
            base = json.loads(base_path.read_text(encoding="utf-8"))
            for network in args.networks:
                for unroll in args.unrolls:
                    for mode in args.modes:
                        stem = f"{model}_{hardware}_{network}_{mode}_u{unroll}"
                        config = base_path.parent / network / f"{stem}_coopinfer.json"
                        payload = _variant(
                            base, model=model, hardware=hardware, network=network,
                            mode=mode, unroll=unroll,
                        )
                        config.parent.mkdir(parents=True, exist_ok=True)
                        config.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                        configs.append(str(config.relative_to(repo_root)).replace("\\", "/"))

                        output = (
                            repo_root / "results" / "wavekv_validation" / model / hardware / network
                            / f"{stem}_solve.json"
                        )
                        if output.exists() and not args.force:
                            saved = json.loads(output.read_text(encoding="utf-8"))
                            row = dict(saved["row"])
                            row.update(
                                _single_baseline_fields(
                                    saved["baselines"], row["latency_ms"]
                                )
                            )
                            rows.append(row)
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
                        all_device = evaluate(state.graph, _assignment(state.graph, 0), **common)
                        all_host = evaluate(state.graph, _assignment(state.graph, 1), **common)
                        if mode == "fix":
                            assignment = _assignment(state.graph, 1)
                            metrics = evaluate(state.graph, assignment, **common)
                            solver_info = {"algorithm": "Fixed Assignment", "iterations": 0, "seed": 7}
                        else:
                            solved = solve(
                                state.graph,
                                algorithm="Random Search",
                                heuristic_iterations=args.iterations,
                                seed=7,
                                **common,
                            )
                            assignment = {str(key): int(value) for key, value in solved.assignment.items()}
                            metrics = solved.metrics
                            solver_info = {
                                "algorithm": "Random Search",
                                "iterations": int(solved.iterations),
                                "iterations_requested": int(args.iterations),
                                "seed": 7,
                                "mode": solved.mode,
                            }
                        summary = _summary(
                            metrics, assignment, state.graph.number_of_nodes(), state.graph.number_of_edges()
                        )
                        row = {
                            "model": model,
                            "hardware": hardware,
                            "network": network,
                            "mode": mode,
                            "pipeline_unroll": unroll,
                            **summary,
                            "result": str(output.relative_to(repo_root)).replace("\\", "/"),
                            "config": str(config.relative_to(repo_root)).replace("\\", "/"),
                        }
                        baselines = {"all_device": _baseline(all_device), "all_host": _baseline(all_host)}
                        row.update(_single_baseline_fields(baselines, summary["latency_ms"]))
                        result = {
                            "version": "1.0",
                            "experiment": payload["experiment"],
                            "input_config": row["config"],
                            "solver": solver_info,
                            "baselines": baselines,
                            "summary": summary,
                            "row": row,
                            "assignment": assignment,
                            "metrics": asdict(metrics),
                        }
                        output.parent.mkdir(parents=True, exist_ok=True)
                        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
                        rows.append(row)
                        print(
                            f"{model} {hardware} {network} {mode} u{unroll}: "
                            f"lat={metrics.latency:.6f} ms II={metrics.initiation_interval:.6f} "
                            f"transfers={len(metrics.transfer_records)}",
                            flush=True,
                        )

    # Fixed configs constrain every node, so their evaluator-side all-host and
    # all-device values collapse to the same legal fixed assignment. Use the
    # matching free config as the source of true single-resource baselines for
    # both free and fix comparisons.
    free_references = {
        (row["model"], row["hardware"], row["network"], row["pipeline_unroll"]): row
        for row in rows
        if row["mode"] == "free"
    }
    for row in rows:
        key = (row["model"], row["hardware"], row["network"], row["pipeline_unroll"])
        reference = free_references[key]
        for field in (
            "single_gpu0_host_latency_ms",
            "single_gpu1_device_latency_ms",
            "best_single_latency_ms",
            "best_single_resource",
        ):
            row[field] = reference[field]
        best_ms = float(row["best_single_latency_ms"])
        row["speedup_vs_best_single_pct"] = (
            (best_ms - float(row["latency_ms"])) / best_ms * 100.0
            if best_ms > 0
            else 0.0
        )
        row["single_baseline_source"] = "matching free config all-resource evaluation"

    rows.sort(key=lambda row: (row["model"], row["hardware"], row["network"], row["mode"], row["pipeline_unroll"]))
    root = repo_root / "results" / "wavekv_validation"
    root.mkdir(parents=True, exist_ok=True)
    (root / "summary.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "experiment": "wavekv_validation_matrix_v1",
                "models": args.models,
                "hardware": args.hardware,
                "networks": {name: NETWORKS[name] for name in args.networks},
                "modes": args.modes,
                "unrolls": args.unrolls,
                "rows": rows,
                "configs": configs,
            }, indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    fields = [key for key in rows[0] if key not in {"config", "result"}] + ["config", "result"]
    with (root / "summary.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    print(f"completed_rows={len(rows)}")
    print(f"summary={root / 'summary.json'}")


if __name__ == "__main__":
    main()
