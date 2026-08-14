from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List


def _add_repo_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _add_script_dir_to_path() -> None:
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))


def _parse_float_list(value: str, *, label: str, allow_zero: bool) -> List[float]:
    rows: List[float] = []
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        try:
            number = float(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid {label} value: {token!r}") from exc
        if allow_zero:
            if number < 0:
                raise argparse.ArgumentTypeError(f"{label} values must be >= 0")
        elif number <= 0:
            raise argparse.ArgumentTypeError(f"{label} values must be > 0")
        rows.append(number)
    if not rows:
        raise argparse.ArgumentTypeError(f"At least one {label} value is required")
    return rows


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep network conditions and structured layer placements on a CoopInfer DAG. "
            "For each bandwidth/latency pair, evaluate k=0..N where all graph nodes stay "
            "on Host except the first k layers of one selected stack on Device."
        )
    )
    parser.add_argument("input", type=Path, help="CoopInfer JSON configuration")
    parser.add_argument(
        "--target-name-contains",
        default="gemma_expert.model.layers",
        help="Substring identifying the layer stack to offload.",
    )
    parser.add_argument(
        "--bandwidths",
        default="100,300,500,750,1000,1250,2500,5000,10000",
        help="Comma-separated MB/s values.",
    )
    parser.add_argument(
        "--latencies",
        default="0,0.1,0.2,0.5,1,2,5",
        help="Comma-separated per-transfer latency values in ms.",
    )
    parser.add_argument("--min-offload", type=int, default=0)
    parser.add_argument("--max-offload", type=int, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--detail-csv", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    args = parser.parse_args()

    bandwidths = _parse_float_list(args.bandwidths, label="bandwidth", allow_zero=False)
    latencies = _parse_float_list(args.latencies, label="latency", allow_zero=True)

    _add_repo_src_to_path()
    _add_script_dir_to_path()

    from coopinfer.model import Environment, load_from_json, validate_environment
    from model_ir_placement_sweep import (
        _assignment_for_prefix_offload,
        _cross_device_edge_count,
        _evaluate_assignment,
        _find_target_layers,
        _result_row,
    )

    state = load_from_json(args.input)
    graph = state.graph
    base_environment = state.environment
    target_layers = _find_target_layers(graph, args.target_name_contains)
    layer_count = len(target_layers)

    min_offload = max(0, int(args.min_offload))
    max_offload = layer_count if args.max_offload is None else int(args.max_offload)
    if max_offload < min_offload or max_offload > layer_count:
        raise ValueError(
            f"Invalid sweep range [{min_offload}, {max_offload}] for {layer_count} layers."
        )

    summary_rows: List[Dict[str, Any]] = []
    detail_rows: List[Dict[str, Any]] = []

    for latency_ms in latencies:
        for bandwidth_mb_s in bandwidths:
            environment = validate_environment(
                Environment(
                    bandwidth=bandwidth_mb_s,
                    latency=latency_ms,
                    weight_avg_latency=base_environment.weight_avg_latency,
                    weight_max_latency=base_environment.weight_max_latency,
                    weight_device_utilization=base_environment.weight_device_utilization,
                    latency_limit=base_environment.latency_limit,
                    batch_transfers=base_environment.batch_transfers,
                    pipeline_unroll=base_environment.pipeline_unroll,
                    max_frame_latency_limit=base_environment.max_frame_latency_limit,
                    solver_threads=base_environment.solver_threads,
                    anneal_initial_temp=base_environment.anneal_initial_temp,
                    anneal_final_temp=base_environment.anneal_final_temp,
                )
            )

            rows_for_condition: List[Dict[str, Any]] = []
            for offload_count in range(min_offload, max_offload + 1):
                assignment = _assignment_for_prefix_offload(
                    graph,
                    target_layers,
                    offload_count,
                )
                result = _evaluate_assignment(graph, environment, assignment)
                row = _result_row(
                    offload_count=offload_count,
                    target_layers=target_layers,
                    assignment=assignment,
                    result=result,
                )
                row["cross_device_edges"] = _cross_device_edge_count(graph, assignment)
                detailed = {
                    "bandwidth_mb_s": float(bandwidth_mb_s),
                    "latency_ms": float(latency_ms),
                    **row,
                }
                rows_for_condition.append(detailed)
                detail_rows.append(detailed)

            baseline = next(
                row for row in rows_for_condition if row["offload_layers"] == min_offload
            )
            if min_offload != 0:
                # A nonzero --min-offload is supported for focused diagnostics, but the
                # reported reference is then the first evaluated k rather than all-Host.
                reference_label = f"k={min_offload}"
            else:
                reference_label = "all-host-k0"

            best = min(rows_for_condition, key=lambda row: row["latency_ms"])
            gain_ms = float(baseline["latency_ms"] - best["latency_ms"])
            speedup_pct = (
                gain_ms / float(baseline["latency_ms"]) * 100.0
                if float(baseline["latency_ms"]) > 0
                else 0.0
            )
            summary_rows.append(
                {
                    "bandwidth_mb_s": float(bandwidth_mb_s),
                    "latency_ms": float(latency_ms),
                    "reference": reference_label,
                    "baseline_latency_ms": float(baseline["latency_ms"]),
                    "best_offload_layers": int(best["offload_layers"]),
                    "best_latency_ms": float(best["latency_ms"]),
                    "gain_ms": gain_ms,
                    "speedup_pct": speedup_pct,
                    "best_device_utilization": float(best["device_utilization"]),
                    "best_host_utilization": float(best["host_utilization"]),
                    "best_network_utilization": float(best["network_utilization"]),
                    "best_cross_device_edges": int(best["cross_device_edges"]),
                    "best_transfer_count": int(best["transfer_count"]),
                    "best_network_active_ms": float(best["network_active_ms"]),
                }
            )

    summary_csv = args.summary_csv or args.input.with_name(
        f"{args.input.stem}_network_placement_summary.csv"
    )
    detail_csv = args.detail_csv or args.input.with_name(
        f"{args.input.stem}_network_placement_detail.csv"
    )
    json_output = args.json_output or args.input.with_name(
        f"{args.input.stem}_network_placement_sweep.json"
    )

    summary_fields = [
        "bandwidth_mb_s",
        "latency_ms",
        "reference",
        "baseline_latency_ms",
        "best_offload_layers",
        "best_latency_ms",
        "gain_ms",
        "speedup_pct",
        "best_device_utilization",
        "best_host_utilization",
        "best_network_utilization",
        "best_cross_device_edges",
        "best_transfer_count",
        "best_network_active_ms",
    ]
    detail_fields = [
        "bandwidth_mb_s",
        "latency_ms",
        "offload_layers",
        "latency_ms",
        "max_frame_latency_ms",
        "device_utilization",
        "host_utilization",
        "network_utilization",
        "cross_device_edges",
        "transfer_count",
        "transfer_size_sum_mb",
        "network_active_ms",
        "loss",
        "device_layer_indices",
    ]

    # detail_rows has two latency concepts, so use an unambiguous column name in the
    # serialized detailed table: network_latency_ms vs E2E latency_ms.
    normalized_detail_rows: List[Dict[str, Any]] = []
    for row in detail_rows:
        normalized_detail_rows.append(
            {
                "bandwidth_mb_s": row["bandwidth_mb_s"],
                "network_latency_ms": row["latency_ms"],
                "offload_layers": row["offload_layers"],
                "e2e_latency_ms": row["latency_ms"],
                "max_frame_latency_ms": row["max_frame_latency_ms"],
                "device_utilization": row["device_utilization"],
                "host_utilization": row["host_utilization"],
                "network_utilization": row["network_utilization"],
                "cross_device_edges": row["cross_device_edges"],
                "transfer_count": row["transfer_count"],
                "transfer_size_sum_mb": row["transfer_size_sum_mb"],
                "network_active_ms": row["network_active_ms"],
                "loss": row["loss"],
                "device_layer_indices": ",".join(
                    str(value) for value in row["device_layer_indices"]
                ),
            }
        )
    detail_fields = [
        "bandwidth_mb_s",
        "network_latency_ms",
        "offload_layers",
        "e2e_latency_ms",
        "max_frame_latency_ms",
        "device_utilization",
        "host_utilization",
        "network_utilization",
        "cross_device_edges",
        "transfer_count",
        "transfer_size_sum_mb",
        "network_active_ms",
        "loss",
        "device_layer_indices",
    ]

    _write_csv(summary_csv, summary_rows, summary_fields)
    _write_csv(detail_csv, normalized_detail_rows, detail_fields)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(
            {
                "input": str(args.input),
                "target_name_contains": args.target_name_contains,
                "target_layer_count": layer_count,
                "bandwidths_mb_s": bandwidths,
                "latencies_ms": latencies,
                "offload_range": [min_offload, max_offload],
                "sweep_rule": (
                    "for each network condition, all graph nodes stay on Host except "
                    "the first k target-stack layers on Device"
                ),
                "summary": summary_rows,
                "detail": normalized_detail_rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("===== network x Layer placement sweep =====")
    print(f"input={args.input}")
    print(f"target={args.target_name_contains}")
    print(f"target_layers={layer_count}")
    print(f"bandwidths_mb_s={bandwidths}")
    print(f"latencies_ms={latencies}")
    print(f"offload_range={min_offload}..{max_offload}")
    print()
    print(
        " bandwidth | latency | best_k | best_e2e | gain_ms | speedup | net_util"
    )
    print(
        "-----------+---------+--------+----------+---------+---------+---------"
    )
    for row in summary_rows:
        print(
            f"{row['bandwidth_mb_s']:10.1f} | "
            f"{row['latency_ms']:7.2f} | "
            f"{row['best_offload_layers']:6d} | "
            f"{row['best_latency_ms']:8.4f} | "
            f"{row['gain_ms']:7.4f} | "
            f"{row['speedup_pct']:6.2f}% | "
            f"{row['best_network_utilization'] * 100:6.2f}%"
        )

    print()
    print(f"summary_csv={summary_csv}")
    print(f"detail_csv={detail_csv}")
    print(f"json={json_output}")
    print("solver_search=not_run (fixed-placement evaluation only)")


if __name__ == "__main__":
    main()
