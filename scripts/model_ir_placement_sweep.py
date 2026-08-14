from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Tuple


def _add_repo_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _layer_index(name: str) -> int | None:
    match = re.search(r"\.layers\[(\d+)\]", str(name))
    return int(match.group(1)) if match else None


def _find_target_layers(graph, name_contains: str) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    for node_id, attrs in graph.nodes(data=True):
        name = str(attrs.get("name", node_id))
        if name_contains not in name:
            continue
        index = _layer_index(name)
        if index is None:
            continue
        rows.append((index, str(node_id)))

    if not rows:
        raise ValueError(
            f"No layer nodes matched --target-name-contains={name_contains!r}."
        )

    rows.sort()
    indices = [index for index, _ in rows]
    if len(indices) != len(set(indices)):
        raise ValueError(
            "Target stack has duplicate layer indices; make --target-name-contains more specific."
        )
    expected = list(range(indices[0], indices[-1] + 1))
    if indices != expected:
        raise ValueError(
            f"Target layer indices are not contiguous: found={indices}, expected={expected}"
        )
    return rows


def _assignment_for_prefix_offload(
    graph,
    target_layers: List[Tuple[int, str]],
    offload_count: int,
) -> Dict[str, int]:
    # Host/A100 is the baseline for every compute node. The first k layers of
    # the selected target stack are moved to Device/RTX4090. This deliberately
    # creates a one-cut structured placement rather than an arbitrary per-layer
    # search result.
    assignment = {str(node_id): 1 for node_id in graph.nodes}
    for _, node_id in target_layers[:offload_count]:
        assignment[node_id] = 0
    return assignment


def _evaluate_assignment(graph, environment, assignment):
    from coopinfer.evaluator import evaluate

    return evaluate(
        graph,
        assignment,
        bandwidth=environment.bandwidth,
        latency=environment.latency,
        weight_avg_latency=environment.weight_avg_latency,
        weight_max_latency=environment.weight_max_latency,
        weight_device_utilization=environment.weight_device_utilization,
        batch_transfers=environment.batch_transfers,
        pipeline_unroll=environment.pipeline_unroll,
    )


def _result_row(
    *,
    offload_count: int,
    target_layers: List[Tuple[int, str]],
    assignment: Dict[str, int],
    result,
) -> Dict[str, Any]:
    device_target_layers = [
        index for index, node_id in target_layers if assignment[node_id] == 0
    ]
    network_active_ms = sum(
        max(0.0, float(record.finish) - float(record.start))
        for record in result.transfer_records
    )
    transfer_mb = sum(float(record.size_mb) for record in result.transfer_records)
    return {
        "offload_layers": int(offload_count),
        "device_layer_indices": device_target_layers,
        "latency_ms": float(result.latency),
        "max_frame_latency_ms": float(result.max_frame_latency),
        "device_utilization": float(result.device_utilization),
        "host_utilization": float(result.host_utilization),
        "network_utilization": float(result.network_utilization),
        "transfer_count": len(result.transfer_records),
        "transfer_size_sum_mb": float(transfer_mb),
        "network_active_ms": float(network_active_ms),
        "loss": float(result.loss),
    }


def _cross_device_edge_count(graph, assignment: Dict[str, int]) -> int:
    return sum(
        1
        for source, target in graph.edges
        if int(assignment[str(source)]) != int(assignment[str(target)])
    )


def _write_csv(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["device_layer_indices"] = ",".join(
                str(value) for value in row["device_layer_indices"]
            )
            writer.writerow({field: out[field] for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep structured layer placements on an exported CoopInfer DAG. "
            "All nodes stay on Host except the first k layers of one selected stack, "
            "which are placed on Device for k=0..N."
        )
    )
    parser.add_argument("input", type=Path, help="CoopInfer JSON configuration")
    parser.add_argument(
        "--target-name-contains",
        default="gemma_expert.model.layers",
        help=(
            "Substring identifying the layer stack to offload. Default targets the "
            "pi0.5 Action Expert stack."
        ),
    )
    parser.add_argument("--min-offload", type=int, default=0)
    parser.add_argument("--max-offload", type=int, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--json-output", type=Path, default=None)
    parser.add_argument(
        "--bandwidth-mb-s",
        type=float,
        default=None,
        help="Override bandwidth from the input JSON.",
    )
    parser.add_argument(
        "--latency-ms",
        type=float,
        default=None,
        help="Override per-transfer latency from the input JSON.",
    )
    args = parser.parse_args()

    _add_repo_src_to_path()
    from coopinfer.model import Environment, load_from_json, validate_environment

    state = load_from_json(args.input)
    graph = state.graph
    environment = state.environment
    if args.bandwidth_mb_s is not None or args.latency_ms is not None:
        environment = validate_environment(
            Environment(
                bandwidth=(
                    args.bandwidth_mb_s
                    if args.bandwidth_mb_s is not None
                    else environment.bandwidth
                ),
                latency=(
                    args.latency_ms
                    if args.latency_ms is not None
                    else environment.latency
                ),
                weight_avg_latency=environment.weight_avg_latency,
                weight_max_latency=environment.weight_max_latency,
                weight_device_utilization=environment.weight_device_utilization,
                latency_limit=environment.latency_limit,
                batch_transfers=environment.batch_transfers,
                pipeline_unroll=environment.pipeline_unroll,
                max_frame_latency_limit=environment.max_frame_latency_limit,
                solver_threads=environment.solver_threads,
                anneal_initial_temp=environment.anneal_initial_temp,
                anneal_final_temp=environment.anneal_final_temp,
            )
        )

    target_layers = _find_target_layers(graph, args.target_name_contains)
    layer_count = len(target_layers)
    min_offload = max(0, int(args.min_offload))
    max_offload = layer_count if args.max_offload is None else int(args.max_offload)
    if max_offload < min_offload or max_offload > layer_count:
        raise ValueError(
            f"Invalid sweep range [{min_offload}, {max_offload}] for {layer_count} layers."
        )

    rows: List[Dict[str, Any]] = []
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
        rows.append(row)

    best = min(rows, key=lambda row: row["latency_ms"])

    csv_output = args.csv_output or args.input.with_name(
        f"{args.input.stem}_placement_sweep.csv"
    )
    json_output = args.json_output or args.input.with_name(
        f"{args.input.stem}_placement_sweep.json"
    )
    _write_csv(csv_output, rows)
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(
            {
                "input": str(args.input),
                "target_name_contains": args.target_name_contains,
                "target_layers": [
                    {"index": index, "node_id": node_id}
                    for index, node_id in target_layers
                ],
                "environment": {
                    "bandwidth_mb_s": environment.bandwidth,
                    "latency_ms": environment.latency,
                    "batch_transfers": environment.batch_transfers,
                    "pipeline_unroll": environment.pipeline_unroll,
                },
                "sweep_rule": (
                    "all graph nodes on Host; first k layers of target stack on Device"
                ),
                "best": best,
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("===== structured Layer placement sweep =====")
    print(f"input={args.input}")
    print(f"target={args.target_name_contains}")
    print(f"target_layers={layer_count}")
    print(
        f"network={environment.bandwidth} MB/s + {environment.latency} ms/transfer "
        f"batch={environment.batch_transfers} unroll={environment.pipeline_unroll}"
    )
    print("rule=all Host except first k target layers on Device")
    print()
    print(
        " k | latency_ms | dev_util | host_util | net_util | xdev_edges | transfers | net_active_ms"
    )
    print("---+------------+----------+-----------+----------+------------+-----------+--------------")
    for row in rows:
        print(
            f"{row['offload_layers']:2d} | "
            f"{row['latency_ms']:10.4f} | "
            f"{row['device_utilization'] * 100:7.2f}% | "
            f"{row['host_utilization'] * 100:8.2f}% | "
            f"{row['network_utilization'] * 100:7.2f}% | "
            f"{row['cross_device_edges']:10d} | "
            f"{row['transfer_count']:9d} | "
            f"{row['network_active_ms']:12.4f}"
        )

    print()
    print(
        f"best_offload_layers={best['offload_layers']} "
        f"best_latency_ms={best['latency_ms']:.6f}"
    )
    print(f"csv={csv_output}")
    print(f"json={json_output}")
    print("solver_search=not_run (fixed-placement evaluation only)")


if __name__ == "__main__":
    main()
