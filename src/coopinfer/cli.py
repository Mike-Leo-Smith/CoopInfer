from __future__ import annotations

import argparse
import csv
import glob
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .model import Environment, load_from_json, validate_environment
from .reporting import export_pipeline_pdf, result_to_mapping
from .solver import solve


SCHEDULER_ALIASES = {
    "auto": "auto",
    "enumerate": "enumerate",
    "brute": "enumerate",
    "random": "random",
    "random_search": "random",
    "anneal": "simulated_annealing",
    "annealing": "simulated_annealing",
    "simulated_annealing": "simulated_annealing",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coopinfer solve",
        description="Solve one or more CoopInfer JSON configs without launching Qt.",
    )
    parser.add_argument(
        "configs",
        nargs="+",
        help="JSON files, directories, or glob patterns. Directories are scanned recursively.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("coopinfer-results"),
        help="directory for per-config result.json/PDF and aggregate summaries",
    )
    parser.add_argument(
        "--scheduler",
        default="auto",
        help=(
            "auto, enumerate, random, or simulated_annealing "
            "(large auto graphs use simulated annealing)"
        ),
    )
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--pipeline-format",
        choices=("pdf", "none"),
        default="pdf",
        help="solved pipeline artifact format (default: pdf)",
    )
    parser.add_argument("--fail-fast", action="store_true")

    bandwidth_group = parser.add_mutually_exclusive_group()
    bandwidth_group.add_argument(
        "--bandwidth",
        type=float,
        help="override effective bandwidth in MB/s (the native config unit)",
    )
    bandwidth_group.add_argument(
        "--bandwidth-mbps",
        type=float,
        help="override effective bandwidth in Mbit/s; converted internally to MB/s",
    )
    parser.add_argument("--latency", type=float, help="override one-way link latency in ms")
    parser.add_argument("--weight-avg-latency", type=float)
    parser.add_argument("--weight-max-latency", type=float)
    parser.add_argument("--weight-device-utilization", type=float)
    parser.add_argument(
        "--latency-limit",
        type=float,
        help="override amortized pipeline-span cap in ms",
    )
    parser.add_argument(
        "--max-frame-latency-limit",
        type=float,
        help="override worst-frame E2E cap in ms",
    )
    parser.add_argument(
        "--initiation-interval-limit",
        type=float,
        help="override steady-state output interval cap in ms",
    )
    parser.add_argument(
        "--batch-transfers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override transfer batching",
    )
    parser.add_argument("--pipeline-unroll", type=int)
    parser.add_argument("--solver-threads", type=int)
    parser.add_argument("--anneal-initial-temp", type=float)
    parser.add_argument("--anneal-final-temp", type=float)
    return parser


def normalize_scheduler(value: str) -> str:
    key = value.strip().lower().replace("-", "_").replace(" ", "_")
    try:
        return SCHEDULER_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown scheduler {value!r}; choose one of {sorted(SCHEDULER_ALIASES)}."
        ) from exc


def expand_config_paths(values: Sequence[str]) -> list[Path]:
    discovered: list[Path] = []
    for value in values:
        matches = [Path(item) for item in glob.glob(value, recursive=True)]
        if not matches:
            matches = [Path(value)]
        for match in matches:
            if match.is_dir():
                discovered.extend(
                    path
                    for path in sorted(match.rglob("*.json"))
                    if path.name not in {"manifest.json", "summary.json"}
                )
            elif match.is_file():
                discovered.append(match)
            else:
                raise ValueError(f"Config path does not exist: {match}")
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in discovered:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(resolved)
    if not unique:
        raise ValueError("No JSON configs were found.")
    return unique


def apply_environment_overrides(
    environment: Environment, args: argparse.Namespace
) -> Environment:
    updates: dict[str, Any] = {}
    direct_fields = {
        "bandwidth": "bandwidth",
        "latency": "latency",
        "weight_avg_latency": "weight_avg_latency",
        "weight_max_latency": "weight_max_latency",
        "weight_device_utilization": "weight_device_utilization",
        "latency_limit": "latency_limit",
        "max_frame_latency_limit": "max_frame_latency_limit",
        "initiation_interval_limit": "initiation_interval_limit",
        "batch_transfers": "batch_transfers",
        "pipeline_unroll": "pipeline_unroll",
        "solver_threads": "solver_threads",
        "anneal_initial_temp": "anneal_initial_temp",
        "anneal_final_temp": "anneal_final_temp",
    }
    for argument, field in direct_fields.items():
        value = getattr(args, argument, None)
        if value is not None:
            updates[field] = value
    if getattr(args, "bandwidth_mbps", None) is not None:
        updates["bandwidth"] = args.bandwidth_mbps / 8.0
    return validate_environment(replace(environment, **updates))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        scheduler = normalize_scheduler(args.scheduler)
        config_paths = expand_config_paths(args.configs)
    except ValueError as exc:
        parser.error(str(exc))

    if args.iterations < 0:
        parser.error("--iterations must be non-negative")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    used_slugs: set[str] = set()
    had_error = False

    for config_path in config_paths:
        slug = _unique_slug(_slug(config_path.stem), used_slugs)
        config_output = output_dir / slug
        config_output.mkdir(parents=True, exist_ok=True)
        metadata: Mapping[str, Any] = {}
        try:
            with config_path.open("r", encoding="utf-8") as file:
                raw_config = json.load(file)
            metadata = raw_config.get("metadata", {})
            if not isinstance(metadata, Mapping):
                metadata = {}
            state = load_from_json(config_path)
            environment = apply_environment_overrides(state.environment, args)
            result = solve(
                state.graph,
                bandwidth=environment.bandwidth,
                latency=environment.latency,
                weight_avg_latency=environment.weight_avg_latency,
                weight_max_latency=environment.weight_max_latency,
                weight_device_utilization=environment.weight_device_utilization,
                algorithm=scheduler,
                heuristic_iterations=args.iterations,
                seed=args.seed,
                latency_limit=environment.latency_limit,
                batch_transfers=environment.batch_transfers,
                pipeline_unroll=environment.pipeline_unroll,
                max_frame_latency_limit=environment.max_frame_latency_limit,
                initiation_interval_limit=environment.initiation_interval_limit,
                solver_threads=environment.solver_threads,
                anneal_initial_temp=environment.anneal_initial_temp,
                anneal_final_temp=environment.anneal_final_temp,
            )
            record = result_to_mapping(
                config_path=config_path,
                graph=state.graph,
                environment=environment,
                result=result,
                metadata=metadata,
            )
            result_path = config_output / "result.json"
            _write_json(result_path, record)
            pdf_path: Path | None = None
            if args.pipeline_format == "pdf":
                pdf_path = config_output / "pipeline.pdf"
                export_pipeline_pdf(
                    pdf_path,
                    config_name=config_path.name,
                    graph=state.graph,
                    environment=environment,
                    result=result,
                    metadata=metadata,
                )
            summary = _success_summary(
                config_path=config_path,
                environment=environment,
                result_record=record,
                result_path=result_path,
                pdf_path=pdf_path,
            )
            summaries.append(summary)
            print(
                f"OK {config_path.name}: {result.mode}; "
                f"mean-frame={result.metrics.mean_frame_latency:.2f} ms, "
                f"max-frame={result.metrics.max_frame_latency:.2f} ms, "
                f"II={result.metrics.initiation_interval:.2f} ms"
            )
        except Exception as exc:
            had_error = True
            summary = {
                "status": "error",
                "config": str(config_path),
                "config_name": config_path.name,
                "batch_size": metadata.get("batch_size", ""),
                "compression": metadata.get("compression", ""),
                "network_profile": metadata.get("network_profile", ""),
                "nominal_bandwidth_mbps": metadata.get(
                    "nominal_bandwidth_mbps", ""
                ),
                "error": str(exc),
            }
            summaries.append(summary)
            _write_json(config_output / "error.json", summary)
            print(f"ERROR {config_path.name}: {exc}", file=sys.stderr)
            if args.fail_fast:
                break

    _write_json(output_dir / "summary.json", summaries)
    _write_summary_csv(output_dir / "summary.csv", summaries)
    print(f"Wrote {len(summaries)} result(s) to {output_dir}")
    return 2 if had_error else 0


def _success_summary(
    *,
    config_path: Path,
    environment: Environment,
    result_record: Mapping[str, Any],
    result_path: Path,
    pdf_path: Path | None,
) -> dict[str, Any]:
    metadata = result_record["metadata"]
    metrics = result_record["metrics"]
    solver = result_record["solver"]
    return {
        "status": "ok",
        "config": str(config_path),
        "config_name": config_path.name,
        "batch_size": metadata.get("batch_size", ""),
        "compression": metadata.get("compression", ""),
        "network_profile": metadata.get("network_profile", ""),
        "nominal_bandwidth_mbps": metadata.get("nominal_bandwidth_mbps", ""),
        "effective_bandwidth_mbps": environment.bandwidth * 8.0,
        "effective_bandwidth_mb_s": environment.bandwidth,
        "scheduler": solver["mode"],
        "iterations": solver["iterations"],
        "amortized_pipeline_span_ms": metrics["amortized_pipeline_span_ms"],
        # Compatibility alias for pre-0.2 CSV consumers.
        "average_latency_ms": metrics["average_latency_ms"],
        "mean_frame_latency_ms": metrics["mean_frame_latency_ms"],
        "max_frame_latency_ms": metrics["max_frame_latency_ms"],
        "initiation_interval_ms": metrics["initiation_interval_ms"],
        "device_utilization": metrics["device_utilization"],
        "host_utilization": metrics["host_utilization"],
        "network_utilization": metrics["network_utilization"],
        "loss": metrics["loss"],
        "placement_stage_count": len(result_record["placement_stages"]),
        "result_json": str(result_path),
        "pipeline_pdf": str(pdf_path) if pdf_path else "",
    }


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")


def _write_summary_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    materialized = list(rows)
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-._")
    return slug or "config"


def _unique_slug(value: str, used: set[str]) -> str:
    candidate = value
    suffix = 2
    while candidate in used:
        candidate = f"{value}-{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate
