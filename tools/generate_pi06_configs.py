#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


IMAGE_NODE_TEMPLATE = {
    "id": "v1",
    "name": "Images",
    "c_dev": 0.0,
    "c_host": 0.0,
    "fixed_dev": True,
    "source_period_ms": 100.0,
    "source_phase_ms": 0.0,
}
VIT_NODE_TEMPLATE = {
    "id": "v2",
    "name": "ViT",
    "c_dev": 13.6,
    "c_host": 3.1,
    "fixed_dev": False,
    "source_period_ms": 0.0,
    "source_phase_ms": 0.0,
}
LLM_NODE_TEMPLATE = {
    "id": "v3",
    "name": "LLM",
    "c_dev": 3.694,
    "c_host": 0.8235,
    "fixed_dev": False,
    "source_period_ms": 0.0,
    "source_phase_ms": 0.0,
}
DIT_NODE_TEMPLATE = {
    "id": "v4",
    "name": "DiT",
    "c_dev": 7.4,
    "c_host": 3.3,
    "fixed_dev": False,
    "source_period_ms": 0.0,
    "source_phase_ms": 0.0,
}
ACTION_NODE_TEMPLATE = {
    "id": "v5",
    "name": "Actions",
    "c_dev": 1.0,
    "c_host": 0.0,
    "fixed_dev": True,
    "source_period_ms": 0.0,
    "source_phase_ms": 0.0,
}
OBSERVATION_NODE_TEMPLATE = {
    "id": "v6",
    "name": "Observations",
    "c_dev": 0.0,
    "c_host": 0.0,
    "fixed_dev": True,
    "source_period_ms": 100.0,
    "source_phase_ms": 0.0,
}

LLM_LAYER_COUNT = 34
IMAGE_TO_VIT_SIZE = 0.0264
VIT_TO_LLM_SIZE = 1.6875
LLM_TO_DIT_SIZE = 0.7324
LLM_TO_LLM_SIZE = 3.6621
OBSERVATION_TO_VIT_SIZE = 5.34e-05
DIT_TO_ACTION_EDGE_SIZE = 0.01335
EFFECTIVENESS = 1.25
ROUND_DIGITS = 8

COMPRESSION = {
    "none": {
        "fp4": 0.0,
        "token": 0.0,
        "steps": 4,
        "transmission": 1.0,
    },
    "conservative": {
        "fp4": 0.6,
        "token": 0.3,
        "steps": 4,
        "transmission": 1.0,
    },
    "normal": {
        "fp4": 0.75,
        "token": 0.5,
        "steps": 2,
        "transmission": 1.0,
    },
    "aggressive": {
        "fp4": 1.0,
        "token": 0.6,
        "steps": 1,
        "transmission": 0.5,
    },
}


@dataclass(frozen=True)
class NetworkProfile:
    name: str
    nominal_mbps: float
    efficiency: float = 0.85

    @property
    def effective_mbps(self) -> float:
        return self.nominal_mbps * self.efficiency

    @property
    def effective_mb_s(self) -> float:
        return self.effective_mbps / 8.0


NETWORK_PROFILES = {
    profile.name: profile
    for profile in (
        NetworkProfile("sparklink2-single", 600.0),
        NetworkProfile("sparklink2-dual", 1200.0),
        NetworkProfile("sparklink2-single-3x", 1800.0),
        NetworkProfile("sparklink2-single-4x", 2400.0),
        NetworkProfile("future-1600", 1600.0),
        NetworkProfile("future-2400", 2400.0),
        NetworkProfile("future-3200", 3200.0),
    )
}
DEFAULT_NETWORK_PROFILES = ("sparklink2-single", "sparklink2-dual")


def rounded(value: float) -> float:
    return round(value, ROUND_DIGITS)


def scale_host_node_batch_time(c_host: float, batch_size: int) -> float:
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    scale = (
        batch_size ** math.log2(1.5)
        if batch_size <= 4
        else batch_size * (9.0 / 16.0)
    )
    return c_host * scale


def scale_vit_node_time(t_vit: float, compression: dict[str, float]) -> float:
    fp4_ratio = compression["fp4"]
    return (t_vit * (1.0 - fp4_ratio) + 0.5 * t_vit * fp4_ratio) * EFFECTIVENESS


def scale_llm_node_time(t_llm: float, compression: dict[str, float]) -> float:
    fp4_ratio = compression["fp4"]
    token_ratio = compression["token"]
    return (
        (t_llm * (1.0 - fp4_ratio) + 0.5 * t_llm * fp4_ratio)
        * (1.0 - token_ratio)
        * EFFECTIVENESS
    )


def scale_dit_node_time(t_dit: float, compression: dict[str, float]) -> float:
    fp4_ratio = compression["fp4"]
    token_ratio = compression["token"]
    return (
        (t_dit * (1.0 - fp4_ratio) + 0.5 * t_dit * fp4_ratio)
        * (1.0 - token_ratio)
        * (compression["steps"] / 4.0)
        * EFFECTIVENESS
        * EFFECTIVENESS
    )


def make_compute_node(
    template: dict[str, Any],
    batch_size: int,
    compression: dict[str, float],
    scaler: Callable[[float, dict[str, float]], float],
) -> dict[str, Any]:
    node = deepcopy(template)
    node["c_dev"] = rounded(scaler(float(template["c_dev"]), compression))
    compressed_host = scaler(float(template["c_host"]), compression)
    node["c_host"] = rounded(scale_host_node_batch_time(compressed_host, batch_size))
    return node


def make_edge(source: str, target: str, size: float) -> dict[str, Any]:
    return {"source": source, "target": target, "size": rounded(size)}


def build_config(
    batch_size: int,
    compression_strategy: str,
    network: NetworkProfile,
    *,
    target_hz: float = 10.0,
    latency_ms: float = 0.0,
    latency_budget_ms: float = 95.0,
    pipeline_unroll: int = 8,
) -> dict[str, Any]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if compression_strategy not in COMPRESSION:
        raise ValueError(f"Unknown compression strategy: {compression_strategy}")
    if target_hz <= 0:
        raise ValueError("target_hz must be positive")
    compression = COMPRESSION[compression_strategy]
    transmission = compression["transmission"]
    period_ms = 1000.0 / target_hz

    image = deepcopy(IMAGE_NODE_TEMPLATE)
    observation = deepcopy(OBSERVATION_NODE_TEMPLATE)
    image["source_period_ms"] = rounded(period_ms)
    observation["source_period_ms"] = rounded(period_ms)
    vit = make_compute_node(
        VIT_NODE_TEMPLATE, batch_size, compression, scale_vit_node_time
    )
    dit = make_compute_node(
        DIT_NODE_TEMPLATE, batch_size, compression, scale_dit_node_time
    )
    llm_nodes = []
    for layer in range(LLM_LAYER_COUNT):
        node = make_compute_node(
            LLM_NODE_TEMPLATE, batch_size, compression, scale_llm_node_time
        )
        node["id"] = f"v3-{layer}"
        node["name"] = f"LLM-{layer}"
        llm_nodes.append(node)

    edges = [
        make_edge("v1", "v2", IMAGE_TO_VIT_SIZE * batch_size),
        make_edge(
            "v6",
            "v2",
            OBSERVATION_TO_VIT_SIZE * batch_size,
        ),
        make_edge(
            "v2",
            "v3-0",
            VIT_TO_LLM_SIZE * batch_size * transmission,
        ),
    ]
    for layer in range(LLM_LAYER_COUNT):
        node_id = f"v3-{layer}"
        if layer + 1 < LLM_LAYER_COUNT:
            edges.append(
                make_edge(
                    node_id,
                    f"v3-{layer + 1}",
                    LLM_TO_LLM_SIZE * batch_size * transmission,
                )
            )
        edges.append(
            make_edge(
                node_id,
                "v4",
                LLM_TO_DIT_SIZE * batch_size * transmission,
            )
        )
    edges.append(
        make_edge("v4", "v5", DIT_TO_ACTION_EDGE_SIZE * batch_size)
    )

    config = {
        "version": "1.0",
        "metadata": {
            "model": "pi0.6-like",
            "host": "960PR-lite",
            "device": "1920",
            "batch_size": batch_size,
            "compression": compression_strategy,
            "compression_parameters": deepcopy(compression),
            "network_profile": network.name,
            "nominal_bandwidth_mbps": network.nominal_mbps,
            "link_efficiency": network.efficiency,
            "effective_bandwidth_mbps": network.effective_mbps,
            "effective_bandwidth_mb_s": network.effective_mb_s,
            "target_hz": target_hz,
            "latency_budget_ms": latency_budget_ms,
            "transfer_payloads_scale_with_batch": True,
        },
        "nodes": [
            image,
            vit,
            *llm_nodes,
            dit,
            deepcopy(ACTION_NODE_TEMPLATE),
            observation,
        ],
        "edges": edges,
        "environment": {
            "bandwidth": rounded(network.effective_mb_s),
            "latency": latency_ms,
            "weight_avg_latency": 0.1,
            "weight_max_latency": 0.1,
            "weight_device_utilization": 1.0,
            # 10 Hz is a steady-state cadence requirement, so constrain II;
            # per-frame responsiveness is protected by the 95 ms E2E budget.
            "latency_limit": 0.0,
            "max_frame_latency_limit": latency_budget_ms,
            "initiation_interval_limit": 1000.0 / target_hz,
            "batch_transfers": True,
            "pipeline_unroll": pipeline_unroll,
            "solver_threads": 0,
            "anneal_initial_temp": 1.0,
            "anneal_final_temp": 0.01,
        },
    }
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    nodes = config["nodes"]
    edges = config["edges"]
    ids = [node["id"] for node in nodes]
    id_set = set(ids)
    if len(nodes) != LLM_LAYER_COUNT + 5:
        raise ValueError("unexpected node count")
    if len(edges) != 2 * LLM_LAYER_COUNT + 3:
        raise ValueError("unexpected edge count")
    if len(ids) != len(id_set):
        raise ValueError("duplicate node ID")
    if any("x" in node for node in nodes):
        raise ValueError("generated configs must not contain solver placement field x")
    pairs: set[tuple[str, str]] = set()
    for edge in edges:
        pair = (edge["source"], edge["target"])
        if pair[0] not in id_set or pair[1] not in id_set:
            raise ValueError(f"dangling edge: {pair}")
        if pair in pairs:
            raise ValueError(f"duplicate edge: {pair}")
        if float(edge["size"]) < 0:
            raise ValueError(f"negative edge size: {pair}")
        pairs.add(pair)
    for layer in range(LLM_LAYER_COUNT):
        if (f"v3-{layer}", "v4") not in pairs:
            raise ValueError(f"missing LLM-{layer} KV-cache edge")


def generate_configs(
    output_dir: Path,
    *,
    batch_sizes: Sequence[int],
    compressions: Sequence[str],
    networks: Sequence[NetworkProfile],
    target_hz: float,
    latency_ms: float,
    latency_budget_ms: float,
    pipeline_unroll: int,
) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for network in networks:
        profile_dir = output_dir / network.name
        profile_dir.mkdir(parents=True, exist_ok=True)
        for batch_size in batch_sizes:
            for compression in compressions:
                config = build_config(
                    batch_size,
                    compression,
                    network,
                    target_hz=target_hz,
                    latency_ms=latency_ms,
                    latency_budget_ms=latency_budget_ms,
                    pipeline_unroll=pipeline_unroll,
                )
                path = profile_dir / f"batch{batch_size:02d}-{compression}.json"
                with path.open("w", encoding="utf-8") as file:
                    json.dump(config, file, indent=2, ensure_ascii=False)
                    file.write("\n")
                manifest.append(
                    {
                        **config["metadata"],
                        "config": str(path),
                    }
                )
    _write_manifest(output_dir, manifest)
    return manifest


def parse_int_ranges(value: str) -> tuple[int, ...]:
    values: set[int] = set()
    for part in value.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            begin_text, end_text = token.split("-", 1)
            begin, end = int(begin_text), int(end_text)
            if end < begin:
                raise argparse.ArgumentTypeError(f"invalid range: {token}")
            values.update(range(begin, end + 1))
        else:
            values.add(int(token))
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("batch sizes must be positive")
    return tuple(sorted(values))


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_float_csv(value: str) -> tuple[float, ...]:
    values = tuple(float(item) for item in parse_csv(value))
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("bandwidths must be positive")
    return values


def resolve_networks(args: argparse.Namespace) -> tuple[NetworkProfile, ...]:
    networks: list[NetworkProfile] = []
    for name in args.network_profiles:
        try:
            networks.append(NETWORK_PROFILES[name])
        except KeyError as exc:
            raise ValueError(
                f"Unknown network profile {name!r}; choose from {sorted(NETWORK_PROFILES)}"
            ) from exc
    for nominal in args.nominal_bandwidths:
        networks.append(
            NetworkProfile(
                name=f"custom-{nominal:g}mbps",
                nominal_mbps=nominal,
                efficiency=args.efficiency,
            )
        )
    deduplicated = {network.name: network for network in networks}
    return tuple(deduplicated.values())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate pi0.6-like 960PR-lite + 1920 CoopInfer configs."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("generated/pi06-960prlite-1920"),
    )
    parser.add_argument("--batch-sizes", type=parse_int_ranges, default=parse_int_ranges("1-8"))
    parser.add_argument(
        "--compressions",
        type=parse_csv,
        default=tuple(COMPRESSION),
    )
    parser.add_argument(
        "--network-profiles",
        type=parse_csv,
        default=DEFAULT_NETWORK_PROFILES,
        help=f"comma-separated names: {', '.join(NETWORK_PROFILES)}",
    )
    parser.add_argument(
        "--nominal-bandwidths",
        type=parse_float_csv,
        default=(),
        help="additional nominal link rates in Mbit/s",
    )
    parser.add_argument("--efficiency", type=float, default=0.85)
    parser.add_argument("--target-hz", type=float, default=10.0)
    parser.add_argument("--latency-ms", type=float, default=0.0)
    parser.add_argument("--latency-budget-ms", type=float, default=95.0)
    parser.add_argument("--pipeline-unroll", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    unknown_compressions = set(args.compressions) - set(COMPRESSION)
    if unknown_compressions:
        raise SystemExit(f"Unknown compressions: {sorted(unknown_compressions)}")
    if not 0 < args.efficiency <= 1:
        raise SystemExit("--efficiency must be in (0, 1]")
    networks = resolve_networks(args)
    manifest = generate_configs(
        args.output_dir.resolve(),
        batch_sizes=args.batch_sizes,
        compressions=args.compressions,
        networks=networks,
        target_hz=args.target_hz,
        latency_ms=args.latency_ms,
        latency_budget_ms=args.latency_budget_ms,
        pipeline_unroll=args.pipeline_unroll,
    )
    print(f"Generated {len(manifest)} configs in {args.output_dir.resolve()}")
    return 0


def _write_manifest(output_dir: Path, manifest: Iterable[dict[str, Any]]) -> None:
    rows = list(manifest)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2, ensure_ascii=False)
        file.write("\n")
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
