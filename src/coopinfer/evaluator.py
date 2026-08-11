from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import networkx as nx

from .model import (
    Environment,
    PLACEMENT_DEVICE,
    PLACEMENT_FREE,
    PLACEMENT_HOST,
    validate_environment,
    validate_graph,
)


@dataclass(frozen=True)
class TransferRecord:
    edges: Tuple[Tuple[str, str], ...]
    start: float
    finish: float
    size_mb: float
    batched: bool


@dataclass(frozen=True)
class EvaluationResult:
    latency: float
    device_utilization: float
    avg_latency_loss: float
    max_frame_latency_loss: float
    device_utilization_loss: float
    loss: float
    start_times: Dict[str, float]
    finish_times: Dict[str, float]
    transfer_records: Tuple[TransferRecord, ...]
    pipeline_unroll: int = 1
    max_frame_latency: float = 0.0
    initiation_interval: float = 0.0
    initiation_interval_loss: float = 0.0
    host_utilization: float = 0.0
    network_utilization: float = 0.0


def edge_transfer_ms(size_mb: float, bandwidth_mb_s: float, latency_ms: float) -> float:
    size = _non_negative_float(size_mb, "Transfer size")
    environment = validate_environment(
        Environment(bandwidth=bandwidth_mb_s, latency=latency_ms)
    )
    return float(
        _core_module().edge_transfer_ms(size, environment.bandwidth, environment.latency)
    )


def base_node_id(op_id: str) -> str:
    return op_id.split("[f", 1)[0] if "[f" in op_id else op_id


def frame_index(op_id: str) -> int:
    if "[f" not in op_id:
        return 0
    suffix = op_id.rsplit("[f", 1)[1].rstrip("]")
    try:
        return int(suffix)
    except ValueError:
        return 0


def infer_latency(
    graph: nx.DiGraph,
    assignment: Mapping[str, int],
    bandwidth: float,
    latency: float,
    batch_transfers: bool = False,
    pipeline_unroll: int = 1,
) -> tuple[float, Dict[str, float], Dict[str, float]]:
    result = evaluate(
        graph,
        assignment,
        bandwidth=bandwidth,
        latency=latency,
        weight_avg_latency=1.0,
        weight_max_latency=0.0,
        weight_device_utilization=0.0,
        batch_transfers=batch_transfers,
        pipeline_unroll=pipeline_unroll,
    )
    return result.latency, result.start_times, result.finish_times


def infer_schedule(
    graph: nx.DiGraph,
    assignment: Mapping[str, int],
    bandwidth: float,
    latency: float,
    batch_transfers: bool = False,
    pipeline_unroll: int = 1,
) -> tuple[float, Dict[str, float], Dict[str, float], Tuple[TransferRecord, ...]]:
    result = evaluate(
        graph,
        assignment,
        bandwidth=bandwidth,
        latency=latency,
        weight_avg_latency=1.0,
        weight_max_latency=0.0,
        weight_device_utilization=0.0,
        batch_transfers=batch_transfers,
        pipeline_unroll=pipeline_unroll,
    )
    return (
        result.latency,
        result.start_times,
        result.finish_times,
        result.transfer_records,
    )


def baseline_scales(
    graph: nx.DiGraph,
    bandwidth: float,
    latency: float,
    batch_transfers: bool = False,
    pipeline_unroll: int = 1,
) -> tuple[float, float]:
    validate_graph(graph, require_dag=True)
    environment = validate_environment(
        Environment(
            bandwidth=bandwidth,
            latency=latency,
            batch_transfers=batch_transfers,
            pipeline_unroll=pipeline_unroll,
        )
    )
    raw = _core_module().baseline_scales_core(
        graph_to_core_data(graph),
        {
            "bandwidth": environment.bandwidth,
            "latency": environment.latency,
            "batch_transfers": environment.batch_transfers,
            "pipeline_unroll": environment.pipeline_unroll,
        },
    )
    return float(raw[0]), float(raw[1])


def evaluate(
    graph: nx.DiGraph,
    assignment: Mapping[str, int],
    bandwidth: float,
    latency: float,
    weight_avg_latency: float = 0.7,
    weight_max_latency: float = 0.3,
    weight_device_utilization: float = 0.3,
    avg_latency_scale: Optional[float] = None,
    max_latency_scale: Optional[float] = None,
    batch_transfers: bool = False,
    pipeline_unroll: int = 1,
) -> EvaluationResult:
    validate_graph(graph, require_dag=True)
    environment = validate_environment(
        Environment(
            bandwidth=bandwidth,
            latency=latency,
            weight_avg_latency=weight_avg_latency,
            weight_max_latency=weight_max_latency,
            weight_device_utilization=weight_device_utilization,
            batch_transfers=batch_transfers,
            pipeline_unroll=pipeline_unroll,
        )
    )
    params: Dict[str, Any] = {
        "bandwidth": environment.bandwidth,
        "latency": environment.latency,
        "weight_avg_latency": environment.weight_avg_latency,
        "weight_max_latency": environment.weight_max_latency,
        "weight_device_utilization": environment.weight_device_utilization,
        "batch_transfers": environment.batch_transfers,
        "pipeline_unroll": environment.pipeline_unroll,
        "assignment": assignment_to_core_vector(graph, assignment),
    }
    if avg_latency_scale is not None and max_latency_scale is not None:
        params["avg_latency_scale"] = _finite_float(avg_latency_scale, "avg_latency_scale")
        params["max_latency_scale"] = _finite_float(max_latency_scale, "max_latency_scale")

    raw_metrics = _core_module().evaluate_core(graph_to_core_data(graph), params)
    return metrics_from_core(raw_metrics)


def _node_placement(attrs: Mapping[str, Any]) -> str:
    placement = str(
        attrs.get(
            "placement",
            PLACEMENT_DEVICE if attrs.get("fixed_dev", False) else PLACEMENT_FREE,
        )
    ).strip().lower()
    if placement not in {PLACEMENT_FREE, PLACEMENT_DEVICE, PLACEMENT_HOST}:
        raise ValueError(f"Unknown placement constraint: {placement!r}")
    return placement


def graph_to_core_data(graph: nx.DiGraph) -> Dict[str, Any]:
    node_ids = [str(node) for node in graph.nodes]
    node_index = {node: index for index, node in enumerate(graph.nodes)}
    placements = [_node_placement(attrs) for _, attrs in graph.nodes(data=True)]
    x_initial = []
    for placement, (_, attrs) in zip(placements, graph.nodes(data=True)):
        if placement == PLACEMENT_DEVICE:
            x_initial.append(0)
        elif placement == PLACEMENT_HOST:
            x_initial.append(1)
        else:
            x_initial.append(int(attrs.get("x", 1)))
    return {
        "ids": node_ids,
        "c_dev": [float(attrs["c_dev"]) for _, attrs in graph.nodes(data=True)],
        "c_host": [float(attrs["c_host"]) for _, attrs in graph.nodes(data=True)],
        # Native core historically calls this field fixed_dev. It now acts as a
        # generic fixed-placement mask; x_initial carries Device(0)/Host(1).
        "fixed_dev": [placement != PLACEMENT_FREE for placement in placements],
        "source_period_ms": [
            float(attrs.get("source_period_ms", 0.0)) for _, attrs in graph.nodes(data=True)
        ],
        "source_phase_ms": [
            float(attrs.get("source_phase_ms", 0.0)) for _, attrs in graph.nodes(data=True)
        ],
        "x_initial": x_initial,
        "edge_sources": [node_index[source] for source, _ in graph.edges],
        "edge_targets": [node_index[target] for _, target in graph.edges],
        "edge_sizes": [float(attrs.get("size", 0.0)) for _, _, attrs in graph.edges(data=True)],
    }


def assignment_to_core_vector(graph: nx.DiGraph, assignment: Mapping[str, int]) -> list[int]:
    vector: list[int] = []
    for node in graph.nodes:
        if node not in assignment and str(node) not in assignment:
            raise ValueError(f"Assignment is missing node {node}.")
        value = assignment[node] if node in assignment else assignment[str(node)]
        value = int(value)
        if value not in {0, 1}:
            raise ValueError(f"Assignment for node {node} must be 0 or 1.")
        vector.append(value)
    return vector


def metrics_from_core(raw_metrics: Mapping[str, Any]) -> EvaluationResult:
    transfer_records = tuple(
        TransferRecord(
            edges=tuple((str(source), str(target)) for source, target in transfer["edges"]),
            start=float(transfer["start"]),
            finish=float(transfer["finish"]),
            size_mb=float(transfer["size_mb"]),
            batched=bool(transfer["batched"]),
        )
        for transfer in raw_metrics["transfer_records"]
    )
    return EvaluationResult(
        latency=float(raw_metrics["latency"]),
        initiation_interval=float(raw_metrics.get("initiation_interval", raw_metrics["latency"])),
        device_utilization=float(raw_metrics["device_utilization"]),
        host_utilization=float(raw_metrics.get("host_utilization", 0.0)),
        network_utilization=float(raw_metrics.get("network_utilization", 0.0)),
        avg_latency_loss=float(raw_metrics["avg_latency_loss"]),
        max_frame_latency_loss=float(raw_metrics["max_frame_latency_loss"]),
        initiation_interval_loss=float(raw_metrics.get("initiation_interval_loss", 0.0)),
        device_utilization_loss=float(raw_metrics["device_utilization_loss"]),
        loss=float(raw_metrics["loss"]),
        start_times={str(node): float(value) for node, value in raw_metrics["start_times"].items()},
        finish_times={str(node): float(value) for node, value in raw_metrics["finish_times"].items()},
        transfer_records=transfer_records,
        pipeline_unroll=int(raw_metrics["pipeline_unroll"]),
        max_frame_latency=float(raw_metrics["max_frame_latency"]),
    )


def _core_module():
    try:
        from . import _core
    except ImportError as exc:
        raise RuntimeError(
            "CoopInfer requires the compiled C++ solver extension. "
            'Build/install the project with `python -m pip install -e ".[dev]"`.'
        ) from exc
    return _core


def _finite_float(value: float, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number


def _non_negative_float(value: float, label: str) -> float:
    number = _finite_float(value, label)
    if number < 0:
        raise ValueError(f"{label} must be non-negative.")
    return number
