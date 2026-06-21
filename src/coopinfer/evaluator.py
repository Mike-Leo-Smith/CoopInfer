from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import networkx as nx


@dataclass(frozen=True)
class EvaluationResult:
    latency: float
    device_utilization: float
    loss: float
    start_times: Dict[str, float]
    finish_times: Dict[str, float]


def edge_transfer_ms(size_mb: float, bandwidth_mb_s: float, latency_ms: float) -> float:
    if bandwidth_mb_s <= 0:
        raise ValueError("Bandwidth must be greater than zero")
    return latency_ms + (size_mb / bandwidth_mb_s * 1000.0)


def infer_latency(
    graph: nx.DiGraph,
    assignment: Mapping[str, int],
    bandwidth: float,
    latency: float,
) -> tuple[float, Dict[str, float], Dict[str, float]]:
    start_times: Dict[str, float] = {}
    finish_times: Dict[str, float] = {}
    time_dev_ready = 0.0
    time_host_ready = 0.0

    for node in nx.topological_sort(graph):
        data_ready_at = 0.0
        node_x = int(assignment[node])
        for predecessor in graph.predecessors(node):
            pred_x = int(assignment[predecessor])
            transfer = 0.0
            if pred_x != node_x:
                edge_size = float(graph.edges[predecessor, node].get("size", 0.0))
                transfer = edge_transfer_ms(edge_size, bandwidth, latency)
            data_ready_at = max(data_ready_at, finish_times[predecessor] + transfer)

        attrs = graph.nodes[node]
        if node_x == 0:
            start_at = max(data_ready_at, time_dev_ready)
            finish_at = start_at + float(attrs["c_dev"])
            time_dev_ready = finish_at
        else:
            start_at = max(data_ready_at, time_host_ready)
            finish_at = start_at + float(attrs["c_host"])
            time_host_ready = finish_at

        start_times[node] = start_at
        finish_times[node] = finish_at

    return (max(finish_times.values()) if finish_times else 0.0, start_times, finish_times)


def baseline_bounds(graph: nx.DiGraph, bandwidth: float, latency: float) -> tuple[float, float]:
    all_device = {node: 0 for node in graph.nodes}
    mostly_host = {
        node: 0 if graph.nodes[node].get("fixed_dev", False) else 1
        for node in graph.nodes
    }
    device_latency, _, _ = infer_latency(graph, all_device, bandwidth, latency)
    host_latency, _, _ = infer_latency(graph, mostly_host, bandwidth, latency)
    return min(device_latency, host_latency), max(device_latency, host_latency)


def evaluate(
    graph: nx.DiGraph,
    assignment: Mapping[str, int],
    bandwidth: float,
    latency: float,
    weight_latency: float,
    tau_min: Optional[float] = None,
    tau_max: Optional[float] = None,
) -> EvaluationResult:
    tau, start_times, finish_times = infer_latency(graph, assignment, bandwidth, latency)
    if tau_min is None or tau_max is None:
        tau_min, tau_max = baseline_bounds(graph, bandwidth, latency)

    denom = tau_max - tau_min
    normalized_latency = 0.0 if abs(denom) < 1e-12 else (tau - tau_min) / denom

    total_dev_cost = sum(float(attrs["c_dev"]) for _, attrs in graph.nodes(data=True))
    used_dev_cost = sum(
        float(graph.nodes[node]["c_dev"]) for node, x_value in assignment.items() if int(x_value) == 0
    )
    device_utilization = 0.0 if total_dev_cost <= 0 else used_dev_cost / total_dev_cost
    utilization_complement = 1.0 - device_utilization
    weight = min(1.0, max(0.0, float(weight_latency)))
    loss = weight * normalized_latency + (1.0 - weight) * utilization_complement

    return EvaluationResult(
        latency=tau,
        device_utilization=device_utilization,
        loss=loss,
        start_times=start_times,
        finish_times=finish_times,
    )
