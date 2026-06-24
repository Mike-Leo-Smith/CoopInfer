from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import networkx as nx

from .evaluator import EvaluationResult, graph_to_core_data, metrics_from_core
from .model import Environment, validate_environment, validate_graph


@dataclass(frozen=True)
class SolverResult:
    assignment: Dict[str, int]
    metrics: EvaluationResult
    mode: str
    iterations: int


def solve(
    graph: nx.DiGraph,
    bandwidth: float,
    latency: float,
    weight_avg_latency: float = 0.7,
    weight_max_latency: float = 0.3,
    weight_device_utilization: float = 0.3,
    algorithm: str = "auto",
    heuristic_iterations: int = 3000,
    seed: int = 7,
    latency_limit: float = 0.0,
    batch_transfers: bool = False,
    pipeline_unroll: int = 1,
    max_frame_latency_limit: float = 0.0,
) -> SolverResult:
    validate_graph(graph, require_dag=True)
    environment = validate_environment(
        Environment(
            bandwidth=bandwidth,
            latency=latency,
            weight_avg_latency=weight_avg_latency,
            weight_max_latency=weight_max_latency,
            weight_device_utilization=weight_device_utilization,
            latency_limit=latency_limit,
            batch_transfers=batch_transfers,
            pipeline_unroll=pipeline_unroll,
            max_frame_latency_limit=max_frame_latency_limit,
        )
    )

    try:
        from . import _core
    except ImportError as exc:
        raise RuntimeError(
            "CoopInfer requires the compiled C++ solver extension. "
            'Build/install the project with `python -m pip install -e ".[dev]"`.'
        ) from exc

    data = graph_to_core_data(graph)
    params = {
        "bandwidth": environment.bandwidth,
        "latency": environment.latency,
        "weight_avg_latency": environment.weight_avg_latency,
        "weight_max_latency": environment.weight_max_latency,
        "weight_device_utilization": environment.weight_device_utilization,
        "algorithm": algorithm,
        "heuristic_iterations": int(heuristic_iterations),
        "seed": int(seed),
        "latency_limit": environment.latency_limit,
        "max_frame_latency_limit": environment.max_frame_latency_limit,
        "batch_transfers": environment.batch_transfers,
        "pipeline_unroll": environment.pipeline_unroll,
    }

    try:
        raw = _core.solve_core(data, params)
    except Exception as exc:
        raise ValueError(str(exc)) from exc

    metrics = metrics_from_core(raw["metrics"])
    return SolverResult(
        assignment={str(node): int(value) for node, value in raw["assignment"].items()},
        metrics=metrics,
        mode=str(raw["mode"]),
        iterations=int(raw["iterations"]),
    )
