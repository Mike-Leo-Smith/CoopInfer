from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import networkx as nx

from .evaluator import EvaluationResult, TransferRecord, baseline_bounds, evaluate


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
    weight_latency: float,
    algorithm: str = "auto",
    heuristic_iterations: int = 3000,
    seed: int = 7,
    latency_limit: float = 0.0,
    batch_transfers: bool = False,
    pipeline_unroll: int = 1,
) -> SolverResult:
    cpp_result = _try_solve_cpp(
        graph,
        bandwidth,
        latency,
        weight_latency,
        algorithm,
        heuristic_iterations,
        seed,
        latency_limit,
        batch_transfers,
        pipeline_unroll,
    )
    if cpp_result is not None:
        return cpp_result

    free_nodes = [
        node for node, attrs in graph.nodes(data=True) if not attrs.get("fixed_dev", False)
    ]
    fixed_assignment = {
        node: 0 if attrs.get("fixed_dev", False) else int(attrs.get("x", 1))
        for node, attrs in graph.nodes(data=True)
    }
    tau_min, tau_max = baseline_bounds(
        graph,
        bandwidth,
        latency,
        batch_transfers=batch_transfers,
        pipeline_unroll=pipeline_unroll,
    )
    algorithm_key = algorithm.strip().lower().replace(" ", "_").replace("-", "_")
    limit = float(latency_limit)

    if algorithm_key in {"auto", ""} and len(free_nodes) <= 12:
        return _brute_force(
            graph, free_nodes, fixed_assignment, bandwidth, latency, weight_latency,
            tau_min, tau_max, limit, batch_transfers, pipeline_unroll
        )
    if algorithm_key in {"auto", ""}:
        return _random_search(
            graph,
            free_nodes,
            fixed_assignment,
            bandwidth,
            latency,
            weight_latency,
            tau_min,
            tau_max,
            heuristic_iterations,
            seed,
            limit,
            batch_transfers,
            pipeline_unroll,
            mode="Auto Random",
        )
    if algorithm_key in {"enumerate", "brute", "brute_force"}:
        return _brute_force(
            graph, free_nodes, fixed_assignment, bandwidth, latency, weight_latency,
            tau_min, tau_max, limit, batch_transfers, pipeline_unroll
        )
    if algorithm_key in {"random", "random_search", "random_n"}:
        return _random_search(
            graph,
            free_nodes,
            fixed_assignment,
            bandwidth,
            latency,
            weight_latency,
            tau_min,
            tau_max,
            heuristic_iterations,
            seed,
            limit,
            batch_transfers,
            pipeline_unroll,
            mode="Random Search",
        )
    if algorithm_key in {"simulated_annealing", "annealing", "sim_anneal", "sim_aneal"}:
        return _simulated_annealing(
            graph,
            free_nodes,
            fixed_assignment,
            bandwidth,
            latency,
            weight_latency,
            tau_min,
            tau_max,
            heuristic_iterations,
            seed,
            limit,
            batch_transfers,
            pipeline_unroll,
        )
    raise ValueError(f"Unknown solver algorithm: {algorithm}")


def _try_solve_cpp(
    graph: nx.DiGraph,
    bandwidth: float,
    latency: float,
    weight_latency: float,
    algorithm: str,
    heuristic_iterations: int,
    seed: int,
    latency_limit: float,
    batch_transfers: bool,
    pipeline_unroll: int,
) -> Optional[SolverResult]:
    try:
        from . import _core
    except ImportError:
        return None

    node_ids = [str(node) for node in graph.nodes]
    node_index = {node: index for index, node in enumerate(graph.nodes)}
    data = {
        "ids": node_ids,
        "c_dev": [float(attrs["c_dev"]) for _, attrs in graph.nodes(data=True)],
        "c_host": [float(attrs["c_host"]) for _, attrs in graph.nodes(data=True)],
        "fixed_dev": [bool(attrs.get("fixed_dev", False)) for _, attrs in graph.nodes(data=True)],
        "source_period_ms": [
            float(attrs.get("source_period_ms", 0.0)) for _, attrs in graph.nodes(data=True)
        ],
        "source_phase_ms": [
            float(attrs.get("source_phase_ms", 0.0)) for _, attrs in graph.nodes(data=True)
        ],
        "x_initial": [
            0 if attrs.get("fixed_dev", False) else int(attrs.get("x", 1))
            for _, attrs in graph.nodes(data=True)
        ],
        "edge_sources": [node_index[source] for source, _ in graph.edges],
        "edge_targets": [node_index[target] for _, target in graph.edges],
        "edge_sizes": [float(attrs.get("size", 0.0)) for _, _, attrs in graph.edges(data=True)],
    }
    params = {
        "bandwidth": float(bandwidth),
        "latency": float(latency),
        "weight_latency": float(weight_latency),
        "algorithm": algorithm,
        "heuristic_iterations": int(heuristic_iterations),
        "seed": int(seed),
        "latency_limit": float(latency_limit),
        "batch_transfers": bool(batch_transfers),
        "pipeline_unroll": int(pipeline_unroll),
    }
    try:
        raw = _core.solve_core(data, params)
    except Exception as exc:
        raise ValueError(str(exc)) from exc

    raw_metrics = raw["metrics"]
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
    metrics = EvaluationResult(
        latency=float(raw_metrics["latency"]),
        device_utilization=float(raw_metrics["device_utilization"]),
        loss=float(raw_metrics["loss"]),
        start_times={str(node): float(value) for node, value in raw_metrics["start_times"].items()},
        finish_times={str(node): float(value) for node, value in raw_metrics["finish_times"].items()},
        transfer_records=transfer_records,
        pipeline_unroll=int(raw_metrics["pipeline_unroll"]),
    )
    return SolverResult(
        assignment={str(node): int(value) for node, value in raw["assignment"].items()},
        metrics=metrics,
        mode=str(raw["mode"]),
        iterations=int(raw["iterations"]),
    )


def _brute_force(
    graph: nx.DiGraph,
    free_nodes: List[str],
    base_assignment: Dict[str, int],
    bandwidth: float,
    latency: float,
    weight_latency: float,
    tau_min: float,
    tau_max: float,
    latency_limit: float,
    batch_transfers: bool,
    pipeline_unroll: int,
) -> SolverResult:
    best_assignment: Optional[Dict[str, int]] = None
    best_metrics: Optional[EvaluationResult] = None
    iterations = 0

    for bits in itertools.product([0, 1], repeat=len(free_nodes)):
        iterations += 1
        assignment = dict(base_assignment)
        assignment.update(dict(zip(free_nodes, bits)))
        metrics = evaluate(
            graph, assignment, bandwidth, latency, weight_latency,
            tau_min=tau_min, tau_max=tau_max, batch_transfers=batch_transfers,
            pipeline_unroll=pipeline_unroll
        )
        if _exceeds_latency_limit(metrics, latency_limit):
            continue
        if best_metrics is None or metrics.loss < best_metrics.loss:
            best_assignment = assignment
            best_metrics = metrics

    if best_assignment is None or best_metrics is None:
        raise ValueError(_no_feasible_message(latency_limit))

    return SolverResult(
        assignment=best_assignment,
        metrics=best_metrics,
        mode="Enumerate",
        iterations=iterations,
    )


def _random_search(
    graph,
    free_nodes,
    base_assignment: Dict[str, int],
    bandwidth: float,
    latency: float,
    weight_latency: float,
    tau_min: float,
    tau_max: float,
    iterations: int,
    seed: int,
    latency_limit: float,
    batch_transfers: bool,
    pipeline_unroll: int,
    mode: str,
) -> SolverResult:
    rng = random.Random(seed)
    best_assignment = _mostly_host_seed(free_nodes, base_assignment)
    best_metrics = evaluate(
        graph, best_assignment, bandwidth, latency, weight_latency,
        tau_min=tau_min, tau_max=tau_max, batch_transfers=batch_transfers,
        pipeline_unroll=pipeline_unroll
    )
    if _exceeds_latency_limit(best_metrics, latency_limit):
        best_assignment = None
        best_metrics = None
    if not free_nodes:
        if best_assignment is None or best_metrics is None:
            raise ValueError(_no_feasible_message(latency_limit))
        return SolverResult(assignment=best_assignment, metrics=best_metrics, mode=mode, iterations=1)

    for _ in range(max(0, iterations)):
        assignment = _mostly_host_seed(free_nodes, base_assignment) if best_assignment is None else dict(best_assignment)
        flips = rng.randint(1, max(1, min(3, len(free_nodes))))
        for node in rng.sample(free_nodes, flips):
            assignment[node] = 1 - int(assignment[node])
        metrics = evaluate(
            graph, assignment, bandwidth, latency, weight_latency,
            tau_min=tau_min, tau_max=tau_max, batch_transfers=batch_transfers,
            pipeline_unroll=pipeline_unroll
        )
        if _exceeds_latency_limit(metrics, latency_limit):
            continue
        if best_metrics is None or metrics.loss < best_metrics.loss:
            best_assignment = assignment
            best_metrics = metrics

    if best_assignment is None or best_metrics is None:
        raise ValueError(_no_feasible_message(latency_limit))

    return SolverResult(
        assignment=best_assignment,
        metrics=best_metrics,
        mode=mode,
        iterations=max(0, iterations),
    )


def _simulated_annealing(
    graph: nx.DiGraph,
    free_nodes: List[str],
    base_assignment: Dict[str, int],
    bandwidth: float,
    latency: float,
    weight_latency: float,
    tau_min: float,
    tau_max: float,
    iterations: int,
    seed: int,
    latency_limit: float,
    batch_transfers: bool,
    pipeline_unroll: int,
) -> SolverResult:
    rng = random.Random(seed)
    current = _mostly_host_seed(free_nodes, base_assignment)
    current_metrics = evaluate(
        graph, current, bandwidth, latency, weight_latency,
        tau_min=tau_min, tau_max=tau_max, batch_transfers=batch_transfers,
        pipeline_unroll=pipeline_unroll
    )
    best_assignment = dict(current)
    best_metrics = None if _exceeds_latency_limit(current_metrics, latency_limit) else current_metrics
    if best_metrics is None:
        best_assignment = None
    if not free_nodes:
        if best_assignment is None or best_metrics is None:
            raise ValueError(_no_feasible_message(latency_limit))
        return SolverResult(
            assignment=best_assignment,
            metrics=best_metrics,
            mode="Simulated Annealing",
            iterations=1,
        )

    total_iterations = max(0, iterations)
    initial_temp = 1.0
    final_temp = 0.01

    for step in range(total_iterations):
        progress = step / max(1, total_iterations - 1)
        temperature = initial_temp * ((final_temp / initial_temp) ** progress)
        candidate = dict(current)
        node = rng.choice(free_nodes)
        candidate[node] = 1 - int(candidate[node])
        metrics = evaluate(
            graph, candidate, bandwidth, latency, weight_latency,
            tau_min=tau_min, tau_max=tau_max, batch_transfers=batch_transfers,
            pipeline_unroll=pipeline_unroll
        )
        if _exceeds_latency_limit(metrics, latency_limit):
            continue
        delta = metrics.loss - current_metrics.loss
        accept = _exceeds_latency_limit(current_metrics, latency_limit) or delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 1e-9))
        if accept:
            current = candidate
            current_metrics = metrics
        if best_metrics is None or metrics.loss < best_metrics.loss:
            best_assignment = candidate
            best_metrics = metrics

    if best_assignment is None or best_metrics is None:
        raise ValueError(_no_feasible_message(latency_limit))

    return SolverResult(
        assignment=best_assignment,
        metrics=best_metrics,
        mode="Simulated Annealing",
        iterations=total_iterations,
    )


def _mostly_host_seed(free_nodes: List[str], base_assignment: Dict[str, int]) -> Dict[str, int]:
    assignment = dict(base_assignment)
    for node in free_nodes:
        assignment[node] = 1
    return assignment


def _exceeds_latency_limit(metrics: EvaluationResult, latency_limit: float) -> bool:
    return latency_limit > 0 and metrics.latency > latency_limit


def _no_feasible_message(latency_limit: float) -> str:
    return f"No feasible assignment satisfies latency limit {latency_limit:.1f} ms."
