from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import networkx as nx

from .evaluator import EvaluationResult, baseline_bounds, evaluate


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
) -> SolverResult:
    free_nodes = [
        node for node, attrs in graph.nodes(data=True) if not attrs.get("fixed_dev", False)
    ]
    fixed_assignment = {
        node: 0 if attrs.get("fixed_dev", False) else int(attrs.get("x", 1))
        for node, attrs in graph.nodes(data=True)
    }
    tau_min, tau_max = baseline_bounds(graph, bandwidth, latency)
    algorithm_key = algorithm.strip().lower().replace(" ", "_").replace("-", "_")

    if algorithm_key in {"auto", ""} and len(free_nodes) <= 12:
        return _brute_force(
            graph, free_nodes, fixed_assignment, bandwidth, latency, weight_latency, tau_min, tau_max
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
            mode="Auto Random",
        )
    if algorithm_key in {"enumerate", "brute", "brute_force"}:
        return _brute_force(
            graph, free_nodes, fixed_assignment, bandwidth, latency, weight_latency, tau_min, tau_max
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
        )
    raise ValueError(f"Unknown solver algorithm: {algorithm}")


def _brute_force(
    graph: nx.DiGraph,
    free_nodes: List[str],
    base_assignment: Dict[str, int],
    bandwidth: float,
    latency: float,
    weight_latency: float,
    tau_min: float,
    tau_max: float,
) -> SolverResult:
    best_assignment: Optional[Dict[str, int]] = None
    best_metrics: Optional[EvaluationResult] = None
    iterations = 0

    for bits in itertools.product([0, 1], repeat=len(free_nodes)):
        iterations += 1
        assignment = dict(base_assignment)
        assignment.update(dict(zip(free_nodes, bits)))
        metrics = evaluate(
            graph, assignment, bandwidth, latency, weight_latency, tau_min=tau_min, tau_max=tau_max
        )
        if best_metrics is None or metrics.loss < best_metrics.loss:
            best_assignment = assignment
            best_metrics = metrics

    return SolverResult(
        assignment=best_assignment or dict(base_assignment),
        metrics=best_metrics
        or evaluate(graph, base_assignment, bandwidth, latency, weight_latency, tau_min, tau_max),
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
    mode: str,
) -> SolverResult:
    rng = random.Random(seed)
    best_assignment = _mostly_host_seed(free_nodes, base_assignment)
    best_metrics = evaluate(
        graph, best_assignment, bandwidth, latency, weight_latency, tau_min=tau_min, tau_max=tau_max
    )
    if not free_nodes:
        return SolverResult(assignment=best_assignment, metrics=best_metrics, mode=mode, iterations=1)

    for _ in range(max(0, iterations)):
        assignment = dict(best_assignment)
        flips = rng.randint(1, max(1, min(3, len(free_nodes))))
        for node in rng.sample(free_nodes, flips):
            assignment[node] = 1 - int(assignment[node])
        metrics = evaluate(
            graph, assignment, bandwidth, latency, weight_latency, tau_min=tau_min, tau_max=tau_max
        )
        if metrics.loss < best_metrics.loss:
            best_assignment = assignment
            best_metrics = metrics

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
) -> SolverResult:
    rng = random.Random(seed)
    current = _mostly_host_seed(free_nodes, base_assignment)
    current_metrics = evaluate(
        graph, current, bandwidth, latency, weight_latency, tau_min=tau_min, tau_max=tau_max
    )
    best_assignment = dict(current)
    best_metrics = current_metrics
    if not free_nodes:
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
            graph, candidate, bandwidth, latency, weight_latency, tau_min=tau_min, tau_max=tau_max
        )
        delta = metrics.loss - current_metrics.loss
        accept = delta <= 0 or rng.random() < math.exp(-delta / max(temperature, 1e-9))
        if accept:
            current = candidate
            current_metrics = metrics
        if metrics.loss < best_metrics.loss:
            best_assignment = candidate
            best_metrics = metrics

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
