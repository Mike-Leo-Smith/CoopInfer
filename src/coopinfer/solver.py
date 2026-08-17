from __future__ import annotations

from dataclasses import dataclass
import math
import random
import re
from typing import Dict

import networkx as nx

from .evaluator import EvaluationResult, baseline_scales, evaluate, graph_to_core_data, metrics_from_core
from .model import Environment, PLACEMENT_DEVICE, PLACEMENT_HOST, validate_environment, validate_graph


_STEP_SUFFIX_RE = re.compile(r"^(?P<base>.+)::step\d+$")


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
    solver_threads: int = 0,
    anneal_initial_temp: float = 1.0,
    anneal_final_temp: float = 0.01,
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
            solver_threads=solver_threads,
            anneal_initial_temp=anneal_initial_temp,
            anneal_final_temp=anneal_final_temp,
        )
    )

    placement_groups = _placement_groups(graph)
    if any(len(nodes) > 1 for nodes in placement_groups.values()):
        return _solve_grouped_placements(
            graph,
            placement_groups,
            environment,
            algorithm=algorithm,
            heuristic_iterations=heuristic_iterations,
            seed=seed,
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
        "solver_threads": environment.solver_threads,
        "anneal_initial_temp": environment.anneal_initial_temp,
        "anneal_final_temp": environment.anneal_final_temp,
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


def _placement_group_id(node_id: str) -> str:
    match = _STEP_SUFFIX_RE.match(str(node_id))
    return match.group("base") if match is not None else str(node_id)


def _placement_groups(graph: nx.DiGraph) -> Dict[str, list[str]]:
    groups: Dict[str, list[str]] = {}
    for node_id in graph.nodes:
        key = _placement_group_id(str(node_id))
        groups.setdefault(key, []).append(str(node_id))
    return groups


def _solve_grouped_placements(
    graph: nx.DiGraph,
    groups: Dict[str, list[str]],
    environment: Environment,
    *,
    algorithm: str,
    heuristic_iterations: int,
    seed: int,
) -> SolverResult:
    """Search one placement variable per base layer across all denoise copies.

    Stage 3 materializes repeated denoise executions as ``base::stepNNN`` nodes.
    Every clone of the same base layer must use the same resource so cached
    prefix KV remains resident and reusable across later denoise steps. The
    native evaluator is still used for each candidate schedule; only the search
    variable space is grouped here.
    """

    fixed: Dict[str, int] = {}
    initial: Dict[str, int] = {}
    for group_id, nodes in groups.items():
        fixed_values = set()
        initial_values = []
        for node_id in nodes:
            attrs = graph.nodes[node_id]
            placement = str(attrs.get("placement", "free")).strip().lower()
            value = int(attrs.get("x", 1))
            if placement == PLACEMENT_DEVICE:
                fixed_values.add(0)
                value = 0
            elif placement == PLACEMENT_HOST:
                fixed_values.add(1)
                value = 1
            initial_values.append(value)
        if len(fixed_values) > 1:
            raise ValueError(f"Placement group {group_id!r} has conflicting fixed placements")
        if fixed_values:
            fixed[group_id] = next(iter(fixed_values))
            initial[group_id] = fixed[group_id]
        else:
            initial[group_id] = initial_values[0] if initial_values else 1

    free_groups = sorted(group_id for group_id in groups if group_id not in fixed)

    def expand(group_assignment: Dict[str, int]) -> Dict[str, int]:
        assignment: Dict[str, int] = {}
        for group_id, nodes in groups.items():
            value = int(group_assignment[group_id])
            for node_id in nodes:
                assignment[node_id] = value
        return assignment

    scales = baseline_scales(
        graph,
        bandwidth=environment.bandwidth,
        latency=environment.latency,
        batch_transfers=environment.batch_transfers,
        pipeline_unroll=environment.pipeline_unroll,
    )

    def candidate_metrics(group_assignment: Dict[str, int]) -> EvaluationResult:
        return evaluate(
            graph,
            expand(group_assignment),
            bandwidth=environment.bandwidth,
            latency=environment.latency,
            weight_avg_latency=environment.weight_avg_latency,
            weight_max_latency=environment.weight_max_latency,
            weight_device_utilization=environment.weight_device_utilization,
            avg_latency_scale=scales[0],
            max_latency_scale=scales[1],
            batch_transfers=environment.batch_transfers,
            pipeline_unroll=environment.pipeline_unroll,
        )

    def feasible(metrics: EvaluationResult) -> bool:
        if environment.latency_limit > 0.0 and metrics.latency > environment.latency_limit:
            return False
        if (
            environment.max_frame_latency_limit > 0.0
            and metrics.max_frame_latency > environment.max_frame_latency_limit
        ):
            return False
        return True

    normalized = str(algorithm).strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in {"", "auto"}:
        normalized = "enumerate" if len(free_groups) <= 12 else "random_search"

    host_seed = dict(initial)
    for group_id in free_groups:
        host_seed[group_id] = 1
    best_assignment = dict(host_seed)
    best_metrics = candidate_metrics(best_assignment)
    has_best = feasible(best_metrics)

    def keep(candidate: Dict[str, int], metrics: EvaluationResult) -> bool:
        nonlocal best_assignment, best_metrics, has_best
        if not feasible(metrics):
            return False
        if not has_best or metrics.loss < best_metrics.loss - 1e-12:
            best_assignment = dict(candidate)
            best_metrics = metrics
            has_best = True
            return True
        return False

    if normalized in {"enumerate", "brute", "brute_force"}:
        if len(free_groups) >= 63:
            raise ValueError("Enumerate has too many free placement groups")
        total = 1 << len(free_groups)
        for mask in range(total):
            candidate = dict(initial)
            for bit, group_id in enumerate(free_groups):
                candidate[group_id] = int((mask >> bit) & 1)
            keep(candidate, candidate_metrics(candidate))
        iterations = total
        mode = "Enumerate (shared denoise placement)"

    elif normalized in {"random", "random_search", "random_n"}:
        rng = random.Random(seed)
        iterations = max(0, int(heuristic_iterations))
        max_flips = max(1, min(3, len(free_groups))) if free_groups else 0
        for _ in range(iterations):
            candidate = dict(best_assignment if has_best else host_seed)
            if free_groups:
                flips = rng.randint(1, max_flips)
                for group_id in rng.sample(free_groups, flips):
                    candidate[group_id] = 1 - candidate[group_id]
            keep(candidate, candidate_metrics(candidate))
        mode = "Random Search (shared denoise placement)"

    elif normalized in {"simulated_annealing", "annealing", "sim_anneal", "sim_aneal"}:
        rng = random.Random(seed)
        iterations = max(0, int(heuristic_iterations))
        current = dict(host_seed)
        current_metrics = candidate_metrics(current)
        keep(current, current_metrics)
        for step in range(iterations):
            if not free_groups:
                break
            candidate = dict(current)
            group_id = rng.choice(free_groups)
            candidate[group_id] = 1 - candidate[group_id]
            metrics = candidate_metrics(candidate)
            if not feasible(metrics):
                continue
            progress = step / max(1, iterations - 1)
            temperature = environment.anneal_initial_temp * math.pow(
                environment.anneal_final_temp / environment.anneal_initial_temp,
                progress,
            )
            delta = metrics.loss - current_metrics.loss
            if delta <= 0.0 or rng.random() < math.exp(-delta / max(temperature, 1e-9)):
                current = candidate
                current_metrics = metrics
            keep(candidate, metrics)
        mode = "Simulated Annealing (shared denoise placement)"

    else:
        raise ValueError(f"Unknown solver algorithm: {algorithm}")

    if not has_best:
        raise ValueError(
            "No feasible shared-placement assignment satisfies the configured latency limits"
        )

    return SolverResult(
        assignment=expand(best_assignment),
        metrics=best_metrics,
        mode=mode,
        iterations=iterations,
    )
