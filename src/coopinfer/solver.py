from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, Tuple

import networkx as nx

from .evaluator import (
    EvaluationResult,
    baseline_scales,
    evaluate,
    graph_to_core_data,
    metrics_from_core,
)
from .model import (
    Environment,
    PLACEMENT_DEVICE,
    PLACEMENT_HOST,
    validate_environment,
    validate_graph,
)


_STEP_TOKEN = "::step"


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

    cache_templates = _persistent_cache_templates(graph)
    if cache_templates:
        return _solve_iterative_persistent_cache(
            graph,
            cache_templates,
            environment,
            algorithm=algorithm,
            heuristic_iterations=heuristic_iterations,
            seed=seed,
        )

    return _solve_native(
        graph,
        environment,
        algorithm=algorithm,
        heuristic_iterations=heuristic_iterations,
        seed=seed,
    )


def _solve_native(
    graph: nx.DiGraph,
    environment: Environment,
    *,
    algorithm: str,
    heuristic_iterations: int,
    seed: int,
) -> SolverResult:
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

    return SolverResult(
        assignment={str(node): int(value) for node, value in raw["assignment"].items()},
        metrics=metrics_from_core(raw["metrics"]),
        mode=str(raw["mode"]),
        iterations=int(raw["iterations"]),
    )


def _split_step_id(node_id: str) -> Tuple[str, int] | None:
    text = str(node_id)
    base, token, suffix = text.rpartition(_STEP_TOKEN)
    if not token or not base or not suffix.isdigit():
        return None
    return base, int(suffix)


def _persistent_cache_templates(
    graph: nx.DiGraph,
) -> list[tuple[str, str, str, float]]:
    """Recover Stage-3 static-to-step0 cache inputs from the execution DAG.

    Stage 3 deliberately emits these inputs only once, targeting step 0. Their
    static source owns the original cache copy. If later denoise steps execute
    on the other resource, the evaluator adds one prefetch/copy edge to the
    first such use and keeps that resource copy resident thereafter.
    """

    templates: list[tuple[str, str, str, float]] = []
    for source, target, attrs in graph.edges(data=True):
        target_step = _split_step_id(str(target))
        if target_step is None or target_step[1] != 0:
            continue
        if _split_step_id(str(source)) is not None:
            continue
        size_mb = float(attrs.get("size", 0.0))
        if size_mb <= 0.0:
            continue
        templates.append((str(source), target_step[0], str(target), size_mb))
    return templates


def _step_nodes_for_base(graph: nx.DiGraph, base: str) -> list[tuple[int, str]]:
    result = []
    for node_id in graph.nodes:
        parsed = _split_step_id(str(node_id))
        if parsed is not None and parsed[0] == base:
            result.append((parsed[1], str(node_id)))
    result.sort()
    return result


def _cache_aware_graph(
    graph: nx.DiGraph,
    assignment: Dict[str, int],
    templates: Iterable[tuple[str, str, str, float]],
) -> nx.DiGraph:
    """Materialize at most one extra cache copy per static input/resource.

    There are two compute resources. The static producer resource already owns
    one copy. The original source->step0 edge transfers to the other resource if
    step 0 needs it. If step 0 stays with the producer but a later step first
    uses the other resource, add one source->that-step prefetch edge. Once that
    copy exists it is reused by every later visit to that resource.
    """

    result = graph.copy()
    for source, target_base, step0_target, size_mb in templates:
        if source not in assignment:
            continue
        source_resource = int(assignment[source])
        opposite = 1 - source_resource
        first_opposite = None
        for _, node_id in _step_nodes_for_base(graph, target_base):
            if int(assignment[node_id]) == opposite:
                first_opposite = node_id
                break
        if first_opposite is None or first_opposite == step0_target:
            continue

        if result.has_edge(source, first_opposite):
            existing = float(result.edges[source, first_opposite].get("size", 0.0))
            result.edges[source, first_opposite]["size"] = existing + size_mb
            result.edges[source, first_opposite]["persistent_cache_copy"] = True
        else:
            result.add_edge(
                source,
                first_opposite,
                size=size_mb,
                persistent_cache_copy=True,
            )
    validate_graph(result, require_dag=True)
    return result


def _warmup_graph(graph: nx.DiGraph) -> nx.DiGraph:
    """Static model plus denoise step 0, used only to seed full-E2E search."""

    keep = []
    for node_id in graph.nodes:
        parsed = _split_step_id(str(node_id))
        if parsed is None or parsed[1] == 0:
            keep.append(node_id)
    result = graph.subgraph(keep).copy()
    validate_graph(result, require_dag=True)
    return result


def _fixed_and_initial(graph: nx.DiGraph) -> tuple[Dict[str, int], Dict[str, int], list[str]]:
    fixed: Dict[str, int] = {}
    initial: Dict[str, int] = {}
    free_nodes = []
    for node_id, attrs in graph.nodes(data=True):
        key = str(node_id)
        placement = str(attrs.get("placement", "free")).strip().lower()
        value = int(attrs.get("x", 1))
        if placement == PLACEMENT_DEVICE:
            value = 0
            fixed[key] = 0
        elif placement == PLACEMENT_HOST:
            value = 1
            fixed[key] = 1
        initial[key] = value

        is_zero_source = (
            graph.in_degree(node_id) == 0
            and float(attrs.get("c_dev", 0.0)) == 0.0
            and float(attrs.get("c_host", 0.0)) == 0.0
        )
        if key not in fixed and not is_zero_source:
            free_nodes.append(key)
    return fixed, initial, sorted(free_nodes)


def _solve_iterative_persistent_cache(
    graph: nx.DiGraph,
    cache_templates: list[tuple[str, str, str, float]],
    environment: Environment,
    *,
    algorithm: str,
    heuristic_iterations: int,
    seed: int,
) -> SolverResult:
    """Search the complete N-step execution with persistent KV reuse.

    Every denoise execution node is a free placement variable unless the input
    graph explicitly fixes it. The full N-step E2E latency is used for every
    candidate evaluation.

    To avoid losing the layerwise VLM/step0 overlap opportunity in a large
    search space, the static+step0 subgraph is solved first and used as one seed
    for the full search. Later denoise steps start on Host in that seed and are
    then free to move independently. This seeding changes search efficiency, not
    the objective: only the full N-step evaluation can become the final answer.
    """

    _, initial, free_nodes = _fixed_and_initial(graph)
    host_seed = dict(initial)
    for node_id in free_nodes:
        host_seed[node_id] = 1

    scales = baseline_scales(
        graph,
        bandwidth=environment.bandwidth,
        latency=environment.latency,
        batch_transfers=environment.batch_transfers,
        pipeline_unroll=environment.pipeline_unroll,
    )

    def candidate_metrics(assignment: Dict[str, int]) -> EvaluationResult:
        effective_graph = _cache_aware_graph(graph, assignment, cache_templates)
        return evaluate(
            effective_graph,
            assignment,
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

    # Explicitly recover a strong first-denoise overlap seed. The warm-up solve
    # may place step0 AE layers on the second resource while later steps remain
    # Host, which is the valid pattern that shared-placement constraints removed.
    warmup = _warmup_graph(graph)
    try:
        warmup_result = _solve_native(
            warmup,
            environment,
            algorithm=algorithm,
            heuristic_iterations=heuristic_iterations,
            seed=seed,
        )
        warm_seed = dict(host_seed)
        for node_id, value in warmup_result.assignment.items():
            if node_id in warm_seed:
                warm_seed[node_id] = int(value)
        keep(warm_seed, candidate_metrics(warm_seed))
    except ValueError:
        # The full search is still valid if a requested warm-up algorithm (for
        # example brute-force enumeration) is unsuitable for the subgraph.
        pass

    normalized = str(algorithm).strip().lower().replace(" ", "_").replace("-", "_")
    if normalized in {"", "auto"}:
        normalized = "enumerate" if len(free_nodes) <= 12 else "random_search"

    if normalized in {"enumerate", "brute", "brute_force"}:
        if len(free_nodes) >= 63:
            raise ValueError("Enumerate has too many free execution nodes")
        total = 1 << len(free_nodes)
        for mask in range(total):
            candidate = dict(initial)
            for bit, node_id in enumerate(free_nodes):
                candidate[node_id] = int((mask >> bit) & 1)
            keep(candidate, candidate_metrics(candidate))
        iterations = total
        mode = "Enumerate (per-step denoise + persistent KV)"

    elif normalized in {"random", "random_search", "random_n"}:
        rng = random.Random(seed)
        iterations = max(0, int(heuristic_iterations))
        max_flips = max(1, min(3, len(free_nodes))) if free_nodes else 0
        for _ in range(iterations):
            candidate = dict(best_assignment if has_best else host_seed)
            if free_nodes:
                flips = rng.randint(1, max_flips)
                for node_id in rng.sample(free_nodes, flips):
                    candidate[node_id] = 1 - candidate[node_id]
            keep(candidate, candidate_metrics(candidate))
        mode = "Random Search (per-step denoise + persistent KV)"

    elif normalized in {"simulated_annealing", "annealing", "sim_anneal", "sim_aneal"}:
        rng = random.Random(seed)
        iterations = max(0, int(heuristic_iterations))
        current = dict(best_assignment if has_best else host_seed)
        current_metrics = candidate_metrics(current)
        keep(current, current_metrics)
        for step in range(iterations):
            if not free_nodes:
                break
            candidate = dict(current)
            node_id = rng.choice(free_nodes)
            candidate[node_id] = 1 - candidate[node_id]
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
        mode = "Simulated Annealing (per-step denoise + persistent KV)"

    else:
        raise ValueError(f"Unknown solver algorithm: {algorithm}")

    if not has_best:
        raise ValueError(
            "No feasible per-step assignment satisfies the configured latency limits"
        )

    return SolverResult(
        assignment=best_assignment,
        metrics=best_metrics,
        mode=mode,
        iterations=iterations,
    )
