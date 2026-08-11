from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple, Union

import networkx as nx


SCHEMA_VERSION = "1.0"
SUPPORTED_SCHEMA_VERSIONS = {"1.0", "1.1"}
PLACEMENT_FREE = "free"
PLACEMENT_DEVICE = "device"
PLACEMENT_HOST = "host"
PLACEMENT_VALUES = {PLACEMENT_FREE, PLACEMENT_DEVICE, PLACEMENT_HOST}
DEFAULT_WEIGHT_AVG_LATENCY = 0.7
DEFAULT_WEIGHT_MAX_LATENCY = 0.3
DEFAULT_WEIGHT_DEVICE_UTILIZATION = 0.3


@dataclass(frozen=True)
class Environment:
    bandwidth: float = 50.0
    latency: float = 5.0
    weight_avg_latency: float = DEFAULT_WEIGHT_AVG_LATENCY
    weight_max_latency: float = DEFAULT_WEIGHT_MAX_LATENCY
    weight_device_utilization: float = DEFAULT_WEIGHT_DEVICE_UTILIZATION
    latency_limit: float = 0.0
    batch_transfers: bool = False
    pipeline_unroll: int = 1
    max_frame_latency_limit: float = 0.0
    solver_threads: int = 0
    anneal_initial_temp: float = 1.0
    anneal_final_temp: float = 0.01


@dataclass(frozen=True)
class ProjectState:
    graph: nx.DiGraph
    environment: Environment


def environment_from_mapping(data: Mapping[str, Any]) -> Environment:
    if any(
        key in data
        for key in (
            "weight_avg_latency",
            "weight_max_latency",
            "weight_device_utilization",
        )
    ):
        weight_avg_latency = data.get("weight_avg_latency", DEFAULT_WEIGHT_AVG_LATENCY)
        weight_max_latency = data.get("weight_max_latency", DEFAULT_WEIGHT_MAX_LATENCY)
        weight_device_utilization = data.get(
            "weight_device_utilization", DEFAULT_WEIGHT_DEVICE_UTILIZATION
        )
    elif "weight_latency" in data:
        legacy_weight = min(
            1.0,
            max(0.0, _finite_float(data["weight_latency"], "Environment weight_latency")),
        )
        weight_avg_latency = legacy_weight
        weight_max_latency = 0.0
        weight_device_utilization = 1.0 - legacy_weight
    else:
        weight_avg_latency = DEFAULT_WEIGHT_AVG_LATENCY
        weight_max_latency = DEFAULT_WEIGHT_MAX_LATENCY
        weight_device_utilization = DEFAULT_WEIGHT_DEVICE_UTILIZATION

    return validate_environment(
        Environment(
            bandwidth=data.get("bandwidth", 50.0),
            latency=data.get("latency", 5.0),
            weight_avg_latency=weight_avg_latency,
            weight_max_latency=weight_max_latency,
            weight_device_utilization=weight_device_utilization,
            latency_limit=data.get("latency_limit", 0.0),
            max_frame_latency_limit=data.get("max_frame_latency_limit", 0.0),
            batch_transfers=data.get("batch_transfers", False),
            pipeline_unroll=data.get("pipeline_unroll", 1),
            solver_threads=data.get("solver_threads", 0),
            anneal_initial_temp=data.get("anneal_initial_temp", 1.0),
            anneal_final_temp=data.get("anneal_final_temp", 0.01),
        )
    )


def validate_environment(environment: Environment) -> Environment:
    bandwidth = _positive_float(environment.bandwidth, "Environment bandwidth")
    latency = _non_negative_float(environment.latency, "Environment latency")
    weight_avg_latency = _unit_float(
        environment.weight_avg_latency, "Environment weight_avg_latency"
    )
    weight_max_latency = _unit_float(
        environment.weight_max_latency, "Environment weight_max_latency"
    )
    weight_device_utilization = _unit_float(
        environment.weight_device_utilization,
        "Environment weight_device_utilization",
    )
    latency_limit = _non_negative_float(environment.latency_limit, "Environment latency_limit")
    max_frame_latency_limit = _non_negative_float(
        environment.max_frame_latency_limit,
        "Environment max_frame_latency_limit",
    )
    pipeline_unroll = _integer(environment.pipeline_unroll, "Environment pipeline_unroll")
    if pipeline_unroll < 1:
        raise ValueError("Environment pipeline_unroll must be at least 1.")
    solver_threads = _integer(environment.solver_threads, "Environment solver_threads")
    if solver_threads < 0:
        raise ValueError("Environment solver_threads must be non-negative.")
    anneal_initial_temp = _positive_float(
        environment.anneal_initial_temp, "Environment anneal_initial_temp"
    )
    anneal_final_temp = _positive_float(
        environment.anneal_final_temp, "Environment anneal_final_temp"
    )

    return Environment(
        bandwidth=bandwidth,
        latency=latency,
        weight_avg_latency=weight_avg_latency,
        weight_max_latency=weight_max_latency,
        weight_device_utilization=weight_device_utilization,
        latency_limit=latency_limit,
        max_frame_latency_limit=max_frame_latency_limit,
        batch_transfers=_bool(environment.batch_transfers, "Environment batch_transfers"),
        pipeline_unroll=pipeline_unroll,
        solver_threads=solver_threads,
        anneal_initial_temp=anneal_initial_temp,
        anneal_final_temp=anneal_final_temp,
    )


def placement_from_record(node: Mapping[str, Any], label: str) -> str:
    if "placement" in node:
        placement = _placement(node["placement"], f"{label} placement")
        if _bool(node.get("fixed_dev", False), f"{label} fixed_dev") and placement != PLACEMENT_DEVICE:
            raise ValueError(
                f"{label} has conflicting placement={placement!r} and fixed_dev=true."
            )
        return placement
    return PLACEMENT_DEVICE if _bool(node.get("fixed_dev", False), f"{label} fixed_dev") else PLACEMENT_FREE


def node_initial_assignment(placement: str, x_value: Any, label: str) -> int:
    if placement == PLACEMENT_DEVICE:
        return 0
    if placement == PLACEMENT_HOST:
        return 1
    return _binary_int(x_value, f"{label} x")


def graph_from_records(
    nodes: Iterable[Dict[str, Any]],
    edges: Iterable[Dict[str, Any]],
) -> nx.DiGraph:
    graph = nx.DiGraph()
    seen: set[str] = set()
    for row, node in enumerate(nodes, start=1):
        node_id = str(_required(node, "id", f"Node row {row}") or "").strip()
        if not node_id:
            raise ValueError(f"Node ID is empty at row {row}.")
        if node_id in seen:
            raise ValueError(f"Duplicate Node ID: {node_id}")
        seen.add(node_id)
        label = f"Node {node_id}"
        placement = placement_from_record(node, label)
        x_value = node_initial_assignment(placement, node.get("x", 1), label)
        graph.add_node(
            node_id,
            name=str(node.get("name", node_id)).strip() or node_id,
            c_dev=_non_negative_float(
                _required(node, "c_dev", label), f"{label} c_dev"
            ),
            c_host=_non_negative_float(
                _required(node, "c_host", label), f"{label} c_host"
            ),
            placement=placement,
            fixed_dev=placement == PLACEMENT_DEVICE,
            source_period_ms=_non_negative_float(
                node.get("source_period_ms", 0.0), f"{label} source_period_ms"
            ),
            source_phase_ms=_non_negative_float(
                node.get("source_phase_ms", 0.0), f"{label} source_phase_ms"
            ),
            x=x_value,
        )

    for row, edge in enumerate(edges, start=1):
        source = str(_required(edge, "source", f"Edge row {row}") or "").strip()
        target = str(_required(edge, "target", f"Edge row {row}") or "").strip()
        if source not in seen or target not in seen:
            raise ValueError(f"Edge row {row} references unknown nodes: {source} -> {target}")
        graph.add_edge(
            source,
            target,
            size=_non_negative_float(
                _required(edge, "size", f"Edge {source}->{target}"),
                f"Edge {source}->{target} size",
            ),
        )
    return graph


def validate_graph(graph: nx.DiGraph, *, require_dag: bool = False) -> None:
    for node_id, attrs in graph.nodes(data=True):
        label = f"Node {node_id}"
        _non_negative_float(_required(attrs, "c_dev", label), f"{label} c_dev")
        _non_negative_float(_required(attrs, "c_host", label), f"{label} c_host")
        _non_negative_float(attrs.get("source_period_ms", 0.0), f"{label} source_period_ms")
        _non_negative_float(attrs.get("source_phase_ms", 0.0), f"{label} source_phase_ms")
        placement = _placement(
            attrs.get(
                "placement",
                PLACEMENT_DEVICE if attrs.get("fixed_dev", False) else PLACEMENT_FREE,
            ),
            f"{label} placement",
        )
        expected = node_initial_assignment(placement, attrs.get("x", 1), label)
        if placement != PLACEMENT_FREE and int(attrs.get("x", expected)) != expected:
            raise ValueError(f"{label} x conflicts with fixed placement {placement!r}.")

    for source, target, attrs in graph.edges(data=True):
        _non_negative_float(
            _required(attrs, "size", f"Edge {source}->{target}"),
            f"Edge {source}->{target} size",
        )

    if require_dag and not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Graph contains a cycle; DAG required.")


def records_from_graph(graph: nx.DiGraph) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    validate_graph(graph)
    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        placement = _placement(
            attrs.get(
                "placement",
                PLACEMENT_DEVICE if attrs.get("fixed_dev", False) else PLACEMENT_FREE,
            ),
            f"Node {node_id} placement",
        )
        x_value = node_initial_assignment(placement, attrs.get("x", 1), f"Node {node_id}")
        nodes.append(
            {
                "id": node_id,
                "name": str(attrs.get("name", node_id)),
                "c_dev": float(attrs["c_dev"]),
                "c_host": float(attrs["c_host"]),
                "placement": placement,
                "source_period_ms": float(attrs.get("source_period_ms", 0.0)),
                "source_phase_ms": float(attrs.get("source_phase_ms", 0.0)),
                "x": x_value,
            }
        )

    edges = []
    for source, target, attrs in graph.edges(data=True):
        edges.append({"source": source, "target": target, "size": float(attrs["size"])})
    return nodes, edges


def load_from_json(path: Union[str, Path]) -> ProjectState:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    version = str(data.get("version"))
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported schema version: {data.get('version')!r}")

    environment = environment_from_mapping(data.get("environment", {}))
    graph = graph_from_records(data.get("nodes", []), data.get("edges", []))
    return ProjectState(graph=graph, environment=environment)


def save_to_json(state: ProjectState, path: Union[str, Path]) -> None:
    environment = validate_environment(state.environment)
    nodes, edges = records_from_graph(state.graph)
    data = {
        "version": SCHEMA_VERSION,
        "nodes": nodes,
        "edges": edges,
        "environment": {
            "bandwidth": float(environment.bandwidth),
            "latency": float(environment.latency),
            "weight_avg_latency": float(environment.weight_avg_latency),
            "weight_max_latency": float(environment.weight_max_latency),
            "weight_device_utilization": float(environment.weight_device_utilization),
            "latency_limit": float(environment.latency_limit),
            "max_frame_latency_limit": float(environment.max_frame_latency_limit),
            "batch_transfers": bool(environment.batch_transfers),
            "pipeline_unroll": int(environment.pipeline_unroll),
            "solver_threads": int(environment.solver_threads),
            "anneal_initial_temp": float(environment.anneal_initial_temp),
            "anneal_final_temp": float(environment.anneal_final_temp),
        },
    }
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def _required(record: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in record:
        raise ValueError(f"{label} is missing required field {key!r}.")
    return record[key]


def _placement(value: Any, label: str) -> str:
    placement = str(value).strip().lower()
    if placement not in PLACEMENT_VALUES:
        allowed = ", ".join(sorted(PLACEMENT_VALUES))
        raise ValueError(f"{label} must be one of: {allowed}.")
    return placement


def _non_negative_float(value: Any, label: str) -> float:
    number = _finite_float(value, label)
    if number < 0:
        raise ValueError(f"{label} must be non-negative.")
    return number


def _unit_float(value: Any, label: str) -> float:
    number = _non_negative_float(value, label)
    if number > 1:
        raise ValueError(f"{label} must be between 0 and 1.")
    return number


def _positive_float(value: Any, label: str) -> float:
    number = _finite_float(value, label)
    if number <= 0:
        raise ValueError(f"{label} must be greater than zero.")
    return number


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be an integer.")
    number = _finite_float(value, label)
    if not number.is_integer():
        raise ValueError(f"{label} must be an integer.")
    return int(number)


def _binary_int(value: Any, label: str) -> int:
    number = _integer(value, label)
    if number not in {0, 1}:
        raise ValueError(f"{label} must be 0 or 1.")
    return number


def _bool(value: Any, label: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"{label} must be boolean.")


def _finite_float(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite.")
    return number
