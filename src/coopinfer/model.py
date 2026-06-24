from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Union

import networkx as nx


SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class Environment:
    bandwidth: float = 50.0
    latency: float = 5.0
    weight_latency: float = 0.7
    latency_limit: float = 0.0
    batch_transfers: bool = False
    pipeline_unroll: int = 1


@dataclass(frozen=True)
class ProjectState:
    graph: nx.DiGraph
    environment: Environment


def graph_from_records(
    nodes: Iterable[Dict[str, Any]],
    edges: Iterable[Dict[str, Any]],
) -> nx.DiGraph:
    graph = nx.DiGraph()
    for node in nodes:
        node_id = str(node["id"]).strip()
        graph.add_node(
            node_id,
            name=str(node.get("name", node_id)).strip() or node_id,
            c_dev=float(node["c_dev"]),
            c_host=float(node["c_host"]),
            fixed_dev=bool(node.get("fixed_dev", False)),
            x=int(node.get("x", 0 if node.get("fixed_dev", False) else 1)),
        )

    for edge in edges:
        graph.add_edge(
            str(edge["source"]).strip(),
            str(edge["target"]).strip(),
            size=float(edge["size"]),
        )
    return graph


def records_from_graph(graph: nx.DiGraph) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    nodes = []
    for node_id, attrs in graph.nodes(data=True):
        nodes.append(
            {
                "id": node_id,
                "name": str(attrs.get("name", node_id)),
                "c_dev": float(attrs["c_dev"]),
                "c_host": float(attrs["c_host"]),
                "fixed_dev": bool(attrs.get("fixed_dev", False)),
            }
        )

    edges = []
    for source, target, attrs in graph.edges(data=True):
        edges.append(
            {
                "source": source,
                "target": target,
                "size": float(attrs["size"]),
            }
        )
    return nodes, edges


def load_from_json(path: Union[str, Path]) -> ProjectState:
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)

    if data.get("version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema version: {data.get('version')!r}")

    environment_data = data.get("environment", {})
    environment = Environment(
        bandwidth=float(environment_data.get("bandwidth", 50.0)),
        latency=float(environment_data.get("latency", 5.0)),
        weight_latency=float(environment_data.get("weight_latency", 0.7)),
        latency_limit=float(environment_data.get("latency_limit", 0.0)),
        batch_transfers=bool(environment_data.get("batch_transfers", False)),
        pipeline_unroll=max(1, int(environment_data.get("pipeline_unroll", 1))),
    )
    graph = graph_from_records(data.get("nodes", []), data.get("edges", []))
    return ProjectState(graph=graph, environment=environment)


def save_to_json(state: ProjectState, path: Union[str, Path]) -> None:
    nodes, edges = records_from_graph(state.graph)
    data = {
        "version": SCHEMA_VERSION,
        "nodes": nodes,
        "edges": edges,
        "environment": {
            "bandwidth": float(state.environment.bandwidth),
            "latency": float(state.environment.latency),
            "weight_latency": float(state.environment.weight_latency),
            "latency_limit": float(state.environment.latency_limit),
            "batch_transfers": bool(state.environment.batch_transfers),
            "pipeline_unroll": int(state.environment.pipeline_unroll),
        },
    }
    with Path(path).open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")
