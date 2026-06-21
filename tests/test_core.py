import json

import networkx as nx

from coopinfer.evaluator import evaluate, infer_latency
from coopinfer.model import Environment, ProjectState, graph_from_records, load_from_json, save_to_json
from coopinfer.solver import solve


def sample_graph():
    return graph_from_records(
        [
            {"id": "v1", "name": "Input", "c_dev": 10.0, "c_host": 1.0, "fixed_dev": True},
            {"id": "v2", "name": "Head", "c_dev": 20.0, "c_host": 2.0, "fixed_dev": False},
        ],
        [{"source": "v1", "target": "v2", "size": 1.0}],
    )


def test_infer_latency_adds_cross_device_transfer():
    graph = sample_graph()
    latency, starts, finishes = infer_latency(
        graph, {"v1": 0, "v2": 1}, bandwidth=10.0, latency=5.0
    )

    assert finishes["v1"] == 10.0
    assert starts["v2"] == 115.0
    assert latency == 117.0


def test_infer_latency_queues_independent_nodes_on_same_device():
    graph = graph_from_records(
        [
            {"id": "a", "c_dev": 10.0, "c_host": 1.0, "fixed_dev": False},
            {"id": "b", "c_dev": 20.0, "c_host": 1.0, "fixed_dev": False},
        ],
        [],
    )
    latency, starts, finishes = infer_latency(
        graph, {"a": 0, "b": 0}, bandwidth=10.0, latency=5.0
    )

    assert starts["a"] == 0.0
    assert finishes["a"] == 10.0
    assert starts["b"] == 10.0
    assert finishes["b"] == 30.0
    assert latency == 30.0


def test_infer_latency_waits_for_host_queue_after_data_ready():
    graph = graph_from_records(
        [
            {"id": "dev_source", "c_dev": 1.0, "c_host": 1.0, "fixed_dev": True},
            {"id": "host_busy", "c_dev": 1.0, "c_host": 50.0, "fixed_dev": False},
            {"id": "host_after_data", "c_dev": 1.0, "c_host": 5.0, "fixed_dev": False},
        ],
        [{"source": "dev_source", "target": "host_after_data", "size": 0.0}],
    )
    latency, starts, finishes = infer_latency(
        graph,
        {"dev_source": 0, "host_busy": 1, "host_after_data": 1},
        bandwidth=10.0,
        latency=5.0,
    )

    assert finishes["dev_source"] == 1.0
    assert starts["host_busy"] == 0.0
    assert finishes["host_busy"] == 50.0
    assert starts["host_after_data"] == 50.0
    assert finishes["host_after_data"] == 55.0
    assert latency == 55.0


def test_evaluate_reports_device_utilization():
    graph = sample_graph()
    result = evaluate(graph, {"v1": 0, "v2": 1}, bandwidth=10.0, latency=5.0, weight_latency=0.7)

    assert result.device_utilization == 10.0 / 30.0
    assert result.loss >= 0.0


def test_solver_keeps_fixed_nodes_on_device():
    graph = sample_graph()
    result = solve(graph, bandwidth=50.0, latency=5.0, weight_latency=1.0)

    assert result.mode == "Enumerate"
    assert result.assignment["v1"] == 0
    assert set(result.assignment) == {"v1", "v2"}


def test_solver_supports_explicit_random_and_annealing_modes():
    graph = sample_graph()
    random_result = solve(
        graph,
        bandwidth=50.0,
        latency=5.0,
        weight_latency=1.0,
        algorithm="Random Search",
        heuristic_iterations=10,
    )
    anneal_result = solve(
        graph,
        bandwidth=50.0,
        latency=5.0,
        weight_latency=1.0,
        algorithm="Simulated Annealing",
        heuristic_iterations=10,
    )

    assert random_result.mode == "Random Search"
    assert anneal_result.mode == "Simulated Annealing"
    assert random_result.assignment["v1"] == 0
    assert anneal_result.assignment["v1"] == 0


def test_json_round_trip(tmp_path):
    graph = sample_graph()
    path = tmp_path / "config.json"
    save_to_json(ProjectState(graph, Environment(50.0, 5.0, 0.7)), path)

    loaded = load_from_json(path)
    assert loaded.environment == Environment(50.0, 5.0, 0.7)
    assert set(loaded.graph.nodes) == {"v1", "v2"}
    assert loaded.graph.nodes["v1"]["name"] == "Input"
    assert loaded.graph.edges["v1", "v2"]["size"] == 1.0

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == "1.0"


def test_graph_from_records_can_be_checked_for_cycles():
    graph = graph_from_records(
        [
            {"id": "a", "c_dev": 1, "c_host": 1, "fixed_dev": False},
            {"id": "b", "c_dev": 1, "c_host": 1, "fixed_dev": False},
        ],
        [
            {"source": "a", "target": "b", "size": 1},
            {"source": "b", "target": "a", "size": 1},
        ],
    )

    assert not nx.is_directed_acyclic_graph(graph)
