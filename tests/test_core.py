import json

import networkx as nx

from coopinfer.evaluator import base_node_id, evaluate, infer_latency
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

    assert finishes["v1"] == 0.0
    assert starts["v2"] == 105.0
    assert latency == 107.0


def test_infer_latency_queues_independent_nodes_on_same_device():
    graph = graph_from_records(
        [
            {"id": "input", "c_dev": 0.0, "c_host": 0.0, "fixed_dev": True},
            {"id": "a", "c_dev": 10.0, "c_host": 1.0, "fixed_dev": False},
            {"id": "b", "c_dev": 20.0, "c_host": 1.0, "fixed_dev": False},
        ],
        [{"source": "input", "target": "a", "size": 0.0}, {"source": "input", "target": "b", "size": 0.0}],
    )
    latency, starts, finishes = infer_latency(
        graph, {"input": 0, "a": 0, "b": 0}, bandwidth=10.0, latency=5.0
    )

    assert starts["a"] == 0.0
    assert finishes["a"] == 10.0
    assert starts["b"] == 10.0
    assert finishes["b"] == 30.0
    assert latency == 30.0


def test_infer_latency_waits_for_host_queue_after_data_ready():
    graph = graph_from_records(
        [
            {"id": "input", "c_dev": 0.0, "c_host": 0.0, "fixed_dev": True},
            {"id": "dev_source", "c_dev": 1.0, "c_host": 1.0, "fixed_dev": True},
            {"id": "host_busy", "c_dev": 1.0, "c_host": 50.0, "fixed_dev": False},
            {"id": "host_after_data", "c_dev": 1.0, "c_host": 5.0, "fixed_dev": False},
        ],
        [
            {"source": "input", "target": "host_busy", "size": 0.0},
            {"source": "dev_source", "target": "host_after_data", "size": 0.0},
        ],
    )
    latency, starts, finishes = infer_latency(
        graph,
        {"input": 1, "dev_source": 0, "host_busy": 1, "host_after_data": 1},
        bandwidth=10.0,
        latency=5.0,
    )

    assert finishes["dev_source"] == 0.0
    assert starts["host_busy"] == 0.0
    assert finishes["host_busy"] == 50.0
    assert starts["host_after_data"] == 50.0
    assert finishes["host_after_data"] == 55.0
    assert latency == 55.0


def test_infer_latency_serializes_cross_device_transfers():
    graph = graph_from_records(
        [
            {"id": "source", "c_dev": 1.0, "c_host": 1.0, "fixed_dev": True},
            {"id": "h1", "c_dev": 1.0, "c_host": 1.0, "fixed_dev": False},
            {"id": "h2", "c_dev": 1.0, "c_host": 1.0, "fixed_dev": False},
        ],
        [
            {"source": "source", "target": "h1", "size": 1.0},
            {"source": "source", "target": "h2", "size": 1.0},
        ],
    )

    latency, starts, finishes = infer_latency(
        graph,
        {"source": 0, "h1": 1, "h2": 1},
        bandwidth=10.0,
        latency=5.0,
    )

    assert finishes["source"] == 0.0
    assert starts["h1"] == 105.0
    assert starts["h2"] == 210.0
    assert latency == 211.0


def test_infer_latency_batches_successive_outgoing_transfers():
    graph = graph_from_records(
        [
            {"id": "source", "c_dev": 1.0, "c_host": 1.0, "fixed_dev": True},
            {"id": "h1", "c_dev": 1.0, "c_host": 1.0, "fixed_dev": False},
            {"id": "h2", "c_dev": 1.0, "c_host": 1.0, "fixed_dev": False},
        ],
        [
            {"source": "source", "target": "h1", "size": 1.0},
            {"source": "source", "target": "h2", "size": 1.0},
        ],
    )

    latency, starts, _ = infer_latency(
        graph,
        {"source": 0, "h1": 1, "h2": 1},
        bandwidth=10.0,
        latency=5.0,
        batch_transfers=True,
    )

    assert starts["h1"] == 205.0
    assert starts["h2"] == 206.0
    assert latency == 207.0


def test_pipeline_unroll_preserves_same_label_fifo():
    graph = graph_from_records(
        [
            {"id": "input", "c_dev": 0.0, "c_host": 0.0, "fixed_dev": True},
            {"id": "stage", "c_dev": 10.0, "c_host": 1.0, "fixed_dev": False},
            {"id": "head", "c_dev": 20.0, "c_host": 2.0, "fixed_dev": False},
        ],
        [
            {"source": "input", "target": "stage", "size": 0.0},
            {"source": "stage", "target": "head", "size": 1.0},
        ],
    )
    latency, starts, finishes = infer_latency(
        graph,
        {"input": 0, "stage": 0, "head": 1},
        bandwidth=1000.0,
        latency=0.0,
        pipeline_unroll=2,
    )

    assert starts["input[f0]"] == 0.0
    assert finishes["input[f0]"] == 0.0
    assert starts["input[f1]"] == 0.0
    assert starts["stage[f0]"] == 0.0
    assert starts["stage[f1]"] == 10.0
    assert starts["head[f0]"] == 11.0
    assert starts["head[f1]"] == 21.0
    assert latency == 11.5


def test_source_period_delays_unrolled_input_frames():
    graph = graph_from_records(
        [
            {
                "id": "camera",
                "name": "Camera",
                "c_dev": 1.0,
                "c_host": 1.0,
                "fixed_dev": True,
                "source_period_ms": 33.0,
            },
            {"id": "head", "c_dev": 2.0, "c_host": 1.0, "fixed_dev": False},
        ],
        [{"source": "camera", "target": "head", "size": 0.0}],
    )

    latency, starts, finishes = infer_latency(
        graph,
        {"camera": 0, "head": 0},
        bandwidth=1000.0,
        latency=0.0,
        pipeline_unroll=3,
    )

    assert starts["camera[f0]"] == 0.0
    assert starts["camera[f1]"] == 33.0
    assert starts["camera[f2]"] == 66.0
    assert finishes["camera[f2]"] == 66.0
    assert finishes["head[f2]"] == 68.0
    assert latency == 68.0 / 3.0


def test_source_nodes_overlap_downstream_pipeline_work():
    graph = graph_from_records(
        [
            {
                "id": "camera",
                "c_dev": 100.0,
                "c_host": 100.0,
                "fixed_dev": True,
                "source_period_ms": 10.0,
            },
            {"id": "slow_out", "c_dev": 50.0, "c_host": 50.0, "fixed_dev": False},
        ],
        [{"source": "camera", "target": "slow_out", "size": 0.0}],
    )

    _, starts, finishes = infer_latency(
        graph,
        {"camera": 0, "slow_out": 1},
        bandwidth=1000.0,
        latency=0.0,
        pipeline_unroll=3,
    )

    assert starts["camera[f1]"] == 10.0
    assert finishes["camera[f1]"] == 10.0
    assert starts["camera[f2]"] == 20.0
    assert starts["slow_out[f1]"] == 50.0


def test_solver_accounts_for_source_period_in_pipeline_result():
    graph = graph_from_records(
        [
            {
                "id": "camera",
                "c_dev": 1.0,
                "c_host": 1.0,
                "fixed_dev": True,
                "source_period_ms": 33.0,
            },
            {"id": "head", "c_dev": 2.0, "c_host": 1.0, "fixed_dev": False},
        ],
        [{"source": "camera", "target": "head", "size": 0.0}],
    )

    result = solve(
        graph,
        bandwidth=1000.0,
        latency=0.0,
        weight_latency=1.0,
        algorithm="Enumerate",
        pipeline_unroll=3,
    )

    assert result.metrics.start_times["camera[f2]"] == 66.0
    assert result.metrics.latency >= 22.0


def test_evaluate_reports_pipeline_device_active_utilization():
    graph = sample_graph()
    result = evaluate(
        graph,
        {"v1": 0, "v2": 0},
        bandwidth=1000.0,
        latency=0.0,
        weight_latency=0.7,
        pipeline_unroll=2,
    )

    assert result.device_utilization == 1.0


def test_evaluate_reports_device_utilization():
    graph = sample_graph()
    result = evaluate(graph, {"v1": 0, "v2": 1}, bandwidth=10.0, latency=5.0, weight_latency=0.7)

    assert result.device_utilization == 0.0
    assert result.loss >= 0.0


def test_solver_keeps_fixed_nodes_on_device():
    graph = sample_graph()
    result = solve(graph, bandwidth=50.0, latency=5.0, weight_latency=1.0)

    assert result.mode == "Enumerate"
    assert result.assignment["v1"] == 0
    assert set(result.assignment) == {"v1", "v2"}


def test_solver_rejects_assignments_above_latency_limit():
    graph = sample_graph()
    result = solve(
        graph,
        bandwidth=50.0,
        latency=5.0,
        weight_latency=1.0,
        latency_limit=35.0,
    )

    assert result.metrics.latency <= 35.0
    assert result.assignment["v2"] == 0


def test_solver_raises_when_no_assignment_satisfies_latency_limit():
    graph = sample_graph()
    try:
        solve(graph, bandwidth=50.0, latency=5.0, weight_latency=1.0, latency_limit=19.0)
    except ValueError as exc:
        assert "No feasible assignment" in str(exc)
    else:
        raise AssertionError("Expected solve to reject all over-limit assignments")


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


def test_unrolled_operations_resolve_to_consistent_base_placement():
    graph = sample_graph()
    result = solve(
        graph,
        bandwidth=1000.0,
        latency=0.0,
        weight_latency=1.0,
        pipeline_unroll=3,
    )

    assert set(result.assignment) == {"v1", "v2"}
    for op_id in result.metrics.start_times:
        assert result.assignment[base_node_id(op_id)] in {0, 1}


def test_json_round_trip(tmp_path):
    graph = sample_graph()
    path = tmp_path / "config.json"
    save_to_json(ProjectState(graph, Environment(50.0, 5.0, 0.7, 120.0, True, 3)), path)

    loaded = load_from_json(path)
    assert loaded.environment == Environment(50.0, 5.0, 0.7, 120.0, True, 3)
    assert set(loaded.graph.nodes) == {"v1", "v2"}
    assert loaded.graph.nodes["v1"]["name"] == "Input"
    assert loaded.graph.edges["v1", "v2"]["size"] == 1.0

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == "1.0"
    assert data["environment"]["latency_limit"] == 120.0
    assert data["environment"]["batch_transfers"] is True
    assert data["environment"]["pipeline_unroll"] == 3
    assert data["nodes"][0]["source_period_ms"] == 0.0
    assert data["nodes"][0]["source_phase_ms"] == 0.0


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
