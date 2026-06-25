import json
import math

import networkx as nx

from coopinfer.evaluator import base_node_id, edge_transfer_ms, evaluate, infer_latency, infer_schedule
from coopinfer.model import (
    Environment,
    ProjectState,
    graph_from_records,
    load_from_json,
    save_to_json,
    validate_environment,
)
from coopinfer.solver import solve


LATENCY_ONLY_WEIGHTS = {
    "weight_avg_latency": 1.0,
    "weight_max_latency": 0.0,
    "weight_device_utilization": 0.0,
}

DEFAULT_OBJECTIVE_WEIGHTS = {
    "weight_avg_latency": 0.7,
    "weight_max_latency": 0.3,
    "weight_device_utilization": 0.3,
}


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


def test_infer_schedule_keeps_legacy_tuple_shape():
    graph = sample_graph()
    schedule = infer_schedule(
        graph,
        {"v1": 0, "v2": 1},
        bandwidth=10.0,
        latency=5.0,
    )

    assert len(schedule) == 4
    latency, starts, finishes, transfers = schedule
    assert latency == 107.0
    assert starts["v2"] == 105.0
    assert finishes["v2"] == 107.0
    assert len(transfers) == 1


def test_edge_transfer_ms_rejects_invalid_size():
    for value in [-1.0, math.inf, math.nan]:
        try:
            edge_transfer_ms(value, 10.0, 5.0)
        except ValueError as exc:
            assert "Transfer size" in str(exc)
        else:
            raise AssertionError(f"Expected invalid transfer size to fail: {value}")


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

    assert starts["a"] in {0.0, 20.0}
    assert starts["b"] in {0.0, 10.0}
    assert starts["a"] != starts["b"]
    assert finishes["a"] - starts["a"] == 10.0
    assert finishes["b"] - starts["b"] == 20.0
    assert latency == 30.0


def test_evaluator_fills_ready_work_while_other_work_waits_for_transfer():
    graph = graph_from_records(
        [
            {"id": "remote_input", "c_dev": 0.0, "c_host": 0.0, "fixed_dev": False},
            {"id": "delayed_dev", "c_dev": 10.0, "c_host": 10.0, "fixed_dev": False},
            {"id": "local_input", "c_dev": 0.0, "c_host": 0.0, "fixed_dev": True},
            {"id": "ready_dev", "c_dev": 5.0, "c_host": 5.0, "fixed_dev": False},
        ],
        [
            {"source": "remote_input", "target": "delayed_dev", "size": 1.0},
            {"source": "local_input", "target": "ready_dev", "size": 0.0},
        ],
    )

    result = evaluate(
        graph,
        {"remote_input": 1, "delayed_dev": 0, "local_input": 0, "ready_dev": 0},
        bandwidth=10.0,
        latency=5.0,
        **LATENCY_ONLY_WEIGHTS,
    )

    assert result.start_times["ready_dev"] == 0.0
    assert result.finish_times["ready_dev"] == 5.0
    assert result.start_times["delayed_dev"] == 105.0
    assert result.finish_times["delayed_dev"] == 115.0
    assert result.latency == 115.0


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


def test_evaluate_reports_max_frame_latency_for_unrolled_pipeline():
    graph = graph_from_records(
        [
            {
                "id": "camera",
                "c_dev": 1.0,
                "c_host": 1.0,
                "fixed_dev": True,
                "source_period_ms": 10.0,
            },
            {"id": "slow_out", "c_dev": 50.0, "c_host": 50.0, "fixed_dev": False},
        ],
        [{"source": "camera", "target": "slow_out", "size": 0.0}],
    )

    result = evaluate(
        graph,
        {"camera": 0, "slow_out": 1},
        bandwidth=1000.0,
        latency=0.0,
        **LATENCY_ONLY_WEIGHTS,
        pipeline_unroll=3,
    )

    assert result.latency == 50.0
    assert result.max_frame_latency == 50.0
    assert result.transfer_records[1].start == 50.0
    assert result.transfer_records[2].start == 100.0
    assert result.avg_latency_loss >= 0.0
    assert result.max_frame_latency_loss >= 0.0
    assert result.device_utilization_loss >= 0.0


def test_evaluator_delays_fast_join_branch_to_reduce_tail_frame_latency():
    graph = graph_from_records(
        [
            {
                "id": "input",
                "c_dev": 0.0,
                "c_host": 0.0,
                "fixed_dev": True,
                "source_period_ms": 10.0,
            },
            {"id": "slow_branch", "c_dev": 50.0, "c_host": 50.0, "fixed_dev": True},
            {"id": "fast_branch", "c_dev": 1.0, "c_host": 1.0, "fixed_dev": False},
            {"id": "merge", "c_dev": 1.0, "c_host": 1.0, "fixed_dev": True},
        ],
        [
            {"source": "input", "target": "slow_branch", "size": 0.0},
            {"source": "input", "target": "fast_branch", "size": 0.0},
            {"source": "slow_branch", "target": "merge", "size": 0.0},
            {"source": "fast_branch", "target": "merge", "size": 0.0},
        ],
    )

    result = evaluate(
        graph,
        {"input": 0, "slow_branch": 0, "fast_branch": 1, "merge": 0},
        bandwidth=1000.0,
        latency=0.0,
        **LATENCY_ONLY_WEIGHTS,
        pipeline_unroll=3,
    )

    assert result.latency == 51.0
    assert result.max_frame_latency == 51.0
    assert result.start_times["fast_branch[f0]"] == 49.0
    assert result.start_times["fast_branch[f1]"] == 100.0
    assert result.start_times["fast_branch[f2]"] == 151.0
    assert result.start_times["slow_branch[f1]"] == 51.0
    assert result.start_times["slow_branch[f2]"] == 102.0


def test_evaluator_preserves_inter_frame_pipeline_overlap():
    graph = graph_from_records(
        [
            {"id": "input", "c_dev": 0.0, "c_host": 0.0, "fixed_dev": True},
            {"id": "stage", "c_dev": 50.0, "c_host": 50.0, "fixed_dev": True},
            {"id": "head", "c_dev": 100.0, "c_host": 100.0, "fixed_dev": False},
        ],
        [
            {"source": "input", "target": "stage", "size": 0.0},
            {"source": "stage", "target": "head", "size": 0.0},
        ],
    )

    result = evaluate(
        graph,
        {"input": 0, "stage": 0, "head": 1},
        bandwidth=1000.0,
        latency=0.0,
        **LATENCY_ONLY_WEIGHTS,
        pipeline_unroll=3,
    )

    assert result.start_times["stage[f0]"] == 0.0
    assert result.start_times["stage[f1]"] == 50.0
    assert result.start_times["stage[f2]"] == 100.0
    assert result.start_times["stage[f1]"] < result.finish_times["head[f0]"]
    assert result.start_times["stage[f2]"] < result.finish_times["head[f0]"]
    assert result.start_times["head[f0]"] == 50.0
    assert result.start_times["head[f1]"] == 150.0
    assert result.latency == 350.0 / 3.0


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
        **LATENCY_ONLY_WEIGHTS,
        algorithm="Enumerate",
        pipeline_unroll=3,
    )

    assert result.metrics.start_times["camera[f2]"] == 66.0
    assert result.metrics.latency == 67.0 / 3.0
    assert result.metrics.max_frame_latency == 1.0


def test_evaluate_reports_pipeline_device_active_utilization():
    graph = sample_graph()
    result = evaluate(
        graph,
        {"v1": 0, "v2": 0},
        bandwidth=1000.0,
        latency=0.0,
        **DEFAULT_OBJECTIVE_WEIGHTS,
        pipeline_unroll=2,
    )

    assert result.device_utilization == 1.0


def test_evaluate_reports_device_utilization():
    graph = sample_graph()
    result = evaluate(
        graph,
        {"v1": 0, "v2": 1},
        bandwidth=10.0,
        latency=5.0,
        **DEFAULT_OBJECTIVE_WEIGHTS,
    )

    assert result.device_utilization == 0.0
    assert result.loss >= 0.0


def test_solver_keeps_fixed_nodes_on_device():
    graph = sample_graph()
    result = solve(
        graph,
        bandwidth=50.0,
        latency=5.0,
        **LATENCY_ONLY_WEIGHTS,
    )

    assert result.mode == "Enumerate"
    assert result.assignment["v1"] == 0
    assert set(result.assignment) == {"v1", "v2"}


def test_solver_rejects_assignments_above_latency_limit():
    graph = sample_graph()
    result = solve(
        graph,
        bandwidth=50.0,
        latency=5.0,
        **LATENCY_ONLY_WEIGHTS,
        latency_limit=35.0,
    )

    assert result.metrics.latency <= 35.0
    assert result.assignment["v2"] == 0


def test_solver_rejects_on_max_frame_latency_limit_not_amortized_latency_limit():
    graph = graph_from_records(
        [
            {"id": "input", "c_dev": 0.0, "c_host": 0.0, "fixed_dev": True},
            {
                "id": "stage",
                "c_dev": 50.0,
                "c_host": 30.0,
                "fixed_dev": False,
            },
            {"id": "head", "c_dev": 999.0, "c_host": 60.0, "fixed_dev": False},
        ],
        [
            {"source": "input", "target": "stage", "size": 0.0},
            {"source": "stage", "target": "head", "size": 0.0},
        ],
    )

    try:
        solve(
            graph,
            bandwidth=1000.0,
            latency=0.0,
            **LATENCY_ONLY_WEIGHTS,
            latency_limit=80.0,
            max_frame_latency_limit=80.0,
            pipeline_unroll=3,
            algorithm="Enumerate",
        )
    except ValueError as exc:
        assert "No feasible assignment" in str(exc)
    else:
        raise AssertionError("Expected max-frame E2E latency to reject the schedule")


def test_solver_allows_high_max_frame_when_only_amortized_limit_is_set():
    graph = graph_from_records(
        [
            {"id": "input", "c_dev": 0.0, "c_host": 0.0, "fixed_dev": True},
            {
                "id": "stage",
                "c_dev": 50.0,
                "c_host": 30.0,
                "fixed_dev": False,
            },
            {"id": "head", "c_dev": 999.0, "c_host": 60.0, "fixed_dev": False},
        ],
        [
            {"source": "input", "target": "stage", "size": 0.0},
            {"source": "stage", "target": "head", "size": 0.0},
        ],
    )

    result = solve(
        graph,
        bandwidth=1000.0,
        latency=0.0,
        **LATENCY_ONLY_WEIGHTS,
        latency_limit=80.0,
        pipeline_unroll=3,
        algorithm="Enumerate",
    )

    assert math.isclose(result.metrics.latency, 230.0 / 3.0)
    assert result.metrics.max_frame_latency == 130.0


def test_solver_max_latency_weight_changes_objective_choice():
    graph = graph_from_records(
        [
            {"id": "input", "c_dev": 0.0, "c_host": 0.0, "fixed_dev": True},
            {
                "id": "stage",
                "c_dev": 50.0,
                "c_host": 30.0,
                "fixed_dev": False,
            },
            {"id": "head", "c_dev": 999.0, "c_host": 60.0, "fixed_dev": False},
        ],
        [
            {"source": "input", "target": "stage", "size": 0.0},
            {"source": "stage", "target": "head", "size": 0.0},
        ],
    )

    avg_result = solve(
        graph,
        bandwidth=1000.0,
        latency=0.0,
        weight_avg_latency=1.0,
        weight_max_latency=0.0,
        weight_device_utilization=0.0,
        pipeline_unroll=3,
        algorithm="Enumerate",
    )
    max_result = solve(
        graph,
        bandwidth=1000.0,
        latency=0.0,
        weight_avg_latency=0.0,
        weight_max_latency=1.0,
        weight_device_utilization=0.0,
        pipeline_unroll=3,
        algorithm="Enumerate",
    )

    assert avg_result.assignment["stage"] == 0
    assert avg_result.assignment["head"] == 1
    assert math.isclose(avg_result.metrics.latency, 230.0 / 3.0)
    assert avg_result.metrics.max_frame_latency == 130.0
    assert max_result.assignment["stage"] == 1
    assert max_result.assignment["head"] == 1
    assert max_result.metrics.latency == 90.0
    assert max_result.metrics.max_frame_latency == 90.0


def test_latency_ignores_delayed_source_phase():
    graph = graph_from_records(
        [
            {
                "id": "sensor",
                "c_dev": 0.0,
                "c_host": 0.0,
                "fixed_dev": True,
                "source_phase_ms": 100.0,
            },
            {"id": "head", "c_dev": 5.0, "c_host": 5.0, "fixed_dev": False},
        ],
        [{"source": "sensor", "target": "head", "size": 0.0}],
    )

    result = evaluate(
        graph,
        {"sensor": 0, "head": 0},
        bandwidth=1000.0,
        latency=0.0,
        **LATENCY_ONLY_WEIGHTS,
    )

    assert result.start_times["sensor"] == 100.0
    assert result.start_times["head"] == 100.0
    assert result.latency == 5.0
    assert result.max_frame_latency == 5.0


def test_solver_uses_latency_when_baseline_scales_collapse():
    graph = graph_from_records(
        [
            {"id": "source", "c_dev": 0.0, "c_host": 0.0, "fixed_dev": False},
            {"id": "a", "c_dev": 10.0, "c_host": 10.0, "fixed_dev": False},
            {"id": "b", "c_dev": 10.0, "c_host": 10.0, "fixed_dev": False},
        ],
        [
            {"source": "source", "target": "a", "size": 0.0},
            {"source": "source", "target": "b", "size": 0.0},
        ],
    )

    result = solve(
        graph,
        bandwidth=1000.0,
        latency=0.0,
        **LATENCY_ONLY_WEIGHTS,
        algorithm="Enumerate",
    )

    assert result.metrics.latency == 10.0
    assert result.metrics.loss >= 0.0


def test_solver_raises_when_no_assignment_satisfies_latency_limit():
    graph = sample_graph()
    try:
        solve(
            graph,
            bandwidth=50.0,
            latency=5.0,
            **LATENCY_ONLY_WEIGHTS,
            latency_limit=19.0,
        )
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
        **LATENCY_ONLY_WEIGHTS,
        algorithm="Random Search",
        heuristic_iterations=10,
    )
    anneal_result = solve(
        graph,
        bandwidth=50.0,
        latency=5.0,
        **LATENCY_ONLY_WEIGHTS,
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
        **LATENCY_ONLY_WEIGHTS,
        pipeline_unroll=3,
    )

    assert set(result.assignment) == {"v1", "v2"}
    for op_id in result.metrics.start_times:
        assert result.assignment[base_node_id(op_id)] in {0, 1}


def test_json_round_trip(tmp_path):
    graph = sample_graph()
    graph.nodes["v2"]["x"] = 0
    path = tmp_path / "config.json"
    environment = Environment(
        bandwidth=50.0,
        latency=5.0,
        **DEFAULT_OBJECTIVE_WEIGHTS,
        latency_limit=120.0,
        batch_transfers=True,
        pipeline_unroll=3,
        max_frame_latency_limit=180.0,
    )
    save_to_json(ProjectState(graph, environment), path)

    loaded = load_from_json(path)
    assert loaded.environment == environment
    assert set(loaded.graph.nodes) == {"v1", "v2"}
    assert loaded.graph.nodes["v1"]["name"] == "Input"
    assert loaded.graph.nodes["v1"]["x"] == 0
    assert loaded.graph.nodes["v2"]["x"] == 0
    assert loaded.graph.edges["v1", "v2"]["size"] == 1.0

    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["version"] == "1.0"
    assert data["nodes"][0]["x"] == 0
    assert data["nodes"][1]["x"] == 0
    assert data["environment"]["latency_limit"] == 120.0
    assert data["environment"]["max_frame_latency_limit"] == 180.0
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


def test_graph_from_records_rejects_invalid_records():
    invalid_cases = [
        (
            [{"id": "a", "c_dev": -1, "c_host": 1}],
            [],
            "non-negative",
        ),
        (
            [{"id": "a", "c_dev": 1, "c_host": 1}],
            [{"source": "a", "target": "missing", "size": 1}],
            "unknown nodes",
        ),
        (
            [
                {"id": "a", "c_dev": 1, "c_host": 1},
                {"id": "b", "c_dev": 1, "c_host": 1},
            ],
            [{"source": "a", "target": "b", "size": -1}],
            "non-negative",
        ),
        (
            [{"id": "a", "c_dev": 1, "c_host": 1, "fixed_dev": "false"}],
            [],
            "boolean",
        ),
        (
            [{"id": "a", "c_dev": 1, "c_host": 1, "x": 1.5}],
            [],
            "integer",
        ),
    ]

    for nodes, edges, message in invalid_cases:
        try:
            graph_from_records(nodes, edges)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"Expected invalid records to fail: {nodes}, {edges}")


def test_environment_validation_rejects_invalid_values():
    invalid_cases = [
        (Environment(bandwidth=0.0), "greater than zero"),
        (Environment(bandwidth=math.inf), "finite"),
        (Environment(latency=-1.0), "non-negative"),
        (Environment(weight_avg_latency=-1.0), "non-negative"),
        (Environment(weight_max_latency=-1.0), "non-negative"),
        (Environment(weight_device_utilization=-1.0), "non-negative"),
        (Environment(weight_avg_latency=1.1), "between 0 and 1"),
        (Environment(weight_max_latency=1.1), "between 0 and 1"),
        (Environment(weight_device_utilization=1.1), "between 0 and 1"),
        (Environment(latency_limit=-1.0), "non-negative"),
        (Environment(max_frame_latency_limit=-1.0), "non-negative"),
        (Environment(pipeline_unroll=0), "at least 1"),
        (Environment(pipeline_unroll=1.5), "integer"),
        (Environment(batch_transfers="false"), "boolean"),
    ]

    for environment, message in invalid_cases:
        try:
            validate_environment(environment)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"Expected invalid environment to fail: {environment}")


def test_environment_validation_keeps_independent_unit_objective_weights():
    environment = validate_environment(
        Environment(
            weight_avg_latency=1.0,
            weight_max_latency=0.5,
            weight_device_utilization=0.25,
        )
    )

    assert environment.weight_avg_latency == 1.0
    assert environment.weight_max_latency == 0.5
    assert environment.weight_device_utilization == 0.25


def test_legacy_weight_latency_json_loads_as_split_weights(tmp_path):
    path = tmp_path / "legacy_environment.json"
    path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "nodes": [{"id": "a", "c_dev": 1, "c_host": 1}],
                "edges": [],
                "environment": {"weight_latency": 0.25},
            }
        ),
        encoding="utf-8",
    )

    loaded = load_from_json(path)

    assert loaded.environment.weight_avg_latency == 0.25
    assert loaded.environment.weight_max_latency == 0.0
    assert loaded.environment.weight_device_utilization == 0.75


def test_load_from_json_rejects_invalid_environment(tmp_path):
    path = tmp_path / "invalid_environment.json"
    path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "nodes": [{"id": "a", "c_dev": 1, "c_host": 1}],
                "edges": [],
                "environment": {"latency": -1},
            }
        ),
        encoding="utf-8",
    )

    try:
        load_from_json(path)
    except ValueError as exc:
        assert "latency" in str(exc)
    else:
        raise AssertionError("Expected invalid environment JSON to fail")
