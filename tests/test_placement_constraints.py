from __future__ import annotations

import json

from coopinfer.evaluator import evaluate, graph_to_core_data
from coopinfer.model import Environment, ProjectState, graph_from_records, load_from_json, save_to_json
from coopinfer.solver import solve


def _placement_graph():
    return graph_from_records(
        [
            {
                "id": "input",
                "name": "Input",
                "c_dev": 0.0,
                "c_host": 0.0,
                "placement": "device",
            },
            {
                "id": "host_stage",
                "name": "Host stage",
                "c_dev": 0.5,
                "c_host": 0.5,
                "placement": "host",
            },
            {
                "id": "free_stage",
                "name": "Free stage",
                "c_dev": 0.1,
                "c_host": 10.0,
                "placement": "free",
                "x": 0,
            },
        ],
        [
            {"source": "input", "target": "host_stage", "size": 0.0},
            {"source": "host_stage", "target": "free_stage", "size": 0.0},
        ],
    )


def test_legacy_fixed_dev_maps_to_device_placement():
    graph = graph_from_records(
        [
            {
                "id": "legacy",
                "c_dev": 1.0,
                "c_host": 2.0,
                "fixed_dev": True,
            }
        ],
        [],
    )
    assert graph.nodes["legacy"]["placement"] == "device"
    assert graph.nodes["legacy"]["x"] == 0


def test_host_placement_round_trips_without_bumping_default_schema(tmp_path):
    graph = _placement_graph()
    path = tmp_path / "placement.json"
    save_to_json(ProjectState(graph=graph, environment=Environment()), path)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == "1.0"
    host_node = next(node for node in payload["nodes"] if node["id"] == "host_stage")
    assert host_node["placement"] == "host"
    assert host_node["x"] == 1

    loaded = load_from_json(path)
    assert loaded.graph.nodes["host_stage"]["placement"] == "host"
    assert loaded.graph.nodes["host_stage"]["x"] == 1


def test_native_wire_uses_fixed_mask_plus_initial_resource():
    graph = _placement_graph()
    data = graph_to_core_data(graph)
    index = {node_id: i for i, node_id in enumerate(data["ids"])}

    assert data["fixed_dev"][index["input"]] is True
    assert data["x_initial"][index["input"]] == 0
    assert data["fixed_dev"][index["host_stage"]] is True
    assert data["x_initial"][index["host_stage"]] == 1
    assert data["fixed_dev"][index["free_stage"]] is False
    assert data["x_initial"][index["free_stage"]] == 0


def test_solver_enforces_both_device_and_host_pins():
    graph = _placement_graph()
    result = solve(
        graph,
        bandwidth=1000.0,
        latency=0.0,
        weight_avg_latency=1.0,
        weight_max_latency=0.0,
        weight_device_utilization=0.0,
        algorithm="Enumerate",
    )

    assert result.assignment["input"] == 0
    assert result.assignment["host_stage"] == 1
    assert result.assignment["free_stage"] == 0
    assert result.iterations == 2


def test_evaluate_keeps_what_if_assignments_independent_of_pins():
    graph = _placement_graph()
    result = evaluate(
        graph,
        {"input": 0, "host_stage": 0, "free_stage": 0},
        bandwidth=1000.0,
        latency=0.0,
        weight_avg_latency=1.0,
        weight_max_latency=0.0,
        weight_device_utilization=0.0,
    )

    assert result.latency >= 0.0
