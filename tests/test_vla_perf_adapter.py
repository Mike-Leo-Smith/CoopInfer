from __future__ import annotations

import networkx as nx
import pytest

from coopinfer.vla_perf_adapter import (
    build_pi05_coopinfer_payload,
    graph_spec_from_profile,
    tensor_sizes_mb,
)


@pytest.fixture
def profile():
    return {
        "schema_version": 1,
        "source": {"project": "fixture"},
        "precision": "bf16",
        "hardware": {"device": "RTX_4090", "host": "H100"},
        "model": {
            "vision_layers": 27,
            "vlm_layers": 18,
            "ae_layers": 18,
            "vision_tokens_per_frame": 256,
            "vision_frames": 3,
            "language_tokens": 32,
            "state_tokens": 32,
            "action_chunk_size": 50,
            "denoising_steps": 10,
            "action_dof": 14,
            "image_resolution": 384,
            "vision_hidden_size": 1152,
            "vlm_hidden_size": 2048,
            "ae_hidden_size": 1024,
            "vlm_num_kv_heads": 1,
            "vlm_head_dim": 256,
        },
        "costs_ms": {
            "device": {
                "vision_total": 27.0,
                "vlm_total": 36.0,
                "ae_pass_total": 18.0,
                "projector": 0.2,
            },
            "host": {
                "vision_total": 13.5,
                "vlm_total": 18.0,
                "ae_pass_total": 9.0,
                "projector": 0.1,
            },
        },
    }


def test_builds_full_pi05_graph(profile):
    payload = build_pi05_coopinfer_payload(
        profile, bandwidth_mb_s=1250.0, latency_ms=0.2
    )
    nodes = {node["id"]: node for node in payload["nodes"]}
    assert len([node for node in nodes if node.startswith("vision_")]) == 27
    assert len([node for node in nodes if node.startswith("vlm_")]) == 18
    assert len([node for node in nodes if node.startswith("ae_s")]) == 180
    assert nodes["vlm_0"]["placement"] == "free"
    assert nodes["ae_s0_l0"]["placement"] == "device"
    assert nodes["vlm_0"]["c_dev"] == pytest.approx(2.0)
    assert nodes["vlm_0"]["c_host"] == pytest.approx(1.0)
    assert nodes["ae_s7_l9"]["c_dev"] == pytest.approx(1.0)
    assert nodes["ae_s7_l9"]["c_host"] == pytest.approx(0.5)


def test_kv_is_streamed_only_once(profile):
    payload = build_pi05_coopinfer_payload(
        profile, bandwidth_mb_s=1250.0, latency_ms=0.2
    )
    kv_edges = [
        edge
        for edge in payload["edges"]
        if edge["source"].startswith("vlm_")
        and edge["target"].startswith("ae_s")
    ]
    assert len(kv_edges) == 18
    assert all(edge["target"].startswith("ae_s0_") for edge in kv_edges)


def test_bf16_tensor_sizes(profile):
    spec = graph_spec_from_profile(profile)
    sizes = tensor_sizes_mb(spec, "bf16")
    # prefix = 3*256 + 32 language + 32 state = 832 tokens
    assert spec.prefix_tokens == 832
    expected_kv_bytes = 2 * 832 * 1 * 256 * 2
    assert sizes["kv_per_layer"] == pytest.approx(
        expected_kv_bytes / (1024 * 1024)
    )


def test_generated_graph_is_dag(profile):
    payload = build_pi05_coopinfer_payload(
        profile, bandwidth_mb_s=1250.0, latency_ms=0.2
    )
    graph = nx.DiGraph()
    for node in payload["nodes"]:
        graph.add_node(node["id"])
    for edge in payload["edges"]:
        graph.add_edge(edge["source"], edge["target"])
    assert nx.is_directed_acyclic_graph(graph)
