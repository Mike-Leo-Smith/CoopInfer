import math

from coopinfer.cost import annotate_vla_perf_profile_costs
from coopinfer.frontend import IRNode, ModelIR, TensorEdge, identity_scheduling_ir


def _profile():
    return {
        "schema_version": 1,
        "source": {"project": "NVlabs/vla-perf", "backend": "GenZ"},
        "precision": "bf16",
        "hardware": {"device": "RTX_4090", "host": "A100_80GB"},
        "models": {
            "vision": "pi0-vision",
            "vlm": "pi0-vlm",
            "action_expert": "pi0-action-expert",
        },
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
                "vision_total": 4.0,
                "vlm_total": 10.0,
                "ae_pass_total": 4.0,
                "projector": 0.1,
            },
            "host": {
                "vision_total": 2.0,
                "vlm_total": 20.0,
                "ae_pass_total": 8.0,
                "projector": 0.05,
            },
        },
    }


def _fine_ir(*, include_source_like_vlm: bool = False):
    nodes = [
        IRNode("input", "input", kind="input"),
        IRNode(
            "vlm0_a",
            "gemm",
            module_path="pi05.paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj",
        ),
        IRNode(
            "vlm0_b",
            "norm",
            module_path="pi05.paligemma_with_expert.paligemma.model.language_model.layers.0.input_layernorm",
        ),
        IRNode("wrapper", "transform", module_path="L__self__"),
        IRNode(
            "ae0_a",
            "gemm",
            module_path="pi05.paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj",
        ),
        IRNode(
            "ae0_b",
            "attention",
            module_path="pi05.paligemma_with_expert.gemma_expert.model.layers.0.self_attn",
        ),
        IRNode(
            "ae1_a",
            "gemm",
            module_path="pi05.paligemma_with_expert.gemma_expert.model.layers.1.mlp.gate_proj",
        ),
        IRNode(
            "ae1_b",
            "norm",
            module_path="pi05.paligemma_with_expert.gemma_expert.model.layers.1.input_layernorm",
        ),
        IRNode("output", "output", kind="output"),
    ]
    if include_source_like_vlm:
        nodes.append(
            IRNode(
                "vlm_static",
                "transform",
                module_path="pi05.paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.k_proj",
            )
        )

    chain_nodes = nodes[:9]
    edges = [
        TensorEdge(chain_nodes[i].id, chain_nodes[i + 1].id, 1024.0, f"t{i}")
        for i in range(len(chain_nodes) - 1)
    ]
    return ModelIR.from_parts(nodes, edges)


def test_vla_perf_costs_preserve_component_totals():
    costed = annotate_vla_perf_profile_costs(_fine_ir(), _profile())

    assert costed.metadata["cost_source"] == "vla-perf-stage-normalized"
    assert costed.metadata["cost_mode"] == "stage-normalized-fine-graph"

    assert costed.nodes["vlm0_a"].costs_ms == {"device": 5.0, "host": 10.0}
    assert costed.nodes["vlm0_b"].costs_ms == {"device": 5.0, "host": 10.0}
    for node_id in ("ae0_a", "ae0_b", "ae1_a", "ae1_b"):
        assert costed.nodes[node_id].costs_ms == {"device": 1.0, "host": 2.0}

    assert costed.nodes["wrapper"].costs_ms == {"device": 0.0, "host": 0.0}
    assert costed.nodes["input"].costs_ms == {"device": 0.0, "host": 0.0}
    assert costed.nodes["output"].costs_ms == {"device": 0.0, "host": 0.0}

    for resource, expected in (("device", 14.0), ("host", 28.0)):
        total = sum(node.costs_ms[resource] for node in costed.nodes.values())
        assert math.isclose(total, expected)

    audit = costed.metadata["cost_audit"]
    assert audit["device"]["vlm"]["node_count"] == 2
    assert audit["device"]["ae"]["node_count"] == 4
    assert audit["device"]["vlm"]["excluded_source_nodes"] == 0
    assert math.isclose(audit["device"]["vlm"]["allocated_total_ms"], 10.0)
    assert math.isclose(audit["host"]["ae"]["allocated_total_ms"], 8.0)


def test_vla_perf_costs_exclude_source_like_export_ops_and_redistribute_total():
    costed = annotate_vla_perf_profile_costs(
        _fine_ir(include_source_like_vlm=True), _profile()
    )

    assert costed.nodes["vlm_static"].costs_ms == {"device": 0.0, "host": 0.0}
    assert costed.nodes["vlm0_a"].costs_ms == {"device": 5.0, "host": 10.0}
    assert costed.nodes["vlm0_b"].costs_ms == {"device": 5.0, "host": 10.0}

    audit = costed.metadata["cost_audit"]
    assert audit["device"]["vlm"]["classified_node_count"] == 3
    assert audit["device"]["vlm"]["node_count"] == 2
    assert audit["device"]["vlm"]["excluded_source_nodes"] == 1
    assert math.isclose(audit["device"]["vlm"]["allocated_total_ms"], 10.0)


def test_vla_perf_costed_full_graph_enters_identity_adapter():
    costed = annotate_vla_perf_profile_costs(_fine_ir(), _profile())
    scheduling = identity_scheduling_ir(costed)

    assert len(scheduling.nodes) == len(costed.nodes)
    assert len(scheduling.edges) == len(costed.edges)
    assert scheduling.metadata["cost_source"] == "vla-perf-stage-normalized"
    assert scheduling.nodes["vlm0_a"].costs_ms["device"] == 5.0
    assert scheduling.nodes["ae0_a"].costs_ms["host"] == 2.0
