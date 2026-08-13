from __future__ import annotations

from coopinfer.frontend.ir import IRNode, ModelIR, TensorEdge
from coopinfer.frontend.pi05_probe import flatten_past_key_values, summarize_pi05_ir


def _node(node_id: str, module_path: str = "", kind: str = "op") -> IRNode:
    return IRNode(
        id=node_id,
        op="gemm" if kind == "op" else kind,
        target="aten.linear.default" if kind == "op" else kind,
        module_path=module_path,
        kind=kind,
    )


def test_wave_kv_candidate_is_discovered_from_existing_dependency():
    nodes = [
        _node("input", kind="input"),
        _node(
            "k0",
            "model.paligemma.model.language_model.layers.0.self_attn.k_proj",
        ),
        _node(
            "v0",
            "model.paligemma.model.language_model.layers.0.self_attn.v_proj",
        ),
        _node("kv_join", "cache.layer.0"),
        _node("expert0", "model.gemma_expert.model.layers.0.self_attn"),
        _node("output", kind="output"),
    ]
    edges = [
        TensorEdge("input", "k0", 8.0),
        TensorEdge("input", "v0", 8.0),
        TensorEdge("k0", "kv_join", 8.0),
        TensorEdge("v0", "kv_join", 8.0),
        TensorEdge("kv_join", "expert0", 16.0),
        TensorEdge("expert0", "output", 8.0),
    ]
    summary = summarize_pi05_ir(ModelIR.from_parts(nodes, edges))

    assert summary.wave_kv_candidate_layers == (0,)
    assert "k0" in summary.k_proj_nodes
    assert "v0" in summary.v_proj_nodes


def test_no_wave_kv_candidate_is_invented_without_dependency():
    nodes = [
        _node(
            "k0",
            "model.paligemma.model.language_model.layers.0.self_attn.k_proj",
        ),
        _node(
            "v0",
            "model.paligemma.model.language_model.layers.0.self_attn.v_proj",
        ),
        _node("expert0", "model.gemma_expert.model.layers.0.self_attn"),
    ]
    summary = summarize_pi05_ir(ModelIR.from_parts(nodes, []))

    assert summary.wave_kv_candidate_layers == ()
    assert any("no same-layer" in note.lower() for note in summary.notes)


def test_flatten_transformers_v5_dynamic_cache_layout():
    class TensorLike:
        def __init__(self, name: str):
            self.name = name
            self.shape = (1, 1, 8, 4)

    class Layer:
        def __init__(self, index: int):
            self.keys = TensorLike(f"k{index}")
            self.values = TensorLike(f"v{index}")

    class DynamicCacheLike:
        def __init__(self):
            self.layers = [Layer(0), Layer(1)]

    flat = flatten_past_key_values(DynamicCacheLike())

    assert [tensor.name for tensor in flat] == ["k0", "v0", "k1", "v1"]
