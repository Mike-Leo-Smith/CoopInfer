from __future__ import annotations

from dataclasses import dataclass

from coopinfer.cost import annotate_layer_genz_costs, infer_layer_work
from coopinfer.frontend import IRNode, ModelIR, TensorEdge, detect_layer_groups


def _tensor_row(shape, *, source="", input_kind="USER_INPUT", dtype="torch.float32"):
    numel = 1
    for dim in shape:
        numel *= dim
    row = {
        "shape": list(shape),
        "dtype": dtype,
        "numel": numel,
        "element_size": 4,
        "nbytes": numel * 4,
    }
    if source:
        row["source"] = source
        row["input_kind"] = input_kind
    return row


def _op(node_id: str, path: str, *, op="memory", target="aten.clone.default") -> IRNode:
    row = _tensor_row((1, 8))
    return IRNode(
        node_id,
        op,
        target=target,
        module_path=path,
        metadata={
            "input_tensors": [dict(row)],
            "output_tensors": [dict(row)],
        },
    )


def test_layer_detection_is_complete_and_separates_repeated_stacks():
    nodes = [
        IRNode("input", "input", kind="input"),
        _op("a0", "model.backbone.layers.0.attn"),
        _op("a1", "model.backbone.layers.1.attn"),
        _op("bridge", "model.bridge"),
        _op("b0", "model.expert.layers.0.attn"),
        _op("b1", "model.expert.layers.1.attn"),
        IRNode("output", "output", kind="output"),
    ]
    edges = [
        TensorEdge("input", "a0", 32, "x"),
        TensorEdge("a0", "a1", 32, "a0_out"),
        TensorEdge("a0", "bridge", 32, "escape"),
        TensorEdge("bridge", "b0", 32, "bridge_out"),
        TensorEdge("a1", "b1", 32, "cross_stack"),
        TensorEdge("b0", "b1", 32, "b0_out"),
        TensorEdge("b1", "output", 32, "y"),
    ]
    ir = ModelIR.from_parts(nodes, edges)
    grouping = detect_layer_groups(ir)

    assert set(grouping.member_to_group) == set(ir.nodes)
    assert len(grouping.stack_layers) == 2
    layer_groups = [g for g in grouping.groups.values() if g.kind == "layer"]
    assert len(layer_groups) == 4
    assert grouping.member_to_group["a0"] != grouping.member_to_group["b0"]
    assert any(g.kind == "aux_region" and "bridge" in g.members for g in grouping.groups.values())


def test_layer_work_does_not_recharge_internal_activation_hbm():
    input_node = IRNode("input", "input", kind="input")
    first = _op("first", "model.layers.0.first")
    second = _op("second", "model.layers.0.second")
    output_node = IRNode("output", "output", kind="output")
    ir = ModelIR.from_parts(
        [input_node, first, second, output_node],
        [
            TensorEdge("input", "first", 32, "x", metadata={"tensors": [_tensor_row((1, 8))]}),
            TensorEdge("first", "second", 32, "internal", metadata={"tensors": [_tensor_row((1, 8))]}),
            TensorEdge("second", "output", 32, "y", metadata={"tensors": [_tensor_row((1, 8))]}),
        ],
    )
    grouping = detect_layer_groups(ir, min_repeated_layers=1)
    layer_id = next(g.id for g in grouping.groups.values() if g.kind == "layer")
    work = infer_layer_work(ir, grouping, layer_id, precision="bf16")

    # bf16 boundary input + output: 16 B + 16 B. The internal 16 B tensor is
    # not separately recharged as layer-level HBM traffic.
    assert work.parameter_bytes == 0.0
    assert work.input_bytes == 16.0
    assert work.output_bytes == 16.0
    assert work.memory_bytes == 32.0


@dataclass
class _FakeSystem:
    flops: float
    offchip_mem_bw: float


def _fake_system_factory(hardware, precision, ceff, meff):
    del precision, ceff, meff
    if hardware == "fast":
        return _FakeSystem(flops=2.0e12, offchip_mem_bw=2.0e12)
    return _FakeSystem(flops=1.0e12, offchip_mem_bw=1.0e12)


def test_layer_genz_cost_is_computed_from_layer_workload_not_fine_latency_sum():
    input_node = IRNode("input", "input", kind="input")
    a = _op("a", "model.layers.0.a")
    b = _op("b", "model.layers.0.b")
    output_node = IRNode("output", "output", kind="output")
    ir = ModelIR.from_parts(
        [input_node, a, b, output_node],
        [
            TensorEdge("input", "a", 32, "x", metadata={"tensors": [_tensor_row((1, 8))]}),
            TensorEdge("a", "b", 32, "internal", metadata={"tensors": [_tensor_row((1, 8))]}),
            TensorEdge("b", "output", 32, "y", metadata={"tensors": [_tensor_row((1, 8))]}),
        ],
    )
    # Deliberately poison fine costs. Layer costing must ignore these values.
    ir.nodes["a"].costs_ms = {"device": 999.0, "host": 999.0}
    ir.nodes["b"].costs_ms = {"device": 999.0, "host": 999.0}

    grouping = detect_layer_groups(ir, min_repeated_layers=1)
    schedule = annotate_layer_genz_costs(
        ir,
        grouping,
        {"device": "fast", "host": "slow"},
        precision="bf16",
        system_factory=_fake_system_factory,
    )
    layer = next(node for node in schedule.nodes.values() if node.metadata["group_kind"] == "layer")

    assert layer.costs_ms["device"] < 1.0
    assert layer.costs_ms["host"] < 1.0
    assert layer.costs_ms["device"] <= layer.costs_ms["host"]
    assert layer.metadata["cost_source"] == "genz-layer-external-roofline"
    assert layer.metadata["layer_work"]["memory_bytes"] == 32.0
