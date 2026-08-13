from __future__ import annotations

import math

import pytest

from coopinfer.cost import annotate_genz_costs, infer_node_work
from coopinfer.frontend import IRNode, ModelIR, TensorEdge, model_ir_from_dict, model_ir_to_dict


def _tensor(shape, dtype="torch.float32"):
    numel = 1
    for dim in shape:
        numel *= dim
    return {
        "shape": list(shape),
        "dtype": dtype,
        "numel": numel,
        "element_size": 4,
        "nbytes": numel * 4,
    }


def _linear_node():
    return IRNode(
        "linear",
        "gemm",
        target="aten.linear.default",
        metadata={
            "input_tensors": [
                _tensor((1, 4, 8)),
                _tensor((16, 8)),
            ],
            "output_tensors": [_tensor((1, 4, 16))],
        },
    )


def test_infer_linear_work_uses_real_tensor_shapes():
    work = infer_node_work(_linear_node(), precision="bf16")

    assert work.model == "gemm"
    assert work.flops == pytest.approx(2 * 4 * 8 * 16)
    # bf16 target precision: (32 activation + 128 weight + 64 output) * 2 B
    assert work.memory_bytes == pytest.approx((32 + 128 + 64) * 2)


def test_genz_backend_produces_per_hardware_fine_costs():
    ir = ModelIR.from_parts(
        [
            IRNode("input", "input", kind="input"),
            _linear_node(),
            IRNode("output", "output", kind="output"),
        ],
        [
            TensorEdge("input", "linear", 128.0),
            TensorEdge("linear", "output", 256.0),
        ],
    )

    class FakeSystem:
        def __init__(self, flops, bandwidth):
            self.flops = float(flops)
            self.offchip_mem_bw = float(bandwidth)

    def factory(hardware, precision, compute_efficiency, memory_efficiency):
        assert precision == "bf16"
        if hardware == "fast":
            return FakeSystem(1.0e12, 1.0e12)
        return FakeSystem(0.5e12, 0.5e12)

    costed = annotate_genz_costs(
        ir,
        {"device": "fast", "host": "slow"},
        precision="bf16",
        system_factory=factory,
    )

    node = costed.nodes["linear"]
    assert node.costs_ms["device"] > 0
    assert node.costs_ms["host"] > node.costs_ms["device"]
    assert node.costs_ms["host"] == pytest.approx(2 * node.costs_ms["device"])
    assert costed.metadata["cost_source"] == "genz-fine-op-roofline"
    assert costed.metadata["cost_mode"] == "fine-op-roofline"

    source = node.metadata["cost_sources"]["device"]
    assert source["source"] == "genz-fine-op-roofline"
    assert source["metadata"]["op_model"] == "gemm"
    assert source["metadata"]["flops"] == pytest.approx(1024.0)


def test_transform_and_source_like_nodes_are_zero_cost():
    source_op = IRNode(
        "parameter_only",
        "gemm",
        target="aten.mm.default",
        metadata={
            "input_tensors": [_tensor((8, 8)), _tensor((8, 8))],
            "output_tensors": [_tensor((8, 8))],
        },
    )
    transform = IRNode(
        "view",
        "transform",
        target="aten.view.default",
        metadata={
            "input_tensors": [_tensor((8, 8))],
            "output_tensors": [_tensor((64,))],
        },
    )
    ir = ModelIR.from_parts(
        [source_op, transform],
        [TensorEdge("parameter_only", "view", 256.0)],
    )

    class FakeSystem:
        flops = 1.0e12
        offchip_mem_bw = 1.0e12

    costed = annotate_genz_costs(
        ir,
        {"device": "d", "host": "h"},
        system_factory=lambda *args: FakeSystem(),
    )
    assert costed.nodes["parameter_only"].costs_ms == {
        "device": 0.0,
        "host": 0.0,
    }
    assert costed.nodes["view"].costs_ms == {"device": 0.0, "host": 0.0}


def test_model_ir_roundtrip_preserves_costs_and_tensor_metadata():
    node = _linear_node()
    node.costs_ms = {"device": 0.1, "host": 0.2}
    ir = ModelIR.from_parts([node], [])

    payload = model_ir_to_dict(ir)
    restored = model_ir_from_dict(payload)

    assert restored.nodes["linear"].costs_ms == {"device": 0.1, "host": 0.2}
    assert restored.nodes["linear"].metadata["input_tensors"][1]["shape"] == [16, 8]


def test_torch_export_captures_parameter_and_tensor_shapes():
    torch = pytest.importorskip("torch")
    from coopinfer.frontend import capture_model

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 16, bias=False)

        def forward(self, x):
            return self.proj(x)

    ir = capture_model(Tiny().eval(), args=(torch.ones(1, 4, 8),), strict=False)
    gemms = [node for node in ir.nodes.values() if node.op == "gemm"]
    assert len(gemms) == 1

    node = gemms[0]
    input_shapes = [row["shape"] for row in node.metadata["input_tensors"]]
    assert [1, 4, 8] in input_shapes
    assert [16, 8] in input_shapes
    assert node.metadata["output_tensors"][0]["shape"] == [1, 4, 16]
    assert ir.metadata["tensor_metadata"] == "shape-dtype-numel-v1"
