from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any, Callable, Dict, Mapping, Optional, Sequence

from coopinfer.frontend.ir import IRNode, ModelIR

from .base import CostBackend, CostEstimate, CostTarget, annotate_costs


SystemFactory = Callable[[str, str, float, float], Any]


_PRECISION_BYTES = {
    "fp32": 4.0,
    "f32": 4.0,
    "tf32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,
    "int8": 1.0,
    "fp6": 0.75,
    "fp4": 0.5,
    "int4": 0.5,
    "int2": 0.25,
}


@dataclass(frozen=True)
class NodeWork:
    """Hardware-neutral work estimate extracted from one fine ModelIR op."""

    flops: float
    memory_bytes: float
    model: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


class GenZCostBackend(CostBackend):
    """Generic fine-op roofline backend using GenZ hardware systems.

    The backend deliberately consumes only canonical ModelIR information:
    operator class, tensor shapes/dtypes, and hardware name. It does not know
    anything about pi0.5, VLMs, Action Experts, or any other model architecture.

    GenZ's operator model evaluates latency as the maximum of compute, memory,
    and communication time. For a single local fine op CoopInfer has no
    inter-device collective here, so this adapter uses the same compute-vs-memory
    roofline with the GenZ System's precision-specific FLOP/s and off-chip
    bandwidth. Inter-device communication remains represented by ModelIR edges
    and is scheduled later by CoopInfer.
    """

    SOURCE = "genz-fine-op-roofline"

    def __init__(
        self,
        model_ir: ModelIR,
        *,
        precision: str = "bf16",
        compute_efficiency: float = 1.0,
        memory_efficiency: float = 1.0,
        system_factory: Optional[SystemFactory] = None,
    ) -> None:
        model_ir.validate()
        precision = str(precision).strip().lower()
        if precision not in _PRECISION_BYTES:
            raise ValueError(f"Unsupported GenZ precision {precision!r}")
        if compute_efficiency <= 0 or memory_efficiency <= 0:
            raise ValueError("GenZ efficiencies must be greater than zero")

        self.precision = precision
        self.compute_efficiency = float(compute_efficiency)
        self.memory_efficiency = float(memory_efficiency)
        self._system_factory = system_factory or _default_system_factory
        self._systems: Dict[str, Any] = {}

        incoming_count = {node_id: 0 for node_id in model_ir.nodes}
        for edge in model_ir.edges:
            incoming_count[edge.target] += 1
        self._source_like = {
            node_id
            for node_id, node in model_ir.nodes.items()
            if node.kind == "op" and incoming_count[node_id] == 0
        }

    def estimate(self, node: IRNode, hardware: str) -> CostEstimate:
        hardware = str(hardware)
        if node.id in self._source_like:
            return CostEstimate(
                0.0,
                self.SOURCE,
                {
                    "hardware": hardware,
                    "precision": self.precision,
                    "model": "source-task-zero",
                    "source_like": True,
                },
            )

        work = infer_node_work(node, precision=self.precision)
        system = self._system(hardware)

        peak_flops = float(system.flops)
        memory_bw = float(system.offchip_mem_bw)
        if peak_flops <= 0 or memory_bw <= 0:
            raise ValueError(
                f"Invalid GenZ system rates for {hardware!r}: "
                f"flops={peak_flops}, offchip_mem_bw={memory_bw}"
            )

        compute_ms = (
            float(work.flops)
            / (peak_flops * self.compute_efficiency)
            * 1000.0
        )
        memory_ms = (
            float(work.memory_bytes)
            / (memory_bw * self.memory_efficiency)
            * 1000.0
        )
        latency_ms = max(compute_ms, memory_ms)
        bound = "compute" if compute_ms >= memory_ms else "memory"

        return CostEstimate(
            latency_ms,
            self.SOURCE,
            {
                "hardware": hardware,
                "precision": self.precision,
                "op_model": work.model,
                "flops": float(work.flops),
                "memory_bytes": float(work.memory_bytes),
                "compute_ms": compute_ms,
                "memory_ms": memory_ms,
                "bound": bound,
                **dict(work.metadata),
            },
        )

    def _system(self, hardware: str) -> Any:
        if hardware not in self._systems:
            self._systems[hardware] = self._system_factory(
                hardware,
                self.precision,
                self.compute_efficiency,
                self.memory_efficiency,
            )
        return self._systems[hardware]


def annotate_genz_costs(
    model_ir: ModelIR,
    targets: Mapping[str, str],
    *,
    precision: str = "bf16",
    compute_efficiency: float = 1.0,
    memory_efficiency: float = 1.0,
    system_factory: Optional[SystemFactory] = None,
) -> ModelIR:
    """Attach generic GenZ fine-op costs to every logical resource."""

    backend = GenZCostBackend(
        model_ir,
        precision=precision,
        compute_efficiency=compute_efficiency,
        memory_efficiency=memory_efficiency,
        system_factory=system_factory,
    )
    cost_targets = {
        str(resource): CostTarget(str(hardware), backend)
        for resource, hardware in targets.items()
    }
    result = annotate_costs(model_ir, cost_targets)
    result.metadata["cost_source"] = GenZCostBackend.SOURCE
    result.metadata["cost_mode"] = "fine-op-roofline"
    result.metadata["cost_precision"] = str(precision).lower()
    result.metadata["cost_hardware"] = {
        str(resource): str(hardware) for resource, hardware in targets.items()
    }
    result.metadata["cost_efficiency"] = {
        "compute": float(compute_efficiency),
        "memory": float(memory_efficiency),
    }
    return result


def infer_node_work(node: IRNode, *, precision: str = "bf16") -> NodeWork:
    """Infer FLOPs and off-chip bytes for a canonical fine op.

    Exact matrix/conv work is derived from captured tensor shapes. Vector ops use
    conventional per-element operation counts. Unknown ops fall back to a
    memory-only model instead of inheriting a model-specific stage average.
    """

    precision = str(precision).lower()
    input_tensors = _tensor_rows(node.metadata.get("input_tensors", ()))
    output_tensors = _tensor_rows(node.metadata.get("output_tensors", ()))
    memory_bytes = _tensor_rows_bytes(
        (*input_tensors, *output_tensors), precision=precision
    )
    op = str(node.op).lower()
    target = str(node.target).lower()
    output_numel = sum(_row_numel(row) or 0 for row in output_tensors)
    input_numel = sum(_row_numel(row) or 0 for row in input_tensors)

    if op == "transform":
        return NodeWork(0.0, 0.0, "metadata-transform")

    if op in {"gemm", "batched_gemm"}:
        flops = _matmul_like_flops(input_tensors, output_tensors)
        return NodeWork(flops, memory_bytes, op)

    if op == "conv":
        flops = _conv_flops(input_tensors, output_tensors)
        return NodeWork(flops, memory_bytes, "conv")

    if op == "attention":
        flops = _attention_flops(input_tensors, output_tensors)
        return NodeWork(flops, memory_bytes, "attention")

    if op == "softmax":
        elements = output_numel or input_numel
        return NodeWork(5.0 * elements, memory_bytes, "softmax")

    if op == "norm":
        elements = output_numel or input_numel
        return NodeWork(5.0 * elements, memory_bytes, "norm")

    if op == "reduction":
        elements = input_numel or output_numel
        return NodeWork(2.0 * elements, memory_bytes, "reduction")

    if op == "elementwise":
        elements = output_numel or input_numel
        expensive = any(
            token in target
            for token in (
                "gelu",
                "silu",
                "tanh",
                "sigmoid",
                "pow",
                "rsqrt",
                "sqrt",
                "exp",
                "sin",
                "cos",
            )
        )
        flops_per_element = 4.0 if expensive else 1.0
        return NodeWork(
            flops_per_element * elements,
            memory_bytes,
            "elementwise-special" if expensive else "elementwise",
        )

    if op == "memory":
        return NodeWork(0.0, memory_bytes, "memory-movement")

    if op == "embedding":
        return NodeWork(0.0, memory_bytes, "embedding-memory")

    return NodeWork(
        0.0,
        memory_bytes,
        "fallback-memory",
        {"canonical_op": node.op},
    )


def _default_system_factory(
    hardware: str,
    precision: str,
    compute_efficiency: float,
    memory_efficiency: float,
) -> Any:
    try:
        from GenZ.LLM_inference.utils import get_inference_system
    except ImportError as exc:
        raise RuntimeError(
            "GenZ is required for GenZCostBackend. Run this cost stage from a "
            "GenZ/VLA-Perf environment or install GenZ as an optional dependency."
        ) from exc

    return get_inference_system(
        system_name=hardware,
        bits=precision,
        ceff=compute_efficiency,
        meff=memory_efficiency,
    )


def _tensor_rows(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(row for row in value if isinstance(row, Mapping))


def _row_shape(row: Mapping[str, Any]) -> tuple[int, ...] | None:
    value = row.get("shape")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    shape = []
    for dim in value:
        if isinstance(dim, bool):
            return None
        try:
            shape.append(int(dim))
        except (TypeError, ValueError):
            return None
    return tuple(shape)


def _row_numel(row: Mapping[str, Any]) -> int | None:
    value = row.get("numel")
    if value is not None:
        try:
            return int(value)
        except (TypeError, ValueError):
            pass
    shape = _row_shape(row)
    if shape is None:
        return None
    return int(prod(shape)) if shape else 1


def _dtype_bytes(dtype: str, *, precision: str) -> float:
    text = str(dtype).lower()
    if any(token in text for token in ("float", "bfloat", "half")):
        return _PRECISION_BYTES[precision]
    if "int64" in text or "long" in text:
        return 8.0
    if "int32" in text:
        return 4.0
    if "int16" in text:
        return 2.0
    if any(token in text for token in ("int8", "uint8", "bool")):
        return 1.0
    return _PRECISION_BYTES[precision]


def _tensor_rows_bytes(
    rows: Sequence[Mapping[str, Any]],
    *,
    precision: str,
) -> float:
    total = 0.0
    for row in rows:
        numel = _row_numel(row)
        if numel is None:
            nbytes = row.get("nbytes")
            if nbytes is not None:
                total += float(nbytes)
            continue
        total += float(numel) * _dtype_bytes(
            str(row.get("dtype", "")), precision=precision
        )
    return total


def _matmul_like_flops(
    inputs: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
) -> float:
    if not inputs or not outputs:
        return 0.0
    a_shape = _row_shape(inputs[0])
    output_numel = sum(_row_numel(row) or 0 for row in outputs)
    if not a_shape or output_numel <= 0:
        return 0.0
    contract = int(a_shape[-1])
    return 2.0 * float(output_numel) * float(contract)


def _conv_flops(
    inputs: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
) -> float:
    if len(inputs) < 2 or not outputs:
        return 0.0
    weight_shape = _row_shape(inputs[1])
    output_numel = sum(_row_numel(row) or 0 for row in outputs)
    if not weight_shape or len(weight_shape) < 2 or output_numel <= 0:
        return 0.0
    kernel_contract = int(prod(weight_shape[1:]))
    return 2.0 * float(output_numel) * float(kernel_contract)


def _attention_flops(
    inputs: Sequence[Mapping[str, Any]],
    outputs: Sequence[Mapping[str, Any]],
) -> float:
    if len(inputs) < 2:
        return 0.0
    q_shape = _row_shape(inputs[0])
    k_shape = _row_shape(inputs[1])
    if not q_shape or not k_shape or len(q_shape) < 2 or len(k_shape) < 2:
        return 0.0

    query_tokens = int(q_shape[-2])
    key_tokens = int(k_shape[-2])
    head_dim = int(q_shape[-1])
    outer = int(prod(q_shape[:-2])) if len(q_shape) > 2 else 1

    qk_and_av = 4.0 * outer * query_tokens * key_tokens * head_dim
    softmax = 5.0 * outer * query_tokens * key_tokens
    return float(qk_and_av + softmax)
