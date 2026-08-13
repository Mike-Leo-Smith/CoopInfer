from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from coopinfer.frontend.ir import ModelIR, SchedulingIR, TensorEdge
from coopinfer.frontend.layerwise import LayerGrouping, build_layer_scheduling_ir

from .genz import infer_node_work


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
class LayerWork:
    """Hardware-neutral workload for one detected layer/region.

    FLOPs are additive across contained fine operations. Off-chip traffic is not
    the sum of fine-op traffic: it includes unique parameter/buffer tensors plus
    only tensors that cross the layer boundary. Internal activations are assumed
    reusable/local to the layer abstraction and are intentionally not recharged
    as HBM traffic for every fine operation.
    """

    flops: float
    parameter_bytes: float
    input_bytes: float
    output_bytes: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def memory_bytes(self) -> float:
        return float(self.parameter_bytes + self.input_bytes + self.output_bytes)


SystemFactory = Callable[[str, str, float, float], Any]


def infer_layer_work(
    model_ir: ModelIR,
    grouping: LayerGrouping,
    group_id: str,
    *,
    precision: str = "bf16",
) -> LayerWork:
    """Build one layer workload from the real Fine ModelIR and provenance map."""

    precision = str(precision).strip().lower()
    if precision not in _PRECISION_BYTES:
        raise ValueError(f"Unsupported layer GenZ precision {precision!r}")
    group = grouping.groups[group_id]
    members = set(group.members)

    flops = 0.0
    op_models: Dict[str, int] = {}
    parameter_rows: Dict[str, Mapping[str, Any]] = {}
    for member in group.members:
        node = model_ir.nodes[member]
        if node.kind != "op":
            continue
        work = infer_node_work(node, precision=precision)
        flops += float(work.flops)
        op_models[work.model] = op_models.get(work.model, 0) + 1
        rows = node.metadata.get("input_tensors", ())
        if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
            continue
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            kind = str(row.get("input_kind", ""))
            if kind not in {"PARAMETER", "BUFFER", "CONSTANT_TENSOR"}:
                continue
            source = str(row.get("source", ""))
            key = source or repr((row.get("shape"), row.get("dtype"), row.get("nbytes")))
            parameter_rows.setdefault(key, row)

    parameter_bytes = sum(
        _row_bytes(row, precision=precision) for row in parameter_rows.values()
    )

    # Boundary activations are deduplicated by producing fine tensor. A tensor
    # fanning out to multiple consumers in the same group is charged once.
    incoming: Dict[Tuple[str, str], float] = {}
    outgoing: Dict[Tuple[str, str], float] = {}
    for edge in model_ir.edges:
        source_inside = edge.source in members
        target_inside = edge.target in members
        tensor_key = (edge.source, edge.tensor_id or edge.source)
        adjusted_bytes = _edge_bytes(edge, precision=precision)
        if target_inside and not source_inside:
            incoming.setdefault(tensor_key, adjusted_bytes)
        if source_inside and not target_inside:
            outgoing.setdefault(tensor_key, adjusted_bytes)

    return LayerWork(
        flops=flops,
        parameter_bytes=float(parameter_bytes),
        input_bytes=float(sum(incoming.values())),
        output_bytes=float(sum(outgoing.values())),
        metadata={
            "member_count": len(group.members),
            "op_models": op_models,
            "unique_parameter_tensors": len(parameter_rows),
            "unique_boundary_inputs": len(incoming),
            "unique_boundary_outputs": len(outgoing),
            "memory_model": "weights-plus-external-io",
            "internal_activation_hbm": "not-recharged",
        },
    )


def annotate_layer_genz_costs(
    model_ir: ModelIR,
    grouping: LayerGrouping,
    targets: Mapping[str, str],
    *,
    precision: str = "bf16",
    compute_efficiency: float = 1.0,
    memory_efficiency: float = 1.0,
    system_factory: Optional[SystemFactory] = None,
) -> SchedulingIR:
    """Create a layer SchedulingIR and cost each group once on GenZ systems.

    This is deliberately not ``sum(fine_latency)``. The layer workload first
    aggregates arithmetic work and reconstructs layer-external HBM traffic, then
    evaluates a single layer-level compute-vs-memory roofline per hardware.
    """

    precision = str(precision).strip().lower()
    if precision not in _PRECISION_BYTES:
        raise ValueError(f"Unsupported layer GenZ precision {precision!r}")
    if compute_efficiency <= 0 or memory_efficiency <= 0:
        raise ValueError("GenZ efficiencies must be greater than zero")

    factory = system_factory or _default_system_factory
    systems = {
        str(resource): factory(
            str(hardware), precision, float(compute_efficiency), float(memory_efficiency)
        )
        for resource, hardware in targets.items()
    }

    incoming_groups = {group_id: 0 for group_id in grouping.groups}
    for edge in model_ir.edges:
        src = grouping.member_to_group[edge.source]
        dst = grouping.member_to_group[edge.target]
        if src != dst:
            incoming_groups[dst] += 1

    costs: Dict[str, Dict[str, float]] = {}
    metadata: Dict[str, Dict[str, object]] = {}
    for group_id, group in grouping.groups.items():
        work = infer_layer_work(model_ir, grouping, group_id, precision=precision)
        resource_meta: Dict[str, object] = {}
        group_costs: Dict[str, float] = {}

        source_semantic_zero = (
            group.kind in {"input", "output", "source_like"}
            or incoming_groups[group_id] == 0
        )
        for resource, system in systems.items():
            hardware = str(targets[resource])
            if source_semantic_zero:
                latency_ms = 0.0
                compute_ms = 0.0
                memory_ms = 0.0
                bound = "source-task-zero"
            else:
                peak_flops = float(system.flops)
                memory_bw = float(system.offchip_mem_bw)
                if peak_flops <= 0 or memory_bw <= 0:
                    raise ValueError(
                        f"Invalid GenZ system rates for {hardware!r}: "
                        f"flops={peak_flops}, offchip_mem_bw={memory_bw}"
                    )
                compute_ms = (
                    float(work.flops)
                    / (peak_flops * float(compute_efficiency))
                    * 1000.0
                )
                memory_ms = (
                    float(work.memory_bytes)
                    / (memory_bw * float(memory_efficiency))
                    * 1000.0
                )
                latency_ms = max(compute_ms, memory_ms)
                bound = "compute" if compute_ms >= memory_ms else "memory"
            group_costs[resource] = float(latency_ms)
            resource_meta[resource] = {
                "hardware": hardware,
                "precision": precision,
                "compute_ms": float(compute_ms),
                "memory_ms": float(memory_ms),
                "latency_ms": float(latency_ms),
                "bound": bound,
            }

        costs[group_id] = group_costs
        metadata[group_id] = {
            "cost_source": "genz-layer-external-roofline",
            "cost_mode": "layer-workload-roofline",
            "layer_work": {
                "flops": float(work.flops),
                "parameter_bytes": float(work.parameter_bytes),
                "input_bytes": float(work.input_bytes),
                "output_bytes": float(work.output_bytes),
                "memory_bytes": float(work.memory_bytes),
                **dict(work.metadata),
            },
            "resource_cost_details": resource_meta,
        }

    # Keep the source Fine ModelIR immutable. A precision-adjusted clone is used
    # only to express layer-boundary communication sizes in the target precision.
    adjusted_ir = _precision_adjusted_model_ir(model_ir, precision=precision)
    result = build_layer_scheduling_ir(
        adjusted_ir,
        grouping,
        group_costs=costs,
        group_metadata=metadata,
    )
    result.metadata.update(
        {
            "cost_source": "genz-layer-external-roofline",
            "cost_mode": "layer-workload-roofline",
            "cost_precision": precision,
            "cost_hardware": {
                str(resource): str(hardware) for resource, hardware in targets.items()
            },
            "cost_efficiency": {
                "compute": float(compute_efficiency),
                "memory": float(memory_efficiency),
            },
            "communication_edge_precision": precision,
        }
    )
    return result


def _precision_adjusted_model_ir(model_ir: ModelIR, *, precision: str) -> ModelIR:
    return ModelIR.from_parts(
        [node.clone() for node in model_ir.nodes.values()],
        [
            TensorEdge(
                source=edge.source,
                target=edge.target,
                size_bytes=_edge_bytes(edge, precision=precision),
                tensor_id=edge.tensor_id,
                metadata=dict(edge.metadata),
            )
            for edge in model_ir.edges
        ],
        metadata=dict(model_ir.metadata),
    )


def _edge_bytes(edge: TensorEdge, *, precision: str) -> float:
    rows = edge.metadata.get("tensors", ())
    if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)):
        tensor_rows = [row for row in rows if isinstance(row, Mapping)]
        if tensor_rows:
            value = sum(_row_bytes(row, precision=precision) for row in tensor_rows)
            if value > 0.0:
                return float(value)
    return float(edge.size_bytes)


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
            "GenZ is required for layer-level costing. Run this stage from the "
            "GenZ/VLA-Perf environment or add that checkout to --genz-root."
        ) from exc
    return get_inference_system(
        system_name=hardware,
        bits=precision,
        ceff=compute_efficiency,
        meff=memory_efficiency,
    )


def _row_shape(row: Mapping[str, Any]) -> Optional[Tuple[int, ...]]:
    value = row.get("shape")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    dims = []
    for dim in value:
        try:
            dims.append(int(dim))
        except (TypeError, ValueError):
            return None
    return tuple(dims)


def _row_numel(row: Mapping[str, Any]) -> Optional[int]:
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


def _row_bytes(row: Mapping[str, Any], *, precision: str) -> float:
    numel = _row_numel(row)
    if numel is None:
        value = row.get("nbytes")
        return float(value) if value is not None else 0.0
    dtype = str(row.get("dtype", "")).lower()
    if "int64" in dtype or "long" in dtype:
        width = 8.0
    elif "int32" in dtype:
        width = 4.0
    elif "int16" in dtype:
        width = 2.0
    elif any(token in dtype for token in ("int8", "uint8", "bool")):
        width = 1.0
    else:
        width = _PRECISION_BYTES[precision]
    return float(numel) * width
