from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Union

from .ir import (
    IRNode,
    ModelIR,
    SchedulingEdge,
    SchedulingIR,
    SchedulingNode,
    TensorEdge,
)


def model_ir_from_dict(data: Mapping[str, Any]) -> ModelIR:
    """Reconstruct a ModelIR from the JSON emitted by the export probe.

    This intentionally preserves the fine-grained node/tensor dependency graph.
    It does not coarsen nodes or infer costs.
    """

    nodes = []
    for index, row in enumerate(data.get("nodes", []), start=1):
        if "id" not in row or "op" not in row:
            raise ValueError(f"ModelIR node row {index} is missing id/op")
        nodes.append(
            IRNode(
                id=str(row["id"]),
                op=str(row["op"]),
                target=str(row.get("target", "")),
                module_path=str(row.get("module_path", "")),
                kind=str(row.get("kind", "op")),
                placement=str(row.get("placement", "free")),
                costs_ms={
                    str(resource): float(value)
                    for resource, value in dict(row.get("costs_ms", {})).items()
                },
                metadata=dict(row.get("metadata", {})),
            )
        )

    edges = []
    for index, row in enumerate(data.get("edges", []), start=1):
        for key in ("source", "target", "size_bytes"):
            if key not in row:
                raise ValueError(f"ModelIR edge row {index} is missing {key!r}")
        edges.append(
            TensorEdge(
                source=str(row["source"]),
                target=str(row["target"]),
                size_bytes=float(row["size_bytes"]),
                tensor_id=str(row.get("tensor_id", "")),
                metadata=dict(row.get("metadata", {})),
            )
        )

    return ModelIR.from_parts(nodes, edges, metadata=dict(data.get("metadata", {})))


def load_model_ir_json(path: Union[str, Path]) -> ModelIR:
    with Path(path).open("r", encoding="utf-8") as file:
        return model_ir_from_dict(json.load(file))


def annotate_synthetic_costs(
    model_ir: ModelIR,
    *,
    device_cost_ms: float = 0.001,
    host_cost_ms: float = 0.002,
    device_resource: str = "device",
    host_resource: str = "host",
) -> ModelIR:
    """Attach deterministic placeholder costs for end-to-end plumbing tests.

    Input/output nodes are zero-cost. Every compute op receives the same
    placeholder Device/Host cost. These values are deliberately synthetic and
    must not be interpreted as a performance model; a real CostBackend can
    replace this step without changing the identity adapter below.
    """

    device_cost_ms = float(device_cost_ms)
    host_cost_ms = float(host_cost_ms)
    if device_cost_ms < 0 or host_cost_ms < 0:
        raise ValueError("Synthetic costs must be non-negative")

    result = model_ir.clone()
    for node in result.nodes.values():
        if node.kind in {"input", "output"}:
            node.costs_ms[device_resource] = 0.0
            node.costs_ms[host_resource] = 0.0
        else:
            node.costs_ms[device_resource] = device_cost_ms
            node.costs_ms[host_resource] = host_cost_ms
        node.metadata["cost_source"] = "synthetic"
    result.metadata["cost_source"] = "synthetic"
    return result


def identity_scheduling_ir(
    model_ir: ModelIR,
    *,
    device_resource: str = "device",
    host_resource: str = "host",
) -> SchedulingIR:
    """Convert ModelIR to SchedulingIR with exactly one scheduling node per IR node.

    No compute nodes are merged and no dependency is invented or removed.
    Parallel tensor edges with the same (source, target) endpoints are aggregated
    because CoopInfer's backend graph is a networkx.DiGraph and therefore stores
    one edge per ordered node pair. Tensor IDs and total bytes are preserved.
    """

    model_ir.validate()

    scheduling_nodes: Dict[str, SchedulingNode] = {}
    for node in model_ir.nodes.values():
        missing = [
            resource
            for resource in (device_resource, host_resource)
            if resource not in node.costs_ms
        ]
        if missing:
            raise ValueError(
                f"ModelIR node {node.id!r} is missing required cost(s): {', '.join(missing)}"
            )
        scheduling_nodes[node.id] = SchedulingNode(
            id=node.id,
            members=(node.id,),
            name=node.id,
            costs_ms={
                device_resource: float(node.costs_ms[device_resource]),
                host_resource: float(node.costs_ms[host_resource]),
            },
            placement=node.placement,
            metadata={
                "member_count": 1,
                "identity_adapter": True,
                "kind": node.kind,
                "op": node.op,
                "module_path": node.module_path,
            },
        )

    edge_accumulator: Dict[tuple[str, str], Dict[str, Any]] = {}
    for edge in model_ir.edges:
        key = (edge.source, edge.target)
        bucket = edge_accumulator.setdefault(
            key,
            {"size_bytes": 0.0, "tensor_ids": []},
        )
        bucket["size_bytes"] = float(bucket["size_bytes"]) + float(edge.size_bytes)
        tensor_id = edge.tensor_id or f"{edge.source}->{edge.target}"
        bucket["tensor_ids"].append(tensor_id)

    scheduling_edges = tuple(
        SchedulingEdge(
            source=source,
            target=target,
            size_bytes=float(values["size_bytes"]),
            tensor_ids=tuple(str(value) for value in values["tensor_ids"]),
        )
        for (source, target), values in edge_accumulator.items()
    )

    result = SchedulingIR(
        nodes=scheduling_nodes,
        edges=scheduling_edges,
        metadata={
            "adapter": "identity",
            "source_node_count": len(model_ir.nodes),
            "source_tensor_edge_count": len(model_ir.edges),
            "backend_edge_count": len(scheduling_edges),
            "cost_source": model_ir.metadata.get("cost_source", "unknown"),
        },
    )
    result.validate()
    return result
