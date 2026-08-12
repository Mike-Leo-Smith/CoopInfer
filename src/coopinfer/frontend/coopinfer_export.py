from __future__ import annotations

from typing import Any, Dict

from .ir import PLACEMENT_DEVICE, SchedulingIR


_BYTES_PER_MB = 1024.0 * 1024.0


def to_coopinfer_payload(
    scheduling_ir: SchedulingIR,
    *,
    bandwidth_mb_s: float,
    latency_ms: float,
    device_resource: str = "device",
    host_resource: str = "host",
    batch_transfers: bool = False,
    pipeline_unroll: int = 1,
) -> Dict[str, Any]:
    """Translate the generic SchedulingIR into CoopInfer's stable DAG schema."""

    scheduling_ir.validate()
    if bandwidth_mb_s <= 0:
        raise ValueError("bandwidth_mb_s must be greater than zero")
    if latency_ms < 0:
        raise ValueError("latency_ms must be non-negative")

    nodes = []
    for node in scheduling_ir.nodes.values():
        if device_resource not in node.costs_ms:
            raise ValueError(
                f"Scheduling node {node.id!r} has no cost for {device_resource!r}"
            )
        if host_resource not in node.costs_ms:
            raise ValueError(
                f"Scheduling node {node.id!r} has no cost for {host_resource!r}"
            )
        x = 0 if node.placement == PLACEMENT_DEVICE else 1
        nodes.append(
            {
                "id": node.id,
                "name": node.name,
                "c_dev": float(node.costs_ms[device_resource]),
                "c_host": float(node.costs_ms[host_resource]),
                "placement": node.placement,
                "source_period_ms": 0.0,
                "source_phase_ms": 0.0,
                "x": x,
            }
        )

    edges = [
        {
            "source": edge.source,
            "target": edge.target,
            "size": float(edge.size_bytes) / _BYTES_PER_MB,
        }
        for edge in scheduling_ir.edges
    ]

    return {
        "version": "1.0",
        "nodes": nodes,
        "edges": edges,
        "environment": {
            "bandwidth": float(bandwidth_mb_s),
            "latency": float(latency_ms),
            "weight_avg_latency": 1.0,
            "weight_max_latency": 0.0,
            "weight_device_utilization": 0.0,
            "latency_limit": 0.0,
            "max_frame_latency_limit": 0.0,
            "batch_transfers": bool(batch_transfers),
            "pipeline_unroll": int(pipeline_unroll),
            "solver_threads": 0,
            "anneal_initial_temp": 1.0,
            "anneal_final_temp": 0.01,
        },
    }
