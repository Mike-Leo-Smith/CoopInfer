from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

from coopinfer.frontend.ir import IRNode, ModelIR
from coopinfer.vla_perf_adapter import validate_vla_perf_profile

from .base import CostBackend, CostEstimate, CostTarget, annotate_costs


@dataclass(frozen=True)
class StageAllocation:
    stage: str
    node_count: int
    classified_node_count: int
    excluded_source_nodes: int
    total_ms: float
    per_node_ms: float


class VlaPerfPi05FineCostBackend(CostBackend):
    """Bridge a component-level VLA-Perf profile onto a fine pi0.5 ModelIR.

    VLA-Perf currently reports aggregate latency for the VLM transformer and one
    Action-Expert pass. This backend preserves those aggregate totals exactly by
    distributing each total uniformly over the fine-grained torch.export nodes
    that become runtime compute tasks in CoopInfer.

    A subtle but important rule is that an exported op with no incoming ModelIR
    edge is not charged here. CoopInfer treats all in-degree-zero DAG nodes as
    source events with zero runtime, so assigning cost to such a node would make
    the allocation audit look correct while the evaluator silently ignores that
    cost. In the torch.export graph these nodes are commonly parameter-only or
    constant-derived helper transforms whose parameter placeholders were removed
    from ModelIR. Their share is therefore redistributed over the stage's actual
    runtime compute nodes.

    Structural/wrapper operations that VLA-Perf does not model also receive zero
    cost. This remains a bridge backend, not the final direct op-level GenZ
    mapper.
    """

    SOURCE = "vla-perf-stage-normalized"

    def __init__(self, model_ir: ModelIR, profile: Mapping[str, Any]) -> None:
        validate_vla_perf_profile(profile)
        model_ir.validate()
        self._profile = profile
        self._hardware_to_side = {
            str(profile["hardware"]["device"]): "device",
            str(profile["hardware"]["host"]): "host",
        }

        incoming_count = {node_id: 0 for node_id in model_ir.nodes}
        for edge in model_ir.edges:
            incoming_count[edge.target] += 1

        self._node_stage = {
            node.id: self.classify_stage(node) for node in model_ir.nodes.values()
        }
        self._runtime_node = {
            node.id: (
                node.kind == "op"
                and incoming_count[node.id] > 0
                and self._node_stage[node.id] in {"vlm", "ae"}
            )
            for node in model_ir.nodes.values()
        }

        self._classified_stage_nodes: Dict[str, tuple[str, ...]] = {
            stage: tuple(
                node_id
                for node_id, node_stage in self._node_stage.items()
                if node_stage == stage
            )
            for stage in ("vlm", "ae")
        }
        self._stage_nodes: Dict[str, tuple[str, ...]] = {
            stage: tuple(
                node_id
                for node_id in self._classified_stage_nodes[stage]
                if self._runtime_node[node_id]
            )
            for stage in ("vlm", "ae")
        }
        for stage in ("vlm", "ae"):
            if not self._stage_nodes[stage]:
                raise ValueError(
                    f"Fine pi0.5 graph contains no runtime nodes classified as {stage!r}"
                )

        self._allocations: Dict[str, Dict[str, StageAllocation]] = {}
        for side in ("device", "host"):
            costs = profile["costs_ms"][side]
            totals = {
                "vlm": float(costs["vlm_total"]),
                "ae": float(costs["ae_pass_total"]),
            }
            self._allocations[side] = {}
            for stage, total in totals.items():
                count = len(self._stage_nodes[stage])
                classified_count = len(self._classified_stage_nodes[stage])
                self._allocations[side][stage] = StageAllocation(
                    stage=stage,
                    node_count=count,
                    classified_node_count=classified_count,
                    excluded_source_nodes=classified_count - count,
                    total_ms=total,
                    per_node_ms=total / count,
                )

    @staticmethod
    def classify_stage(node: IRNode) -> str:
        path = node.module_path.lower()
        if "gemma_expert" in path:
            return "ae"
        if "paligemma.model.language_model" in path:
            return "vlm"
        return "structural"

    def estimate(self, node: IRNode, hardware: str) -> CostEstimate:
        side = self._resolve_side(hardware)
        stage = self._node_stage.get(node.id, self.classify_stage(node))
        if stage not in {"vlm", "ae"}:
            return CostEstimate(
                0.0,
                self.SOURCE,
                {
                    "stage": "structural",
                    "profile_side": side,
                    "modeled": False,
                    "allocation": "unmodeled-zero",
                },
            )

        if not self._runtime_node.get(node.id, False):
            return CostEstimate(
                0.0,
                self.SOURCE,
                {
                    "stage": stage,
                    "profile_side": side,
                    "modeled": False,
                    "allocation": "source-like-zero",
                },
            )

        allocation = self._allocations[side][stage]
        return CostEstimate(
            allocation.per_node_ms,
            self.SOURCE,
            {
                "stage": stage,
                "profile_side": side,
                "modeled": True,
                "allocation": "uniform-runtime-nodes-within-stage",
                "stage_total_ms": allocation.total_ms,
                "stage_node_count": allocation.node_count,
            },
        )

    def audit(self) -> Dict[str, Dict[str, Dict[str, float | int]]]:
        result: Dict[str, Dict[str, Dict[str, float | int]]] = {}
        for side, stages in self._allocations.items():
            result[side] = {}
            for stage, allocation in stages.items():
                result[side][stage] = {
                    "node_count": allocation.node_count,
                    "classified_node_count": allocation.classified_node_count,
                    "excluded_source_nodes": allocation.excluded_source_nodes,
                    "profile_total_ms": allocation.total_ms,
                    "allocated_total_ms": allocation.per_node_ms
                    * allocation.node_count,
                    "per_node_ms": allocation.per_node_ms,
                }
        return result

    def _resolve_side(self, hardware: str) -> str:
        hardware = str(hardware)
        if hardware in self._hardware_to_side:
            return self._hardware_to_side[hardware]
        if hardware in {"device", "host"}:
            return hardware
        available = ", ".join(sorted(self._hardware_to_side))
        raise KeyError(
            f"Hardware {hardware!r} is not present in VLA-Perf profile; "
            f"available: {available}"
        )


def annotate_vla_perf_profile_costs(
    model_ir: ModelIR,
    profile: Mapping[str, Any],
    *,
    device_resource: str = "device",
    host_resource: str = "host",
) -> ModelIR:
    """Attach stage-normalized VLA-Perf costs to a fine pi0.5 ModelIR."""

    backend = VlaPerfPi05FineCostBackend(model_ir, profile)
    result = annotate_costs(
        model_ir,
        {
            device_resource: CostTarget(
                hardware=str(profile["hardware"]["device"]), backend=backend
            ),
            host_resource: CostTarget(
                hardware=str(profile["hardware"]["host"]), backend=backend
            ),
        },
    )
    result.metadata["cost_source"] = VlaPerfPi05FineCostBackend.SOURCE
    result.metadata["cost_mode"] = "stage-normalized-fine-graph"
    result.metadata["cost_hardware"] = {
        device_resource: str(profile["hardware"]["device"]),
        host_resource: str(profile["hardware"]["host"]),
    }
    result.metadata["cost_audit"] = backend.audit()
    return result
