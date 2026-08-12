from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, Union

from coopinfer.frontend.ir import IRNode, ModelIR


@dataclass(frozen=True)
class CostEstimate:
    latency_ms: float
    source: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if float(self.latency_ms) < 0:
            raise ValueError("Cost latency must be non-negative")


class CostBackend(ABC):
    """Pluggable hardware cost backend for a canonical IR node."""

    @abstractmethod
    def estimate(self, node: IRNode, hardware: str) -> CostEstimate:
        raise NotImplementedError


@dataclass(frozen=True)
class CostTarget:
    hardware: str
    backend: CostBackend


class MappingCostBackend(CostBackend):
    """Deterministic backend for tests, calibration tables, and POCs."""

    def __init__(
        self,
        costs_ms: Mapping[Tuple[str, str], float],
        *,
        default_ms: Optional[float] = None,
        source: str = "mapping",
    ) -> None:
        self._costs = {(str(k[0]), str(k[1])): float(v) for k, v in costs_ms.items()}
        self._default = None if default_ms is None else float(default_ms)
        self._source = source

    def estimate(self, node: IRNode, hardware: str) -> CostEstimate:
        for key in ((node.id, hardware), (node.op, hardware)):
            if key in self._costs:
                return CostEstimate(self._costs[key], self._source, {"key": key[0]})
        if self._default is not None:
            return CostEstimate(self._default, self._source, {"key": "default"})
        raise KeyError(
            f"No cost for node={node.id!r}, op={node.op!r}, hardware={hardware!r}"
        )


class CallableCostBackend(CostBackend):
    """Adapter for analytical models or profilers exposed as a Python callback."""

    def __init__(
        self,
        estimator: Callable[[IRNode, str], Union[CostEstimate, float]],
        *,
        source: str = "callable",
    ) -> None:
        self._estimator = estimator
        self._source = source

    def estimate(self, node: IRNode, hardware: str) -> CostEstimate:
        value = self._estimator(node, hardware)
        if isinstance(value, CostEstimate):
            return value
        return CostEstimate(float(value), self._source)


def annotate_costs(
    model_ir: ModelIR,
    targets: Mapping[str, CostTarget],
) -> ModelIR:
    """Return a cloned ModelIR with one latency value per logical resource."""

    result = model_ir.clone()
    for node in result.nodes.values():
        node.costs_ms.clear()
        cost_sources: Dict[str, Dict[str, Any]] = {}
        for resource, target in targets.items():
            if node.kind in {"input", "output"}:
                estimate = CostEstimate(0.0, "structural")
            else:
                estimate = target.backend.estimate(node, target.hardware)
            node.costs_ms[str(resource)] = float(estimate.latency_ms)
            cost_sources[str(resource)] = {
                "hardware": target.hardware,
                "source": estimate.source,
                "metadata": dict(estimate.metadata),
            }
        node.metadata["cost_sources"] = cost_sources
    return result
