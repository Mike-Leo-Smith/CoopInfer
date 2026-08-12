from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple


PLACEMENT_FREE = "free"
PLACEMENT_DEVICE = "device"
PLACEMENT_HOST = "host"
PLACEMENTS = {PLACEMENT_FREE, PLACEMENT_DEVICE, PLACEMENT_HOST}


@dataclass
class IRNode:
    id: str
    op: str
    target: str = ""
    module_path: str = ""
    kind: str = "op"
    placement: str = PLACEMENT_FREE
    costs_ms: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id)
        if not self.id:
            raise ValueError("IRNode.id must be non-empty")
        self.placement = str(self.placement).strip().lower()
        if self.placement not in PLACEMENTS:
            raise ValueError(f"Unsupported placement {self.placement!r}")
        for resource, value in self.costs_ms.items():
            if float(value) < 0:
                raise ValueError(f"Negative cost for {self.id}/{resource}: {value}")

    def clone(self) -> "IRNode":
        return IRNode(
            id=self.id,
            op=self.op,
            target=self.target,
            module_path=self.module_path,
            kind=self.kind,
            placement=self.placement,
            costs_ms=dict(self.costs_ms),
            metadata=dict(self.metadata),
        )


@dataclass(frozen=True)
class TensorEdge:
    source: str
    target: str
    size_bytes: float
    tensor_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if float(self.size_bytes) < 0:
            raise ValueError(
                f"Tensor edge {self.source}->{self.target} has negative size"
            )


@dataclass
class ModelIR:
    nodes: Dict[str, IRNode]
    edges: Tuple[TensorEdge, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_parts(
        cls,
        nodes: Iterable[IRNode],
        edges: Iterable[TensorEdge],
        *,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "ModelIR":
        node_map: Dict[str, IRNode] = {}
        for node in nodes:
            if node.id in node_map:
                raise ValueError(f"Duplicate IR node id {node.id!r}")
            node_map[node.id] = node
        graph = cls(
            nodes=node_map,
            edges=tuple(edges),
            metadata=dict(metadata or {}),
        )
        graph.validate()
        return graph

    def validate(self) -> None:
        for edge in self.edges:
            if edge.source not in self.nodes:
                raise ValueError(f"Unknown edge source {edge.source!r}")
            if edge.target not in self.nodes:
                raise ValueError(f"Unknown edge target {edge.target!r}")

    def clone(self) -> "ModelIR":
        return ModelIR.from_parts(
            [node.clone() for node in self.nodes.values()],
            [
                TensorEdge(
                    source=edge.source,
                    target=edge.target,
                    size_bytes=edge.size_bytes,
                    tensor_id=edge.tensor_id,
                    metadata=dict(edge.metadata),
                )
                for edge in self.edges
            ],
            metadata=dict(self.metadata),
        )


@dataclass
class SchedulingNode:
    id: str
    members: Tuple[str, ...]
    name: str
    costs_ms: Dict[str, float]
    placement: str = PLACEMENT_FREE
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SchedulingEdge:
    source: str
    target: str
    size_bytes: float
    tensor_ids: Tuple[str, ...] = ()


@dataclass
class SchedulingIR:
    nodes: Dict[str, SchedulingNode]
    edges: Tuple[SchedulingEdge, ...]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        for edge in self.edges:
            if edge.source not in self.nodes or edge.target not in self.nodes:
                raise ValueError(
                    f"Scheduling edge references unknown node: "
                    f"{edge.source}->{edge.target}"
                )
