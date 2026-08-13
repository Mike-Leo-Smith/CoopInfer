from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .ir import ModelIR


_LAYER_RE = re.compile(r"(?:layers|layer)[._](\d+)")


@dataclass(frozen=True)
class Pi05ProbeSummary:
    node_count: int
    edge_count: int
    q_proj_nodes: Tuple[str, ...]
    k_proj_nodes: Tuple[str, ...]
    v_proj_nodes: Tuple[str, ...]
    vlm_nodes: Tuple[str, ...]
    expert_nodes: Tuple[str, ...]
    output_predecessors: Tuple[str, ...]
    wave_kv_candidate_layers: Tuple[int, ...]
    notes: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "q_proj_nodes": list(self.q_proj_nodes),
            "k_proj_nodes": list(self.k_proj_nodes),
            "v_proj_nodes": list(self.v_proj_nodes),
            "vlm_nodes": list(self.vlm_nodes),
            "expert_nodes": list(self.expert_nodes),
            "output_predecessors": list(self.output_predecessors),
            "wave_kv_candidate_layers": list(self.wave_kv_candidate_layers),
            "notes": list(self.notes),
        }


def flatten_past_key_values(cache: Any) -> tuple:
    """Flatten supported Hugging Face cache containers into K/V tensor pairs.

    Transformers 5.x stores DynamicCache data as ``cache.layers[i].keys`` and
    ``cache.layers[i].values``. Older releases exposed either a legacy tuple or
    ``key_cache`` / ``value_cache`` lists. This helper accepts all three forms
    without importing Transformers, keeping the probe version-tolerant.
    """

    if cache is None:
        raise RuntimeError("Prefix forward returned no past_key_values")

    legacy = getattr(cache, "to_legacy_cache", None)
    if callable(legacy):
        cache = legacy()

    flat: List[Any] = []
    if isinstance(cache, (tuple, list)):
        for layer in cache:
            if isinstance(layer, (tuple, list)) and len(layer) >= 2:
                flat.extend((layer[0], layer[1]))
            else:
                flat.append(layer)
    else:
        # Transformers >=5: DynamicCache.layers contains CacheLayer objects
        # whose actual tensors live in .keys and .values.
        layers = getattr(cache, "layers", None)
        if layers is not None:
            for layer in layers:
                key = getattr(layer, "keys", None)
                value = getattr(layer, "values", None)
                if key is None and value is None:
                    continue
                if key is None or value is None:
                    raise RuntimeError(
                        "DynamicCache layer exposed only one of keys/values"
                    )
                flat.extend((key, value))

        # Transformers 4.x compatibility.
        if not flat:
            key_cache = getattr(cache, "key_cache", None)
            value_cache = getattr(cache, "value_cache", None)
            if key_cache is not None and value_cache is not None:
                for key, value in zip(key_cache, value_cache, strict=True):
                    flat.extend((key, value))

    if not flat or not all(hasattr(item, "shape") for item in flat):
        raise RuntimeError(
            f"Could not flatten cache type {type(cache)!r} into tensor K/V pairs"
        )
    if len(flat) % 2 != 0:
        raise RuntimeError(
            f"Flattened cache contained {len(flat)} tensors; expected K/V pairs"
        )
    return tuple(flat)


def summarize_pi05_ir(model_ir: ModelIR) -> Pi05ProbeSummary:
    """Report pi0.5 graph visibility without inventing any VLM->AE edge."""

    nodes = model_ir.nodes
    adjacency: Dict[str, List[str]] = defaultdict(list)
    predecessors: Dict[str, List[str]] = defaultdict(list)
    for edge in model_ir.edges:
        adjacency[edge.source].append(edge.target)
        predecessors[edge.target].append(edge.source)

    def matching(*tokens: str) -> Tuple[str, ...]:
        lowered_tokens = tuple(token.lower() for token in tokens)
        out = []
        for node in nodes.values():
            haystack = f"{node.id} {node.target} {node.module_path}".lower()
            if any(token in haystack for token in lowered_tokens):
                out.append(node.id)
        return tuple(out)

    q_nodes = matching("q_proj")
    k_nodes = matching("k_proj")
    v_nodes = matching("v_proj")
    vlm_nodes = matching("paligemma", "language_model")
    expert_nodes = matching("gemma_expert")

    output_ids = [node.id for node in nodes.values() if node.kind == "output"]
    output_predecessors: List[str] = []
    for output_id in output_ids:
        output_predecessors.extend(predecessors.get(output_id, ()))

    expert_by_layer: Dict[int, List[str]] = defaultdict(list)
    for node_id in expert_nodes:
        layer = _layer_index(nodes[node_id].module_path)
        if layer is not None:
            expert_by_layer[layer].append(node_id)

    kv_by_layer: Dict[int, Dict[str, List[str]]] = defaultdict(
        lambda: {"k": [], "v": []}
    )
    for kind, node_ids in (("k", k_nodes), ("v", v_nodes)):
        for node_id in node_ids:
            node = nodes[node_id]
            if "gemma_expert" in node.module_path.lower():
                continue
            layer = _layer_index(node.module_path)
            if layer is not None:
                kv_by_layer[layer][kind].append(node_id)

    candidate_layers: List[int] = []
    for layer, kv_nodes in sorted(kv_by_layer.items()):
        expert_targets = expert_by_layer.get(layer, [])
        if not kv_nodes["k"] or not kv_nodes["v"] or not expert_targets:
            continue
        k_reaches = any(
            _reaches_any(source, expert_targets, adjacency)
            for source in kv_nodes["k"]
        )
        v_reaches = any(
            _reaches_any(source, expert_targets, adjacency)
            for source in kv_nodes["v"]
        )
        if k_reaches and v_reaches:
            candidate_layers.append(layer)

    notes: List[str] = []
    if not k_nodes or not v_nodes:
        notes.append(
            "Q/K/V projection module names were not visible in the captured IR; "
            "the graph may be fused, decomposed differently, or missing module metadata."
        )
    if output_ids and len(output_predecessors) >= 2:
        notes.append(
            "The exported output has multiple tensor producers; for the prefix probe "
            "this is consistent with flattened per-layer KV tensors becoming graph-visible."
        )
    if expert_nodes and not candidate_layers:
        notes.append(
            "VLM/AE modules are visible but no same-layer VLM K/V -> expert dependency "
            "was proven by graph reachability. Inspect the raw IR before adding annotations."
        )
    if candidate_layers:
        notes.append(
            "Same-layer VLM K/V producers reach Action Expert nodes in the captured graph; "
            "these are Wave-KV candidate boundaries for dependency-aware coarsening."
        )

    return Pi05ProbeSummary(
        node_count=len(nodes),
        edge_count=len(model_ir.edges),
        q_proj_nodes=q_nodes,
        k_proj_nodes=k_nodes,
        v_proj_nodes=v_nodes,
        vlm_nodes=vlm_nodes,
        expert_nodes=expert_nodes,
        output_predecessors=tuple(dict.fromkeys(output_predecessors)),
        wave_kv_candidate_layers=tuple(candidate_layers),
        notes=tuple(notes),
    )


def model_ir_to_dict(model_ir: ModelIR) -> Dict[str, Any]:
    return {
        "metadata": dict(model_ir.metadata),
        "nodes": [
            {
                "id": node.id,
                "kind": node.kind,
                "op": node.op,
                "target": node.target,
                "module_path": node.module_path,
                "placement": node.placement,
                "metadata": {
                    key: value
                    for key, value in node.metadata.items()
                    if key != "stack_trace"
                },
            }
            for node in model_ir.nodes.values()
        ],
        "edges": [
            {
                "source": edge.source,
                "target": edge.target,
                "size_bytes": edge.size_bytes,
                "tensor_id": edge.tensor_id,
            }
            for edge in model_ir.edges
        ],
    }


def format_model_ir(model_ir: ModelIR) -> str:
    lines = ["# nodes"]
    for node in model_ir.nodes.values():
        lines.append(
            f"{node.id:48s} kind={node.kind:6s} op={node.op:20s} module={node.module_path}"
        )
    lines.append("")
    lines.append("# edges")
    for edge in model_ir.edges:
        lines.append(
            f"{edge.source} -> {edge.target} size_bytes={edge.size_bytes:.0f}"
        )
    return "\n".join(lines) + "\n"


def _layer_index(module_path: str) -> Optional[int]:
    text = module_path.replace("[", ".").replace("]", ".").replace("'", "")
    match = _LAYER_RE.search(text)
    if match is None:
        return None
    return int(match.group(1))


def _reaches_any(
    source: str,
    targets: Sequence[str],
    adjacency: Mapping[str, Sequence[str]],
) -> bool:
    target_set = set(targets)
    queue = deque([source])
    seen = {source}
    while queue:
        current = queue.popleft()
        if current in target_set and current != source:
            return True
        for nxt in adjacency.get(current, ()):
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return False
