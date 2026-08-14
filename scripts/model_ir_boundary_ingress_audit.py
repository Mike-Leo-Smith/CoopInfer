from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys


def _add_repo_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _short(root: str) -> str:
    parts = [p for p in root.split(".") if p]
    return ".".join(parts[-4:]) if len(parts) > 4 else root


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare current source-origin layer payloads with tensors immediately entering "
            "the target layer. Read-only; no production scheduling code is modified."
        )
    )
    parser.add_argument("model_ir", type=Path)
    parser.add_argument("--precision", default="bf16")
    args = parser.parse_args()

    _add_repo_src_to_path()
    from coopinfer.frontend import detect_layer_groups, discover_layer_dependencies, load_model_ir_json
    from coopinfer.frontend.discovered_layer_schedule import (
        _edge_bytes_for_precision,
        discover_layer_frontier_payloads,
    )

    ir = load_model_ir_json(args.model_ir)
    grouping = detect_layer_groups(ir)
    deps = discover_layer_dependencies(ir, grouping)
    old_payloads = discover_layer_frontier_payloads(ir, grouping, precision=args.precision)

    incoming = defaultdict(list)
    for edge in ir.edges:
        incoming[edge.target].append(edge)

    # For each dependency S->T, inspect Fine edges that enter members of T. Keep an
    # ingress tensor when walking backwards through non-layer glue reaches S before
    # any other detected layer. This approximates the payload under the convention
    # that transparent/residual glue between layers is folded into the producer side.
    layer_ids = {gid for gid, g in grouping.groups.items() if g.kind == "layer"}
    reverse_adj = defaultdict(list)
    for edge in ir.edges:
        reverse_adj[edge.target].append(edge.source)

    def reaches_source_without_other_layer(start: str, source_gid: str, target_gid: str) -> bool:
        stack = [start]
        seen = set()
        while stack:
            node_id = stack.pop()
            if node_id in seen:
                continue
            seen.add(node_id)
            gid = grouping.member_to_group[node_id]
            if gid == source_gid:
                return True
            if gid in layer_ids and gid != target_gid:
                continue
            stack.extend(reverse_adj.get(node_id, ()))
        return False

    print("===== target-ingress boundary audit =====")
    print(f"model_ir={args.model_ir}")
    print(f"precision={args.precision}")
    print(f"dependencies={len(deps)}")

    same_ratios = []
    cross_ratios = []
    for dep in deps:
        source = grouping.groups[dep.source_group]
        target = grouping.groups[dep.target_group]
        tensors = {}
        for target_member in target.members:
            for edge in incoming.get(target_member, ()):
                pred_gid = grouping.member_to_group[edge.source]
                if pred_gid == target.id:
                    continue
                if not reaches_source_without_other_layer(edge.source, source.id, target.id):
                    continue
                key = edge.tensor_id or edge.source
                size = _edge_bytes_for_precision(
                    edge,
                    source_node=ir.nodes.get(edge.source),
                    precision=args.precision,
                )
                tensors[f"{edge.source}:{key}"] = max(size, tensors.get(f"{edge.source}:{key}", 0.0))

        ingress_bytes = float(sum(tensors.values()))
        old = old_payloads.get((source.id, target.id))
        old_bytes = float(old.size_bytes) if old is not None else 0.0
        ratio = (old_bytes / ingress_bytes) if ingress_bytes > 0 else float("inf")
        relation = "cross" if dep.cross_stack else "same"
        print(
            f"[{relation}] {_short(source.stack_root)}[{source.layer_index}] -> "
            f"{_short(target.stack_root)}[{target.layer_index}] "
            f"source_origin={old_bytes:.0f} ingress={ingress_bytes:.0f} "
            f"ratio={ratio:.3f} ingress_tensors={len(tensors)}"
        )
        if dep.cross_stack:
            cross_ratios.append(ratio)
        else:
            same_ratios.append(ratio)

    if same_ratios:
        finite = [x for x in same_ratios if x != float("inf")]
        print("\n===== same-stack ratio summary =====")
        print(f"count={len(same_ratios)}")
        if finite:
            print(f"min={min(finite):.3f} max={max(finite):.3f}")
    if cross_ratios:
        finite = [x for x in cross_ratios if x != float("inf")]
        print("\n===== cross-stack ratio summary =====")
        print(f"count={len(cross_ratios)}")
        if finite:
            print(f"min={min(finite):.3f} max={max(finite):.3f}")


if __name__ == "__main__":
    main()
