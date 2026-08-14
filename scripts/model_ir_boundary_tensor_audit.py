from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys


def _add_repo_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _short_root(root: str) -> str:
    parts = [part for part in root.split(".") if part]
    if len(parts) <= 4:
        return root
    return ".".join(parts[-4:])


def _fmt_shape(row: dict) -> str:
    shape = row.get("shape", [])
    dtype = row.get("dtype", "?")
    numel = row.get("numel", "?")
    nbytes = row.get("nbytes", "?")
    return f"shape={shape} dtype={dtype} numel={numel} native_bytes={nbytes}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Read-only audit of automatically recovered layer-boundary payloads. "
            "It reports dependency edges, payload bytes, tensor identities, producer ops, "
            "module paths, and tensor shapes without model-specific dependency rules."
        )
    )
    parser.add_argument("model_ir", type=Path)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument(
        "--cross-only",
        action="store_true",
        help="Print tensor details only for cross-stack layer dependencies.",
    )
    parser.add_argument(
        "--details-limit",
        type=int,
        default=8,
        help="Maximum tensor detail rows printed per layer edge.",
    )
    args = parser.parse_args()

    _add_repo_src_to_path()
    from coopinfer.frontend import (
        detect_layer_groups,
        discover_layer_dependencies,
        load_model_ir_json,
    )
    from coopinfer.frontend.discovered_layer_schedule import discover_layer_frontier_payloads

    model_ir = load_model_ir_json(args.model_ir)
    grouping = detect_layer_groups(model_ir)
    dependencies = discover_layer_dependencies(model_ir, grouping)
    payloads = discover_layer_frontier_payloads(
        model_ir,
        grouping,
        precision=args.precision,
    )

    print("===== boundary tensor audit =====")
    print(f"model_ir={args.model_ir}")
    print(f"precision={args.precision}")
    print(f"fine_nodes={len(model_ir.nodes)}")
    print(f"fine_edges={len(model_ir.edges)}")
    print(f"layer_dependencies={len(dependencies)}")

    cross = [dependency for dependency in dependencies if dependency.cross_stack]
    same_index_cross = [
        dependency
        for dependency in cross
        if dependency.source_layer is not None
        and dependency.source_layer == dependency.target_layer
    ]
    print(f"cross_stack_dependencies={len(cross)}")
    print(f"same_index_cross_stack_dependencies={len(same_index_cross)}")

    rows = []
    missing = []
    for dependency in dependencies:
        key = (dependency.source_group, dependency.target_group)
        payload = payloads.get(key)
        if payload is None:
            missing.append(key)
            continue
        rows.append((dependency, payload))

    if missing:
        print(f"missing_payload_dependencies={len(missing)}")
        for source, target in missing[:20]:
            print(f"  MISSING {source} -> {target}")
    else:
        print("missing_payload_dependencies=0")

    print("\n===== cross-stack payload summary =====")
    cross_rows = [(dependency, payload) for dependency, payload in rows if dependency.cross_stack]
    for dependency, payload in cross_rows:
        source = grouping.groups[dependency.source_group]
        target = grouping.groups[dependency.target_group]
        print(
            f"  {_short_root(source.stack_root)}[{source.layer_index}] -> "
            f"{_short_root(target.stack_root)}[{target.layer_index}] "
            f"bytes={payload.size_bytes:.0f} "
            f"MiB={payload.size_bytes / (1024 ** 2):.6f} "
            f"tensors={len(payload.tensor_ids)}"
        )

    pattern_counter = Counter()
    pair_sizes = defaultdict(list)
    for dependency, payload in cross_rows:
        source = grouping.groups[dependency.source_group]
        target = grouping.groups[dependency.target_group]
        pair = (source.stack_root, target.stack_root)
        pair_sizes[pair].append(float(payload.size_bytes))
        pattern_counter[
            (
                source.stack_root,
                target.stack_root,
                round(float(payload.size_bytes), 6),
                len(payload.tensor_ids),
            )
        ] += 1

    print("\n===== cross-stack payload patterns =====")
    for (source_root, target_root, size_bytes, tensor_count), count in sorted(
        pattern_counter.items(), key=lambda item: (-item[1], item[0])
    ):
        print(
            f"  count={count:3d} bytes={size_bytes:.0f} "
            f"MiB={size_bytes / (1024 ** 2):.6f} tensors={tensor_count} "
            f"{_short_root(source_root)} -> {_short_root(target_root)}"
        )

    print("\n===== tensor details =====")
    detail_rows = cross_rows if args.cross_only else rows
    for dependency, payload in detail_rows:
        source_group = grouping.groups[dependency.source_group]
        target_group = grouping.groups[dependency.target_group]
        relation = "cross" if dependency.cross_stack else "same"
        print(
            f"\n[{relation}] {_short_root(source_group.stack_root)}[{source_group.layer_index}] -> "
            f"{_short_root(target_group.stack_root)}[{target_group.layer_index}] "
            f"payload_bytes={payload.size_bytes:.0f} tensors={len(payload.tensor_ids)}"
        )
        for tensor_key in payload.tensor_ids[: args.details_limit]:
            # discover_layer_frontier_payloads stores keys as "source_node:tensor_id".
            source_node_id = tensor_key.split(":", 1)[0]
            node = model_ir.nodes.get(source_node_id)
            if node is None:
                print(f"  tensor={tensor_key} producer=<missing>")
                continue
            print(
                f"  tensor={tensor_key}\n"
                f"    producer={source_node_id}\n"
                f"    op={node.op}\n"
                f"    target={node.target}\n"
                f"    module_path={node.module_path}"
            )
            tensors = node.metadata.get("output_tensors", ())
            if tensors:
                for row in tensors:
                    if isinstance(row, dict):
                        print(f"    {_fmt_shape(row)}")
            else:
                print("    output_tensors=<none>")
        remaining = len(payload.tensor_ids) - args.details_limit
        if remaining > 0:
            print(f"  ... {remaining} more tensors")

    same_index_sizes = []
    for dependency, payload in cross_rows:
        if (
            dependency.source_layer is not None
            and dependency.source_layer == dependency.target_layer
        ):
            same_index_sizes.append(float(payload.size_bytes))

    print("\n===== same-index cross-stack size check =====")
    if not same_index_sizes:
        print("count=0")
    else:
        unique = Counter(round(value, 6) for value in same_index_sizes)
        print(f"count={len(same_index_sizes)}")
        print(f"min_bytes={min(same_index_sizes):.0f}")
        print(f"max_bytes={max(same_index_sizes):.0f}")
        print(f"total_MiB={sum(same_index_sizes) / (1024 ** 2):.6f}")
        print("unique_sizes:")
        for value, count in sorted(unique.items()):
            print(f"  bytes={value:.0f} MiB={value / (1024 ** 2):.6f} count={count}")

    if not missing:
        print("BOUNDARY_AUDIT_STATUS=PASS_ALL_DEPENDENCIES_HAVE_PAYLOADS")
    else:
        print("BOUNDARY_AUDIT_STATUS=CHECK_MISSING_PAYLOADS")


if __name__ == "__main__":
    main()
