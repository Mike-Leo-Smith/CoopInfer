from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def _add_repo_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _default_output(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_coarsened_coopinfer.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a costed fine ModelIR, coarsen it with generic DAG/cost "
            "signals, and export a CoopInfer GUI/backend JSON. This command does "
            "not evaluate or solve the graph."
        )
    )
    parser.add_argument("input", type=Path, help="Costed fine ModelIR JSON")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bandwidth-mb-s", type=float, default=1250.0)
    parser.add_argument("--latency-ms", type=float, default=0.2)
    parser.add_argument("--max-ops-per-group", type=int, default=16)
    parser.add_argument("--max-group-cost-ms", type=float, default=None)
    parser.add_argument("--reference-resource", default=None)
    parser.add_argument("--min-merge-affinity", type=float, default=0.45)
    parser.add_argument("--boundary-threshold", type=float, default=0.55)
    parser.add_argument("--long-range-span-threshold", type=float, default=0.10)
    parser.add_argument("--long-range-min-skipped-nodes", type=int, default=2)
    parser.add_argument(
        "--communication-exposure-threshold", type=float, default=0.18
    )
    args = parser.parse_args()

    _add_repo_src_to_path()

    from coopinfer.frontend import (
        DependencyAnalysisConfig,
        DependencyAwarePolicy,
        analyze_dependencies,
        load_model_ir_json,
        to_coopinfer_payload,
    )

    model_ir = load_model_ir_json(args.input)
    analyzed = analyze_dependencies(
        model_ir,
        config=DependencyAnalysisConfig(
            boundary_threshold=args.boundary_threshold,
            long_range_span_threshold=args.long_range_span_threshold,
            communication_exposure_threshold=args.communication_exposure_threshold,
            long_range_min_skipped_nodes=args.long_range_min_skipped_nodes,
        ),
    )
    scheduling_ir = DependencyAwarePolicy(
        max_ops_per_group=args.max_ops_per_group,
        max_group_cost_ms=args.max_group_cost_ms,
        reference_resource=args.reference_resource,
        min_merge_affinity=args.min_merge_affinity,
    ).apply(analyzed)

    payload = to_coopinfer_payload(
        scheduling_ir,
        bandwidth_mb_s=args.bandwidth_mb_s,
        latency_ms=args.latency_ms,
    )
    output = args.output or _default_output(args.input)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    dep_meta = analyzed.metadata.get("dependency_analysis", {})
    coarse_meta = scheduling_ir.metadata
    fine_nodes = len(model_ir.nodes)
    coarse_nodes = len(scheduling_ir.nodes)
    ratio = fine_nodes / coarse_nodes if coarse_nodes else 0.0

    print("===== generic dependency-aware coarsening =====")
    print(f"input={args.input}")
    print(f"output={output}")
    print(f"fine_nodes={fine_nodes}")
    print(f"fine_edges={len(model_ir.edges)}")
    print(f"coarse_nodes={coarse_nodes}")
    print(f"coarse_edges={len(scheduling_ir.edges)}")
    print(f"compression_ratio={ratio:.3f}x")
    print(f"largest_group={coarse_meta.get('largest_group', 0)}")
    print(f"average_group_size={coarse_meta.get('average_group_size', 0.0):.3f}")
    print(
        "preserved_dependency_edges="
        f"{dep_meta.get('preserved_edges', 0)}/{dep_meta.get('edge_count', 0)}"
    )
    print(
        "thresholds="
        f"boundary:{args.boundary_threshold} "
        f"span:{args.long_range_span_threshold} "
        f"min_skipped:{args.long_range_min_skipped_nodes} "
        f"exposure:{args.communication_exposure_threshold} "
        f"merge_affinity:{args.min_merge_affinity}"
    )
    print(f"network={args.bandwidth_mb_s} MB/s + {args.latency_ms} ms/transfer")
    print("evaluation=not_run")
    print("solver=not_run")


if __name__ == "__main__":
    main()