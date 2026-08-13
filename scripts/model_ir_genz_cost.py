from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys


def _add_repo_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _add_genz_checkout_to_path(root: Path | None) -> None:
    if root is None:
        return
    root = root.resolve()
    candidates = (root, root / "genz", root / "vla-perf")
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _default_output(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_genz_costed.json")


def _has_tensor_metadata(model_ir) -> bool:
    return any(
        node.kind == "op"
        and node.metadata.get("input_tensors")
        and node.metadata.get("output_tensors")
        for node in model_ir.nodes.values()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Attach generic GenZ fine-op costs to a ModelIR. The input graph may "
            "come from any torch.export frontend; no model-specific profiler is used."
        )
    )
    parser.add_argument("input", type=Path, help="Fine ModelIR JSON")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--genz-root",
        type=Path,
        default=None,
        help=(
            "Optional checkout root containing GenZ and Systems. For NVlabs/vla-perf "
            "this is the repository root, e.g. D:\\Project\\vla-perf."
        ),
    )
    parser.add_argument("--device", default="RTX_4090")
    parser.add_argument("--host", default="A100_80GB")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--compute-efficiency", type=float, default=1.0)
    parser.add_argument("--memory-efficiency", type=float, default=1.0)
    args = parser.parse_args()

    _add_repo_src_to_path()
    _add_genz_checkout_to_path(args.genz_root)

    from coopinfer.cost import annotate_genz_costs
    from coopinfer.frontend import load_model_ir_json, write_model_ir_json

    model_ir = load_model_ir_json(args.input)
    if not _has_tensor_metadata(model_ir):
        raise SystemExit(
            "ModelIR has no fine tensor shape metadata. Re-capture it with the "
            "current torch.export frontend before running GenZ costing."
        )

    costed_ir = annotate_genz_costs(
        model_ir,
        {
            "device": args.device,
            "host": args.host,
        },
        precision=args.precision,
        compute_efficiency=args.compute_efficiency,
        memory_efficiency=args.memory_efficiency,
    )

    output = args.output or _default_output(args.input)
    write_model_ir_json(costed_ir, output)

    print("===== generic GenZ fine-op costing =====")
    print(f"input={args.input}")
    print(f"output={output}")
    print(f"nodes={len(costed_ir.nodes)}")
    print(f"edges={len(costed_ir.edges)}")
    print(f"precision={args.precision}")
    print(f"hardware=device:{args.device} host:{args.host}")

    for resource in ("device", "host"):
        total_ms = sum(
            float(node.costs_ms.get(resource, 0.0))
            for node in costed_ir.nodes.values()
        )
        positive_nodes = sum(
            1
            for node in costed_ir.nodes.values()
            if float(node.costs_ms.get(resource, 0.0)) > 0.0
        )
        models = Counter()
        bounds = Counter()
        for node in costed_ir.nodes.values():
            sources = node.metadata.get("cost_sources", {})
            row = sources.get(resource, {}) if isinstance(sources, dict) else {}
            metadata = row.get("metadata", {}) if isinstance(row, dict) else {}
            if isinstance(metadata, dict):
                if metadata.get("op_model"):
                    models[str(metadata["op_model"])] += 1
                if metadata.get("bound"):
                    bounds[str(metadata["bound"])] += 1
        print(
            f"{resource}: total_ms={total_ms:.6f} positive_nodes={positive_nodes} "
            f"op_models={dict(models)} bounds={dict(bounds)}"
        )


if __name__ == "__main__":
    main()
