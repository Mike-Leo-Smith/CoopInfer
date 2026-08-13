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
    return input_path.with_name(f"{input_path.stem}_coopinfer.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an already-costed ModelIR into CoopInfer backend JSON. "
            "This command performs no evaluation and no placement search."
        )
    )
    parser.add_argument("input", type=Path, help="Costed ModelIR JSON")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bandwidth-mb-s", type=float, default=1250.0)
    parser.add_argument("--latency-ms", type=float, default=0.2)
    args = parser.parse_args()

    _add_repo_src_to_path()

    from coopinfer.frontend import (
        identity_scheduling_ir,
        load_model_ir_json,
        to_coopinfer_payload,
    )

    model_ir = load_model_ir_json(args.input)
    scheduling_ir = identity_scheduling_ir(model_ir)
    payload = to_coopinfer_payload(
        scheduling_ir,
        bandwidth_mb_s=args.bandwidth_mb_s,
        latency_ms=args.latency_ms,
    )

    output = args.output or _default_output(args.input)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("===== ModelIR -> CoopInfer export =====")
    print(f"input={args.input}")
    print(f"output={output}")
    print(f"model_ir_nodes={len(model_ir.nodes)}")
    print(f"model_ir_edges={len(model_ir.edges)}")
    print(f"scheduling_nodes={len(scheduling_ir.nodes)}")
    print(f"scheduling_edges={len(scheduling_ir.edges)}")
    print(f"cost_source={model_ir.metadata.get('cost_source', 'unknown')}")
    print(
        "network="
        f"{args.bandwidth_mb_s} MB/s + {args.latency_ms} ms/transfer"
    )
    print("evaluation=not_run")
    print("solver=not_run")


if __name__ == "__main__":
    main()
