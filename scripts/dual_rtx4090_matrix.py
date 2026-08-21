from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path


DEFAULT_MODELS = (
    "smolvla",
    "pi0",
    "pi05",
    "pi0_fast",
    "xvla",
    "groot_n17",
)


def _is_iterative_node(node_id: object) -> bool:
    return "::step" in str(node_id)


def _variant(
    base: dict,
    *,
    model: str,
    mode: str,
    pipeline_unroll: int,
    bandwidth_mb_s: float,
    latency_ms: float,
) -> dict:
    payload = deepcopy(base)
    payload["environment"]["bandwidth"] = float(bandwidth_mb_s)
    payload["environment"]["latency"] = float(latency_ms)
    payload["environment"]["pipeline_unroll"] = int(pipeline_unroll)
    payload["environment"]["weight_avg_latency"] = 1.0
    payload["environment"]["weight_max_latency"] = 0.0
    payload["environment"]["weight_device_utilization"] = 0.0

    for node in payload["nodes"]:
        if mode == "free":
            node["placement"] = "free"
            node["x"] = 1
        elif mode == "split":
            # In this experiment Host means GPU0 and Device means GPU1.
            # Static vision/VLM/prefill stays on GPU0; every materialized
            # denoise/decode execution node runs on GPU1.
            iterative = _is_iterative_node(node["id"])
            node["placement"] = "device" if iterative else "host"
            node["x"] = 0 if iterative else 1
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    payload["experiment"] = {
        "name": "dual_rtx4090_pipeline_matrix_v1",
        "model": model,
        "gpu0": "RTX_4090 (Host label)",
        "gpu1": "RTX_4090 (Device label)",
        "placement_mode": mode,
        "pipeline_unroll": int(pipeline_unroll),
        "bandwidth_mb_s": float(bandwidth_mb_s),
        "latency_ms": float(latency_ms),
        "source_release_policy": "saturated_zero_period",
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate GUI-loadable dual-RTX-4090 free/split CoopInfer configs "
            "for pipeline unroll experiments."
        )
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--unrolls", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--bandwidth-mb-s", type=float, default=20000.0)
    parser.add_argument("--latency-ms", type=float, default=0.01)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    if any(unroll < 1 for unroll in args.unrolls):
        raise ValueError("Every pipeline unroll must be at least 1")

    generated = []
    for model in args.models:
        model_dir = args.repo_root / "examples" / "models" / model / "dual_rtx4090"
        base_path = model_dir / f"{model}_dual4090_base_coopinfer.json"
        base = json.loads(base_path.read_text(encoding="utf-8"))
        for unroll in args.unrolls:
            for mode in ("free", "split"):
                output = model_dir / f"{model}_dual4090_{mode}_u{unroll}_coopinfer.json"
                payload = _variant(
                    base,
                    model=model,
                    mode=mode,
                    pipeline_unroll=unroll,
                    bandwidth_mb_s=args.bandwidth_mb_s,
                    latency_ms=args.latency_ms,
                )
                output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                generated.append(str(output.relative_to(args.repo_root)))

    manifest_path = args.repo_root / "results" / "dual_rtx4090" / "config_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "experiment": "dual_rtx4090_pipeline_matrix_v1",
                "models": list(args.models),
                "unrolls": list(args.unrolls),
                "modes": ["free", "split"],
                "bandwidth_mb_s": args.bandwidth_mb_s,
                "latency_ms": args.latency_ms,
                "configs": generated,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"generated_configs={len(generated)}")
    print(f"manifest={manifest_path}")


if __name__ == "__main__":
    main()
