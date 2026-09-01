from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def _scalar(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        if hasattr(value, "iloc"):
            return float(value.iloc[0])
        raise


def _model_df_latency_sum(output: dict[str, Any]) -> float | None:
    df = output.get("model_df")
    if df is None:
        return None
    for column in ("Latency (msec)", "Latency (ms)", "Latency"):
        if column in df.columns:
            return float(df[column].sum())
    return None


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _projector_roofline_ms(
    *,
    system_config: dict[str, Any],
    precision: str,
    tokens: int,
    input_dim: int,
    output_dim: int,
    bytes_per_element: float,
) -> dict[str, float]:
    flops_table = system_config.get("Flops", {})
    if not isinstance(flops_table, dict) or precision not in flops_table:
        aliases = {
            "bf16": "fp16",
            "fp16": "bf16",
            "fp32": "tf32",
            "tf32": "fp32",
        }
        alt = aliases.get(precision)
        if alt is None or alt not in flops_table:
            raise ValueError(
                f"System does not define {precision!r} throughput: {flops_table}"
            )
        precision = alt

    peak_tflops = float(flops_table[precision])
    memory_bw_gb_s = float(system_config["Memory_BW"])
    flops = 2.0 * tokens * input_dim * output_dim
    bytes_moved = (
        input_dim * output_dim
        + tokens * input_dim
        + tokens * output_dim
    ) * bytes_per_element

    compute_ms = flops / (peak_tflops * 1e12) * 1e3
    memory_ms = bytes_moved / (memory_bw_gb_s * 1e9) * 1e3
    return {
        "latency_ms": max(compute_ms, memory_ms),
        "compute_ms": compute_ms,
        "memory_ms": memory_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Profile a pi0/pi0.5-shaped VLA with NVIDIA VLA-Perf/GenZ and "
            "export a hardware-neutral CoopInfer cost profile."
        )
    )
    parser.add_argument("--vla-perf-root", type=Path, required=True)
    parser.add_argument("--device", default="RTX_4090")
    parser.add_argument("--host", default="H100")
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--vision-tokens", type=int, default=256)
    parser.add_argument("--vision-frames", type=int, default=3)
    parser.add_argument("--language-tokens", type=int, default=32)
    parser.add_argument("--state-tokens", type=int, default=32)
    parser.add_argument("--image-resolution", type=int, default=384)
    parser.add_argument("--action-chunk-size", type=int, default=50)
    parser.add_argument("--denoising-steps", type=int, default=10)
    parser.add_argument("--action-dof", type=int, default=14)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/vla_perf_pi05_profile.json"),
    )
    args = parser.parse_args()

    root = args.vla_perf_root.resolve()
    genz_root = root / "genz"
    vla_perf_dir = root / "vla-perf"
    if not genz_root.exists() or not vla_perf_dir.exists():
        raise SystemExit(
            f"{root} does not look like an NVlabs/vla-perf checkout "
            "(expected genz/ and vla-perf/)."
        )

    # Official VLA-Perf installs GenZ as a package; these paths also support
    # running straight from an editable checkout.
    sys.path.insert(0, str(genz_root))
    sys.path.insert(0, str(vla_perf_dir))

    from GenZ import prefill_moddeling, parallel_decode_modeling
    from GenZ.Models import get_configs
    import GenZ.Models.Model_sets.vla_models  # noqa: F401 - registers VLA configs
    from Systems.system_configs import system_configs

    precision = args.precision.lower()
    dtype_bytes = {
        "fp32": 4.0,
        "tf32": 4.0,
        "bf16": 2.0,
        "fp16": 2.0,
        "fp8": 1.0,
        "int8": 1.0,
        "fp4": 0.5,
        "int4": 0.5,
    }
    if precision not in dtype_bytes:
        raise SystemExit(f"Unsupported precision for this adapter: {precision}")

    vision_model = "pi0-vision"
    vlm_model = "pi0-vlm"
    ae_model = "pi0-action-expert"

    vision_cfg = get_configs(vision_model)
    vlm_cfg = get_configs(vlm_model)
    ae_cfg = get_configs(ae_model)

    prefix_tokens = (
        args.vision_tokens * args.vision_frames
        + args.language_tokens
        + args.state_tokens
    )

    def profile_system(system: str) -> dict[str, Any]:
        if system not in system_configs:
            raise ValueError(
                f"Unknown VLA-Perf/GenZ system {system!r}. "
                "Available examples include RTX_4090, A100_80GB, H100."
            )
        vision = prefill_moddeling(
            model=vision_model,
            batch_size=args.vision_frames,
            input_tokens=args.vision_tokens,
            system_name=system,
            bits=precision,
            tensor_parallel=1,
            pipeline_parallel=1,
            debug=False,
        )
        vlm = prefill_moddeling(
            model=vlm_model,
            batch_size=1,
            input_tokens=prefix_tokens,
            system_name=system,
            bits=precision,
            tensor_parallel=1,
            pipeline_parallel=1,
            debug=False,
        )
        ae = parallel_decode_modeling(
            model=ae_model,
            batch_size=1,
            input_tokens=prefix_tokens,
            output_tokens_parallel=args.action_chunk_size,
            self_attention=True,
            system_name=system,
            bits=precision,
            tensor_parallel=1,
            pipeline_parallel=1,
            debug=False,
        )
        projector = _projector_roofline_ms(
            system_config=system_configs[system],
            precision=precision,
            tokens=args.vision_tokens * args.vision_frames,
            input_dim=int(vision_cfg.hidden_size),
            output_dim=int(vlm_cfg.hidden_size),
            bytes_per_element=dtype_bytes[precision],
        )
        return {
            "vision_total": _scalar(vision["Latency"]),
            "vlm_total": _scalar(vlm["Latency"]),
            "ae_pass_total": _scalar(ae["Latency"]),
            "projector": projector["latency_ms"],
            "diagnostics": {
                "vision_model_df_latency_sum_ms": _model_df_latency_sum(vision),
                "vlm_model_df_latency_sum_ms": _model_df_latency_sum(vlm),
                "ae_model_df_latency_sum_ms": _model_df_latency_sum(ae),
                "projector_compute_ms": projector["compute_ms"],
                "projector_memory_ms": projector["memory_ms"],
                "peak_flops": system_configs[system].get("Flops"),
                "memory_bw_gb_s": system_configs[system].get("Memory_BW"),
            },
        }

    costs = {
        "device": profile_system(args.device),
        "host": profile_system(args.host),
    }

    profile = {
        "schema_version": 1,
        "source": {
            "project": "NVlabs/vla-perf",
            "backend": "GenZ",
            "checkout_commit": _git_commit(root),
            "note": (
                "pi0.5 uses the pi0 transformer dimensions registered by "
                "VLA-Perf; state tokens are explicitly added to prefix length."
            ),
        },
        "precision": precision,
        "hardware": {
            "device": args.device,
            "host": args.host,
        },
        "models": {
            "vision": vision_model,
            "vlm": vlm_model,
            "action_expert": ae_model,
        },
        "model": {
            "vision_layers": int(vision_cfg.num_encoder_layers),
            "vlm_layers": int(vlm_cfg.num_decoder_layers),
            "ae_layers": int(ae_cfg.num_decoder_layers),
            "vision_tokens_per_frame": args.vision_tokens,
            "vision_frames": args.vision_frames,
            "language_tokens": args.language_tokens,
            "state_tokens": args.state_tokens,
            "action_chunk_size": args.action_chunk_size,
            "denoising_steps": args.denoising_steps,
            "action_dof": args.action_dof,
            "image_resolution": args.image_resolution,
            "vision_hidden_size": int(vision_cfg.hidden_size),
            "vlm_hidden_size": int(vlm_cfg.hidden_size),
            "ae_hidden_size": int(ae_cfg.hidden_size),
            "vlm_num_kv_heads": int(
                vlm_cfg.num_key_value_heads or vlm_cfg.num_attention_heads
            ),
            "vlm_head_dim": int(
                vlm_cfg.head_dim
                or (vlm_cfg.hidden_size // vlm_cfg.num_attention_heads)
            ),
            "prefix_tokens": prefix_tokens,
        },
        "costs_ms": costs,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(profile, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote VLA-Perf profile: {args.output}")
    print(
        f"{args.device}: vision={costs['device']['vision_total']:.4f} ms, "
        f"vlm={costs['device']['vlm_total']:.4f} ms, "
        f"AE/pass={costs['device']['ae_pass_total']:.4f} ms"
    )
    print(
        f"{args.host}: vision={costs['host']['vision_total']:.4f} ms, "
        f"vlm={costs['host']['vlm_total']:.4f} ms, "
        f"AE/pass={costs['host']['ae_pass_total']:.4f} ms"
    )


if __name__ == "__main__":
    main()
