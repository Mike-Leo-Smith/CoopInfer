from __future__ import annotations

import argparse

from coopinfer.vla_perf_adapter import (
    build_pi05_coopinfer_payload,
    graph_spec_from_profile,
    load_vla_perf_profile,
    tensor_sizes_mb,
    write_coopinfer_payload,
)


def _fix_separate_token_id_edges(payload, spec) -> None:
    """Keep language/state int32 token transport sizes independent.

    The core adapter currently exposes a combined token-id helper size for
    convenience. These are two distinct source tensors in the generated DAG,
    so correct each edge to its own int32 payload before writing the JSON.
    """
    language_mb = spec.language_tokens * 4.0 / (1024.0 * 1024.0)
    state_mb = spec.state_tokens * 4.0 / (1024.0 * 1024.0)
    for edge in payload["edges"]:
        if edge["source"] == "language_input" and edge["target"] == "vlm_0":
            edge["size"] = language_mb
        elif edge["source"] == "state_input" and edge["target"] == "vlm_0":
            edge["size"] = state_mb


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a VLA-Perf/GenZ pi0.5 profile into a CoopInfer JSON."
    )
    parser.add_argument("profile")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bandwidth", type=float, default=1250.0, help="MB/s")
    parser.add_argument("--latency", type=float, default=0.2, help="ms")
    parser.add_argument(
        "--ae-placement",
        choices=["device", "host"],
        default="device",
        help=(
            "Pin the whole Action Expert to one side. This preserves persistent "
            "KV residency with the current stateless CoopInfer DAG."
        ),
    )
    parser.add_argument("--source-period-ms", type=float, default=100.0)
    parser.add_argument("--pipeline-unroll", type=int, default=1)
    parser.add_argument("--batch-transfers", action="store_true")
    args = parser.parse_args()

    profile = load_vla_perf_profile(args.profile)
    spec = graph_spec_from_profile(profile)
    payload = build_pi05_coopinfer_payload(
        profile,
        bandwidth_mb_s=args.bandwidth,
        latency_ms=args.latency,
        ae_placement=args.ae_placement,
        source_period_ms=args.source_period_ms,
        pipeline_unroll=args.pipeline_unroll,
        batch_transfers=args.batch_transfers,
    )
    _fix_separate_token_id_edges(payload, spec)
    write_coopinfer_payload(payload, args.output)

    sizes = tensor_sizes_mb(spec, profile["precision"])
    print(f"Wrote CoopInfer graph: {args.output}")
    print(
        f"nodes={len(payload['nodes'])}, edges={len(payload['edges'])}, "
        f"prefix_tokens={spec.prefix_tokens}"
    )
    print(
        f"KV/layer={sizes['kv_per_layer']:.4f} MB, "
        f"VLM hidden={sizes['vlm_hidden']:.4f} MB, "
        f"AE hidden={sizes['ae_hidden']:.4f} MB"
    )


if __name__ == "__main__":
    main()
