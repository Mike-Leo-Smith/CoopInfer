from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_MODELS = ("pi0", "pi05", "smolvla")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Wave-KV u1/u2 results against the matching best single-GPU baseline."
    )
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/wavekv_validation/summary.json"),
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("results/wavekv_validation/pi_smolvla_single_gpu_comparison"),
    )
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    source_rows = [row for row in payload["rows"] if row["model"] in args.models]
    by_key = {
        (
            row["model"],
            row["hardware"],
            row["network"],
            row["mode"],
            row["pipeline_unroll"],
        ): row
        for row in source_rows
    }
    rows = []
    for model in args.models:
        for hardware in ("dual_rtx4090", "a100_rtx4090"):
            for network in ("normal", "fast"):
                for mode in ("fix", "free"):
                    u1 = by_key[(model, hardware, network, mode, 1)]
                    u2 = by_key[(model, hardware, network, mode, 2)]
                    rows.append(
                        {
                            "model": model,
                            "hardware": hardware,
                            "network": network,
                            "mode": mode,
                            "u1_latency_ms": u1["latency_ms"],
                            "u2_avg_latency_ms": u2["latency_ms"],
                            "u2_max_frame_latency_ms": u2["max_frame_latency_ms"],
                            "best_single_u2_latency_ms": u2["best_single_latency_ms"],
                            "best_single_resource": u2["best_single_resource"],
                            "speedup_vs_best_single_pct": u2["speedup_vs_best_single_pct"],
                            "u2_transfers": u2["num_transfers"],
                            "result": u2["result"],
                            "config": u2["config"],
                        }
                    )

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = args.output_prefix.with_suffix(".json")
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "models": list(args.models),
                "comparison": "same unroll=2 result versus matching free-config best single resource",
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"rows={len(rows)}")
    print(f"json={json_path}")
    print(f"csv={csv_path}")


if __name__ == "__main__":
    main()
