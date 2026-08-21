from __future__ import annotations

import csv
import json
from pathlib import Path


MODELS = ("pi0", "pi05", "smolvla", "molmoact2", "eo1")
HARDWARE = ("dual_rtx4090", "a100_rtx4090")
NETWORKS = ("normal", "fast")


def main() -> None:
    source = Path("results/wavekv_validation/summary.json")
    output_prefix = Path("results/wavekv_validation/unified_single_dual_comparison")
    payload = json.loads(source.read_text(encoding="utf-8"))
    by_key = {
        (
            row["model"], row["hardware"], row["network"],
            row["mode"], row["pipeline_unroll"],
        ): row
        for row in payload["rows"]
    }
    rows = []
    for unroll in (1, 2):
        for model in MODELS:
            for hardware in HARDWARE:
                for network in NETWORKS:
                    fix = by_key[(model, hardware, network, "fix", unroll)]
                    free = by_key[(model, hardware, network, "free", unroll)]
                    gpu0_ms = float(free["single_gpu0_host_latency_ms"])
                    gpu1_ms = float(free["single_gpu1_device_latency_ms"])
                    fix_ms = float(fix["latency_ms"])
                    free_ms = float(free["latency_ms"])
                    rows.append(
                        {
                            "pipeline_unroll": unroll,
                            "model": model,
                            "hardware": hardware,
                            "network": network,
                            "best_single_latency_ms": free["best_single_latency_ms"],
                            "best_single_resource": free["best_single_resource"],
                            "single_gpu0_host_latency_ms": gpu0_ms,
                            "single_gpu1_device_latency_ms": gpu1_ms,
                            "fix_latency_ms": fix_ms,
                            "fix_speedup_vs_single_pct": fix["speedup_vs_best_single_pct"],
                            "fix_speedup_vs_gpu0_host_pct": (gpu0_ms - fix_ms) / gpu0_ms * 100.0,
                            "fix_speedup_vs_gpu1_device_pct": (gpu1_ms - fix_ms) / gpu1_ms * 100.0,
                            "fix_transfers": fix["num_transfers"],
                            "free_latency_ms": free_ms,
                            "free_speedup_vs_single_pct": free["speedup_vs_best_single_pct"],
                            "free_speedup_vs_gpu0_host_pct": (gpu0_ms - free_ms) / gpu0_ms * 100.0,
                            "free_speedup_vs_gpu1_device_pct": (gpu1_ms - free_ms) / gpu1_ms * 100.0,
                            "free_transfers": free["num_transfers"],
                        }
                    )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = output_prefix.with_suffix(".json")
    csv_path = output_prefix.with_suffix(".csv")
    json_path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "models": list(MODELS),
                "comparison": (
                    "u1 and u2 results versus the matching-unroll best single resource "
                    "from the free configuration"
                ),
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
