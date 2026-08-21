#!/usr/bin/env python3
"""Aggregate the completed dual-RTX-4090 experiment matrix."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


MODELS = ("smolvla", "pi0", "pi05", "pi0_fast", "xvla", "groot_n17")
MODES = ("free", "split")
UNROLLS = (1, 2, 3, 4)


def _row(path: Path, data: dict) -> dict:
    summary = data["summary"]
    baseline = data["baselines"]["all_device"]
    return {
        "model": path.parent.name,
        "mode": "split" if "_split_" in path.name else "free",
        "pipeline_unroll": int(summary["pipeline_unroll"]),
        "latency_ms": float(summary["latency_ms"]),
        "max_frame_latency_ms": float(summary["max_frame_latency_ms"]),
        "initiation_interval_ms": float(summary["initiation_interval_ms"]),
        "all_device_latency_ms": float(baseline["latency_ms"]),
        "speedup_vs_single_gpu": float(baseline["latency_ms"]) / float(summary["latency_ms"]),
        "device_utilization": float(summary["device_utilization"]),
        "host_utilization": float(summary["host_utilization"]),
        "network_utilization": float(summary["network_utilization"]),
        "num_transfers": int(summary["num_transfers"]),
        "num_nodes": int(summary["num_nodes"]),
        "result": path.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.repo_root.resolve()
    result_root = root / "results" / "dual_rtx4090"

    rows = []
    missing = []
    for model in MODELS:
        for mode in MODES:
            for unroll in UNROLLS:
                path = result_root / model / f"{model}_dual4090_{mode}_u{unroll}_solve.json"
                if not path.exists():
                    missing.append(path.relative_to(root).as_posix())
                    continue
                rows.append(_row(path.relative_to(root), json.loads(path.read_text(encoding="utf-8"))))

    payload = {
        "experiment": "dual_rtx4090_pipeline_matrix_v1",
        "environment": {
            "gpu0": "RTX_4090 (Host label)",
            "gpu1": "RTX_4090 (Device label)",
            "bandwidth_mb_s": 20000.0,
            "latency_ms": 0.01,
            "source_release_policy": "saturated_zero_period",
        },
        "methodology": {
            "split": "exact evaluation of fixed Vision/VLM@GPU0 + Action/Decode@GPU1",
            "free": "300-iteration seeded random search; grouped autoregressive placement is verified on the full token DAG",
        },
        "expected_results": len(MODELS) * len(MODES) * len(UNROLLS),
        "completed_results": len(rows),
        "missing": missing,
        "rows": rows,
    }
    (result_root / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (result_root / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    for model in MODELS:
        model_rows = [row for row in rows if row["model"] == model]
        (result_root / model / "summary.json").write_text(
            json.dumps({"model": model, "rows": model_rows}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"completed={len(rows)} missing={len(missing)}")


if __name__ == "__main__":
    main()
