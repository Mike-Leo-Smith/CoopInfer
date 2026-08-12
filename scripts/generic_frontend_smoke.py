from __future__ import annotations

import json
from pathlib import Path

from coopinfer.cost import CallableCostBackend, CostTarget, annotate_costs
from coopinfer.frontend import DependencyAwarePolicy, capture_model, to_coopinfer_payload


def main() -> None:
    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Install PyTorch to run the generic frontend smoke demo.") from exc

    class TinyBranch(torch.nn.Module):
        def forward(self, x):
            shared = x * 2
            left = shared + 1
            right = shared - 1
            return left + right

    model_ir = capture_model(TinyBranch().eval(), args=(torch.ones(2, 4),))

    backend = CallableCostBackend(
        lambda node, hardware: 0.05 if hardware == "A100_80GB" else 0.10,
        source="synthetic-smoke",
    )
    costed = annotate_costs(
        model_ir,
        {
            "device": CostTarget("RTX_4090", backend),
            "host": CostTarget("A100_80GB", backend),
        },
    )
    scheduling_ir = DependencyAwarePolicy(max_ops_per_group=8).apply(costed)
    payload = to_coopinfer_payload(
        scheduling_ir,
        bandwidth_mb_s=1250.0,
        latency_ms=0.2,
    )

    output = Path("results/generic_frontend_smoke.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"captured_ops={len(model_ir.nodes)} "
        f"schedule_nodes={len(scheduling_ir.nodes)} "
        f"schedule_edges={len(scheduling_ir.edges)}"
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
