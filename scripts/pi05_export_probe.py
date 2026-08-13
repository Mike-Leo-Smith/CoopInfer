from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback


def _add_lerobot_to_path(root: Path) -> None:
    root = root.resolve()
    for candidate in (root, root / "src"):
        if (candidate / "lerobot").is_dir():
            sys.path.insert(0, str(candidate))
            return
    raise SystemExit(
        f"{root} does not contain a LeRobot package; expected <root>/src/lerobot or <root>/lerobot."
    )


def _load_pi05(args):
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy, PI05Pytorch

    if args.model_path is None:
        config = PI05Config(dtype=args.dtype, device="cpu", compile_model=False)
        model = PI05Pytorch(config)
        source = "architecture-only"
    else:
        model_path = str(args.model_path.resolve())
        config = PI05Config.from_pretrained(
            model_path,
            local_files_only=args.local_files_only,
        )
        config.device = "cpu"
        config.compile_model = False
        config.dtype = args.dtype
        policy = PI05Policy.from_pretrained(
            model_path,
            config=config,
            local_files_only=args.local_files_only,
        )
        model = policy.model
        source = model_path

    model.eval()
    return model, source


def _make_wrappers(model, prefix_tokens: int):
    import torch

    from coopinfer.frontend.pi05_probe import flatten_past_key_values

    vlm = model.paligemma_with_expert.paligemma.model.language_model
    layer0 = vlm.layers[0]

    class QKVWrapper(torch.nn.Module):
        def __init__(self, layer):
            super().__init__()
            self.layer = layer

        def forward(self, hidden_states):
            attn = self.layer.self_attn
            return (
                attn.q_proj(hidden_states),
                attn.k_proj(hidden_states),
                attn.v_proj(hidden_states),
            )

    class PrefixWrapper(torch.nn.Module):
        def __init__(self, pi05):
            super().__init__()
            self.pi05 = pi05

        def forward(self, prefix_embs, attention_mask, position_ids):
            _, cache = self.pi05.paligemma_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            return flatten_past_key_values(cache)

    class PrefixOneStepWrapper(torch.nn.Module):
        def __init__(self, pi05):
            super().__init__()
            self.pi05 = pi05

        def forward(
            self,
            prefix_embs,
            prefix_pad_masks,
            attention_mask,
            position_ids,
            x_t,
            timestep,
        ):
            _, cache = self.pi05.paligemma_with_expert.forward(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            return self.pi05.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=cache,
                x_t=x_t,
                timestep=timestep,
            )

    weight_dtype = layer0.self_attn.q_proj.weight.dtype
    hidden_size = int(layer0.self_attn.q_proj.in_features)
    batch = 1
    hidden = torch.zeros(batch, prefix_tokens, hidden_size, dtype=weight_dtype)
    position_ids = torch.arange(prefix_tokens, dtype=torch.long)[None, :]
    prefix_pad_masks = torch.ones(batch, prefix_tokens, dtype=torch.bool)
    attention_mask = torch.zeros(
        batch,
        1,
        prefix_tokens,
        prefix_tokens,
        dtype=weight_dtype,
    )
    x_t = torch.zeros(
        batch,
        int(model.config.chunk_size),
        int(model.config.max_action_dim),
        dtype=torch.float32,
    )
    timestep = torch.full((batch,), 0.5, dtype=torch.float32)

    return {
        "qkv": (QKVWrapper(layer0).eval(), (hidden,)),
        "prefix": (
            PrefixWrapper(model).eval(),
            (hidden, attention_mask, position_ids),
        ),
        "prefix_ae": (
            PrefixOneStepWrapper(model).eval(),
            (
                hidden,
                prefix_pad_masks,
                attention_mask,
                position_ids,
                x_t,
                timestep,
            ),
        ),
    }


def _run_probe(name, module, probe_args, output_dir: Path) -> dict:
    from coopinfer.frontend import capture_model
    from coopinfer.frontend.pi05_probe import (
        format_model_ir,
        model_ir_to_dict,
        summarize_pi05_ir,
    )

    record = {"probe": name, "status": "failed"}
    try:
        model_ir = capture_model(module, args=tuple(probe_args), strict=False)
        summary = summarize_pi05_ir(model_ir)
        record.update({"status": "ok", "summary": summary.to_dict()})
        (output_dir / f"{name}_ir.json").write_text(
            json.dumps(model_ir_to_dict(model_ir), indent=2) + "\n",
            encoding="utf-8",
        )
        (output_dir / f"{name}_graph.txt").write_text(
            format_model_ir(model_ir), encoding="utf-8"
        )
    except Exception as exc:
        record.update(
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Probe the real LeRobot pi0.5 PyTorch graph on CPU with torch.export. "
            "This reports graph visibility only and does not benchmark latency."
        )
    )
    parser.add_argument("--lerobot-root", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional local/HF pi0.5 checkpoint; omit for architecture-only random weights",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="float32 is the safest CPU default; bfloat16 reduces host RAM",
    )
    parser.add_argument("--prefix-tokens", type=int, default=832)
    parser.add_argument(
        "--probe", choices=("qkv", "prefix", "prefix_ae", "all"), default="all"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/pi05_export_probe")
    )
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    if args.prefix_tokens <= 0:
        raise SystemExit("--prefix-tokens must be positive")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    _add_lerobot_to_path(args.lerobot_root)

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("PyTorch is required for the pi0.5 export probe") from exc

    print(f"torch={torch.__version__}")
    print("device=cpu")
    print(f"prefix_tokens={args.prefix_tokens}")

    model, model_source = _load_pi05(args)
    wrappers = _make_wrappers(model, args.prefix_tokens)
    names = ("qkv", "prefix", "prefix_ae") if args.probe == "all" else (args.probe,)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name in names:
        print(f"\n===== {name} =====")
        module, probe_args = wrappers[name]
        record = _run_probe(name, module, probe_args, args.output_dir)
        results.append(record)
        if record["status"] == "ok":
            summary = record["summary"]
            print(
                f"nodes={summary['node_count']} edges={summary['edge_count']} "
                f"q={len(summary['q_proj_nodes'])} "
                f"k={len(summary['k_proj_nodes'])} "
                f"v={len(summary['v_proj_nodes'])}"
            )
            print(f"wave_kv_candidate_layers={summary['wave_kv_candidate_layers']}")
        else:
            print(f"FAILED: {record['error_type']}: {record['error']}")

    report = {
        "model_source": model_source,
        "torch_version": torch.__version__,
        "device": "cpu",
        "prefix_tokens": args.prefix_tokens,
        "results": results,
        "interpretation": {
            "qkv": "Can torch.export see the real layer-0 Q/K/V projections?",
            "prefix": "Does the prefix path expose per-layer cache tensors as graph outputs?",
            "prefix_ae": (
                "Can graph reachability prove VLM K/V -> Action Expert dependencies "
                "without manually adding Wave-KV edges?"
            ),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {report_path}")

    if any(item["status"] != "ok" for item in results):
        print(
            "Some probes failed. Inspect report.json to decide whether the next step "
            "is decomposition, cache flattening, or a thinner export wrapper."
        )


if __name__ == "__main__":
    main()
