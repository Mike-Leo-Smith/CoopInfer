from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def _add_repo_src_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src = repo_root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _add_lerobot_src_to_path(lerobot_root: Path) -> None:
    src = lerobot_root.resolve() / "src"
    if not src.exists():
        raise FileNotFoundError(f"LeRobot src directory not found: {src}")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-1 SmolVLA torch.export adapter. Captures the native model tensor-input "
            "inference path: image vision frontend + language/state prefix + prefix KV prefill + "
            "one Action Expert denoise step. It emits Fine ModelIR only."
        )
    )
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        default=Path(r"D:\Project\lerobot-main"),
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path(r"D:\Project\models\SmolVLM2-500M-metadata"),
        help="Local SmolVLM config/tokenizer/processor directory; checkpoint weights are not loaded.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=3,
        help="Workload input: number of camera images supplied to embed_prefix.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/smolvla_full_pipeline/smolvla_fine_ir.json"),
    )
    args = parser.parse_args()

    if args.num_images < 1:
        raise ValueError("--num-images must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if not args.metadata_dir.exists():
        raise FileNotFoundError(f"metadata directory not found: {args.metadata_dir}")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    _add_repo_src_to_path()
    _add_lerobot_src_to_path(args.lerobot_root)

    import torch
    from torch import nn

    from coopinfer.frontend import capture_model, write_model_ir_json
    from lerobot.policies.common.vla_utils import make_att_2d_masks
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

    torch.manual_seed(0)

    # Use the native LeRobot SmolVLA architecture/configuration. Only the local
    # VLM metadata path and execution environment are supplied by this adapter.
    config = SmolVLAConfig(
        device="cpu",
        vlm_model_name=str(args.metadata_dir.resolve()),
        load_vlm_weights=False,
        use_cache=True,
        compile_model=False,
    )

    core = VLAFlowMatching(config).eval()
    image_h, image_w = (int(config.resize_imgs_with_padding[0]), int(config.resize_imgs_with_padding[1]))
    lang_len = int(config.tokenizer_max_length)
    chunk_size = int(config.chunk_size)
    state_dim = int(config.max_state_dim)
    action_dim = int(config.max_action_dim)

    # Match the real inference structure while capturing one denoise execution.
    class InferenceOneStepWrapper(nn.Module):
        def __init__(self, model: VLAFlowMatching):
            super().__init__()
            self.model = model

        def forward(
            self,
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            x_t,
            timestep,
        ):
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state=state,
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            _, past_key_values = self.model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            return self.model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=timestep,
            )

    batch = args.batch_size
    images = tuple(
        torch.rand(batch, 3, image_h, image_w, dtype=torch.float32)
        for _ in range(args.num_images)
    )
    img_masks = tuple(torch.ones(batch, dtype=torch.bool) for _ in range(args.num_images))
    lang_tokens = torch.zeros(batch, lang_len, dtype=torch.long)
    lang_masks = torch.ones(batch, lang_len, dtype=torch.bool)
    state = torch.randn(batch, state_dim, dtype=torch.float32)
    x_t = torch.randn(batch, chunk_size, action_dim, dtype=torch.float32)
    timestep = torch.full((batch,), 0.5, dtype=torch.float32)

    print("===== Stage 1: SmolVLA torch.export =====")
    print(f"torch={torch.__version__}")
    print(f"lerobot_root={args.lerobot_root.resolve()}")
    print(f"metadata_dir={args.metadata_dir.resolve()}")
    print("pretrained_weights=False")
    print("capture_scope=inference_one_step")
    print(f"actual_vlm_layers={core.vlm_with_expert.num_vlm_layers}")
    print(f"actual_expert_layers={core.vlm_with_expert.num_expert_layers}")
    print(f"vlm_hidden_size={core.vlm_with_expert.config.text_config.hidden_size}")
    print(f"expert_hidden_size={core.vlm_with_expert.expert_hidden_size}")
    print(f"num_images={args.num_images}")
    print(f"image_resolution={image_h}x{image_w}")
    print(f"lang_len={lang_len}")
    print(f"chunk_size={chunk_size}")
    print(f"num_inference_steps={config.num_steps}")

    wrapper = InferenceOneStepWrapper(core).eval()
    model_ir = capture_model(
        wrapper,
        args=(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            x_t,
            timestep,
        ),
        strict=False,
    )
    model_ir.metadata.update(
        {
            "model_family": "smolvla",
            "capture_scope": "inference_one_step",
            "random_init": True,
            "pretrained_weights": False,
            "metadata_dir": str(args.metadata_dir.resolve()),
            "num_images": int(args.num_images),
            "image_resolution": [image_h, image_w],
            "lang_len": lang_len,
            "chunk_size": chunk_size,
            "max_state_dim": state_dim,
            "max_action_dim": action_dim,
            "num_inference_steps": int(config.num_steps),
            "captured_denoise_steps": 1,
            "persistent_context_kind": "per_layer_prefix_kv",
            "kv_reuse_across_denoise_steps": True,
            "wave_kv_first_step": True,
            "vlm_layers": int(core.vlm_with_expert.num_vlm_layers),
            "expert_layers": int(core.vlm_with_expert.num_expert_layers),
        }
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_model_ir_json(model_ir, args.output)

    print(f"fine_nodes={len(model_ir.nodes)}")
    print(f"fine_edges={len(model_ir.edges)}")
    print(f"model_ir={args.output}")
    print("next_stage=python scripts/model_ir_layer_analysis.py <model_ir> --precision bf16")


if __name__ == "__main__":
    main()
