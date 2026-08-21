from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def _add_repo_src_to_path() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _add_lerobot_src_to_path(lerobot_root: Path) -> None:
    root = lerobot_root.resolve()
    for candidate in (root / "src", root):
        if (candidate / "lerobot").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return
    raise FileNotFoundError(f"LeRobot package not found under {root}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-1 pi0 torch.export adapter. Captures image vision frontend, "
            "language prefix, prefix KV prefill, continuous state suffix, and one "
            "Action Expert denoise step."
        )
    )
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "lerobot-main",
    )
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--num-images", type=int, default=3)
    parser.add_argument("--lang-len", type=int, default=48)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/models/pi0/pi0_fine_ir.json"),
    )
    args = parser.parse_args()

    if args.num_images < 1 or args.lang_len < 1 or args.batch_size < 1:
        raise ValueError("num-images, lang-len, and batch-size must be positive")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    _add_repo_src_to_path()
    _add_lerobot_src_to_path(args.lerobot_root)

    import torch
    from torch import nn

    from coopinfer.frontend import capture_model, write_model_ir_json
    from lerobot.policies.common.vla_utils import make_att_2d_masks, prepare_attention_masks_4d
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.policies.pi0.modeling_pi0 import PI0Pytorch

    torch.manual_seed(0)
    config = PI0Config()
    config.device = "cpu"
    config.compile_model = False
    config.dtype = args.dtype
    model = PI0Pytorch(config).eval()

    model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

    class InferenceOneStepWrapper(nn.Module):
        def __init__(self, core: PI0Pytorch):
            super().__init__()
            self.core = core

        def forward(
            self,
            images,
            img_masks,
            tokens,
            token_masks,
            state,
            x_t,
            timestep,
        ):
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.core.embed_prefix(
                images, img_masks, tokens, token_masks
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            prefix_att_2d_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)
            _, past_key_values = self.core.paligemma_with_expert.forward(
                attention_mask=prefix_att_2d_masks_4d,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=True,
            )
            return self.core.denoise_step(
                state=state,
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=timestep,
            )

    batch = args.batch_size
    image_h, image_w = map(int, config.image_resolution)
    images = tuple(
        torch.rand(batch, 3, image_h, image_w, dtype=torch.float32)
        for _ in range(args.num_images)
    )
    img_masks = tuple(torch.ones(batch, dtype=torch.bool) for _ in range(args.num_images))
    tokens = torch.zeros(batch, args.lang_len, dtype=torch.long)
    token_masks = torch.ones(batch, args.lang_len, dtype=torch.bool)
    state = torch.zeros(batch, int(config.max_state_dim), dtype=torch.float32)
    x_t = torch.randn(
        batch,
        int(config.chunk_size),
        int(config.max_action_dim),
        dtype=torch.float32,
    )
    timestep = torch.full((batch,), 0.5, dtype=torch.float32)

    print("===== Stage 1: pi0 torch.export =====")
    print(f"torch={torch.__version__}")
    print(f"lerobot_root={args.lerobot_root.resolve()}")
    print("config_source=native-LeRobot-PI0Config")
    print("pretrained_weights=False")
    print("capture_scope=inference_one_step")
    print(f"dtype={args.dtype}")
    print(f"num_images={args.num_images}")
    print(f"image_resolution={image_h}x{image_w}")
    print(f"lang_len={args.lang_len}")
    print(f"chunk_size={config.chunk_size}")
    print(f"num_inference_steps={config.num_inference_steps}")
    print(f"paligemma_variant={config.paligemma_variant}")
    print(f"action_expert_variant={config.action_expert_variant}")

    model_ir = capture_model(
        InferenceOneStepWrapper(model).eval(),
        args=(images, img_masks, tokens, token_masks, state, x_t, timestep),
        strict=False,
    )
    model_ir.metadata.update(
        {
            "model_family": "pi0",
            "capture_scope": "inference_one_step",
            "random_init": True,
            "pretrained_weights": False,
            "config_source": "native-LeRobot-PI0Config",
            "num_images": int(args.num_images),
            "image_resolution": [image_h, image_w],
            "lang_len": int(args.lang_len),
            "chunk_size": int(config.chunk_size),
            "max_state_dim": int(config.max_state_dim),
            "max_action_dim": int(config.max_action_dim),
            "num_inference_steps": int(config.num_inference_steps),
            "captured_denoise_steps": 1,
            "persistent_context_kind": "per_layer_prefix_kv",
            "kv_reuse_across_denoise_steps": True,
            "wave_kv_first_step": True,
            "state_conditioning": "continuous_suffix_projection",
            "paligemma_variant": str(config.paligemma_variant),
            "action_expert_variant": str(config.action_expert_variant),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_model_ir_json(model_ir, args.output)
    print(f"fine_nodes={len(model_ir.nodes)}")
    print(f"fine_edges={len(model_ir.edges)}")
    print(f"model_ir={args.output}")


if __name__ == "__main__":
    main()
