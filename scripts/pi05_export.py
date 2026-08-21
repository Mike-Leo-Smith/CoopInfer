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
    root = lerobot_root.resolve()
    for candidate in (root / "src", root):
        if (candidate / "lerobot").is_dir():
            if str(candidate) not in sys.path:
                sys.path.insert(0, str(candidate))
            return
    raise FileNotFoundError(
        f"LeRobot package not found under {root}; expected <root>/src/lerobot or <root>/lerobot"
    )


def _load_config(config_path: Path | None, dtype: str):
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    if config_path is None:
        config = PI05Config()
        source = "native-LeRobot-PI05Config"
    else:
        config = PI05Config.from_pretrained(
            str(config_path.resolve()),
            local_files_only=True,
        )
        source = str(config_path.resolve())

    config.device = "cpu"
    config.compile_model = False
    config.dtype = dtype
    return config, source


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage-1 pi0.5 torch.export adapter. Captures the real model tensor-input "
            "inference path: image vision frontend + language prefix + prefix KV prefill + "
            "one Action Expert denoise step. It emits Fine ModelIR only."
        )
    )
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "lerobot-main",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help=(
            "Optional local pi0.5 config/checkpoint directory. Only configuration is read; "
            "weights are intentionally random for structural export. Omit to use native "
            "LeRobot PI05Config defaults."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16"),
        default="float32",
        help="Model construction dtype. float32 is the safest CPU export default.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=3,
        help="Workload input: number of camera images supplied to embed_prefix.",
    )
    parser.add_argument(
        "--lang-len",
        type=int,
        default=None,
        help="Workload input token length. Omit to use config.tokenizer_max_length.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/models/pi05/pi05_fine_ir.json"),
    )
    args = parser.parse_args()

    if args.num_images < 1:
        raise ValueError("--num-images must be >= 1")
    if args.lang_len is not None and args.lang_len < 1:
        raise ValueError("--lang-len must be >= 1 when provided")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.config_path is not None and not args.config_path.exists():
        raise FileNotFoundError(f"config path not found: {args.config_path}")

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
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch

    torch.manual_seed(0)
    config, config_source = _load_config(args.config_path, args.dtype)
    model = PI05Pytorch(config).eval()

    model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001
    model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

    lang_len = int(config.tokenizer_max_length if args.lang_len is None else args.lang_len)
    image_h, image_w = (int(config.image_resolution[0]), int(config.image_resolution[1]))

    class InferenceOneStepWrapper(nn.Module):
        def __init__(self, core: PI05Pytorch):
            super().__init__()
            self.core = core

        def forward(
            self,
            images,
            img_masks,
            tokens,
            token_masks,
            states,
            state_masks,
            x_t,
            timestep,
        ):
            prefix_states = states if self.core.config.use_proprioceptive_memory else None
            prefix_state_masks = (
                state_masks if self.core.config.use_proprioceptive_memory else None
            )
            prefix_embs, prefix_pad_masks, prefix_att_masks = self.core.embed_prefix(
                images,
                img_masks,
                tokens,
                token_masks,
                prefix_states,
                prefix_state_masks,
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
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=timestep,
            )

    batch = args.batch_size
    if config.use_visual_memory:
        frames = int(config.memory_frames)
        images = tuple(
            torch.rand(batch, frames, 3, image_h, image_w, dtype=torch.float32)
            for _ in range(args.num_images)
        )
        img_masks = tuple(
            torch.ones(batch, frames, dtype=torch.bool) for _ in range(args.num_images)
        )
    else:
        images = tuple(
            torch.rand(batch, 3, image_h, image_w, dtype=torch.float32)
            for _ in range(args.num_images)
        )
        img_masks = tuple(torch.ones(batch, dtype=torch.bool) for _ in range(args.num_images))

    tokens = torch.zeros(batch, lang_len, dtype=torch.long)
    token_masks = torch.ones(batch, lang_len, dtype=torch.bool)

    state_frames = int(config.memory_frames) if config.use_proprioceptive_memory else 1
    states = torch.zeros(
        batch,
        state_frames,
        int(config.max_state_dim),
        dtype=torch.float32,
    )
    state_masks = torch.ones(batch, state_frames, dtype=torch.bool)

    x_t = torch.randn(
        batch,
        int(config.chunk_size),
        int(config.max_action_dim),
        dtype=torch.float32,
    )
    timestep = torch.full((batch,), 0.5, dtype=torch.float32)

    print("===== Stage 1: pi0.5 torch.export =====")
    print(f"torch={torch.__version__}")
    print(f"lerobot_root={args.lerobot_root.resolve()}")
    print(f"config_source={config_source}")
    print("pretrained_weights=False")
    print("capture_scope=inference_one_step")
    print(f"num_images={args.num_images}")
    print(f"image_resolution={image_h}x{image_w}")
    print(f"lang_len={lang_len}")
    print(f"chunk_size={config.chunk_size}")
    print(f"num_inference_steps={config.num_inference_steps}")
    print(f"use_visual_memory={config.use_visual_memory}")
    print(f"use_proprioceptive_memory={config.use_proprioceptive_memory}")
    print(f"paligemma_variant={config.paligemma_variant}")
    print(f"action_expert_variant={config.action_expert_variant}")

    wrapper = InferenceOneStepWrapper(model).eval()
    model_ir = capture_model(
        wrapper,
        args=(
            images,
            img_masks,
            tokens,
            token_masks,
            states,
            state_masks,
            x_t,
            timestep,
        ),
        strict=False,
    )
    model_ir.metadata.update(
        {
            "model_family": "pi05",
            "capture_scope": "inference_one_step",
            "random_init": True,
            "pretrained_weights": False,
            "config_source": config_source,
            "num_images": int(args.num_images),
            "image_resolution": [image_h, image_w],
            "lang_len": lang_len,
            "chunk_size": int(config.chunk_size),
            "max_state_dim": int(config.max_state_dim),
            "max_action_dim": int(config.max_action_dim),
            "num_inference_steps": int(config.num_inference_steps),
            "captured_denoise_steps": 1,
            "persistent_context_kind": "per_layer_prefix_kv",
            "kv_reuse_across_denoise_steps": True,
            "wave_kv_first_step": True,
            "use_visual_memory": bool(config.use_visual_memory),
            "use_proprioceptive_memory": bool(config.use_proprioceptive_memory),
            "paligemma_variant": str(config.paligemma_variant),
            "action_expert_variant": str(config.action_expert_variant),
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
