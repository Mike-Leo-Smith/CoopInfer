from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def _add_paths(lerobot_root: Path) -> None:
    repo_src = Path(__file__).resolve().parents[1] / "src"
    if str(repo_src) not in sys.path:
        sys.path.insert(0, str(repo_src))
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
            "Stage-1 pi0-FAST export: one multimodal prefill and one cached "
            "autoregressive action-token decode step."
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
        "--max-action-tokens",
        type=int,
        default=64,
        help="Total action-token budget recorded for Stage-3 expansion (default: 64).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/models/pi0_fast/pi0_fast_fine_ir.json"),
    )
    args = parser.parse_args()
    if args.num_images < 1 or args.lang_len < 1 or args.batch_size < 1:
        raise ValueError("num-images, lang-len, and batch-size must be positive")
    if args.max_action_tokens < 2:
        raise ValueError("max-action-tokens must be at least 2")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _add_paths(args.lerobot_root)

    import torch
    from torch import nn

    from coopinfer.frontend import capture_model, write_model_ir_json
    from lerobot.policies.common.vla_utils import prepare_attention_masks_4d
    from lerobot.policies.pi0_fast.configuration_pi0_fast import PI0FastConfig
    from lerobot.policies.pi0_fast.modeling_pi0_fast import PI0FastPytorch

    torch.manual_seed(0)
    config = PI0FastConfig()
    native_max_action_tokens = int(config.max_action_tokens)
    if args.max_action_tokens > native_max_action_tokens:
        raise ValueError(
            f"max-action-tokens={args.max_action_tokens} exceeds model limit "
            f"{native_max_action_tokens}"
        )
    config.device = "cpu"
    config.compile_model = False
    config.dtype = args.dtype
    core = PI0FastPytorch(config, paligemma_tokenizer=None).eval()
    core.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = "eager"  # noqa: SLF001

    class PrefillPhase(nn.Module):
        def __init__(self, model: PI0FastPytorch):
            super().__init__()
            self.core = model

        def forward(self, images, img_masks, tokens_with_bos, masks_with_bos):
            embs, pad_masks, att_masks, _, _ = self.core.embed_prefix_fast(
                images,
                img_masks,
                tokens_with_bos,
                masks_with_bos,
                fast_action_tokens=None,
                fast_action_masks=None,
            )
            if self.core.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
                embs = embs.to(torch.bfloat16)
            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            att_4d = prepare_attention_masks_4d(att_masks, dtype=embs.dtype)
            (output, _), cache = self.core.paligemma_with_expert.forward(
                attention_mask=att_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[embs, None],
                use_cache=True,
                adarms_cond=[None, None],
            )
            logits = self.core.paligemma_with_expert.paligemma.lm_head(output[:, -1:, :])
            return logits, cache, pad_masks

    class DecodePhase(nn.Module):
        def __init__(self, model: PI0FastPytorch):
            super().__init__()
            self.core = model

        def forward(self, token, cache, prefix_pad_masks):
            token_emb = self.core.paligemma_with_expert.embed_language_tokens(token)
            if self.core.paligemma_with_expert.paligemma.model.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
                token_emb = token_emb.to(torch.bfloat16)
            current_pad_mask = torch.cat(
                [
                    prefix_pad_masks,
                    torch.ones((token.shape[0], 1), dtype=torch.bool, device=token.device),
                ],
                dim=1,
            )
            position_ids = (torch.sum(current_pad_mask, dim=1, keepdim=True) - 1).long()
            att_4d = prepare_attention_masks_4d(
                current_pad_mask.unsqueeze(1), dtype=token_emb.dtype
            )
            (output, _), _ = self.core.paligemma_with_expert.forward(
                attention_mask=att_4d,
                position_ids=position_ids,
                past_key_values=cache,
                inputs_embeds=[token_emb, None],
                use_cache=True,
                adarms_cond=[None, None],
            )
            return self.core.paligemma_with_expert.paligemma.lm_head(output[:, -1:, :])

    class PrefillDecodeWrapper(nn.Module):
        def __init__(self, model: PI0FastPytorch):
            super().__init__()
            # Intentional shared parameters under two call-site paths. torch.export
            # preserves prefill.* and decode.* module stacks while lifting one set
            # of aliased parameters.
            self.prefill = PrefillPhase(model)
            self.decode = DecodePhase(model)

        def forward(self, images, img_masks, tokens_with_bos, masks_with_bos, decode_token):
            prefill_logits, cache, pad_masks = self.prefill(
                images, img_masks, tokens_with_bos, masks_with_bos
            )
            decode_logits = self.decode(decode_token, cache, pad_masks)
            return prefill_logits, decode_logits

    batch = args.batch_size
    image_h, image_w = map(int, config.image_resolution)
    images = tuple(
        torch.rand(batch, 3, image_h, image_w, dtype=torch.float32)
        for _ in range(args.num_images)
    )
    img_masks = tuple(torch.ones(batch, dtype=torch.bool) for _ in range(args.num_images))
    # Tokenizer is intentionally outside the neural DAG. The final input token
    # occupies the BOS position; its numeric value does not affect graph shape.
    tokens_with_bos = torch.zeros(batch, args.lang_len + 1, dtype=torch.long)
    masks_with_bos = torch.ones(batch, args.lang_len + 1, dtype=torch.bool)
    decode_token = torch.zeros(batch, 1, dtype=torch.long)

    print("===== Stage 1: pi0-FAST prefill + cached decode export =====")
    print(f"torch={torch.__version__}")
    print(f"lerobot_root={args.lerobot_root.resolve()}")
    print("config_source=native-LeRobot-PI0FastConfig")
    print("pretrained_weights=False")
    print("capture_scope=prefill_plus_one_cached_decode")
    print(f"dtype={args.dtype}")
    print(f"num_images={args.num_images}")
    print(f"image_resolution={image_h}x{image_w}")
    print(f"lang_len={args.lang_len}")
    print(f"max_action_tokens={args.max_action_tokens}")
    print(f"native_max_action_tokens={native_max_action_tokens}")

    model_ir = capture_model(
        PrefillDecodeWrapper(core).eval(),
        args=(images, img_masks, tokens_with_bos, masks_with_bos, decode_token),
        strict=False,
    )
    model_ir.metadata.update(
        {
            "model_family": "pi0_fast",
            "capture_scope": "prefill_plus_one_cached_decode",
            "random_init": True,
            "pretrained_weights": False,
            "config_source": "native-LeRobot-PI0FastConfig",
            "num_images": int(args.num_images),
            "image_resolution": [image_h, image_w],
            "lang_len": int(args.lang_len),
            "chunk_size": int(config.chunk_size),
            "max_action_dim": int(config.max_action_dim),
            "native_max_action_tokens": native_max_action_tokens,
            "max_action_tokens": int(args.max_action_tokens),
            "num_decode_steps": int(args.max_action_tokens - 1),
            "captured_decode_steps": 1,
            "iterative_execution_kind": "autoregressive_decode",
            "decode_token_bytes": 8,
            "decode_kv_increment_bytes_per_layer": 1024,
            "kv_cache": True,
            "paligemma_variant": str(config.paligemma_variant),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_model_ir_json(model_ir, args.output)
    print(f"fine_nodes={len(model_ir.nodes)}")
    print(f"fine_edges={len(model_ir.edges)}")
    print(f"model_ir={args.output}")


if __name__ == "__main__":
    main()
