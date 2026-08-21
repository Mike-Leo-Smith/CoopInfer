from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from types import MethodType


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
            "Export the complete GR00T N1.7 inference architecture: Qwen3-VL "
            "vision/language backbone plus one structural flow-matching step; "
            "Stage 3 materializes all four inference steps."
        )
    )
    parser.add_argument(
        "--lerobot-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "lerobot-main",
    )
    parser.add_argument("--text-tokens", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/models/groot_n17/groot_n17_fine_ir.json"),
    )
    args = parser.parse_args()
    if args.text_tokens < 1:
        raise ValueError("--text-tokens must be positive")

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _add_paths(args.lerobot_root)

    import torch
    from torch import nn
    from transformers.feature_extraction_utils import BatchFeature

    from coopinfer.frontend import capture_model, write_model_ir_json
    from lerobot.policies.groot.groot_n1_7 import GR00TN17, GR00TN17Config

    torch.manual_seed(0)
    config = GR00TN17Config(
        load_bf16=True,
        backbone_trainable_params_fp32=False,
        use_flash_attention=False,
    )
    core = GR00TN17(config, load_backbone_weights=False).eval().to(dtype=torch.bfloat16)
    core.backbone.model.config._attn_implementation = "eager"
    core.backbone.language_model.config._attn_implementation = "eager"

    class FullInferenceWrapper(nn.Module):
        def __init__(self, model: GR00TN17):
            super().__init__()
            self.core = model

        def forward(
            self,
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            mm_token_type_ids,
            state,
            embodiment_id,
            noisy_action,
            timestep,
        ):
            backbone_output = self.core.backbone(
                BatchFeature(
                    data={
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "pixel_values": pixel_values,
                        "image_grid_thw": image_grid_thw,
                        "mm_token_type_ids": mm_token_type_ids,
                    }
                )
            )
            head = self.core.action_head
            backbone_output = head.process_backbone_output(backbone_output)
            state_features = head.state_encoder(
                state.view(state.shape[0], 1, -1), embodiment_id
            )
            action_features = head.action_encoder(noisy_action, timestep, embodiment_id)
            pos_ids = torch.arange(
                action_features.shape[1], dtype=torch.long, device=action_features.device
            )
            action_features = action_features + head.position_embedding(pos_ids).unsqueeze(0)
            hidden_states = torch.cat((state_features, action_features), dim=1)
            model_output = head.model(
                hidden_states=hidden_states,
                encoder_hidden_states=backbone_output.backbone_features,
                timestep=timestep,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
            return head.action_decoder(model_output, embodiment_id)[:, -noisy_action.shape[1] :]

    # One 256x256 image becomes a 16x16 patch grid. Spatial 2x2 merging
    # produces 64 image placeholders in the language sequence.
    grid_h = grid_w = 16
    raw_patch_count = grid_h * grid_w
    image_token_count = raw_patch_count // 4
    seq_len = image_token_count + args.text_tokens
    image_token_id = int(core.backbone.model.config.image_token_id)
    input_ids = torch.zeros((1, seq_len), dtype=torch.long)
    input_ids[:, :image_token_count] = image_token_id
    attention_mask = torch.ones_like(input_ids)
    mm_token_type_ids = torch.zeros_like(input_ids, dtype=torch.int32)
    mm_token_type_ids[:, :image_token_count] = 1
    pixel_values = torch.rand(
        raw_patch_count,
        3 * 2 * 16 * 16,
        dtype=torch.bfloat16,
    )
    image_grid_thw = torch.tensor([[1, grid_h, grid_w]], dtype=torch.long)
    state = torch.zeros(
        1,
        int(config.state_history_length),
        int(config.max_state_dim),
        dtype=torch.bfloat16,
    )
    embodiment_id = torch.zeros(1, dtype=torch.long)
    noisy_action = torch.randn(
        1,
        int(config.action_horizon),
        int(config.max_action_dim),
        dtype=torch.bfloat16,
    )
    timestep = torch.zeros(1, dtype=torch.long)

    # Transformers' Qwen3-VL position helpers use Python ``.tolist()`` on the
    # fixed image grid/token types, which torch.export intentionally rejects
    # for FakeTensor inputs.  Specialize those parameter-free helpers for this
    # declared 256x256 workload while retaining every learned vision/language
    # layer in the exported graph.
    visual = core.backbone.visual
    visual.register_buffer(
        "_coopinfer_export_pos_embeds",
        visual.fast_pos_embed_interpolate(image_grid_thw).detach(),
        persistent=False,
    )
    visual.register_buffer(
        "_coopinfer_export_rotary_pos_embeds",
        visual.rot_pos_emb(image_grid_thw).detach(),
        persistent=False,
    )

    def fixed_visual_pos(self, _grid_thw):
        return self._coopinfer_export_pos_embeds

    def fixed_visual_rotary(self, _grid_thw):
        return self._coopinfer_export_rotary_pos_embeds

    visual.fast_pos_embed_interpolate = MethodType(fixed_visual_pos, visual)
    visual.rot_pos_emb = MethodType(fixed_visual_rotary, visual)

    def single_image_vision_attention(
        self,
        hidden_states,
        cu_seqlens,
        rotary_pos_emb=None,
        position_embeddings=None,
        **_kwargs,
    ):
        # This workload contains exactly one image, hence cu_seqlens describes
        # one attention chunk. Avoid the upstream ``lengths.tolist()`` split
        # while performing the same eager attention over that single chunk.
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            apply_rotary_pos_emb_vision,
        )

        seq_length = hidden_states.shape[0]
        query, key, value = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query, key = apply_rotary_pos_emb_vision(query, key, cos, sin)
        query = query.transpose(0, 1).unsqueeze(0)
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)
        weights = torch.matmul(query, key.transpose(2, 3)) * self.scaling
        weights = torch.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        output = torch.matmul(weights, value).transpose(1, 2).contiguous()
        return self.proj(output.reshape(seq_length, -1).contiguous())

    for block in visual.blocks:
        block.attn.forward = MethodType(single_image_vision_attention, block.attn)

    core.backbone.register_buffer(
        "_coopinfer_export_position_ids",
        torch.arange(seq_len, dtype=torch.long).view(1, 1, seq_len).expand(3, 1, seq_len),
        persistent=False,
    )

    def fixed_language_position_ids(self, model_input):
        model_input["position_ids"] = self._coopinfer_export_position_ids

    core.backbone._ensure_legacy_qwen3_position_ids = MethodType(  # noqa: SLF001
        fixed_language_position_ids, core.backbone
    )

    parameter_count = sum(parameter.numel() for parameter in core.parameters())
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in core.parameters())
    print("===== Stage 1: GR00T N1.7 full inference export =====")
    print(f"torch={torch.__version__}")
    print(f"lerobot_root={args.lerobot_root.resolve()}")
    print("config_source=native-LeRobot-GR00TN17Config")
    print("pretrained_weights=False")
    print("dtype=bfloat16")
    print(f"parameters={parameter_count}")
    print(f"parameter_bytes={parameter_bytes}")
    print("image_resolution=256x256")
    print(f"language_sequence_length={seq_len}")
    print(f"backbone_language_layers={len(core.backbone.language_model.layers)}")
    print(f"backbone_vision_layers={len(core.backbone.visual.blocks)}")
    print(f"action_layers={config.diffusion_model_cfg['num_layers']}")
    print(f"num_inference_steps={config.num_inference_timesteps}")

    model_ir = capture_model(
        FullInferenceWrapper(core).eval(),
        args=(
            input_ids,
            attention_mask,
            pixel_values,
            image_grid_thw,
            mm_token_type_ids,
            state,
            embodiment_id,
            noisy_action,
            timestep,
        ),
        strict=False,
    )
    model_ir.metadata.update(
        {
            "model_family": "groot_n17",
            "model_name": "nvidia/GR00T-N1.7-3B",
            "backbone_name": str(config.model_name),
            "capture_scope": "full_backbone_plus_one_flow_step",
            "random_init": True,
            "pretrained_weights": False,
            "config_source": "native-LeRobot-GR00TN17Config",
            "dtype": "bfloat16",
            "parameter_count": int(parameter_count),
            "parameter_bytes": int(parameter_bytes),
            "num_images": 1,
            "image_resolution": [256, 256],
            "language_sequence_length": int(seq_len),
            "max_state_dim": int(config.max_state_dim),
            "max_action_dim": int(config.max_action_dim),
            "action_horizon": int(config.action_horizon),
            "chunk_size": int(config.action_horizon),
            "num_inference_steps": int(config.num_inference_timesteps),
            "captured_denoise_steps": 1,
            "iterative_stack_root_hint": "core.action_head.model.transformer_blocks",
            "backbone_language_layers": len(core.backbone.language_model.layers),
            "backbone_vision_layers": len(core.backbone.visual.blocks),
            "action_layers": int(config.diffusion_model_cfg["num_layers"]),
            "flow_matching": True,
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_model_ir_json(model_ir, args.output)
    print(f"fine_nodes={len(model_ir.nodes)}")
    print(f"fine_edges={len(model_ir.edges)}")
    print(f"model_ir={args.output}")


if __name__ == "__main__":
    main()
