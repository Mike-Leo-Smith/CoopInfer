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
            "Export official XVLA-0.9B architecture using the checkpoint config "
            "without downloading checkpoint weights."
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
        default=Path("examples/models/xvla/source_config"),
    )
    parser.add_argument("--text-tokens", type=int, default=32)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/models/xvla/xvla_fine_ir.json"),
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

    from coopinfer.frontend import capture_model, write_model_ir_json
    from lerobot.policies.xvla.configuration_xvla import XVLAConfig
    from lerobot.policies.xvla.modeling_xvla import XVLAModel

    torch.manual_seed(0)
    config = XVLAConfig.from_pretrained(
        str(args.config_path.resolve()),
        local_files_only=True,
    )
    config.device = "cpu"
    config.dtype = "bfloat16"
    florence_config = config.get_florence_config()
    core = XVLAModel(
        config=config,
        florence_config=florence_config,
        proprio_dim=int(config.max_state_dim),
    ).eval()

    class OneFlowStepWrapper(nn.Module):
        def __init__(self, model: XVLAModel):
            super().__init__()
            self.core = model

        def forward(
            self,
            input_ids,
            image_input,
            image_mask,
            domain_id,
            proprio,
            noisy_action,
            timestep,
        ):
            # The declared validation workload has all three camera views
            # present. Specializing that invariant avoids XVLA's Python
            # ``int(mask.sum().item())`` branch, which cannot consume an
            # export FakeTensor, while preserving the exact all-valid path.
            batch_size, num_views = image_input.shape[:2]
            flat_images = image_input.flatten(0, 1)
            valid_features = self.core.vlm.get_image_features(flat_images).pooler_output
            tokens_per_view, hidden_dim = valid_features.shape[1:]
            image_features = valid_features.view(
                batch_size, num_views, tokens_per_view, hidden_dim
            )
            inputs_embeds = self.core.vlm.get_input_embeddings()(input_ids)
            merged_embeds = torch.cat([image_features[:, 0], inputs_embeds], dim=1)
            attention_mask = torch.ones(
                merged_embeds.shape[:2], dtype=torch.long, device=merged_embeds.device
            )
            vlm_features = self.core.vlm.language_model.encoder(
                attention_mask=attention_mask,
                inputs_embeds=merged_embeds,
            )[0]
            aux_visual_inputs = image_features[:, 1:].reshape(
                batch_size, -1, hidden_dim
            )
            proprio_model, action_model = self.core.action_space.preprocess(
                proprio, noisy_action
            )
            return self.core.transformer(
                domain_id=domain_id,
                action_with_noise=action_model,
                proprio=proprio_model,
                t=timestep,
                vlm_features=vlm_features,
                aux_visual_inputs=aux_visual_inputs,
            )

    views = int(config.num_image_views or 1)
    resize = config.resize_imgs_with_padding or (224, 224)
    image_h, image_w = int(resize[0]), int(resize[1])
    input_ids = torch.zeros((1, args.text_tokens), dtype=torch.long)
    image_input = torch.rand(
        1, views, 3, image_h, image_w, dtype=torch.bfloat16
    )
    image_mask = torch.ones((1, views), dtype=torch.bool)
    domain_id = torch.zeros(1, dtype=torch.long)
    proprio = torch.zeros(1, int(config.max_state_dim), dtype=torch.bfloat16)
    noisy_action = torch.randn(
        1, int(config.chunk_size), int(core.dim_action), dtype=torch.bfloat16
    )
    timestep = torch.ones(1, dtype=torch.bfloat16)

    parameter_count = sum(parameter.numel() for parameter in core.parameters())
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in core.parameters())
    print("===== Stage 1: XVLA-0.9B architecture export =====")
    print(f"torch={torch.__version__}")
    print(f"lerobot_root={args.lerobot_root.resolve()}")
    print(f"config_source={args.config_path.resolve()}")
    print("pretrained_weights=False")
    print("dtype=bfloat16")
    print(f"parameters={parameter_count}")
    print(f"parameter_bytes={parameter_bytes}")
    print(f"num_image_views={views}")
    print(f"image_resolution={image_h}x{image_w}")
    print(f"text_tokens={args.text_tokens}")
    print(f"florence_language_layers={florence_config.text_config.encoder_layers}")
    print(f"policy_transformer_layers={config.depth}")
    print(f"num_denoising_steps={config.num_denoising_steps}")

    model_ir = capture_model(
        OneFlowStepWrapper(core).eval(),
        args=(
            input_ids,
            image_input,
            image_mask,
            domain_id,
            proprio,
            noisy_action,
            timestep,
        ),
        strict=False,
    )
    model_ir.metadata.update(
        {
            "model_family": "xvla",
            "model_name": "lerobot/xvla-base",
            # cat_6 also contains per-step action embeddings and local learned
            # prompts. Only the 182-token VLM context crosses the FM boundary:
            # 182 * 1024 * 2 bytes (bf16).
            "static_conditioning_tensor_bytes": {"cat_6": 372736},
            "capture_scope": "full_florence_encoders_plus_one_flow_step",
            "random_init": True,
            "pretrained_weights": False,
            "config_source": str(args.config_path.resolve()),
            "dtype": "bfloat16",
            "parameter_count": int(parameter_count),
            "parameter_bytes": int(parameter_bytes),
            "num_images": views,
            "image_resolution": [image_h, image_w],
            "language_sequence_length": int(args.text_tokens),
            "chunk_size": int(config.chunk_size),
            "max_state_dim": int(config.max_state_dim),
            "max_action_dim": int(core.dim_action),
            "num_inference_steps": int(config.num_denoising_steps),
            "captured_denoise_steps": 1,
            "iterative_stack_root_hint": "core.transformer.blocks",
            "florence_language_layers": int(florence_config.text_config.encoder_layers),
            "policy_transformer_layers": int(config.depth),
            "flow_matching": True,
            "soft_prompts": True,
            "num_domains": int(config.num_domains),
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_model_ir_json(model_ir, args.output)
    print(f"fine_nodes={len(model_ir.nodes)}")
    print(f"fine_edges={len(model_ir.edges)}")
    print(f"model_ir={args.output}")


if __name__ == "__main__":
    main()
