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


def _default_output(repo_root: Path, mode: str, vlm_layers: int, expert_layers: int) -> Path:
    return (
        repo_root
        / "results"
        / "smolvla_export_probe"
        / f"smolvla_{mode}_v{vlm_layers}_e{expert_layers}_ir.json"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Random-init SmolVLA torch.export probe. No pretrained checkpoint is loaded. "
            "The preferred mode captures prefix computation plus one inference denoise step."
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
        help="Local SmolVLM config/tokenizer/processor directory; no checkpoint weights required.",
    )
    parser.add_argument(
        "--mode",
        choices=("inference_one_step", "joint_forward"),
        default="inference_one_step",
        help=(
            "inference_one_step: prefix KV build + one cached AE denoise pass; "
            "joint_forward: one training-style prefix+suffix pass without cache, useful as a fallback."
        ),
    )
    parser.add_argument("--vlm-layers", type=int, default=16)
    parser.add_argument("--expert-layers", type=int, default=16)
    parser.add_argument("--num-images", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--lang-len", type=int, default=48)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--state-dim", type=int, default=32)
    parser.add_argument("--action-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if args.vlm_layers < 1 or args.expert_layers < 1:
        raise ValueError("--vlm-layers and --expert-layers must be positive")
    if args.vlm_layers % args.expert_layers != 0:
        raise ValueError(
            "SmolVLA requires VLM layer count to be divisible by expert layer count "
            f"when --expert-layers is explicit: {args.vlm_layers} % {args.expert_layers} != 0"
        )
    if args.num_images < 1:
        raise ValueError("--num-images must be >= 1")
    if not args.metadata_dir.exists():
        raise FileNotFoundError(f"metadata directory not found: {args.metadata_dir}")

    # Make this probe deterministic and offline. The metadata path is local and
    # load_vlm_weights=False, so no pretrained checkpoint is requested.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    repo_root = Path(__file__).resolve().parents[1]
    _add_repo_src_to_path()
    _add_lerobot_src_to_path(args.lerobot_root)

    import torch
    from torch import nn

    from coopinfer.frontend import (
        capture_model,
        detect_layer_groups,
        discover_layer_dependencies,
        layer_dependency_summary,
        write_model_ir_json,
    )
    from lerobot.policies.common.vla_utils import make_att_2d_masks
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching

    torch.manual_seed(0)

    config = SmolVLAConfig(
        device="cpu",
        vlm_model_name=str(args.metadata_dir.resolve()),
        load_vlm_weights=False,
        num_vlm_layers=args.vlm_layers,
        num_expert_layers=args.expert_layers,
        chunk_size=args.chunk_size,
        n_action_steps=args.chunk_size,
        tokenizer_max_length=args.lang_len,
        max_state_dim=args.state_dim,
        max_action_dim=args.action_dim,
        resize_imgs_with_padding=(args.image_size, args.image_size),
        use_cache=True,
        compile_model=False,
    )

    print("===== SmolVLA random-init construction =====")
    print(f"torch={torch.__version__}")
    print(f"lerobot_root={args.lerobot_root.resolve()}")
    print(f"metadata_dir={args.metadata_dir.resolve()}")
    print("pretrained_weights=False")
    print(f"mode={args.mode}")
    print(f"requested_vlm_layers={args.vlm_layers}")
    print(f"requested_expert_layers={args.expert_layers}")

    core = VLAFlowMatching(config)
    core.eval()

    actual_vlm_layers = core.vlm_with_expert.num_vlm_layers
    actual_expert_layers = core.vlm_with_expert.num_expert_layers
    print(f"actual_vlm_layers={actual_vlm_layers}")
    print(f"actual_expert_layers={actual_expert_layers}")
    print(f"vlm_hidden_size={core.vlm_with_expert.config.text_config.hidden_size}")
    print(f"expert_hidden_size={core.vlm_with_expert.expert_hidden_size}")

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

    class JointForwardWrapper(nn.Module):
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
            actions,
            noise,
            timestep,
        ):
            return self.model(
                images,
                img_masks,
                lang_tokens,
                lang_masks,
                state,
                actions,
                noise,
                timestep,
            )

    batch = args.batch_size
    images = tuple(
        torch.rand(batch, 3, args.image_size, args.image_size, dtype=torch.float32)
        for _ in range(args.num_images)
    )
    img_masks = tuple(torch.ones(batch, dtype=torch.bool) for _ in range(args.num_images))
    lang_tokens = torch.zeros(batch, args.lang_len, dtype=torch.long)
    lang_masks = torch.ones(batch, args.lang_len, dtype=torch.bool)
    state = torch.randn(batch, args.state_dim, dtype=torch.float32)
    x_t = torch.randn(batch, args.chunk_size, args.action_dim, dtype=torch.float32)
    timestep = torch.full((batch,), 0.5, dtype=torch.float32)

    if args.mode == "inference_one_step":
        wrapper = InferenceOneStepWrapper(core).eval()
        export_args = (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            x_t,
            timestep,
        )
    else:
        wrapper = JointForwardWrapper(core).eval()
        actions = torch.randn(batch, args.chunk_size, args.action_dim, dtype=torch.float32)
        noise = torch.randn_like(actions)
        export_args = (
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            timestep,
        )

    print("===== torch.export capture =====")
    model_ir = capture_model(wrapper, args=export_args, strict=False)
    model_ir.metadata.update(
        {
            "model_family": "smolvla",
            "capture_scope": args.mode,
            "random_init": True,
            "pretrained_weights": False,
            "metadata_dir": str(args.metadata_dir.resolve()),
            "num_images": args.num_images,
            "image_size": args.image_size,
            "lang_len": args.lang_len,
            "chunk_size": args.chunk_size,
            "state_dim": args.state_dim,
            "action_dim": args.action_dim,
            "vlm_layers": actual_vlm_layers,
            "expert_layers": actual_expert_layers,
        }
    )

    output = args.output or _default_output(
        repo_root,
        args.mode,
        actual_vlm_layers,
        actual_expert_layers,
    )
    write_model_ir_json(model_ir, output)

    grouping = detect_layer_groups(model_ir)
    dependencies = discover_layer_dependencies(model_ir, grouping)
    summary = layer_dependency_summary(dependencies)

    layer_groups = [group for group in grouping.groups.values() if group.kind == "layer"]
    print(f"fine_nodes={len(model_ir.nodes)}")
    print(f"fine_edges={len(model_ir.edges)}")
    print(f"detected_layer_stacks={len(grouping.stack_layers)}")
    for index, (root, layers) in enumerate(grouping.stack_layers.items()):
        print(f"  stack[{index}] root={root} layers={list(layers)} count={len(layers)}")
    print(f"detected_layer_nodes={len(layer_groups)}")
    print(f"layer_frontier_dependencies={summary['layer_frontier_dependencies']}")
    print(f"cross_stack_dependencies={summary['cross_stack_dependencies']}")
    print(f"same_index_cross_stack_dependencies={summary['same_index_cross_stack_dependencies']}")

    cross = [dependency for dependency in dependencies if dependency.cross_stack]
    if cross:
        print("cross_stack_edges:")
        for dependency in cross:
            print(
                "  "
                f"{dependency.source_layer} -> {dependency.target_layer} "
                f"glue_hops={dependency.glue_hops} "
                f"{dependency.source_stack} -> {dependency.target_stack}"
            )
    else:
        print("cross_stack_edges: none")

    print(f"model_ir={output}")
    print("genz=not_run")
    print("solver=not_run")


if __name__ == "__main__":
    main()
