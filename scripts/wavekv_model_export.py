from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys


def _add_repo_src_to_path() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


@dataclass(frozen=True)
class Architecture:
    model_family: str
    source_model: str
    vision_layers: int
    vision_hidden: int
    vision_heads: int
    vision_intermediate: int
    vision_tokens: int
    prefix_layers: int
    prefix_hidden: int
    prefix_heads: int
    prefix_kv_heads: int
    prefix_head_dim: int
    prefix_intermediate: int
    prefix_tokens: int
    action_hidden: int
    action_heads: int
    action_intermediate: int
    action_layers: int
    action_tokens: int
    action_dim: int
    denoise_steps: int
    action_stack_name: str
    shared_backbone_weights: bool


ARCHITECTURES = {
    "molmoact2": Architecture(
        model_family="molmoact2",
        source_model="allenai/MolmoAct2",
        vision_layers=27,
        vision_hidden=1152,
        vision_heads=16,
        vision_intermediate=4304,
        vision_tokens=729,
        prefix_layers=36,
        prefix_hidden=2560,
        prefix_heads=32,
        prefix_kv_heads=8,
        prefix_head_dim=128,
        prefix_intermediate=9728,
        prefix_tokens=832,
        action_hidden=768,
        action_heads=8,
        action_intermediate=3072,
        action_layers=36,
        action_tokens=30,
        action_dim=32,
        denoise_steps=10,
        action_stack_name="action_expert_layers",
        shared_backbone_weights=False,
    ),
    "eo1": Architecture(
        model_family="eo1",
        source_model="Qwen/Qwen2.5-VL-3B-Instruct",
        vision_layers=32,
        vision_hidden=1280,
        vision_heads=16,
        vision_intermediate=3420,
        vision_tokens=128,
        prefix_layers=36,
        prefix_hidden=2048,
        prefix_heads=16,
        prefix_kv_heads=2,
        prefix_head_dim=128,
        prefix_intermediate=11008,
        prefix_tokens=192,
        action_hidden=2048,
        action_heads=16,
        action_intermediate=11008,
        action_layers=36,
        action_tokens=8,
        action_dim=32,
        denoise_steps=10,
        action_stack_name="action_suffix_layers",
        shared_backbone_weights=True,
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Export a weight-free, full-width Wave-KV execution graph using the released "
            "MolmoAct2 or EO-1 architecture dimensions. Parameters and inputs live on "
            "the PyTorch meta device; no checkpoint weights are downloaded."
        )
    )
    parser.add_argument("model", choices=tuple(ARCHITECTURES))
    parser.add_argument("--prefix-tokens", type=int, default=None)
    parser.add_argument("--vision-tokens", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    _add_repo_src_to_path()

    import torch
    from torch import nn
    import torch.nn.functional as F

    from coopinfer.frontend import capture_model, write_model_ir_json

    base = ARCHITECTURES[args.model]
    arch = Architecture(
        **{
            **base.__dict__,
            "prefix_tokens": args.prefix_tokens or base.prefix_tokens,
            "vision_tokens": args.vision_tokens or base.vision_tokens,
        }
    )
    if arch.prefix_tokens < 1 or arch.vision_tokens < 1:
        raise ValueError("token counts must be positive")

    class VisionLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(arch.vision_hidden)
            self.qkv = nn.Linear(arch.vision_hidden, 3 * arch.vision_hidden)
            self.out = nn.Linear(arch.vision_hidden, arch.vision_hidden)
            self.norm2 = nn.LayerNorm(arch.vision_hidden)
            self.up = nn.Linear(arch.vision_hidden, arch.vision_intermediate)
            self.down = nn.Linear(arch.vision_intermediate, arch.vision_hidden)

        def forward(self, x):
            residual = x
            q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
            head_dim = arch.vision_hidden // arch.vision_heads
            shape = (x.shape[0], x.shape[1], arch.vision_heads, head_dim)
            q = q.view(shape).transpose(1, 2)
            k = k.view(shape).transpose(1, 2)
            v = v.view(shape).transpose(1, 2)
            y = F.scaled_dot_product_attention(q, k, v)
            y = y.transpose(1, 2).reshape(x.shape[0], x.shape[1], arch.vision_hidden)
            x = residual + self.out(y)
            return x + self.down(F.silu(self.up(self.norm2(x))))

    class PrefixLayer(nn.Module):
        def __init__(self):
            super().__init__()
            q_dim = arch.prefix_heads * arch.prefix_head_dim
            kv_dim = arch.prefix_kv_heads * arch.prefix_head_dim
            self.norm1 = nn.LayerNorm(arch.prefix_hidden)
            self.q = nn.Linear(arch.prefix_hidden, q_dim, bias=False)
            self.k = nn.Linear(arch.prefix_hidden, kv_dim, bias=False)
            self.v = nn.Linear(arch.prefix_hidden, kv_dim, bias=False)
            self.out = nn.Linear(q_dim, arch.prefix_hidden, bias=False)
            self.norm2 = nn.LayerNorm(arch.prefix_hidden)
            self.gate = nn.Linear(arch.prefix_hidden, arch.prefix_intermediate, bias=False)
            self.up = nn.Linear(arch.prefix_hidden, arch.prefix_intermediate, bias=False)
            self.down = nn.Linear(arch.prefix_intermediate, arch.prefix_hidden, bias=False)

        def forward(self, x):
            residual = x
            z = self.norm1(x)
            q = self.q(z).view(x.shape[0], x.shape[1], arch.prefix_heads, arch.prefix_head_dim)
            k = self.k(z).view(x.shape[0], x.shape[1], arch.prefix_kv_heads, arch.prefix_head_dim)
            v = self.v(z).view(x.shape[0], x.shape[1], arch.prefix_kv_heads, arch.prefix_head_dim)
            repeat = arch.prefix_heads // arch.prefix_kv_heads
            k_attn = k.repeat_interleave(repeat, dim=2)
            v_attn = v.repeat_interleave(repeat, dim=2)
            y = F.scaled_dot_product_attention(
                q.transpose(1, 2), k_attn.transpose(1, 2), v_attn.transpose(1, 2)
            )
            y = y.transpose(1, 2).reshape(x.shape[0], x.shape[1], -1)
            x = residual + self.out(y)
            z = self.norm2(x)
            return x + self.down(F.silu(self.gate(z)) * self.up(z)), k, v

    class ActionLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm1 = nn.LayerNorm(arch.action_hidden)
            self.self_qkv = nn.Linear(arch.action_hidden, 3 * arch.action_hidden, bias=False)
            self.self_out = nn.Linear(arch.action_hidden, arch.action_hidden, bias=False)
            self.norm2 = nn.LayerNorm(arch.action_hidden)
            self.cross_q = nn.Linear(arch.action_hidden, arch.action_hidden, bias=False)
            kv_dim = arch.prefix_kv_heads * arch.prefix_head_dim
            self.cross_k = nn.Linear(kv_dim, arch.action_hidden, bias=False)
            self.cross_v = nn.Linear(kv_dim, arch.action_hidden, bias=False)
            self.cross_out = nn.Linear(arch.action_hidden, arch.action_hidden, bias=False)
            self.norm3 = nn.LayerNorm(arch.action_hidden)
            self.gate = nn.Linear(arch.action_hidden, arch.action_intermediate, bias=False)
            self.up = nn.Linear(arch.action_hidden, arch.action_intermediate, bias=False)
            self.down = nn.Linear(arch.action_intermediate, arch.action_hidden, bias=False)

        def _heads(self, value):
            head_dim = arch.action_hidden // arch.action_heads
            return value.view(value.shape[0], value.shape[1], arch.action_heads, head_dim).transpose(1, 2)

        def forward(self, x, prefix_k, prefix_v):
            residual = x
            q, k, v = self.self_qkv(self.norm1(x)).chunk(3, dim=-1)
            y = F.scaled_dot_product_attention(self._heads(q), self._heads(k), self._heads(v))
            y = y.transpose(1, 2).reshape_as(x)
            x = residual + self.self_out(y)
            residual = x
            q = self._heads(self.cross_q(self.norm2(x)))
            k = self._heads(self.cross_k(prefix_k.flatten(2)))
            v = self._heads(self.cross_v(prefix_v.flatten(2)))
            y = F.scaled_dot_product_attention(q, k, v)
            y = y.transpose(1, 2).reshape_as(x)
            x = residual + self.cross_out(y)
            z = self.norm3(x)
            return x + self.down(F.silu(self.gate(z)) * self.up(z))

    class WaveKVModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.vision_layers = nn.ModuleList([VisionLayer() for _ in range(arch.vision_layers)])
            self.vision_to_prefix = nn.Linear(arch.vision_hidden, arch.prefix_hidden)
            self.prefix_layers = nn.ModuleList([PrefixLayer() for _ in range(arch.prefix_layers)])
            self.action_in = nn.Linear(arch.action_dim, arch.action_hidden)
            action_layers = nn.ModuleList([ActionLayer() for _ in range(arch.action_layers)])
            setattr(self, arch.action_stack_name, action_layers)
            self.action_out = nn.Linear(arch.action_hidden, arch.action_dim)

        def forward(self, vision, prefix, noisy_action):
            for layer in self.vision_layers:
                vision = layer(vision)
            vision_summary = self.vision_to_prefix(vision.mean(dim=1, keepdim=True))
            prefix = prefix + vision_summary
            action = self.action_in(noisy_action)
            action_layers = getattr(self, arch.action_stack_name)
            # The FX DAG expresses the Wave-KV frontier: action layer i depends on
            # action layer i-1 and the K/V produced by prefix layer i. Python loop
            # order does not add a false dependency on later prefix layers.
            for prefix_layer, action_layer in zip(self.prefix_layers, action_layers, strict=True):
                prefix, key, value = prefix_layer(prefix)
                action = action_layer(action, key, value)
            return self.action_out(action)

    print(f"===== Stage 1: {arch.model_family} Wave-KV structural export =====")
    print(f"source_model={arch.source_model}")
    print("checkpoint_weights=False")
    print("parameter_device=meta")
    print(f"vision_layers={arch.vision_layers}")
    print(f"prefix_layers={arch.prefix_layers}")
    print(f"action_layers={arch.action_layers}")
    print(f"prefix_tokens={arch.prefix_tokens}")
    print(f"action_tokens={arch.action_tokens}")
    print(f"num_inference_steps={arch.denoise_steps}")

    with torch.device("meta"):
        model = WaveKVModel().eval()
        vision = torch.empty(1, arch.vision_tokens, arch.vision_hidden)
        prefix = torch.empty(1, arch.prefix_tokens, arch.prefix_hidden)
        noisy_action = torch.empty(1, arch.action_tokens, arch.action_dim)
    # Importing torch._dynamo while the global default device context is meta can
    # make Dynamo's own scalar bootstrap tensors meta tensors. Trace only after
    # leaving that context; model parameters and example inputs remain on meta.
    model_ir = capture_model(model, args=(vision, prefix, noisy_action), strict=False)

    # detect_layer_groups normalizes ModuleList names to these exact roots.
    action_root = arch.action_stack_name
    model_ir.metadata.update(
        {
            "model_family": arch.model_family,
            "source_model": arch.source_model,
            "capture_scope": "full_width_structural_wavekv_one_denoise_step",
            "capture_fidelity": "released_dimensions_weight_free_meta_graph",
            "pretrained_weights": False,
            "random_init": False,
            "vision_layers": arch.vision_layers,
            "vlm_layers": arch.prefix_layers,
            "action_layers": arch.action_layers,
            "vision_tokens": arch.vision_tokens,
            "prefix_tokens": arch.prefix_tokens,
            "chunk_size": arch.action_tokens,
            "max_action_dim": arch.action_dim,
            "num_inference_steps": arch.denoise_steps,
            "captured_denoise_steps": 1,
            "iterative_execution_kind": "denoise",
            "iterative_stack_root_hint": action_root,
            "persistent_context_kind": "per_layer_prefix_kv",
            "kv_reuse_across_denoise_steps": True,
            "wave_kv_first_step": True,
            "shared_backbone_weights_between_prefix_and_action": arch.shared_backbone_weights,
            "workload_note": "token counts are explicit export workload assumptions",
        }
    )
    output = args.output or Path(f"examples/models/{arch.model_family}/{arch.model_family}_fine_ir.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    write_model_ir_json(model_ir, output)
    print(f"fine_nodes={len(model_ir.nodes)}")
    print(f"fine_edges={len(model_ir.edges)}")
    print(f"model_ir={output}")


if __name__ == "__main__":
    main()
