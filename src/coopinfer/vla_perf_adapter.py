from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


PROFILE_SCHEMA_VERSION = 1


def bytes_per_element(precision: str) -> float:
    """Return storage bytes per scalar for the modeled precision."""
    key = precision.strip().lower()
    mapping = {
        "fp32": 4.0,
        "tf32": 4.0,
        "bf16": 2.0,
        "fp16": 2.0,
        "fp8": 1.0,
        "int8": 1.0,
        "fp4": 0.5,
        "int4": 0.5,
        "int2": 0.25,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported precision for tensor sizing: {precision!r}")
    return mapping[key]


def mb(num_bytes: float) -> float:
    return float(num_bytes) / (1024.0 * 1024.0)


@dataclass(frozen=True)
class Pi05GraphSpec:
    vision_layers: int = 27
    vlm_layers: int = 18
    ae_layers: int = 18
    vision_tokens_per_frame: int = 256
    vision_frames: int = 3
    language_tokens: int = 32
    state_tokens: int = 32
    action_chunk_size: int = 50
    denoising_steps: int = 10
    action_dof: int = 14
    image_resolution: int = 384
    vision_hidden_size: int = 1152
    vlm_hidden_size: int = 2048
    ae_hidden_size: int = 1024
    vlm_num_kv_heads: int = 1
    vlm_head_dim: int = 256

    @property
    def prefix_tokens(self) -> int:
        return (
            self.vision_tokens_per_frame * self.vision_frames
            + self.language_tokens
            + self.state_tokens
        )


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object.")
    return value


def load_vla_perf_profile(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_vla_perf_profile(payload)
    return payload


def validate_vla_perf_profile(profile: Mapping[str, Any]) -> None:
    version = int(profile.get("schema_version", -1))
    if version != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported VLA-Perf profile schema {version}; "
            f"expected {PROFILE_SCHEMA_VERSION}."
        )
    model = _require_mapping(profile.get("model"), "profile.model")
    hardware = _require_mapping(profile.get("hardware"), "profile.hardware")
    costs = _require_mapping(profile.get("costs_ms"), "profile.costs_ms")
    for key in ("device", "host"):
        if key not in hardware:
            raise ValueError(f"profile.hardware is missing {key!r}.")
        side = _require_mapping(costs.get(key), f"profile.costs_ms.{key}")
        for cost_key in ("vision_total", "vlm_total", "ae_pass_total", "projector"):
            value = float(side.get(cost_key, -1.0))
            if value < 0:
                raise ValueError(
                    f"profile.costs_ms.{key}.{cost_key} must be non-negative."
                )
    for key in (
        "vision_layers",
        "vlm_layers",
        "ae_layers",
        "vision_tokens_per_frame",
        "vision_frames",
        "language_tokens",
        "state_tokens",
        "action_chunk_size",
        "denoising_steps",
        "action_dof",
        "image_resolution",
        "vision_hidden_size",
        "vlm_hidden_size",
        "ae_hidden_size",
        "vlm_num_kv_heads",
        "vlm_head_dim",
    ):
        if int(model.get(key, 0)) <= 0:
            raise ValueError(f"profile.model.{key} must be positive.")


def graph_spec_from_profile(profile: Mapping[str, Any]) -> Pi05GraphSpec:
    validate_vla_perf_profile(profile)
    model = profile["model"]
    return Pi05GraphSpec(
        vision_layers=int(model["vision_layers"]),
        vlm_layers=int(model["vlm_layers"]),
        ae_layers=int(model["ae_layers"]),
        vision_tokens_per_frame=int(model["vision_tokens_per_frame"]),
        vision_frames=int(model["vision_frames"]),
        language_tokens=int(model["language_tokens"]),
        state_tokens=int(model["state_tokens"]),
        action_chunk_size=int(model["action_chunk_size"]),
        denoising_steps=int(model["denoising_steps"]),
        action_dof=int(model["action_dof"]),
        image_resolution=int(model["image_resolution"]),
        vision_hidden_size=int(model["vision_hidden_size"]),
        vlm_hidden_size=int(model["vlm_hidden_size"]),
        ae_hidden_size=int(model["ae_hidden_size"]),
        vlm_num_kv_heads=int(model["vlm_num_kv_heads"]),
        vlm_head_dim=int(model["vlm_head_dim"]),
    )


def tensor_sizes_mb(spec: Pi05GraphSpec, precision: str) -> dict[str, float]:
    element_bytes = bytes_per_element(precision)
    vision_tokens = spec.vision_tokens_per_frame * spec.vision_frames

    # Camera transport is raw uint8 RGB. Compute precision does not change it.
    camera_input = mb(
        spec.vision_frames
        * spec.image_resolution
        * spec.image_resolution
        * 3
    )
    vision_hidden = mb(
        vision_tokens * spec.vision_hidden_size * element_bytes
    )
    projected_vision = mb(
        vision_tokens * spec.vlm_hidden_size * element_bytes
    )
    vlm_hidden = mb(
        spec.prefix_tokens * spec.vlm_hidden_size * element_bytes
    )
    kv_per_layer = mb(
        2
        * spec.prefix_tokens
        * spec.vlm_num_kv_heads
        * spec.vlm_head_dim
        * element_bytes
    )
    ae_hidden = mb(
        spec.action_chunk_size * spec.ae_hidden_size * element_bytes
    )
    action = mb(
        spec.action_chunk_size * spec.action_dof * element_bytes
    )
    language_token_ids = mb(spec.language_tokens * 4)
    state_token_ids = mb(spec.state_tokens * 4)

    return {
        "camera_input": camera_input,
        "vision_hidden": vision_hidden,
        "projected_vision": projected_vision,
        "vlm_hidden": vlm_hidden,
        "kv_per_layer": kv_per_layer,
        "ae_hidden": ae_hidden,
        "action": action,
        "language_token_ids": language_token_ids,
        "state_token_ids": state_token_ids,
    }


def build_pi05_coopinfer_payload(
    profile: Mapping[str, Any],
    *,
    bandwidth_mb_s: float,
    latency_ms: float,
    ae_placement: str = "device",
    source_period_ms: float = 100.0,
    batch_transfers: bool = False,
    pipeline_unroll: int = 1,
) -> dict[str, Any]:
    """Build a fine-grained pi0.5-like CoopInfer JSON from VLA-Perf costs.

    Vision/VLM/projector placement stays free. The Action Expert is pinned to
    one side (Device by default) so persistent VLM KV residency is represented
    correctly with today's stateless CoopInfer DAG.
    """
    validate_vla_perf_profile(profile)
    if bandwidth_mb_s <= 0:
        raise ValueError("bandwidth_mb_s must be greater than zero.")
    if latency_ms < 0:
        raise ValueError("latency_ms must be non-negative.")
    ae_placement = ae_placement.strip().lower()
    if ae_placement not in {"device", "host"}:
        raise ValueError("ae_placement must be 'device' or 'host'.")

    spec = graph_spec_from_profile(profile)
    precision = str(profile.get("precision", "bf16"))
    sizes = tensor_sizes_mb(spec, precision)
    costs = profile["costs_ms"]

    per_layer = {
        "device": {
            "vision": float(costs["device"]["vision_total"]) / spec.vision_layers,
            "vlm": float(costs["device"]["vlm_total"]) / spec.vlm_layers,
            "ae": float(costs["device"]["ae_pass_total"]) / spec.ae_layers,
            "projector": float(costs["device"]["projector"]),
        },
        "host": {
            "vision": float(costs["host"]["vision_total"]) / spec.vision_layers,
            "vlm": float(costs["host"]["vlm_total"]) / spec.vlm_layers,
            "ae": float(costs["host"]["ae_pass_total"]) / spec.ae_layers,
            "projector": float(costs["host"]["projector"]),
        },
    }

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def add_node(
        node_id: str,
        name: str,
        c_dev: float,
        c_host: float,
        placement: str = "free",
        *,
        period: float = 0.0,
    ) -> None:
        x = 0 if placement == "device" else 1
        nodes.append(
            {
                "id": node_id,
                "name": name,
                "c_dev": float(c_dev),
                "c_host": float(c_host),
                "placement": placement,
                "source_period_ms": float(period),
                "source_phase_ms": 0.0,
                "x": x,
            }
        )

    def add_edge(source: str, target: str, size_mb: float) -> None:
        edges.append(
            {"source": source, "target": target, "size": float(size_mb)}
        )

    # Physical observations and FM schedule originate on the robot.
    add_node("camera_input", "Camera Input", 0.0, 0.0, "device", period=source_period_ms)
    add_node("language_input", "Language Token IDs", 0.0, 0.0, "device", period=source_period_ms)
    add_node("state_input", "Discretized State Token IDs", 0.0, 0.0, "device", period=source_period_ms)
    add_node("noise_input", "Initial FM Noise", 0.0, 0.0, ae_placement, period=source_period_ms)
    add_node("fm_schedule", "FM Timestep Schedule", 0.0, 0.0, ae_placement, period=source_period_ms)

    # SigLIP layer costs are the VLA-Perf/GenZ total distributed over the
    # homogeneous transformer blocks.
    for layer in range(spec.vision_layers):
        add_node(
            f"vision_{layer}",
            f"SigLIP Vision Layer {layer}",
            per_layer["device"]["vision"],
            per_layer["host"]["vision"],
        )
    add_edge("camera_input", "vision_0", sizes["camera_input"])
    for layer in range(spec.vision_layers - 1):
        add_edge(
            f"vision_{layer}",
            f"vision_{layer + 1}",
            sizes["vision_hidden"],
        )

    # Small PaliGemma multimodal projection. This cost is a roofline estimate
    # built from VLA-Perf's hardware system_configs during the profiling stage.
    add_node(
        "multimodal_projector",
        "Vision-to-VLM Projector",
        per_layer["device"]["projector"],
        per_layer["host"]["projector"],
    )
    add_edge(
        f"vision_{spec.vision_layers - 1}",
        "multimodal_projector",
        sizes["vision_hidden"],
    )

    # Gemma VLM is fully free for placement search.
    for layer in range(spec.vlm_layers):
        add_node(
            f"vlm_{layer}",
            f"VLM Gemma Layer {layer}",
            per_layer["device"]["vlm"],
            per_layer["host"]["vlm"],
        )
    add_edge("multimodal_projector", "vlm_0", sizes["projected_vision"])
    add_edge("language_input", "vlm_0", sizes["language_token_ids"])
    add_edge("state_input", "vlm_0", sizes["state_token_ids"])
    for layer in range(spec.vlm_layers - 1):
        add_edge(f"vlm_{layer}", f"vlm_{layer + 1}", sizes["vlm_hidden"])

    # Full Flow-Matching loop. VLM KV is streamed exactly once into step 0;
    # later denoising steps reuse the persistent cache on the AE resource.
    previous_update: str | None = None
    for step in range(spec.denoising_steps):
        action_source = "noise_input" if step == 0 else previous_update
        assert action_source is not None

        add_node(
            f"fm_input_s{step}",
            f"FM Step {step}: Action State",
            0.0,
            0.0,
            ae_placement,
        )
        add_edge(action_source, f"fm_input_s{step}", sizes["action"])
        add_edge("fm_schedule", f"fm_input_s{step}", 0.0)

        for layer in range(spec.ae_layers):
            add_node(
                f"ae_s{step}_l{layer}",
                f"FM Step {step}: Action Expert Layer {layer}",
                per_layer["device"]["ae"],
                per_layer["host"]["ae"],
                ae_placement,
            )
        add_edge(f"fm_input_s{step}", f"ae_s{step}_l0", sizes["ae_hidden"])
        for layer in range(spec.ae_layers - 1):
            add_edge(
                f"ae_s{step}_l{layer}",
                f"ae_s{step}_l{layer + 1}",
                sizes["ae_hidden"],
            )

        update_id = f"fm_update_s{step}"
        add_node(
            update_id,
            f"FM Step {step}: ODE Update",
            0.0,
            0.0,
            ae_placement,
        )
        add_edge(
            f"ae_s{step}_l{spec.ae_layers - 1}",
            update_id,
            sizes["action"],
        )
        previous_update = update_id

    for layer in range(spec.vlm_layers):
        add_edge(
            f"vlm_{layer}",
            f"ae_s0_l{layer}",
            sizes["kv_per_layer"],
        )

    add_node("action_output", "Final Action Chunk", 0.0, 0.0, "device")
    assert previous_update is not None
    add_edge(previous_update, "action_output", sizes["action"])

    return {
        "version": "1.0",
        "nodes": nodes,
        "edges": edges,
        "environment": {
            "bandwidth": float(bandwidth_mb_s),
            "latency": float(latency_ms),
            "weight_avg_latency": 1.0,
            "weight_max_latency": 0.0,
            "weight_device_utilization": 0.0,
            "latency_limit": 0.0,
            "max_frame_latency_limit": 0.0,
            "batch_transfers": bool(batch_transfers),
            "pipeline_unroll": int(pipeline_unroll),
            "solver_threads": 0,
            "anneal_initial_temp": 1.0,
            "anneal_final_temp": 0.01,
        },
    }


def write_coopinfer_payload(payload: Mapping[str, Any], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
