# VLA-Perf -> CoopInfer integration

This integration turns NVIDIA VLA-Perf / GenZ analytical hardware costs into a fine-grained CoopInfer DAG for a pi0/pi0.5-shaped VLA.

## Why two stages?

CoopInfer and VLA-Perf can use different Python environments. The official VLA-Perf quick start uses Python 3.10, while CoopInfer may run on a newer Python. The adapter therefore separates:

1. **VLA-Perf profiling** -> hardware-neutral cost profile JSON.
2. **CoopInfer graph generation** -> fine-grained placement/scheduling JSON.

No VLA-Perf package is added as a hard CoopInfer dependency.

## Modeled architecture

The default profile follows the model dimensions registered by VLA-Perf for pi0 / pi0.5:

- SigLIP SoViT-400m vision tower: 27 transformer layers, hidden size 1152.
- Gemma-2B VLM: 18 layers, hidden size 2048, 1 KV head, head dimension 256.
- Gemma-300M Action Expert: 18 layers, hidden size 1024.
- Three camera views, 256 visual tokens per view.
- 32 language tokens + 32 discretized state tokens by default.
- 50 action tokens and 10 flow-matching denoising steps.

The token counts, image resolution, action chunk size, and denoising steps are CLI parameters.

## Cost modeling

`scripts/profile_pi05_vla_perf.py` calls GenZ directly:

- `prefill_moddeling()` for the vision tower.
- `prefill_moddeling()` for the VLM prefill.
- `parallel_decode_modeling()` for one Action Expert flow-matching pass.

The resulting total transformer latency is distributed over homogeneous transformer blocks when generating the CoopInfer layer-wise graph. The script also records the `model_df` latency sum when available for a sanity check.

The small vision-to-VLM projector is not modeled as a transformer by VLA-Perf. The profiling script estimates it with a simple roofline model using the same VLA-Perf `system_configs` peak FLOPS and memory bandwidth.

## Tensor / communication sizing

The CoopInfer generator derives communication sizes from model shapes and precision:

- raw camera payload: RGB uint8 frames;
- vision hidden state;
- projected visual tokens;
- VLM hidden state;
- layer-wise VLM KV (`K + V`);
- Action Expert hidden state;
- final action chunk.

For BF16 and the default 832-token prefix, layer-wise VLM KV is about 0.8125 MB.

Only step 0 contains `VLM_i -> AE_step0_i` KV transfer edges. Denoising steps 1-9 reuse the already resident KV cache and do not retransmit it.

## Current placement search semantics

- camera/language/state inputs are fixed on Device;
- Vision, projector, and all VLM layers are **free**;
- the full Action Expert is pinned to Device by default;
- final action output is fixed on Device.

The AE pin is deliberate. Current CoopInfer represents a stateless DAG and does not yet model persistent tensor residency as a first-class state. Allowing different AE denoising steps to migrate independently could produce an invalid schedule where a later step moves to a resource that does not own the cached VLM KV.

Use `--ae-placement host` to generate the symmetric all-AE-on-Host control.

## Setup VLA-Perf on Windows

Clone NVIDIA VLA-Perf next to CoopInfer:

```powershell
cd D:\Project
git clone https://github.com/NVlabs/vla-perf.git vla-perf
cd .\vla-perf

py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .\genz
```

## 1. Profile H100 Host + RTX 4090 Device

From the CoopInfer repository root, invoke the profiling script with the VLA-Perf Python environment:

```powershell
D:\Project\vla-perf\.venv\Scripts\python.exe `
  .\scripts\profile_pi05_vla_perf.py `
  --vla-perf-root D:\Project\vla-perf `
  --device RTX_4090 `
  --host H100 `
  --precision bf16 `
  --output .\results\pi05_h100_4090_profile.json
```

For the A100 calibration pair, change only:

```powershell
--host A100_80GB
```

The profile JSON records the VLA-Perf checkout commit, hardware names, model dimensions, prefix length, component latencies, and GenZ diagnostics.

## 2. Generate a CoopInfer graph

Activate the CoopInfer environment and run:

```powershell
.\.venv\Scripts\Activate.ps1

python .\scripts\vla_perf_to_coopinfer.py `
  .\results\pi05_h100_4090_profile.json `
  --output .\examples\pi05_h100_4090_wavekv.json `
  --bandwidth 1250 `
  --latency 0.2
```

The command prints node/edge counts and derived KV/hidden-state sizes.

## 3. Run CoopInfer

Load the generated JSON in the GUI or use the existing solver/export workflow.

Recommended first experiment:

- Simulated Annealing;
- 30,000 iterations;
- H100 Host / RTX 4090 Device profile;
- 1250 MB/s network bandwidth;
- 0.2 ms per-message latency.

Then sweep bandwidth while keeping the same cost profile:

```text
1250, 1000, 750, 500, 250, 125 MB/s
```

The important question is whether the optimizer naturally transitions between Device-heavy execution and Host Vision/VLM + Device AE Wave-KV as network conditions change.

## Notes / limitations

1. VLA-Perf currently registers pi0.5 with the same transformer dimensions as pi0; pi0.5-specific adaRMSNorm overhead is described as small and is not separately modeled here.
2. The generator distributes total GenZ transformer latency evenly across homogeneous blocks. A future version can map `model_df` operator rows to exact transformer blocks if non-uniform per-layer costs become important.
3. Camera transport is raw RGB by default. JPEG/codec preprocessing can be introduced as explicit DAG nodes later.
4. This is analytical modeling, not a replacement for real A100/4090 measurements. The A100 pair is useful for calibration against existing measured pi0.5 profiling.
