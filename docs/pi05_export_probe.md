# Real pi0.5 torch.export probe

This experiment checks whether the generic CoopInfer frontend can recover
useful dependencies from the **real LeRobot pi0.5 PyTorch implementation**
without manually drawing a Wave-KV DAG.

It is intentionally separate from cost modeling and scheduling.

```text
LeRobot PI05Pytorch
        |
        v
CPU torch.export
        |
        v
fine ATen/FX dependency graph
        |
        +-- probe 1: are Q/K/V projections visible?
        |
        +-- probe 2: are per-layer prefix KV tensors graph-visible?
        |
        +-- probe 3: can VLM K/V reach same-layer Action Expert nodes?
```

A successful third probe is evidence that a Wave-KV scheduling opportunity
already exists in the model data flow and can be preserved by dependency-aware
coarsening. The probe never inserts VLM->AE edges itself.

## Why CPU is enough

This experiment validates graph structure, tensor dependencies, and module
metadata. It does **not** measure runtime latency, so a CUDA GPU is not required.
Real A100/4090 profiling belongs to a later `CostBackend` validation step.

The default architecture-only path instantiates the real LeRobot modules with
random weights. Numerical pretrained weights are not required to ask whether
the graph contains a dependency.

## Requirements

Use an environment with:

- PyTorch with `torch.export`;
- a LeRobot checkout with the pi0.5 policy dependencies installed;
- enough host RAM to instantiate the model.

PyTorch and LeRobot remain optional dependencies of CoopInfer itself.

## Run on Windows CPU

For a LeRobot checkout at `D:\Project\lerobot`:

```powershell
cd D:\Project\CoopInfer-pr
git fetch origin
git switch --track origin/agent/pi05-export-probe

python .\scripts\pi05_export_probe.py `
  --lerobot-root D:\Project\lerobot `
  --probe all
```

The default uses the real architecture with random float32 weights.

If memory pressure is high, retry with:

```powershell
python .\scripts\pi05_export_probe.py `
  --lerobot-root D:\Project\lerobot `
  --dtype bfloat16 `
  --probe all
```

## Run with an existing pretrained checkpoint

```powershell
python .\scripts\pi05_export_probe.py `
  --lerobot-root D:\Project\lerobot `
  --model-path D:\Models\pi05_base `
  --local-files-only `
  --probe all
```

Pretrained weights are not required for the first graph-visibility experiment.

## Three probes

### 1. `qkv`

Wraps the actual layer-0 `q_proj`, `k_proj`, and `v_proj` modules and exports
them with a representative prefix tensor.

Question:

> Does the real LeRobot layer expose Q/K/V projection operations to
> `torch.export` and preserve module metadata?

This is the smallest diagnostic. If it fails, there is no reason to attempt the
larger prefix graph yet.

### 2. `prefix`

Calls the real prefix branch of `PaliGemmaWithExpertModel.forward(...,
use_cache=True)` and flattens the returned cache only at the wrapper output.

Question:

> Are per-layer K/V cache tensors visible as graph outputs?

Flattening the cache object does not create any dependency. It only turns an
opaque Python/cache container into tensor outputs that `torch.export` can
represent.

### 3. `prefix_ae`

Runs the real prefix path and immediately feeds its returned
`past_key_values` into one real `denoise_step`.

Question:

> Can captured graph reachability prove a VLM-layer K/V producer -> matching
> Action-Expert-layer consumer path?

The report field:

```text
wave_kv_candidate_layers
```

contains only layers where the captured graph itself proves both K and V can
reach a same-layer Action Expert node. No Wave-KV edge is added by the probe.

## Outputs

By default:

```text
results/pi05_export_probe/
├── report.json
├── qkv_ir.json
├── qkv_graph.txt
├── prefix_ir.json
├── prefix_graph.txt
├── prefix_ae_ir.json
└── prefix_ae_graph.txt
```

Failed probes are recorded in `report.json` with the exception and traceback,
so failure is still actionable evidence.

## How to interpret failures

Typical outcomes:

```text
A. qkv + prefix + prefix_ae succeed
   -> best case: dependency is directly recoverable.

B. qkv succeeds, prefix fails
   -> cache object / attention path needs a thinner export wrapper or
      decomposition.

C. prefix succeeds, prefix_ae has no VLM->AE reachability
   -> inspect whether cache cloning / Hugging Face cache semantics obscure
      the dependency in exported IR.

D. Q/K/V are replaced by an opaque custom/fused op
   -> add decomposition before changing the scheduler.
```

The important rule is to fix graph visibility in the frontend rather than
manually writing the desired Wave-KV dependency into the Scheduling DAG.
