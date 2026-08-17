# CoopInfer VLA Pipeline

The current VLA path is intentionally organized into four stages. Development-time audit experiments are not part of the production pipeline.

```text
PyTorch / Hugging Face model
        |
        v
1. torch.export
   capture real Fine computation DAG
        |
        v
   Fine ModelIR
        |
        v
2. Layer Graph Analysis
   - automatic repeated-layer detection
   - causal dependency recovery
   - causal ownership boundary / payload recovery
   - sanity validation
        |
        v
   LayerGraphIR
        |
        v
3. Cost + SchedulingIR
   - extract per-layer workload
   - GenZ hardware cost modeling
   - attach ownership-derived communication payloads
   - network model
   - build solver-ready SchedulingIR
        |
        v
   Costed SchedulingIR / CoopInfer JSON
        |
        v
4. CoopInfer Solver
   - placement search
   - compute/communication overlap
   - resource scheduling
   - E2E latency evaluation
```

## Stage 1 - torch.export

Model-specific wrappers are allowed only to construct the real inference path and example tensor inputs. The generic frontend receives the resulting Fine ModelIR; it is not given stack roles, layer counts, KV annotations, or split points.

The canonical model adapters are `pi05_export.py` and `smolvla_export.py`.

### pi0.5 Stage-1 scope

The pi0.5 adapter has one production/research scope only. The old `qkv`, `prefix`, and `prefix_ae` debug modes have been removed.

```text
model tensor inputs
  images + image masks
  language tokens + masks
  optional proprioceptive-memory state tensors when enabled by native config
        |
        v
PaliGemma vision frontend + multimodal projection
        |
        v
prefix embedding + language backbone prefill
        |
        v
per-layer prefix KV cache
        |
        v
one Action Expert denoise step
        |
        v
Fine ModelIR
```

The adapter uses native `PI05Config` by default. A local `--config-path` may be supplied to instantiate the same architecture configuration as a checkpoint without loading checkpoint weights. `--num-images` and `--lang-len` are workload inputs, not architecture annotations; when `--lang-len` is omitted the native tokenizer maximum length is used.

Stock pi0.5 puts normalized robot state into the text prompt before tokenization; the optional proprioceptive-memory variant instead passes continuous state history into the backbone. Stage 1 begins at model tensor inputs, so prompt construction/tokenization and image normalization/resizing in the policy processor are outside the exported neural DAG.

One denoise step is captured structurally. `num_inference_steps` and `captured_denoise_steps=1` are recorded in ModelIR metadata so repeated execution can be modeled separately rather than statically cloning the same Expert layer stack in the Fine DAG.

Canonical pi0.5 Stage-1 command:

```powershell
python .\scripts\pi05_export.py `
  --lerobot-root D:\Project\lerobot-main `
  --num-images 3 `
  --output .\results\pi05_full_pipeline\pi05_fine_ir.json
```

### SmolVLA Stage-1 scope

SmolVLA now has the same single canonical one-step inference scope: native LeRobot SmolVLA architecture/configuration, image/language/state prefix frontend, VLM prefix prefill, KV construction, and one cached Action Expert denoise execution. The former `joint_forward` and manual layer-truncation export options were development aids and are not part of the canonical Stage-1 script.

Canonical SmolVLA Stage-1 command:

```powershell
python .\scripts\smolvla_export.py `
  --lerobot-root D:\Project\lerobot-main `
  --metadata-dir D:\Project\models\SmolVLM2-500M-metadata `
  --num-images 3 `
  --output .\results\smolvla_full_pipeline\smolvla_fine_ir.json
```

## Stage 2 - Layer Graph Analysis

Public library API:

```python
layer_graph = analyze_layer_graph(model_ir, precision="bf16")
```

The returned `LayerGraphIR` contains:

- detected repeated layer groups and stacks;
- immediate causal layer dependencies;
- communication payloads recovered from causal ownership crossings;
- validation status.

The production boundary rule is **causal ownership**, not source-origin reachability, target-ingress fan-out, or unconstrained graph min-cut. Those alternatives were useful during development but can over-count residual intermediates or choose semantically invalid cuts at multi-input operators.

CLI:

```powershell
python .\scripts\model_ir_layer_analysis.py `
  <fine_model_ir.json> `
  --precision bf16
```

A healthy graph should end with:

```text
layer_graph_dag=True
missing_payload_dependencies=0
LAYER_GRAPH_STATUS=PASS
```

`ambiguous_owner_nodes` and non-adjacent same-stack edges are reported as diagnostic information. They are not automatically assumed to be model bugs, because real architectures may contain parallel joins or skip connections.

## Stage 3 - Cost + SchedulingIR

This stage does not choose placement. It describes the hardware-aware scheduling problem.

```powershell
python .\scripts\model_ir_build_scheduling.py `
  <fine_model_ir.json> `
  --genz-root D:\Project\vla-perf `
  --device RTX_4090 `
  --host A100_80GB `
  --precision bf16 `
  --bandwidth-mb-s 1250 `
  --latency-ms 0.2
```

Outputs:

- `*_scheduling_ir.json`: layer DAG with per-resource compute costs and communication payloads;
- `*_coopinfer.json`: CoopInfer-compatible solver input.

Stage 3 uses the same `LayerGraphIR` boundary semantics validated in Stage 2.

## Stage 4 - CoopInfer Solver

The solver consumes the `*_coopinfer.json` produced by Stage 3 and searches placement and execution schedules. Solver policy is intentionally separated from model parsing and cost construction, so the same frontend can be evaluated with different search methods.

Canonical CLI:

```powershell
python .\scripts\coopinfer_solve.py `
  <fine_model_ir_stem>_coopinfer.json `
  --algorithm "Random Search" `
  --heuristic-iterations 3000 `
  --output <fine_model_ir_stem>_solve_result.json
```

When `--output` is provided, Stage 4 writes a self-contained solve-result JSON containing:

- solver algorithm, iteration count, and seed;
- network/environment parameters;
- all-Device and all-Host baselines;
- best latency/utilization summary;
- complete node assignment;
- per-node placement and Device/Host compute costs;
- detailed schedule metrics, including start/finish times and transfer records.

The network configuration stored by Stage 3 is used by default; `--bandwidth-mb-s` and `--latency-ms` can override it for sensitivity studies.

## Canonical scripts

```text
Stage 1  model_ir_export.py / pi05_export.py / smolvla_export.py
Stage 2  model_ir_layer_analysis.py
Stage 3  model_ir_build_scheduling.py
Stage 4  coopinfer_solve.py
```

Placement/network sweep scripts remain experiment helpers. The old dependency, ingress, min-cut, frontier-payload, ownership-audit, legacy layerwise CLIs, old model-specific structural probes, and pi0.5-specific Full-Graph solver wrapper have been removed after their useful logic was either promoted into the library or superseded by the canonical four-stage path.

## Regression tests

The production regression set is intentionally small and focused:

```powershell
python -m pytest .\tests\test_layer_dependencies.py .\tests\test_layer_graph.py .\tests\test_discovered_layer_schedule.py -q
```

For a real-model regression, generate a fresh Stage-1 Fine ModelIR and then run Stage 2. Do not use the retired pi0.5 `prefix_ae_ir.json` as the formal full-model baseline.

SmolVLA's currently validated full Stage-1 capture has these reference points:

- 3 repeated stacks: Vision x12, VLM x16, Expert x16;
- 44 layer nodes and 58 immediate causal dependencies;
- 16 same-index VLM-to-Expert dependencies;
- ownership payload patterns include 72,000 B Expert hidden, 462,720 B VLM hidden, 308,480 B per-layer VLM-to-Expert prefix K/V, and 4.5 MiB between adjacent Vision layers at bf16 communication precision.

A new pi0.5 structural reference should be recorded only after the new full Stage-1 adapter is run, because the previous 2-stack / 36-layer / 52-dependency numbers came from a prefix-only capture that intentionally omitted the vision frontend.

These values are regression references for specific saved workloads, not architecture priors used by the analyzer.
