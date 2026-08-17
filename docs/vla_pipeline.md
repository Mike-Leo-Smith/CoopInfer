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
   - materialize iterative inference execution from model metadata
   - reuse static prefix/KV inputs across denoise steps
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
   - shared layer placement across denoise steps
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

One denoise step is captured structurally. The adapter reads the model's native `config.num_inference_steps` and records it as the normalized `ModelIR.metadata["num_inference_steps"]`, together with `captured_denoise_steps=1`. The current pi0.5 config yields 10 steps, but CoopInfer does not contain a pi0.5-specific `10` rule.

Canonical pi0.5 Stage-1 command:

```powershell
python .\scripts\pi05_export.py `
  --lerobot-root D:\Project\lerobot-main `
  --num-images 3 `
  --output .\results\pi05_full_pipeline\pi05_fine_ir.json
```

### SmolVLA Stage-1 scope

SmolVLA now has the same single canonical one-step inference scope: native LeRobot SmolVLA architecture/configuration, image/language/state prefix frontend, VLM prefix prefill, KV construction, and one cached Action Expert denoise execution. Its native `config.num_steps` is normalized into the same `num_inference_steps` ModelIR metadata field, so Stage 3 uses the model's own step count without model-family branching. The former `joint_forward` and manual layer-truncation export options were development aids and are not part of the canonical Stage-1 script.

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

This stage does not choose placement. It describes the hardware-aware scheduling problem and materializes repeated inference execution when the Stage-1 metadata says the captured one-step graph is executed multiple times.

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

For iterative Flow-Matching-style inference, Stage 3 applies these rules generically:

1. Read `num_inference_steps=N` and `captured_denoise_steps` from ModelIR metadata. The value is supplied by the model adapter from the model's native config; it is not hardcoded in Stage 3.
2. Infer the iterative layer stack structurally. The current rule requires one unique target stack that receives a complete set of same-index cross-stack dependencies; it does not search for strings such as `expert`, `pi05`, or a fixed layer count. If the stack cannot be resolved uniquely, Stage 3 fails explicitly rather than guessing.
3. Execute static Vision/VLM/prefix work once.
4. Materialize the iterative stack for N denoise executions in true step order (`AE0 -> ... -> AEL` repeated N times), rather than multiplying each layer cost by N in place.
5. Materialize static-to-iterative prefix/KV dependencies only for denoise step 0. Once transferred, the KV/cache is considered resident and is reused by steps 1..N-1.
6. Repeat iterative-layer internal communication on every denoise step because the evolving action hidden state is newly produced each iteration.
7. Add a loop-carried action-state dependency between consecutive denoise passes. Its payload size is inferred from the exported `x_t`-shaped input and the selected communication precision.
8. All copies of one base iterative layer share one placement decision across denoise steps. This preserves cache residency and prevents the solver from choosing an unrealistically different device for the same layer on each iteration.

For the current pi0.5 workload, the model config supplies `N=10`, so the expected execution shape is:

```text
Vision x1
  -> VLM/prefix x1
       -> per-layer KV transfer x1
            -> AE stack step 0
            -> AE stack step 1 (reuse KV)
            -> ...
            -> AE stack step 9 (reuse KV)
```

Outputs:

- `*_scheduling_ir.json`: execution-aware layer DAG with per-resource compute costs and communication payloads;
- `*_coopinfer.json`: CoopInfer-compatible solver input.

Stage 3 prints both the one-step structural layer cost and the materialized full-execution compute total. For iterative graphs it also reports `num_inference_steps`, iterative layer count, whether KV reuse is enabled, the count of static-to-iterative edges charged once, and loop-carried state size.

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

Execution-aware nodes use the internal `base_layer::stepNNN` convention. Stage 4 groups all copies with the same base layer into one placement variable, so every denoise execution of a given layer stays on the same Device/Host. Non-iterative graphs continue to use the native C++ solver path unchanged.

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

The scheduling regression includes an iterative three-step toy graph that verifies all of the following at once:

- the downstream iterative stack is inferred from graph structure;
- static prefix/KV dependencies are emitted only for step 0;
- iterative internal edges repeat once per denoise execution;
- loop-carried state edges connect consecutive passes;
- the SchedulingIR stays a DAG.

For a real-model regression, generate a fresh Stage-1 Fine ModelIR and then run Stages 2 and 3. Do not use the retired pi0.5 `prefix_ae_ir.json` as the formal full-model baseline.

SmolVLA's currently validated full Stage-1 capture has these reference points:

- 3 repeated stacks: Vision x12, VLM x16, Expert x16;
- 44 layer nodes and 58 immediate causal dependencies;
- 16 same-index VLM-to-Expert dependencies;
- ownership payload patterns include 72,000 B Expert hidden, 462,720 B VLM hidden, 308,480 B per-layer VLM-to-Expert prefix K/V, and 4.5 MiB between adjacent Vision layers at bf16 communication precision.

The current full pi0.5 Stage-2 structural reference is Vision x27 + VLM x18 + iterative stack x18 = 63 layer nodes and 79 immediate dependencies. Its current native config supplies 10 denoise steps; Stage 3 is responsible for expanding those execution semantics without changing the Stage-2 structural graph.

These values are regression references for specific saved workloads, not architecture priors used by the analyzer.
