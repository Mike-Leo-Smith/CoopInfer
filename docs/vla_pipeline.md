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

Model-specific wrappers are allowed only to construct the real inference path and example inputs. The generic frontend receives the resulting Fine ModelIR; it is not given stack roles, layer counts, KV annotations, or split points.

Examples currently include `pi05_export_probe.py` and `smolvla_export_probe.py`.

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
  .\results\smolvla_export_probe\smolvla_inference_one_step_v16_e16_ir.json `
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
  --heuristic-iterations 3000
```

The network configuration stored by Stage 3 is used by default; `--bandwidth-mb-s` and `--latency-ms` can override it for sensitivity studies.

## Canonical scripts

The production VLA path is now centered on these scripts:

```text
Stage 1  model_ir_export.py / pi05_export_probe.py / smolvla_export_probe.py
Stage 2  model_ir_layer_analysis.py
Stage 3  model_ir_build_scheduling.py
Stage 4  coopinfer_solve.py
```

Placement/network sweep scripts remain useful experiment helpers. The old dependency, ingress, min-cut, frontier-payload, ownership-audit, and legacy layerwise CLIs have been removed after their validated logic was promoted into the library.

## Regression tests

The production regression set is intentionally small and focused:

```powershell
python -m pytest .\tests\test_layer_dependencies.py .\tests\test_layer_graph.py .\tests\test_discovered_layer_schedule.py -q
```

For real-model regression, run Stage 2 on both saved Fine ModelIRs:

```powershell
python .\scripts\model_ir_layer_analysis.py `
  .\results\pi05_export_probe\prefix_ae_ir.json `
  --precision bf16

python .\scripts\model_ir_layer_analysis.py `
  .\results\smolvla_export_probe\smolvla_inference_one_step_v16_e16_ir.json `
  --precision bf16
```

Known structural reference points from the validated captures:

- pi0.5: 2 stacks, 36 layer nodes, 52 dependencies, 18 same-index cross-stack dependencies;
- SmolVLA: 3 stacks, 44 layer nodes, 58 dependencies, 16 same-index VLM-to-Expert dependencies;
- SmolVLA ownership payload patterns include 72,000 B Expert hidden, 462,720 B VLM hidden, 308,480 B per-layer VLM-to-Expert prefix K/V, and 4.5 MiB between adjacent Vision layers at bf16 communication precision.

These numbers are regression references for the current saved workloads, not architecture priors used by the analyzer.
