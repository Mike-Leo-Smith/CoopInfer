# Scripts

## Canonical VLA pipeline

Use these entry points for the current production/research path:

```text
1. torch.export
   model_ir_export.py
   pi05_export_probe.py
   smolvla_export_probe.py

2. Layer Graph Analysis
   model_ir_layer_analysis.py

3. Cost + SchedulingIR
   model_ir_build_scheduling.py

4. CoopInfer Solver
   coopinfer_solve.py
```

The detailed workflow and validation commands are documented in `docs/vla_pipeline.md`.

### Model Stage-1 adapters

`pi05_export_probe.py` is now the single pi0.5 Stage-1 adapter. Despite the historical filename, it no longer exposes the old `qkv`, `prefix`, or `prefix_ae` debug modes. It captures the complete model tensor-input path used for structural scheduling analysis:

```text
images + language/token inputs
        -> vision frontend / multimodal projection
        -> prefix VLM prefill + KV cache
        -> one Action Expert denoise step
        -> Fine ModelIR
```

The number of denoise iterations is stored as workload metadata; Stage 1 captures one denoise execution rather than statically duplicating the same Expert layers N times.

`smolvla_export_probe.py` follows the same one-step structural scope.

## Experiment helpers / references

The remaining scripts are not additional required pipeline stages. They are retained for controlled experiments or historical reference, for example:

- `model_ir_placement_sweep.py`: structured placement sweeps;
- `model_ir_network_placement_sweep.py`: bandwidth/RTT sensitivity sweeps;
- `model_ir_genz_cost.py`: Fine-graph GenZ costing reference;
- `model_ir_coarsen.py`: generic coarsening experiments;
- `model_ir_solve.py`: legacy Full-Graph reference path;
- `profile_pi05_vla_perf.py`: VLA-Perf/pi0.5 profiling experiments;
- `generic_frontend_smoke.py`: small generic frontend smoke test.

The old pi0.5 `qkv` / `prefix` / `prefix_ae` probe implementation, its dedicated frontend helper, and the pi0.5-specific Full-Graph solve wrapper were removed. Development-only dependency/frontier/ingress/min-cut/ownership audit CLIs were also removed after the validated causal dependency and ownership-boundary logic was promoted into `src/coopinfer/frontend/layer_graph.py`.
