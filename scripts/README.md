# Scripts

## Canonical VLA pipeline

Use these entry points for the current production/research path:

```text
1. torch.export
   model_ir_export.py
   pi05_export.py
   smolvla_export.py

2. Layer Graph Analysis
   model_ir_layer_analysis.py

3. Cost + SchedulingIR
   model_ir_build_scheduling.py

4. CoopInfer Solver
   coopinfer_solve.py
```

The detailed workflow and validation commands are documented in `docs/vla_pipeline.md`.

### Model Stage-1 adapters

`pi05_export.py` and `smolvla_export.py` are the single canonical Stage-1 adapters for the two current VLA models. Both capture the model tensor-input inference path through the visual/prefix frontend, prefix KV construction, and one Action Expert denoise execution, then emit Fine ModelIR only.

```text
model tensor inputs
        -> vision / prefix frontend
        -> VLM prefix prefill + KV cache
        -> one Action Expert denoise step
        -> Fine ModelIR
```

The repeated denoise count is stored as workload metadata; Stage 1 captures one denoise execution rather than statically duplicating the same Expert stack N times.

The old pi0.5 `qkv`, `prefix`, and `prefix_ae` probe modes and helper module have been removed. The old SmolVLA `joint_forward` / manual layer-truncation export entrypoint has also been removed from the canonical path.

## Experiment helpers / references

The remaining scripts are not additional required pipeline stages. They are retained for controlled experiments or historical reference, for example:

- `model_ir_placement_sweep.py`: structured placement sweeps;
- `model_ir_network_placement_sweep.py`: bandwidth/RTT sensitivity sweeps;
- `model_ir_genz_cost.py`: Fine-graph GenZ costing reference;
- `model_ir_coarsen.py`: generic coarsening experiments;
- `model_ir_solve.py`: legacy Full-Graph reference path;
- `profile_pi05_vla_perf.py`: VLA-Perf/pi0.5 profiling experiments;
- `generic_frontend_smoke.py`: small generic frontend smoke test.

Development-only dependency/frontier/ingress/min-cut/ownership audit CLIs were removed after the validated causal dependency and ownership-boundary logic was promoted into `src/coopinfer/frontend/layer_graph.py`. The pi0.5-specific Full-Graph solve wrapper was also removed because Stage 4 is now shared through `coopinfer_solve.py`.
