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

## Experiment helpers / references

The remaining scripts are not additional required pipeline stages. They are retained for controlled experiments or historical reference, for example:

- `model_ir_placement_sweep.py`: structured placement sweeps;
- `model_ir_network_placement_sweep.py`: bandwidth/RTT sensitivity sweeps;
- `model_ir_genz_cost.py`: Fine-graph GenZ costing reference;
- `model_ir_coarsen.py`: generic coarsening experiments;
- `model_ir_solve.py`: legacy Full-Graph reference path;
- `pi05_full_graph_solve.py`: pi0.5 Full-Graph reference;
- `profile_pi05_vla_perf.py`: VLA-Perf/pi0.5 profiling experiments;
- `generic_frontend_smoke.py`: small generic frontend smoke test.

Development-only dependency/frontier/ingress/min-cut/ownership audit CLIs were removed after the validated causal dependency and ownership-boundary logic was promoted into `src/coopinfer/frontend/layer_graph.py`.
