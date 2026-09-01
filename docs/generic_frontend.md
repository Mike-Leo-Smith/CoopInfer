# Generic model frontend MVP

This frontend separates model capture, cost modeling, graph coarsening, and
CoopInfer scheduling so each layer can evolve independently.

```text
PyTorch / HuggingFace tensor model
        |
        v
torch.export / FX
fine op + tensor dependency graph
        |
        v
ModelIR
        |
        v
CostBackend
(GenZ / real profiler / table / callback)
        |
        v
DependencyAwarePolicy
        |
        v
SchedulingIR
        |
        v
to_coopinfer_payload()
        |
        v
existing CoopInfer solver
        |
        v
placement + communication/pipeline schedule
```

## Design boundary

The existing solver remains the stable scheduling backend. It only consumes:

- per-node Device and Host cost;
- optional fixed placement;
- tensor size on each dependency edge;
- network bandwidth and message latency.

The new code lives above that boundary:

- `frontend/torch_export.py` captures a PyTorch model into a fine-grained IR;
- `cost/base.py` defines a pluggable cost backend interface;
- `frontend/coarsening.py` reduces the op graph without hiding fan-out/join
  dependencies;
- `frontend/coopinfer_export.py` translates the scheduling IR into the existing
  CoopInfer JSON schema.

The earlier `vla_perf_adapter.py` remains intact as a pi0/pi0.5 VLA-Perf proof
of concept and regression path. The generic frontend does not depend on it.

## Why dependency-aware coarsening?

Capturing at op granularity preserves possible scheduling opportunities, but
searching every ATen op is unnecessarily expensive and often meaningless.
The MVP therefore merges only simple one-producer/one-consumer chains.

Fan-out, joins, model I/O, explicit hard boundaries, and fixed-placement
conflicts are preserved. This is enough to express the key Wave-KV pattern when
the captured graph exposes it:

```text
pre -> KV -----> VLM remainder
          \
           +----> Action Expert
```

A naive layer-wise merge can hide this branch. `DependencyAwarePolicy` keeps
the fan-out visible in the SchedulingIR, while the CoopInfer search still
decides whether moving/streaming that tensor is beneficial for the selected
hardware and network.

## Cost backend scope

This PR defines the interface rather than binding the generic frontend to a
specific GenZ version or profiler runtime:

```python
class CostBackend:
    def estimate(node, hardware) -> CostEstimate:
        ...
```

`MappingCostBackend` and `CallableCostBackend` make the pipeline immediately
usable for tests, calibration tables, external analytical models, or profiler
callbacks. The existing VLA-Perf integration remains the current concrete GenZ
path. A direct ATen/canonical-op -> GenZ operator backend can be added behind the
same interface without changing capture, coarsening, or CoopInfer.

## PyTorch dependency

PyTorch is intentionally not added as a hard CoopInfer dependency. The import
is lazy and only required when `capture_model()` is used.

## Validation

Core frontend tests do not require PyTorch. The torch.export smoke test is
skipped automatically when torch is absent.

```bash
python -m pytest tests/test_generic_frontend.py -q
```

With PyTorch installed, the end-to-end plumbing demo is:

```bash
python scripts/generic_frontend_smoke.py
```

The demo uses synthetic costs on purpose. It verifies interface plumbing, not
hardware accuracy. Hardware cost accuracy belongs to the selected CostBackend.
