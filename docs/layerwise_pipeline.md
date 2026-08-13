# Automatic Layer-wise Pipeline

## Goal

Build a scheduling graph at repeated-layer granularity without manually encoding
model-specific Prefix/Expert/KV dependencies.

```text
Real Fine ModelIR
      |
      | repeated module-path detection
      v
Layer grouping + conservative auxiliary regions
      |
      | quotient projection of real Fine edges only
      v
LayerIR
      |
      | layer workload -> GenZ System roofline
      v
Costed Layer SchedulingIR
      |
      v
CoopInfer JSON
```

## Does Layer Auto Detection lose the Fine ModelIR?

No. The transform is non-destructive and produces new artifacts. The source Fine
ModelIR is never rewritten.

The abstraction does hide layer-internal fine topology from the LayerIR view, but
that information remains recoverable through the source Fine ModelIR and the
layer mapping/provenance artifact.

The following invariants are enforced:

1. Every Fine node maps to exactly one layer/region/singleton group.
2. A LayerIR edge exists only when at least one real Fine edge crosses the two groups.
3. Every real cross-group Fine edge is recorded in quotient-edge provenance.
4. Unrecognized nodes are kept in auxiliary regions rather than discarded.
5. Input/output/source-like nodes are kept singleton to preserve current CoopInfer source semantics.

## Automatic layer detection

The detector reads `module_path` metadata produced by `torch.export`. It recognizes
common repeated containers such as:

- `layers.<index>` / `layer.<index>`
- `blocks.<index>` / `block.<index>`
- `h.<index>`

A container is considered a repeated layer stack only when multiple distinct layer
indices are visible. Different stack roots remain separate even when both contain
layers numbered 0..N.

No VLM, Action Expert, KV, Gemma, PaliGemma, or pi0.5 name is required to construct
a dependency edge.

## Dependency recovery

Layer dependencies are quotient-graph dependencies:

```text
Fine u in Group A
Fine v in Group B
real Fine edge u -> v
        |
        v
Layer edge A -> B
```

Multiple Fine edges between the same group pair are aggregated. Tensor IDs and a
complete Fine-edge provenance table are retained in the mapping artifact.

## Layer-level GenZ cost

Layer latency is not the sum of previously computed Fine-op latencies.

For each group we reconstruct a hardware-neutral workload:

- FLOPs: sum of arithmetic work represented by the real Fine ops;
- parameter bytes: unique parameter/buffer tensors referenced in the layer;
- input bytes: unique activation tensors entering the layer;
- output bytes: unique activation tensors leaving the layer.

Internal intermediate activations are not recharged as off-chip HBM traffic for
every Fine op. The layer workload is then evaluated once per hardware using the
GenZ System's precision-specific compute throughput and off-chip memory bandwidth:

```text
compute_ms = layer_flops / GenZ_peak_flops
memory_ms  = layer_external_bytes / GenZ_HBM_bandwidth
layer_ms   = max(compute_ms, memory_ms)
```

This first version is named `genz-layer-external-roofline`. It is a layer-level
analytical model using GenZ hardware semantics, not a measured CUDA latency and
not yet GenZ's full transformer-operator sequence model. A later refinement can
map reconstructed transformer parameters into GenZ native FC/Logit/Attend/etc.
operators and use memory pinning/cache assumptions explicitly.

## Artifacts

`scripts/model_ir_layerwise.py` writes three independent files:

- `*_layer_ir.json`: costed layer/region quotient graph;
- `*_layer_mapping.json`: Fine node membership and quotient-edge provenance;
- `*_layer_genz_coopinfer.json`: GUI/solver payload.

It does not run evaluation or search.
