# Dependency-aware multilevel coarsening

## Goal

CoopInfer receives a fine-grained costed DAG from a generic model frontend. A
real model can contain thousands of fine operators, which makes direct binary
Host/Device placement search unnecessarily expensive. The coarsening stage
reduces the number of placement variables while retaining boundaries that are
valuable for communication/computation overlap.

This stage is intentionally model agnostic. It does not recognize KV cache,
VLM, Action Expert, Transformer layers, or any model-specific operator name.

## Classical ideas reused

The design combines two standard families of ideas.

1. **Multilevel graph partitioning.** METIS-style multilevel methods repeatedly
   contract strongly coupled vertices/edges, solve a smaller graph, and later
   refine if needed. Heavy local edges are attractive contraction targets
   because cutting them would be expensive.
2. **Heterogeneous DAG scheduling.** HEFT ranks tasks by average heterogeneous
   compute cost plus the longest remaining successor path. This gives a cheap
   estimate of scheduling criticality.

The CoopInfer adaptation differs from ordinary graph partitioning because a
large edge is not always something to contract. A large *long-range* dependency
can be a useful cross-device pipeline boundary, whereas a large *local* edge is
usually a good contraction target.

## Fine-node quantities

For a fine node `v` and hardware resource `r`, the cost backend provides
`c_v^r`.

The mean heterogeneous compute cost is

```text
c_bar(v) = mean_r c_v^r
```

The upward rank is

```text
rank_up(v) = c_bar(v) + max_{w in succ(v)} rank_up(w)
```

and the downward rank is defined symmetrically over predecessors. These ranks
provide a critical-path/slack estimate without committing to a placement.

A resource-cost signature is also built:

```text
p_v(r) = c_v^r / sum_j c_v^j
```

For edge `u -> v`, the placement gradient is the total-variation distance
between these signatures:

```text
delta(u,v) = 0.5 * ||p_u - p_v||_1
```

A high gradient means the relative hardware behavior changes across the edge,
so merging the two nodes removes more placement freedom.

## Fine-edge quantities

For edge `e=(u,v)` with payload size `s_e`:

### Topological span

```text
span(e) = (topo_index(v) - topo_index(u)) / (|V| - 1)
```

This is a cheap normalized proxy for whether the dependency escapes far across
the DAG.

### Robust tensor-size score

Tensor sizes are log-scaled against the graph's p95 edge size:

```text
size(e) = clip(log(1+s_e) / log(1+p95(s)), 0, 1)
```

### Communication exposure

```text
exposure(e) = size(e) * span(e)
```

A large tensor that travels far across topological order is a valuable boundary
to retain because it can expose communication/computation overlap.

### Local heavy-edge affinity

```text
local_heavy(e) = size(e) * (1 - span(e))
```

This adapts the heavy-edge intuition: a large tensor on a local dependency is a
good contraction candidate.

## Boundary score

The current generic boundary score combines normalized graph, compute, and
placement features:

```text
B(e) =
    0.25 * span
  + 0.12 * fanout
  + 0.10 * join
  + 0.18 * criticality
  + 0.15 * placement_gradient
  + 0.10 * compute_importance
  + 0.10 * communication_exposure
```

All terms lie in `[0,1]`; the score is clipped to `[0,1]`.

The complementary merge affinity is

```text
A(e) =
    0.45 * (1 - B(e))
  + 0.35 * local_heavy(e)
  + 0.20 * (1 - placement_gradient)
```

Interpretation:

- high `B(e)` -> keep the boundary visible to the scheduler;
- high `A(e)` -> contract the edge into one coarse placement variable.

The initial generic defaults are:

```text
boundary_threshold                  = 0.55
long_range_span_threshold           = 0.10
communication_exposure_threshold    = 0.18
min_merge_affinity                  = 0.45
max_ops_per_group                   = 16
```

An edge is preserved when its boundary score exceeds the threshold, or when it
is simultaneously long-range and communication-exposed.

These are normalized heuristic defaults, not model-specific constants. They are
CLI parameters and should later be swept on representative graphs.

## Safe contraction

Contraction must preserve DAG semantics. It is not sufficient to inspect only
the immediate edge. For example:

```text
a -> b -> c
|         ^
+-> x ----+
```

Contracting `{a,b,c}` would turn the outside path `a -> x -> c` into
`group -> x -> group`, creating a cycle in the quotient graph.

Before extending a group, CoopInfer therefore checks whether an existing group
member reaches an external predecessor of the proposed next node. Such a
contraction is rejected. The final quotient graph is also checked explicitly for
acyclicity.

## Greedy coarsening

Starting in topological order:

1. create a new group at the first unassigned node;
2. consider outgoing edges from the group's current tail;
3. reject I/O boundaries, explicit hard boundaries, placement conflicts,
   preserved dependency edges, low-affinity edges, cost/size-cap violations,
   and cycle-producing contractions;
4. among the remaining candidates, choose the highest merge affinity;
5. extend until no safe candidate remains or `max_ops_per_group` is reached.

Coarse compute cost is initially additive:

```text
c_r(group) = sum_{v in group} c_v^r
```

This is only the first coarse cost. A later **Coarse Cost Refinement** stage can
correct the additive estimate for fusion, cache/locality, launch overhead, and
other block-level effects.

## Why this formulation fits cooperative inference

Ordinary partitioning mostly tries to avoid expensive cuts. Cooperative
inference has a second objective: some cuts are useful because they create
parallel resources and allow communication to overlap with independent compute.
Therefore the policy distinguishes:

```text
large + local      -> strong merge affinity
large + long-range -> strong boundary importance
```

If a model contains a layerwise producer/consumer dependency that enables a
pipeline, it should survive because of its generic graph/cost properties rather
than because the frontend names it specially.

## References / conceptual sources

- G. Karypis and V. Kumar, multilevel graph partitioning / METIS: graph
  contraction, heavy-edge coarsening, coarse solve, and refinement.
- H. Topcuoglu, S. Hariri, and M.-Y. Wu, HEFT/CPOP: heterogeneous task costs and
  upward-rank critical-path scheduling.
- General communication-aware DAG scheduling literature: exact heterogeneous
  DAG scheduling is computationally hard, motivating structured heuristics and
  reduced search spaces.
