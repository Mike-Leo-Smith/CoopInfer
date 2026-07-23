# Pi0.6-like 10 Hz study on 960PR-Lite + 1920

## Scope and acceptance criteria

- Topology: `{Images, Observations} -> ViT -> 34x LLM -> DiT -> Actions`.
- Every LLM layer also sends a KV-cache edge to DiT.
- Batch sizes: 1 through 8.
- Compression: none, conservative, normal, and aggressive.
- StarFlash 2.0 nominal bandwidths: 600, 1200, 1800, and 2400 Mbit/s.
- Effective bandwidth applies the requested 85% factor: 510, 1020, 1530,
  and 2040 Mbit/s.
- A result is feasible at 10 Hz when initiation interval is at most 100 ms and
  worst-frame E2E latency is at most 95 ms.
- Large graphs use simulated annealing when `scheduler=auto`. The boundary
  cases were repeated with 256 evaluations.

Host compute time receives both compression and batch scaling. Device compute
time receives compression scaling only, following the requested model. Edge
payloads scale with batch size; ViT/LLM/DiT edges additionally scale with the
compression strategy's transmission ratio.

## Result

The feasible batch boundary is unchanged across all four link profiles.

| Compression | 600 Mbit/s | 1200 Mbit/s | 1800 Mbit/s | 2400 Mbit/s | Placement at boundary |
| --- | ---: | ---: | ---: | ---: | --- |
| None | 3 | 3 | 3 | 3 | Inputs/device -> ViT+LLM+DiT/host -> Actions/device |
| Conservative | 7 | 7 | 7 | 7 | Inputs/device -> ViT+LLM+DiT/host -> Actions/device |
| Normal | >=8 | >=8 | >=8 | >=8 | All device |
| Aggressive | >=8 | >=8 | >=8 | >=8 | All device |

`>=8` means that batch 8 is feasible at the scan cap; it is not a proof that
batch 8 is the physical maximum. Under the requested scaling assumptions,
normal and aggressive compression make the all-device placement independent
of batch size.

At the two host-assisted boundaries:

| Effective link | None, batch 3 E2E | Conservative, batch 7 E2E |
| ---: | ---: | ---: |
| 510 Mbit/s | 86.60 ms | 93.53 ms |
| 1020 Mbit/s | 85.66 ms | 91.34 ms |
| 1530 Mbit/s | 85.35 ms | 90.61 ms |
| 2040 Mbit/s | 85.19 ms | 90.25 ms |

Increasing effective bandwidth from 510 to 2040 Mbit/s improves conservative
batch-7 E2E by only 3.28 ms and does not make batch 8 feasible. The boundary is
therefore compute-limited rather than network-limited.

## Current recommendation

- If host offload is required, use conservative compression and batch 7. One
  600 Mbit/s StarFlash 2.0 link is sufficient under the modeled payloads.
- If all-device execution is allowed and the compression quality is acceptable,
  normal or aggressive compression reaches at least batch 8 within this scan.
- Do not buy a 3x or 4x link solely to increase batch size for the current
  placement: the modeled batch boundary does not move.

## Future link requirements for batch 8 at 10 Hz

The following values use the requested 85% realized-link factor and then add
20% engineering headroom. They are nominal link recommendations, not measured
protocol limits.

| Candidate cross-device cut | Full transmission | Aggressive transmission |
| --- | ---: | ---: |
| Raw inputs and actions only | 0.036 Gbit/s | 0.036 Gbit/s |
| ViT output to host LLM+DiT | 1.54 Gbit/s | 0.77 Gbit/s |
| One LLM inter-layer boundary | 3.31 Gbit/s | 1.65 Gbit/s |
| All 34 LLM KV-cache edges to remote DiT | 22.50 Gbit/s | 11.25 Gbit/s |

Consequences:

- A nominal 1.8 Gbit/s link is the practical floor for a batch-8
  ViT-device/LLM-host split without aggressive transmission compression.
- A nominal 3.3 Gbit/s link is needed for one uncompressed LLM-layer boundary;
  the 2.4 Gbit/s 4x profile remains insufficient.
- Streaming all 34 KV-cache edges to a remote DiT is not a sensible StarFlash
  target. Co-locate LLM and DiT, aggregate the cache, or change the interface.

## Metric clarification

The legacy `latency` field is an amortized pipeline span, not the mean of
per-frame E2E latency. With eight unrolled frames at a 100 ms source period,
an all-device aggressive case can report:

```text
amortized span = (7 * 100 ms + 41.48 ms) / 8 = 92.68 ms/frame
mean frame E2E = 41.48 ms
max frame E2E  = 41.48 ms
```

This is why the old UI could show "average 92.68 ms" above "max frame
41.48 ms". The CLI, PDF, and GUI now label the 92.68 ms value as amortized
pipeline span and report mean-frame E2E separately.
