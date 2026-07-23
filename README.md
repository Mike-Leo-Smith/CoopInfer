# coopinfer

策略驱动的具身智能边端协同推理系统原型。

## Run

```bash
python -m pip install -e ".[dev]"
python -m coopinfer solve examples/sample_config.json
```

The default install is headless and includes PDF pipeline export. Install the
optional Qt GUI with:

```bash
python -m pip install -e ".[dev,gui]"
python -m coopinfer gui
```

The Python package metadata includes the runtime dependencies and the native build tools used for the C++ extension (`cmake`, `ninja`, `pybind11`, and `scikit-build-core`). A C++ compiler is still required by the platform: MSVC Build Tools on Windows, Xcode Command Line Tools on macOS, or a standard GCC/Clang toolchain on Linux.

You can also launch the GUI module directly:

```bash
python -m coopinfer.app
```

The application stores graph topology, node/edge costs, environment parameters, and scheduling decisions through the JSON schema described in the project brief.

## Prototype Behavior

- Nodes are stored in a `networkx.DiGraph`. Placement is solver output and is
  not required in input JSON. Legacy `x=0/1` fields remain accepted as optional
  warm starts.
- Node IDs remain the stable edge references; optional node names are used as topology display labels.
- Each assignment is evaluated in the native C++ core by expanding the unrolled pipeline into source, compute, and transfer tasks.
- The evaluator schedules serialized Device, Host, and Network resources with greedy ready-list and bounded lookahead modes. Ready tasks are ordered by feasible start time and deterministic rule priorities.
- Cross-device edges become explicit network tasks on one serialized network queue, so network transfers do not overlap.
- Each cross-device transfer costs `latency_ms + size_mb / bandwidth_mb_s * 1000`.
- The optional `batch_transfers` environment flag groups multiple outgoing cross-device transfers from the same source node into one network transaction. The batch pays fixed latency once and still pays bandwidth time for the summed payload size.
- `pipeline_unroll` repeats the same DAG for multiple frames to model pipeline parallelism across consecutive inputs. The evaluator enforces FIFO ordering for the same logical node label across frames, while device, host, and network queues can overlap different labels from different frames.
- For multi-frame schedules, the evaluator also probes a one-frame non-unrolled schedule and packs adjacent linear same-resource operators into internal meta-nodes before rerunning the full unrolled scheduler. Public metrics are expanded back to original node timings.
- As a fallback, the evaluator can tile copies of the one-frame schedule into available Device, Host, and Network gaps by shifting each copied frame right until resource intervals no longer overlap.
- For each candidate assignment, the native core builds the task DAG and critical-path ranks once, simulates a best-of-candidates ensemble of transfer timing, slack retiming, greedy/lookahead scheduler modes, and ready-list priority rules, and keeps the schedule with the lowest configured split loss.
- The evaluator also applies tail-latency-oriented retiming: blocked transfers can be made just-in-time, and a conservative postprocess right-shifts slack work without moving output completions or adding frame-wide gates. This avoids counting early no-op work as the start of a long tail-frame E2E window while preserving inter-frame pipeline overlap.
- Source nodes can define `source_period_ms` and `source_phase_ms`. For unrolled pipelines, frame `f` of that input is released at `source_phase_ms + f * source_period_ms`. These nodes are independent zero-duration input events: they do not consume device/host compute queues and their successive frames may overlap downstream work.
- `latency` is retained for compatibility but means input-excluded pipeline
  makespan divided by `pipeline_unroll`; it is an amortized pipeline-span
  metric, not average per-frame E2E latency.
- Mean-frame latency is the arithmetic mean of per-frame input-excluded E2E
  windows. Max-frame latency is their maximum.
- Initiation interval is the average spacing between consecutive frame
  completions. A 10 Hz workload should normally constrain
  `initiation_interval_limit <= 100` together with a per-frame E2E limit.
- Device, Host, and Network utilization are reported as active time over the same input-excluded pipeline span, not as static sums of assigned costs.
- The objective is an explicit weighted sum: `weight_avg_latency * L_avg + weight_max_latency * L_max + weight_device_utilization * L_util`. Each public weight is in `[0, 1]`. `L_avg` and `L_max` are normalized by all-device/all-host baseline scales; `L_util` is `1 - device_utilization`.
- The solver can run Auto, Enumerate, Random Search, or Simulated Annealing.
  Auto uses enumeration for up to 12 free nodes and simulated annealing beyond
  that. Constrained annealing can traverse infeasible intermediate placements
  while retaining only feasible final results.
- Candidate evaluation is parallelized in the native core. `solver_threads=0` uses the hardware thread count; Simulated Annealing chains synchronize on the shared best result every 32 local steps and keep the best feasible chain result.
- `latency_limit` is an optional amortized pipeline-span cap in milliseconds.
  `max_frame_latency_limit` is an optional worst-frame E2E cap. Use `0` to
  disable either limit; scheduler candidates that satisfy positive limits are
  selected before lower-loss candidates that violate them, and assignments
  with no satisfying schedule are rejected.
- `initiation_interval_limit` is an optional steady-state output interval cap in
  milliseconds. Use `100` for a 10 Hz throughput requirement and `0` to disable.
- The GUI has two topology views: the config DAG before solving, and the solved unrolled pipeline with placement, transfer, and FIFO edges. In solved views, node fill color identifies the frame; device/host placement is shown by the node outline.
- The timeline visualizes the evaluator's actual compute and transfer records, including serialized, batched, and unrolled pipeline transfers. Input releases, transfer bars, and vertical dependency/release lines reuse the same high-contrast frame colors as the filled pipeline nodes.
- Solver search and all schedule/performance evaluation run through a required C++ extension built with CMake/scikit-build-core. Python modules handle validation, JSON IO, GUI wiring, and native-core bindings.

For the full evaluator algorithm, see [docs/native_scheduler.md](docs/native_scheduler.md).

## JSON Environment Fields

```json
{
  "bandwidth": 50.0,
  "latency": 5.0,
  "weight_avg_latency": 0.7,
  "weight_max_latency": 0.3,
  "weight_device_utilization": 0.3,
  "latency_limit": 120.0,
  "max_frame_latency_limit": 180.0,
  "initiation_interval_limit": 100.0,
  "batch_transfers": false,
  "pipeline_unroll": 1,
  "solver_threads": 0,
  "anneal_initial_temp": 1.0,
  "anneal_final_temp": 0.01
}
```

## Headless CLI

Solve files, directories, or glob patterns. Every successful config writes
`result.json` and, by default, a multi-page `pipeline.pdf`; the output root also
contains `summary.json` and `summary.csv`.

```bash
python -m coopinfer solve configs/ \
  --output-dir results/ \
  --scheduler auto \
  --iterations 3000
```

CLI values override the environment embedded in every input config:

```bash
python -m coopinfer solve configs/*.json \
  --scheduler simulated_annealing \
  --bandwidth-mbps 510 \
  --latency 0 \
  --max-frame-latency-limit 95 \
  --initiation-interval-limit 100 \
  --pipeline-unroll 8 \
  --batch-transfers
```

`--bandwidth` uses the native config unit MB/s. `--bandwidth-mbps` accepts
Mbit/s and divides by eight. Other overrides include objective weights,
latency limits, pipeline unroll, transfer batching, solver threads, annealing
temperatures, iteration count, and seed. Use `--pipeline-format none` to skip
PDF creation during large sweeps.

## Pi0.6-like Config Sweep

The generator creates `{Images, Observations} -> ViT -> 34x LLM -> DiT ->
Actions`, including every LLM-to-DiT KV-cache edge. Host time receives
compression and batch scaling; device time receives compression scaling only.
No generated node contains an `x` placement field. Transfer payloads scale
with batch size; ViT/LLM/DiT model edges additionally scale with the selected
compression strategy's transmission ratio.

```bash
python tools/generate_pi06_configs.py \
  --output-dir generated/pi06 \
  --batch-sizes 1-8 \
  --compressions none,conservative,normal,aggressive \
  --network-profiles \
    sparklink2-single,sparklink2-dual,sparklink2-single-3x,sparklink2-single-4x
```

The StarFlash profiles apply 85% link efficiency: nominal 600/1200/1800/2400
Mbit/s become effective 510/1020/1530/2040 Mbit/s.

## Example

Load one of the bundled examples from the GUI:

- `examples/sample_config.json`: small branching DAG for quick checks.
- `examples/moderate_multisensor_dag.json`: medium RGB/depth/IMU pipeline with source periods and pipeline unroll.
- `examples/complex_embodied_dag.json`: larger embodied perception, fusion, and control DAG.

## Test

```bash
python -m pytest
```

## Wheels

GitHub Actions builds wheel artifacts for Linux, Windows, and macOS through `cibuildwheel` on pushes, pull requests, and manual dispatches.
