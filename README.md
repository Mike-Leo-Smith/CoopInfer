# coopinfer

策略驱动的具身智能边端协同推理系统原型。

## Run

```bash
python -m pip install -e ".[dev]"
python -m coopinfer
```

The Python package metadata includes the runtime dependencies and the native build tools used for the C++ extension (`cmake`, `ninja`, `pybind11`, and `scikit-build-core`). A C++ compiler is still required by the platform: MSVC Build Tools on Windows, Xcode Command Line Tools on macOS, or a standard GCC/Clang toolchain on Linux.

You can also launch the GUI module directly:

```bash
python -m coopinfer.app
```

The application stores graph topology, node/edge costs, environment parameters, and scheduling decisions through the JSON schema described in the project brief.

## Prototype Behavior

- Nodes are stored in a `networkx.DiGraph`; scheduling assignment `x=0` means Device and `x=1` means Host.
- Node IDs remain the stable edge references; optional node names are used as topology display labels.
- Each assignment is evaluated in the native C++ core by expanding the unrolled pipeline into source, compute, and transfer tasks.
- The evaluator is a work-conserving list scheduler over serialized Device, Host, and Network resources. It schedules ready tasks by earliest feasible start time, then applies deterministic tie-break rules.
- Cross-device edges become explicit network tasks on one serialized network queue, so network transfers do not overlap.
- Each cross-device transfer costs `latency_ms + size_mb / bandwidth_mb_s * 1000`.
- The optional `batch_transfers` environment flag groups multiple outgoing cross-device transfers from the same source node into one network transaction. The batch pays fixed latency once and still pays bandwidth time for the summed payload size.
- `pipeline_unroll` repeats the same DAG for multiple frames to model pipeline parallelism across consecutive inputs. The evaluator enforces FIFO ordering for the same logical node label across frames, while device, host, and network queues can overlap different labels from different frames.
- For each candidate assignment, the native core builds the task DAG and critical-path ranks once, simulates `CriticalPath`, `DeviceFirst`, and `FifoReady` scheduling rules, and keeps the schedule with the lowest configured split loss.
- The evaluator also applies tail-latency-oriented retiming: blocked transfers can be made just-in-time, and a conservative postprocess right-shifts slack work without moving output completions or adding frame-wide gates. This avoids counting early no-op work as the start of a long tail-frame E2E window while preserving inter-frame pipeline overlap.
- Source nodes can define `source_period_ms` and `source_phase_ms`. For unrolled pipelines, frame `f` of that input is released at `source_phase_ms + f * source_period_ms`. These nodes are independent zero-duration input events: they do not consume device/host compute queues and their successive frames may overlap downstream work.
- End-to-end latency excludes source/input events. It is measured from the first non-input compute or transfer event to the last non-input output node finish, then amortized over `pipeline_unroll`.
- Max-frame latency is the worst per-frame version of the same input-excluded window.
- Device utilization is reported as active device compute time over the same input-excluded pipeline span, not as a static sum of assigned node costs.
- The objective is an explicit weighted sum: `weight_avg_latency * L_avg + weight_max_latency * L_max + weight_device_utilization * L_util`. Each weight is in `[0, 1]`. `L_avg` and `L_max` are normalized by all-device/all-host baseline scales; `L_util` is `1 - device_utilization`.
- The solver can run Auto, Enumerate, Random Search, or Simulated Annealing. Auto uses enumeration for up to 12 free nodes and random search beyond that.
- `latency_limit` is an optional amortized E2E latency cap in milliseconds. `max_frame_latency_limit` is an optional worst-frame E2E cap. Use `0` to disable either limit; assignments above a positive limit are rejected.
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
  "batch_transfers": false,
  "pipeline_unroll": 1
}
```

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
