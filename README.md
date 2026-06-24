# coopinfer

策略驱动的具身智能边端协同推理系统原型。

## Run

```bash
python -m pip install -e ".[dev]"
python -m coopinfer
```

You can also launch the GUI module directly:

```bash
python -m coopinfer.app
```

The application stores graph topology, node/edge costs, environment parameters, and scheduling decisions through the JSON schema described in the project brief.

## Prototype Behavior

- Nodes are stored in a `networkx.DiGraph`; scheduling assignment `x=0` means Device and `x=1` means Host.
- Node IDs remain the stable edge references; optional node names are used as topology display labels.
- The evaluator walks the DAG in topological order and enforces one execution queue per device through `time_dev_ready` and `time_host_ready`.
- Cross-device transfers use one serialized network queue, so network transfers do not overlap.
- Each cross-device transfer costs `latency_ms + size_mb / bandwidth_mb_s * 1000`.
- The optional `batch_transfers` environment flag groups multiple outgoing cross-device transfers from the same source node into one network transaction. The batch pays fixed latency once and still pays bandwidth time for the summed payload size.
- `pipeline_unroll` repeats the same DAG for multiple frames to model pipeline parallelism across consecutive inputs. The evaluator enforces FIFO ordering for the same logical node label across frames, while device, host, and network queues can overlap different labels from different frames.
- Device utilization is reported as `device_active_time / total_pipeline_makespan`, not as a static sum of assigned node costs.
- The solver can run Auto, Enumerate, Random Search, or Simulated Annealing. Auto uses enumeration for up to 12 free nodes and random search beyond that.
- `latency_limit` is an optional E2E latency cap in milliseconds. Use `0` to disable it; assignments above a positive limit are rejected.
- The GUI has two topology views: the config DAG before solving, and the solved unrolled pipeline with placement, transfer, and FIFO edges.
- The timeline visualizes the evaluator's actual compute and transfer records, including serialized, batched, and unrolled pipeline transfers.

## JSON Environment Fields

```json
{
  "bandwidth": 50.0,
  "latency": 5.0,
  "weight_latency": 0.7,
  "latency_limit": 120.0,
  "batch_transfers": false,
  "pipeline_unroll": 1
}
```

## Example

Load `examples/sample_config.json` from the GUI to try a branching DAG with fixed device input/output nodes.

## Test

```bash
python -m pytest
```
