# coopinfer

策略驱动的具身智能边端协同推理系统原型。

## Run

```bash
python -m pip install -e ".[dev]"
python -m coopinfer
```

The application stores graph topology, node/edge costs, environment parameters, and scheduling decisions through the JSON schema described in the project brief.

## Prototype Behavior

- Nodes are stored in a `networkx.DiGraph`; scheduling assignment `x=0` means Device and `x=1` means Host.
- Node IDs remain the stable edge references; optional node names are used as topology display labels.
- The evaluator walks the DAG in topological order and enforces one execution queue per device through `time_dev_ready` and `time_host_ready`.
- Cross-device edges pay `latency_ms + size_mb / bandwidth_mb_s * 1000`.
- The solver can run Auto, Enumerate, Random Search, or Simulated Annealing. Auto uses enumeration for up to 12 free nodes and random search beyond that.

## Example

Load `examples/sample_config.json` from the GUI to try a branching DAG with fixed device input/output nodes.

## Test

```bash
python -m pytest
```
