# Native Scheduler

CoopInfer evaluates every candidate placement in the C++ native core. The
Python layer validates inputs and converts the NetworkX graph into compact
arrays; all schedule construction, timing simulation, metric extraction,
objective scoring, and solver search happen in `src/coopinfer/cpp/core.cpp`.

## Inputs

For a fixed assignment `x`, the evaluator receives:

- DAG nodes with device and host compute costs.
- DAG edges with transfer payload sizes.
- Source cadence fields `source_period_ms` and `source_phase_ms`.
- Environment bandwidth, fixed transfer latency, batching flag, and
  `pipeline_unroll`.
- Objective weights for average E2E latency, max-frame E2E latency, initiation
  interval, and device utilization. The public GUI/model path exposes average
  E2E, max-frame E2E, and device-utilization weights; the native parser accepts
  `weight_initiation_interval` for direct/internal callers and otherwise
  defaults it to zero.

Placement values are binary: `x=0` means Device and `x=1` means Host. Fixed
device nodes are enforced by the solver before evaluation.

## Expanded Task DAG

The native evaluator expands the single-frame DAG into `pipeline_unroll` frames.
Each expanded node is represented as a task. Cross-device edges are also
represented as explicit network tasks, so the evaluator schedules compute and
communication with the same dependency machinery.

### Source Tasks

Input/source graph nodes are nodes with no incoming edges. Their expanded tasks
are zero duration and do not consume Device or Host compute resources.

For frame `f`, a source task is released at:

```text
source_phase_ms + f * source_period_ms
```

Source releases can delay downstream work, but source events themselves are
excluded from E2E latency windows.

### Compute Tasks

Every non-source graph node becomes one compute task per frame:

- Device task duration: `c_dev`
- Host task duration: `c_host`

Device and Host are each modeled as one serialized queue. Compute tasks on
different resources may overlap when their dependencies allow it.

### Network Tasks

Each cross-device edge becomes one Network task unless batching combines
multiple outgoing cross-device edges from the same source operation.

Transfer duration is:

```text
latency_ms + size_mb / bandwidth_mb_s * 1000
```

The Network resource is a single serialized queue, so transfers never overlap.

When `batch_transfers` is enabled, multiple outgoing cross-device edges from the
same source operation are grouped into one Network task. The batch pays fixed
latency once and bandwidth time for the summed payload.

## Dependencies

The expanded task graph contains these dependency classes:

- Same-device data dependency: source compute task must finish before target
  compute task can run.
- Cross-device data dependency: source compute/source task -> Network task ->
  target compute task.
- Same-label FIFO dependency: for every non-source graph node, frame `f` must
  finish before the same graph node in frame `f+1` can run.

The FIFO dependency prevents younger frames from overtaking older frames at the
same logical operator. This protects frame order and avoids schedules that look
good in aggregate but produce pathological control-tail latency.

## Packed Single-Frame Stages

For multi-frame evaluations, the native core also tests a packed-stage variant
of the task graph. This variant is inferred from a non-unrolled, one-frame
schedule for the same assignment:

1. Run the normal one-frame evaluator.
2. Find adjacent non-source operators where the first operator's finish time is
   exactly the second operator's start time.
3. Pack only linear same-resource pairs/chains: the upstream operator must have
   one outgoing edge, the downstream operator must have one incoming edge, and
   both operators must be assigned to the same serialized resource.
4. Build the full `pipeline_unroll` task graph using each packed chain as one
   internal scheduler task.

The packed task keeps the sum of the original operator durations and preserves
all original DAG, transfer, source-release, and same-label FIFO constraints at
the packed-stage boundary. Metrics are expanded back to the original operator
IDs, so GUI timelines and API results still report per-node start/finish times
and original transfer edges.

This packed graph is an additional candidate, not a replacement. If no adjacent
linear chain is found, or if the packed schedule does not improve the configured
objective/tie-breaks, the evaluator returns the normal unpacked schedule.

## Tail-Latency Retiming

The evaluator simulates several deterministic variants for the same assignment
and chooses the one with the best configured loss. It also applies a conservative
postprocess sweep to reduce artificial tail-frame latency without blocking
legitimate inter-frame pipeline overlap.

### Deferred Blocked Transfers

A cross-device transfer can be ready before the target frame can use its data.
If the target compute task is also blocked by other prerequisites, starting the
transfer early can create a misleadingly early non-input event for that frame.
That inflates the measured frame E2E window without improving output completion.

The deferred-transfer variant adds dependencies from the target's other
non-network prerequisites to the transfer. This turns the transfer into
just-in-time work: it starts only when the downstream frame is close enough to
consume it.

This is most visible when an input source crosses devices into a slow serialized
compute stage. Earlier behavior could start frame-2 transfer near frame-2 input
release, then wait a long time for the compute queue. The new variant waits
until that compute slot is actually reachable, reducing max-frame E2E without
changing the output finish time.

### Postprocess Right-Shift Sweep

Fast side branches can also start very early and then wait at a join for a slow
branch. A frame-wide gate fixes that symptom but breaks pipeline parallelism by
forcing younger frames to wait for older frame outputs. CoopInfer does not add
that gate.

Instead, after the work-conserving list schedule is built, the native core runs a
backward right-shift sweep over the scheduled task order:

1. Output compute tasks are anchored; their finish times are not moved.
2. Source tasks are anchored at their release times.
3. Network tasks and same-frame join side-branch compute tasks may move later.
4. A moved task must still finish before every successor starts.
5. A moved task must still finish before the next task on the same serialized
   resource starts.
6. A moved task must still start after its release time and all predecessor
   finishes.

The sweep is a tiling/compaction pass over the already feasible schedule: it
removes idle gaps that only make a frame appear to have started earlier, but it
does not change output completion times, resource order, or data dependencies.
Linear feeder stages remain left-packed, so a Device stage from frame `f+1` can
overlap a Host output from frame `f`.

## Scheduling Rules

For every timing, retiming, and scheduler-mode variant, the evaluator runs seven
deterministic ready-task rules over the same expanded task graph.

### CriticalPath

The evaluator computes an upward rank for each task:

```text
rank(task) = duration(task) + max(rank(successor))
```

The CriticalPath rule prioritizes higher rank and adds a large older-frame aging
bonus:

```text
priority = rank + 1_000_000 * max(0, unroll - 1 - frame)
```

This pushes the scheduler to drain older frames and critical downstream work.

### DeviceFirst

The DeviceFirst rule prefers:

1. Source release tasks.
2. Device compute tasks.
3. Network tasks.
4. Host compute tasks.

Within each class it uses task duration and rank as deterministic tie-breakers.
This rule is useful when the objective rewards Device utilization and when
keeping the on-device queue occupied is beneficial.

### HostFirst

The HostFirst rule mirrors DeviceFirst for host-heavy placements: source tasks
come first, then Host compute, then Network, then Device compute.

### NetworkFirst

The NetworkFirst rule prioritizes serialized cross-device transfers when they
tie with other ready work. This helps the ensemble test schedules that drain the
network queue early instead of letting same-time compute decisions hide transfer
contention.

### OutputFirst

The OutputFirst rule gives output compute tasks a large bonus, then applies the
same older-frame aging used by CriticalPath. This favors schedules that close
frames promptly when the objective is sensitive to worst-frame latency.

### Throughput

The Throughput rule gives younger frames a large tie-break bonus before rank.
This lets the ensemble test schedules that favor steady-state pipeline
initiation interval over strictly draining older frames first.

### FifoReady

The FifoReady rule uses task creation order as the priority tie-breaker. Combined
with earliest feasible start time, this behaves like a stable first-ready list
scheduler.

## Event Loop

For a greedy variant and rule, the simulator maintains:

- `pending`: number of unsatisfied dependencies per task.
- `ready_time`: earliest data/release-ready time per task.
- `time_dev_ready`: next Device availability.
- `time_host_ready`: next Host availability.
- `time_network_ready`: next Network availability.
- `start` and `finish` arrays for compute/source tasks.
- transfer records for Network tasks.

At each step:

1. Put all zero-pending tasks into the ready set.
2. For every ready task, compute feasible start:

   ```text
   start = max(ready_time[task], resource_ready_time[task], release[task])
   ```

3. Pick the ready task with the smallest feasible start time.
4. If multiple tasks have the same feasible start, apply the active scheduling
   rule priority.
5. Schedule the task, update the matching resource ready time, and record the
   task or transfer interval.
6. Propagate the finish time to successors. A successor enters the ready set
   when all dependencies are satisfied.

This greedy event loop is work-conserving with respect to the selected variant:
it does not idle a resource for a high-priority task whose data is not ready when
another ready task can start earlier.

## Bounded Lookahead

The ensemble also runs a deterministic bounded lookahead scheduler for each
timing, retiming, and ready-rule combination. It uses the same ready-task start
times and rule priorities as the greedy loop, but keeps several partial
schedules instead of committing immediately to one ready task.

At each scheduling step:

1. Sort ready candidates by earliest feasible start time, then active rule
   priority, then task id.
2. For each current beam state, branch on the first three ready candidates.
3. Prune the merged states to a beam width of four.
4. Rank partial states by a lower bound on serialized resource completion:

   ```text
   max(
     time_dev_ready + remaining_device_work,
     time_host_ready + remaining_host_work,
     time_network_ready + remaining_network_work
   )
   ```

5. Break partial-state ties by lower current resource readiness, lower summed
   starts, higher summed priorities, then scheduled order.

The lookahead brancher can explore a small number of non-greedy orderings, but
it does not change dependencies, resource serialization, source releases, metric
extraction, or objective scoring. Each completed beam state is materialized into
the same metrics as a greedy schedule. If right-shift retiming is active, the
same postprocess sweep is applied before scoring.

## Winner Selection

The evaluator currently tests all combinations of:

- `defer_blocked_transfers = false/true`
- `right_shift_slack = false/true`
- `lookahead_beam = false/true`
- `CriticalPath`, `DeviceFirst`, `HostFirst`, `NetworkFirst`, `OutputFirst`,
  `Throughput`, `FifoReady`

That gives 56 deterministic unpacked schedule candidates per candidate
assignment. For multi-frame packable graphs, the same 56 combinations are also
tested on the packed-stage task graph inferred from the one-frame schedule. The
native core computes the configured split loss for each schedule:

```text
loss =
  weight_avg_latency * L_avg
+ weight_max_latency * L_max
+ weight_initiation_interval * L_ii
+ weight_device_utilization * L_util
```

`L_avg`, `L_max`, and `L_ii` are normalized by all-device/all-host baseline
scales. `L_util` is `1 - device_utilization`. Public GUI/model calls leave
`weight_initiation_interval` at zero while still reporting the initiation
interval metric and loss term.

The schedule with the lowest loss wins, so the inner evaluator follows the same
global split objective used by the solver. If losses tie, the evaluator picks
lower max-frame latency, then lower average latency, then lower initiation
interval, then higher Device utilization.

## Metric Extraction

Latency excludes input/source events.

For each frame:

1. `frame_work_start` is the earliest non-source compute start or Network
   transfer start in that frame.
2. `frame_finish` is the latest non-source output-node finish in that frame.
3. `frame_latency = frame_finish - frame_work_start`.

The reported max-frame latency is:

```text
max_frame_latency = max(frame_latency)
```

The reported average E2E latency is amortized over the unrolled pipeline:

```text
latency = (latest_frame_finish - earliest_frame_work_start) / pipeline_unroll
```

For `pipeline_unroll = 1`, this is exactly the wall-clock E2E window from the
first non-input work to the last output finish. For larger unroll factors, it is
the amortized pipeline latency per frame.

Device, Host, and Network utilization are active time over the same
input-excluded pipeline span. Device utilization also remains the utilization
term used by the current objective.

## Solver Integration

The solver calls the evaluator for each candidate assignment. Enumeration,
random search, and simulated annealing all use the same native evaluator and the
same objective/constraint logic.

Candidate evaluation is parallelized inside the native core. Enumeration splits
the assignment mask space across worker threads and then reduces the exact best
feasible result. Random search and simulated annealing run independent per-thread
chains with deterministic mixed seeds; simulated annealing uses the configured
initial and final temperatures for each chain schedule. `solver_threads = 0`
selects the hardware thread count.

Positive `latency_limit` rejects assignments whose amortized E2E latency exceeds
the limit. Positive `max_frame_latency_limit` rejects assignments whose worst
single-frame E2E exceeds the limit. A value of `0` disables the corresponding
limit.
