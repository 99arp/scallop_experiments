# Scallop STL Salvage

This folder now contains the salvaged STL monitor pieces from
`/home/qnc/Desktop/command_agent` and intentionally leaves out the reasoning,
oPIEC, persistence, frontend, and export layers.

## Salvaged Pieces

- `drone_rules/drone_rules.yaml`: declarative drone distance and height rules.
- `stl/formula.py` and `stl/utils.py`: STL formula primitives and robustness helpers.
- `stl/monitor.py`: YAML rule loader plus pure in-memory STL evaluator.
- `stl/visualizer.py` and `stl/viz.py`: Graphviz `dot` renderer for the live STL rule graph.
- `stl/ros_monitor.py` and `stl/main.py`: thin ROS2 listener around the STL monitor.
- `stl_prob/`: direct STL robustness to Scallop fact conversion and live export.
- `event_calculus/`: probabilistic Event Calculus over exported STL fact streams.
- `tests/test_salvaged_stl.py`: smoke tests proving the salvage works without the old stack.

## Local Salvage Test

Run this from `/home/qnc/scallop`:

```bash
.venv/bin/python -m unittest -v tests.test_salvaged_stl tests.test_stl_prob tests.test_event_calculus
```

Successful salvage looks like:

```text
Ran 12 tests

OK
```

Those tests verify that:

- the salvaged `formula.py` still evaluates STL robustness;
- `drone_rules/drone_rules.yaml` loads and compiles into rules;
- synthetic drone poses evaluate through the monitor;
- the Graphviz visualizer writes `.dot`, `.svg`, and `.json` artifacts.
- STL predicate probabilities can be exported as Scallop facts.
- exported STL facts can be consumed by the Event Calculus module.

## Manual Graphviz Check

The unit test writes artifacts to a temporary directory. To generate visible
artifacts in the repo:

```bash
.venv/bin/python - <<'PY'
import jax.numpy as jnp
from stl.monitor import STLDistanceMonitor
from stl.visualizer import STLRuntimeVisualizer

monitor = STLDistanceMonitor(threshold_m=100.0, window_steps=2)
result = None
for i in range(2):
    result = monitor.add_sample(
        jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.array([120.0 + i, 0.0, 15.0], dtype=jnp.float32),
        jnp.array([0.0, 130.0 + i, 15.0], dtype=jnp.float32),
    )

artifacts = STLRuntimeVisualizer().render(
    compiled_rules=monitor.compiled_rules,
    combine_operator=monitor.combine_operator,
    result=result,
)
print(artifacts)
PY
```

Expected files:

- `stl/stl_viz/stl_live.dot`
- `stl/stl_viz/stl_live.svg`
- `stl/stl_viz/stl_live.json`

The visualizer uses the system `dot` binary. On this machine it was found at
`/usr/bin/dot`.

## STL Predicate Probability Facts

`stl_prob` converts STL robustness (`rho`) into a value in `[0, 1]` using:

```text
p = sigmoid(scale * (rho - midpoint))
```

By default it uses each rule's instantaneous robustness. That keeps the
probability conversion separate from the STL temporal operator, instead of
double-smoothing a window robustness value.

Example:

```bash
.venv/bin/python - <<'PY'
import jax.numpy as jnp
from stl.monitor import STLDistanceMonitor
from stl_prob import monitor_result_to_facts

monitor = STLDistanceMonitor(threshold_m=100.0, window_steps=2)
result = None
for i in range(2):
    result = monitor.add_sample(
        jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
        jnp.array([120.0 + i, 0.0, 15.0], dtype=jnp.float32),
        jnp.array([0.0, 130.0 + i, 15.0], dtype=jnp.float32),
    )

facts = monitor_result_to_facts(result)
for fact in facts:
    print(fact.to_event_calculus())
PY
```

Example output:

```text
0.999999999::happensAt(stl_predicate(rule,origin_to_drone1,true),1).
```

The fields are:

- event predicate: `happensAt`
- event term: `stl_predicate(source_kind, name, value)`
- `sample_index` as the EC time point
- probability prefix from the sigmoid

## Event Calculus Over Facts

`event_calculus/` reads the exported probabilistic `happensAt(...)` stream and
applies a Tensor-pEC-style tensor backend implemented with JAX:

- `happensAt(stl_predicate(...,true),T)` contributes to `initiatedAt`;
- `happensAt(stl_predicate(...,false),T)` contributes to `terminatedAt`;
- input events are grounded into initiation and termination tensors;
- `holdsAt` is computed with a vectorized JAX `lax.scan` recurrence;
- intervals are extracted from `holdsAt(...)` using a probability threshold.

Run EC over a live fact export:

```bash
python -m event_calculus \
  stl_prob/exports/stl_live_facts.scl \
  --output stl_prob/exports/stl_live_ec.scl \
  --hold-threshold 0.5
```

Output facts look like:

```text
0.999999999::holdsAt(stl_predicate(rule,origin_to_drone1)=true,1).
holdsFor(stl_predicate(rule,origin_to_drone1)=true,[(1,12)]).
```

`holdsFor` uses `[start_sample_index, end_sample_index)` intervals.

During a ROS live run, ready STL samples are exported automatically to:

```text
stl_prob/exports/stl_live_facts.scl
```

The export file is reset when the node starts unless
`scallop_fact_reset_on_start:=false` is passed.

## ROS2 Run

After sourcing your ROS2 environment, run:

```bash
python -m stl
```

The node subscribes to:

- `/origin/pose`
- `/Drone_1/pose`
- `/Drone_2/pose`

Useful ROS parameters:

```bash
python -m stl --ros-args \
  -p threshold_m:=100.0 \
  -p window_sec:=5.0 \
  -p sample_rate_hz:=10.0 \
  -p rules_path:=/home/qnc/scallop/drone_rules/drone_rules.yaml \
  -p visualization_dir:=/tmp/stl_viz \
  -p scallop_fact_export_path:=/tmp/stl_live_facts.scl
```

Scallop fact export parameters:

- `scallop_fact_export_enabled`: defaults to `true`
- `scallop_fact_export_path`: defaults to `stl_prob/exports/stl_live_facts.scl`
- `scallop_fact_signal`: `instant` by default, or `window`
- `scallop_fact_scale`: sigmoid scale, default `1.0`
- `scallop_fact_midpoint`: sigmoid midpoint, default `0.0`
- `scallop_fact_include_groups`: defaults to `true`
- `scallop_fact_reset_on_start`: defaults to `true`

If salvage is successful under ROS, the node logs first-pose messages for each
topic, then rule robustness lines and `scallop_facts_exported=...` once the
sample window is full. The live Graphviz outputs are written to the configured
visualization directory.
