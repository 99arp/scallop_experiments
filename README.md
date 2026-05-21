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
- `tests/test_salvaged_stl.py`: smoke tests proving the salvage works without the old stack.

## Local Salvage Test

Run this from `/home/qnc/scallop`:

```bash
.venv/bin/python -m unittest -v tests.test_salvaged_stl
```

Successful salvage looks like:

```text
Ran 3 tests

OK
```

Those tests verify that:

- the salvaged `formula.py` still evaluates STL robustness;
- `drone_rules/drone_rules.yaml` loads and compiles into rules;
- synthetic drone poses evaluate through the monitor;
- the Graphviz visualizer writes `.dot`, `.svg`, and `.json` artifacts.

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
  -p visualization_dir:=/tmp/stl_viz
```

If salvage is successful under ROS, the node logs first-pose messages for each
topic, then rule robustness lines once the sample window is full. The live
Graphviz outputs are written to the configured visualization directory.
