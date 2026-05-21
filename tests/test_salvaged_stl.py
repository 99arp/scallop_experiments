from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import jax.numpy as jnp

from stl.formula import Always, Identity
from stl.monitor import STLDistanceMonitor
from stl.visualizer import STLRuntimeVisualizer


class SalvagedSTLTests(unittest.TestCase):
    def test_formula_always_with_infinite_interval(self) -> None:
        formula = Always(Identity(), interval=[0, jnp.inf])

        result = formula.robustness_trace(jnp.array([1.0, 2.0], dtype=jnp.float32))

        self.assertEqual(result.shape, (2,))
        self.assertTrue(
            jnp.allclose(result, jnp.array([1.0, -1.0e9], dtype=jnp.float32))
        )

    def test_drone_rules_load_and_evaluate(self) -> None:
        monitor = STLDistanceMonitor(threshold_m=100.0, window_steps=2)

        first = monitor.add_sample(
            jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
            jnp.array([120.0, 0.0, 15.0], dtype=jnp.float32),
            jnp.array([0.0, 130.0, 15.0], dtype=jnp.float32),
        )
        second = monitor.add_sample(
            jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
            jnp.array([125.0, 0.0, 15.0], dtype=jnp.float32),
            jnp.array([0.0, 135.0, 15.0], dtype=jnp.float32),
        )

        self.assertFalse(first.is_ready)
        self.assertTrue(second.is_ready)
        self.assertEqual(monitor.compiled_rules[0].name, "origin_to_drone1")
        self.assertTrue(second.ok_now)
        self.assertTrue(second.ok_window)

    def test_graphviz_visualizer_writes_artifacts(self) -> None:
        monitor = STLDistanceMonitor(threshold_m=100.0, window_steps=2)
        result = None
        for i in range(2):
            result = monitor.add_sample(
                jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
                jnp.array([120.0 + i, 0.0, 15.0], dtype=jnp.float32),
                jnp.array([0.0, 130.0 + i, 15.0], dtype=jnp.float32),
            )

        with TemporaryDirectory() as tmp_dir:
            visualizer = STLRuntimeVisualizer(output_dir=tmp_dir)
            artifacts = visualizer.render(
                compiled_rules=monitor.compiled_rules,
                combine_operator=monitor.combine_operator,
                result=result,
            )

            self.assertTrue(artifacts.dot_path.is_file())
            self.assertTrue(artifacts.svg_path.is_file())
            self.assertTrue(artifacts.json_path.is_file())
            self.assertIn("origin_to_drone1", artifacts.dot_path.read_text())
            snapshot = json.loads(Path(artifacts.json_path).read_text())
            self.assertTrue(snapshot["is_ready"])
            self.assertGreater(len(snapshot["rules"]), 0)


if __name__ == "__main__":
    unittest.main()
