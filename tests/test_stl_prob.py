from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import jax.numpy as jnp

from stl.monitor import STLDistanceMonitor
from stl_prob import (
    ProbabilityConfig,
    ScallopFactExporter,
    monitor_result_to_facts,
    render_scallop_facts,
    sigmoid_probability,
)


class STLProbTests(unittest.TestCase):
    def test_sigmoid_probability_maps_rho_to_unit_interval(self) -> None:
        self.assertAlmostEqual(sigmoid_probability(0.0), 0.5)
        self.assertGreater(sigmoid_probability(3.0), 0.5)
        self.assertLess(sigmoid_probability(-3.0), 0.5)
        self.assertGreaterEqual(sigmoid_probability(-1000.0), 0.0)
        self.assertLessEqual(sigmoid_probability(1000.0), 1.0)

    def test_monitor_result_to_scallop_facts_uses_sample_index(self) -> None:
        monitor = STLDistanceMonitor(threshold_m=100.0, window_steps=2)
        result = None
        for i in range(2):
            result = monitor.add_sample(
                jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
                jnp.array([120.0 + i, 0.0, 15.0], dtype=jnp.float32),
                jnp.array([0.0, 130.0 + i, 15.0], dtype=jnp.float32),
                sample_time_ns=10_000 + i,
            )

        facts = monitor_result_to_facts(result)
        rendered = render_scallop_facts(facts)

        self.assertGreater(len(facts), 0)
        self.assertEqual(facts[0].source_kind, "rule")
        self.assertEqual(facts[0].name, "origin_to_drone1")
        self.assertEqual(facts[0].sample_index, 1)
        self.assertIn('stl_predicate_probability("rule", "origin_to_drone1", 1, true,', rendered)

    def test_window_signal_can_be_selected_explicitly(self) -> None:
        monitor = STLDistanceMonitor(threshold_m=100.0, window_steps=2)
        result = None
        for i in range(2):
            result = monitor.add_sample(
                jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
                jnp.array([120.0 + i, 0.0, 15.0], dtype=jnp.float32),
                jnp.array([0.0, 130.0 + i, 15.0], dtype=jnp.float32),
            )

        facts = monitor_result_to_facts(
            result,
            config=ProbabilityConfig(signal="window"),
            include_groups=False,
        )

        self.assertEqual(len(facts), len(result.rule_evaluations))
        self.assertAlmostEqual(facts[0].rho, result.rule_evaluations[0].window_robustness)

    def test_exporter_writes_live_fact_file(self) -> None:
        monitor = STLDistanceMonitor(threshold_m=100.0, window_steps=2)
        result = None
        for i in range(2):
            result = monitor.add_sample(
                jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32),
                jnp.array([120.0 + i, 0.0, 15.0], dtype=jnp.float32),
                jnp.array([0.0, 130.0 + i, 15.0], dtype=jnp.float32),
            )

        with TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "stl_live_facts.scl"
            exporter = ScallopFactExporter(output_path)
            facts = exporter.export_result(result)

            self.assertEqual(len(facts), len(result.rule_evaluations) + len(result.group_evaluations))
            self.assertIn(
                'stl_predicate_probability("rule", "origin_to_drone1", 1, true,',
                output_path.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
