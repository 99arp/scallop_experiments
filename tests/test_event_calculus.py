from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from event_calculus import (
    EventCalculusConfig,
    ReferenceEventCalculusEngine,
    infer_event_calculus,
    infer_event_calculus_from_file,
    parse_scallop_facts,
    render_event_calculus_facts,
)


class EventCalculusTests(unittest.TestCase):
    def test_parse_probabilistic_happens_at_facts(self) -> None:
        facts = parse_scallop_facts(
            "0.900000000::happensAt(stl_predicate(rule,origin_to_drone1,true),4).\n"
        )

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].source_kind, "rule")
        self.assertEqual(facts[0].name, "origin_to_drone1")
        self.assertEqual(facts[0].sample_index, 4)
        self.assertTrue(facts[0].value)
        self.assertAlmostEqual(facts[0].probability, 0.9)

    def test_parse_legacy_stl_probability_facts(self) -> None:
        facts = parse_scallop_facts(
            'stl_predicate_probability("rule", "origin_to_drone1", 4, true, 0.9, 2.0).\n'
        )

        self.assertEqual(len(facts), 1)
        self.assertEqual(facts[0].name, "origin_to_drone1")

    def test_event_calculus_inertia_and_termination(self) -> None:
        facts = parse_scallop_facts(
            "\n".join(
                [
                    "0.800000000::happensAt(stl_predicate(rule,origin_to_drone1,true),1).",
                    "0.750000000::happensAt(stl_predicate(rule,origin_to_drone1,false),3).",
                ]
            )
        )

        result = infer_event_calculus(
            facts,
            config=EventCalculusConfig(hold_threshold=0.5),
        )
        series = result.holds_at("rule", "origin_to_drone1")

        self.assertEqual([point.sample_index for point in series], [1, 2, 3])
        self.assertAlmostEqual(series[0].probability, 0.8)
        self.assertAlmostEqual(series[1].probability, 0.8)
        self.assertAlmostEqual(series[2].probability, 0.2)
        self.assertEqual(len(result.intervals), 1)
        self.assertEqual(result.intervals[0].start_sample_index, 1)
        self.assertEqual(result.intervals[0].end_sample_index, 3)

        reference = ReferenceEventCalculusEngine(
            config=EventCalculusConfig(hold_threshold=0.5)
        ).infer(facts)
        reference_series = reference.holds_at("rule", "origin_to_drone1")
        for tensor_point, reference_point in zip(series, reference_series):
            self.assertAlmostEqual(tensor_point.probability, reference_point.probability)

    def test_event_calculus_renders_holds_and_intervals(self) -> None:
        facts = parse_scallop_facts(
            "0.700000000::happensAt(stl_predicate(group,distance,true),1).\n"
        )

        result = infer_event_calculus(facts)
        rendered = render_event_calculus_facts(result)

        self.assertIn("::holdsAt(stl_predicate(group,distance)=true,1).", rendered)
        self.assertIn("holdsFor(stl_predicate(group,distance)=true,[(1,2)]).", rendered)

    def test_event_calculus_file_api(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "facts.scl"
            path.write_text(
                "0.800000000::happensAt(stl_predicate(rule,origin_to_drone1,true),1).\n",
                encoding="utf-8",
            )

            result = infer_event_calculus_from_file(path)

            self.assertEqual(len(result.holds), 1)
            self.assertEqual(result.holds[0].sample_index, 1)


if __name__ == "__main__":
    unittest.main()
