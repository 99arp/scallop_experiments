from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from .models import (
    EventCalculusConfig,
    EventCalculusResult,
    FluentInterval,
    FluentProbability,
    ProbabilisticFact,
)
from .parser import parse_scallop_facts
from .tensor_engine import TensorEventCalculusEngine


class ReferenceEventCalculusEngine:
    """
    Compact probabilistic Event Calculus over STL predicate facts.

    Default event effects:
    - `value=true` initiates the fluent identified by `(source_kind, name)`.
    - `value=false` terminates that fluent.
    - if no event affects a fluent at a sample, inertia carries it forward.
    """

    def __init__(self, config: EventCalculusConfig | None = None):
        self.config = config or EventCalculusConfig()

    def infer(self, facts: tuple[ProbabilisticFact, ...]) -> EventCalculusResult:
        if not facts:
            return EventCalculusResult(holds=(), intervals=())

        facts_by_time: dict[int, list[ProbabilisticFact]] = defaultdict(list)
        fluent_keys: set[tuple[str, str]] = set()
        for fact in facts:
            facts_by_time[fact.sample_index].append(fact)
            fluent_keys.add(fact.fluent_key)

        start = min(facts_by_time)
        end = max(facts_by_time)
        state = {key: 0.0 for key in fluent_keys}
        holds: list[FluentProbability] = []

        for sample_index in range(start, end + 1):
            for fact in facts_by_time.get(sample_index, ()):
                current = state.get(fact.fluent_key, 0.0)
                if fact.value:
                    state[fact.fluent_key] = _noisy_or(current, fact.probability)
                else:
                    state[fact.fluent_key] = current * (1.0 - fact.probability)

            for source_kind, name in sorted(state):
                probability = state[(source_kind, name)]
                if probability > self.config.min_probability:
                    holds.append(
                        FluentProbability(
                            source_kind=source_kind,
                            name=name,
                            sample_index=sample_index,
                            probability=probability,
                        )
                    )

        return EventCalculusResult(
            holds=tuple(holds),
            intervals=self._build_intervals(tuple(holds)),
        )

    def _build_intervals(
        self,
        holds: tuple[FluentProbability, ...],
    ) -> tuple[FluentInterval, ...]:
        by_fluent: dict[tuple[str, str], list[FluentProbability]] = defaultdict(list)
        for fluent in holds:
            by_fluent[fluent.fluent_key].append(fluent)

        intervals: list[FluentInterval] = []
        for (source_kind, name), series in sorted(by_fluent.items()):
            active_start: int | None = None
            active_min = 1.0
            active_max = 0.0
            previous_sample: int | None = None

            for point in sorted(series, key=lambda item: item.sample_index):
                is_active = point.probability >= self.config.hold_threshold
                if is_active and active_start is None:
                    active_start = point.sample_index
                    active_min = point.probability
                    active_max = point.probability
                elif is_active:
                    active_min = min(active_min, point.probability)
                    active_max = max(active_max, point.probability)
                elif active_start is not None:
                    intervals.append(
                        FluentInterval(
                            source_kind=source_kind,
                            name=name,
                            start_sample_index=active_start,
                            end_sample_index=point.sample_index,
                            min_probability=active_min,
                            max_probability=active_max,
                        )
                    )
                    active_start = None
                previous_sample = point.sample_index

            if active_start is not None and previous_sample is not None:
                intervals.append(
                    FluentInterval(
                        source_kind=source_kind,
                        name=name,
                        start_sample_index=active_start,
                        end_sample_index=previous_sample + 1,
                        min_probability=active_min,
                        max_probability=active_max,
                    )
                )

        return tuple(intervals)


def infer_event_calculus(
    facts: tuple[ProbabilisticFact, ...],
    *,
    config: EventCalculusConfig | None = None,
) -> EventCalculusResult:
    return EventCalculusEngine(config=config).infer(facts)


def infer_event_calculus_from_file(
    path: str | Path,
    *,
    config: EventCalculusConfig | None = None,
) -> EventCalculusResult:
    resolved_config = config or EventCalculusConfig()
    text = Path(path).read_text(encoding="utf-8")
    facts = parse_scallop_facts(
        text,
        expected_relation_name=resolved_config.input_relation_name,
    )
    return EventCalculusEngine(config=resolved_config).infer(facts)


def _noisy_or(existing_probability: float, event_probability: float) -> float:
    return 1.0 - (1.0 - existing_probability) * (1.0 - event_probability)


class EventCalculusEngine(TensorEventCalculusEngine):
    """Default Tensor-pEC-style JAX backend."""
