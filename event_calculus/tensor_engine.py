from __future__ import annotations

from collections import defaultdict

import jax
import jax.numpy as jnp

from .models import (
    EventCalculusConfig,
    EventCalculusResult,
    FluentInterval,
    FluentProbability,
    ProbabilisticFact,
)


class TensorEventCalculusEngine:
    """
    Tensor-pEC-style backend using JAX tensors.

    Input facts are grounded into initiation and termination matrices with shape
    `(fluents, time)`. A vectorized `lax.scan` applies the EC recurrence:

    holds_t = initiated_t OR (holds_{t-1} AND NOT terminated_t)
    """

    def __init__(self, config: EventCalculusConfig | None = None):
        self.config = config or EventCalculusConfig()

    def infer(self, facts: tuple[ProbabilisticFact, ...]) -> EventCalculusResult:
        if not facts:
            return EventCalculusResult(holds=(), intervals=())

        fluent_keys = sorted({fact.fluent_key for fact in facts})
        start = min(fact.sample_index for fact in facts)
        end = max(fact.sample_index for fact in facts)
        key_to_index = {key: index for index, key in enumerate(fluent_keys)}
        time_steps = end - start + 1
        fluent_count = len(fluent_keys)

        initiated = jnp.zeros((fluent_count, time_steps), dtype=jnp.float32)
        terminated = jnp.zeros((fluent_count, time_steps), dtype=jnp.float32)
        for (fluent_key, sample_index, value), probabilities in _group_fact_probs(facts).items():
            probability = _noisy_or_probabilities(probabilities)
            fluent_index = key_to_index[fluent_key]
            time_index = sample_index - start
            if value:
                initiated = initiated.at[fluent_index, time_index].set(probability)
            else:
                terminated = terminated.at[fluent_index, time_index].set(probability)

        holds_matrix = _holds_at_matrix(initiated, terminated)
        holds = self._matrix_to_holds(
            holds_matrix=holds_matrix,
            fluent_keys=fluent_keys,
            start_sample_index=start,
        )
        return EventCalculusResult(
            holds=holds,
            intervals=self._build_intervals(holds),
        )

    def _matrix_to_holds(
        self,
        *,
        holds_matrix,
        fluent_keys: list[tuple[str, str]],
        start_sample_index: int,
    ) -> tuple[FluentProbability, ...]:
        holds = []
        matrix = jax.device_get(holds_matrix)
        for fluent_index, (source_kind, name) in enumerate(fluent_keys):
            for time_index, probability_value in enumerate(matrix[fluent_index].tolist()):
                probability = float(probability_value)
                if probability > self.config.min_probability:
                    holds.append(
                        FluentProbability(
                            source_kind=source_kind,
                            name=name,
                            sample_index=start_sample_index + time_index,
                            probability=probability,
                        )
                    )
        return tuple(holds)

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


@jax.jit
def _holds_at_matrix(initiated, terminated):
    events_by_time = (initiated.T, terminated.T)
    initial_state = jnp.zeros((initiated.shape[0],), dtype=initiated.dtype)

    def step(previous_holds, events):
        initiated_t, terminated_t = events
        carried = previous_holds * (1.0 - terminated_t)
        holds_t = 1.0 - (1.0 - initiated_t) * (1.0 - carried)
        return holds_t, holds_t

    _, holds_by_time = jax.lax.scan(step, initial_state, events_by_time)
    return holds_by_time.T


def _group_fact_probs(
    facts: tuple[ProbabilisticFact, ...],
) -> dict[tuple[tuple[str, str], int, bool], list[float]]:
    grouped: dict[tuple[tuple[str, str], int, bool], list[float]] = defaultdict(list)
    for fact in facts:
        grouped[(fact.fluent_key, fact.sample_index, fact.value)].append(fact.probability)
    return grouped


def _noisy_or_probabilities(probabilities: list[float]) -> float:
    product = 1.0
    for probability in probabilities:
        product *= 1.0 - probability
    return 1.0 - product
