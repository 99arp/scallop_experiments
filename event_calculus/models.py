from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProbabilisticFact:
    relation_name: str
    source_kind: str
    name: str
    sample_index: int
    value: bool
    probability: float
    rho: float

    @property
    def fluent_key(self) -> tuple[str, str]:
        return self.source_kind, self.name


@dataclass(frozen=True)
class FluentProbability:
    source_kind: str
    name: str
    sample_index: int
    probability: float

    @property
    def fluent_key(self) -> tuple[str, str]:
        return self.source_kind, self.name


@dataclass(frozen=True)
class FluentInterval:
    source_kind: str
    name: str
    start_sample_index: int
    end_sample_index: int
    min_probability: float
    max_probability: float

    @property
    def fluent_key(self) -> tuple[str, str]:
        return self.source_kind, self.name


@dataclass(frozen=True)
class EventCalculusResult:
    holds: tuple[FluentProbability, ...]
    intervals: tuple[FluentInterval, ...]

    def holds_at(self, source_kind: str, name: str) -> tuple[FluentProbability, ...]:
        return tuple(
            fluent
            for fluent in self.holds
            if fluent.source_kind == source_kind and fluent.name == name
        )


@dataclass(frozen=True)
class EventCalculusConfig:
    """
    Configuration for the compact probabilistic EC engine.

    `hold_threshold` is used only when extracting intervals from the continuous
    probability trace. `min_probability` prunes near-zero values from output.
    """

    input_relation_name: str = "happensAt"
    holds_relation_name: str = "holdsAt"
    interval_relation_name: str = "holdsFor"
    hold_threshold: float = 0.5
    min_probability: float = 1.0e-9

    def __post_init__(self) -> None:
        if not self.input_relation_name:
            raise ValueError("input_relation_name must be a non-empty string")
        if not self.holds_relation_name:
            raise ValueError("holds_relation_name must be a non-empty string")
        if not self.interval_relation_name:
            raise ValueError("interval_relation_name must be a non-empty string")
        if not 0.0 <= self.hold_threshold <= 1.0:
            raise ValueError("hold_threshold must be in [0, 1]")
        if not 0.0 <= self.min_probability <= 1.0:
            raise ValueError("min_probability must be in [0, 1]")
