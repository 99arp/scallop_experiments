from __future__ import annotations

from dataclasses import dataclass
import math

from stl.monitor import GroupEvaluation, MonitorResult, RuleEvaluation
from event_calculus.models import ProbabilisticFact
from event_calculus.syntax import stl_predicate_term


@dataclass(frozen=True)
class ProbabilityConfig:
    """
    Controls the direct STL robustness -> probability mapping.

    `signal` selects which robustness surface to convert. Use `instant` by
    default so temporal smoothing is not applied twice.
    """

    scale: float = 1.0
    midpoint: float = 0.0
    min_probability: float = 1.0e-9
    max_probability: float = 1.0 - 1.0e-9
    signal: str = "instant"
    relation_name: str = "stl_predicate_probability"

    def __post_init__(self) -> None:
        if self.scale <= 0.0:
            raise ValueError("scale must be greater than 0.0")
        if not 0.0 <= self.min_probability < self.max_probability <= 1.0:
            raise ValueError(
                "probability bounds must satisfy 0 <= min_probability < max_probability <= 1"
            )
        if self.signal not in {"instant", "window"}:
            raise ValueError("signal must be either `instant` or `window`")
        if not self.relation_name:
            raise ValueError("relation_name must be a non-empty string")


@dataclass(frozen=True)
class ScallopFact:
    relation_name: str
    source_kind: str
    name: str
    sample_index: int
    value: bool
    probability: float
    rho: float

    def to_scallop(self) -> str:
        value_text = "true" if self.value else "false"
        return (
            f'{self.relation_name}("{_escape_string(self.source_kind)}", '
            f'"{_escape_string(self.name)}", '
            f"{self.sample_index}, "
            f"{value_text}, "
            f"{self.probability:.9f}, "
            f"{self.rho:.9f})."
        )

    def to_event_calculus(self) -> str:
        return (
            f"{self.probability:.9f}::happensAt("
            f"{stl_predicate_term(self.source_kind, self.name, self.value)},"
            f"{self.sample_index})."
        )

    def to_event_calculus_fact(self) -> ProbabilisticFact:
        return ProbabilisticFact(
            relation_name="happensAt",
            source_kind=self.source_kind,
            name=self.name,
            sample_index=self.sample_index,
            value=self.value,
            probability=self.probability,
            rho=self.rho,
        )


def sigmoid_probability(
    rho: float,
    *,
    scale: float = 1.0,
    midpoint: float = 0.0,
    min_probability: float = 1.0e-9,
    max_probability: float = 1.0 - 1.0e-9,
) -> float:
    """
    Convert STL robustness to a probability in [0, 1] with a stable sigmoid.

    rho == midpoint maps to 0.5. Positive robustness moves toward 1.0 and
    negative robustness moves toward 0.0.
    """

    if scale <= 0.0:
        raise ValueError("scale must be greater than 0.0")
    if not 0.0 <= min_probability < max_probability <= 1.0:
        raise ValueError(
            "probability bounds must satisfy 0 <= min_probability < max_probability <= 1"
        )
    x = scale * (float(rho) - midpoint)
    if x >= 0.0:
        z = math.exp(-x) if x < 709.0 else 0.0
        probability = 1.0 / (1.0 + z)
    else:
        z = math.exp(x) if x > -709.0 else 0.0
        probability = z / (1.0 + z)
    return min(max(probability, min_probability), max_probability)


def rule_evaluation_to_fact(
    rule: RuleEvaluation,
    *,
    sample_index: int,
    config: ProbabilityConfig | None = None,
) -> ScallopFact:
    config = config or ProbabilityConfig()
    if config.signal == "instant":
        rho = rule.instant_robustness
        value = rule.ok_now
    else:
        rho = rule.window_robustness
        value = rule.ok_window
    raw_prob = sigmoid_probability(
        rho,
        scale=config.scale,
        midpoint=config.midpoint,
        min_probability=config.min_probability,
        max_probability=config.max_probability,
    )
    return ScallopFact(
        relation_name=config.relation_name,
        source_kind="rule",
        name=rule.name,
        sample_index=sample_index,
        value=value,
        probability=raw_prob if value else 1.0 - raw_prob,
        rho=rho,
    )


def group_evaluation_to_fact(
    group: GroupEvaluation,
    *,
    sample_index: int,
    config: ProbabilityConfig | None = None,
) -> ScallopFact:
    config = config or ProbabilityConfig()
    rho = group.window_robustness
    value = group.ok_window
    raw_prob = sigmoid_probability(
        rho,
        scale=config.scale,
        midpoint=config.midpoint,
        min_probability=config.min_probability,
        max_probability=config.max_probability,
    )
    return ScallopFact(
        relation_name=config.relation_name,
        source_kind="group",
        name=group.name,
        sample_index=sample_index,
        value=value,
        probability=raw_prob if value else 1.0 - raw_prob,
        rho=rho,
    )


def monitor_result_to_facts(
    result: MonitorResult,
    *,
    config: ProbabilityConfig | None = None,
    include_groups: bool = True,
) -> tuple[ScallopFact, ...]:
    if not result.is_ready:
        return ()
    config = config or ProbabilityConfig()
    facts: list[ScallopFact] = [
        rule_evaluation_to_fact(
            rule,
            sample_index=result.sample_index,
            config=config,
        )
        for rule in result.rule_evaluations
    ]
    if include_groups:
        facts.extend(
            group_evaluation_to_fact(
                group,
                sample_index=result.sample_index,
                config=config,
            )
            for group in result.group_evaluations
        )
    return tuple(facts)


def render_scallop_facts(facts: tuple[ScallopFact, ...]) -> str:
    if not facts:
        return ""
    return "\n".join(fact.to_scallop() for fact in facts) + "\n"


def render_event_calculus_events(facts: tuple[ScallopFact, ...]) -> str:
    if not facts:
        return ""
    return "\n".join(fact.to_event_calculus() for fact in facts) + "\n"


def scallop_facts_to_event_calculus_facts(
    facts: tuple[ScallopFact, ...],
) -> tuple[ProbabilisticFact, ...]:
    return tuple(fact.to_event_calculus_fact() for fact in facts)


def _escape_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
