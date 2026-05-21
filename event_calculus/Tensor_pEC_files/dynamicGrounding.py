"""Dynamic grounding for STL predicate fluents."""

from event_calculus.models import ProbabilisticFact


def dynamicGrounding(
    facts: tuple[ProbabilisticFact, ...],
) -> tuple[tuple[str, str], ...]:
    return tuple(sorted({fact.fluent_key for fact in facts}))
