from __future__ import annotations

from .models import EventCalculusResult
from .syntax import fluent_term


def render_event_calculus_facts(result: EventCalculusResult) -> str:
    lines = []
    for fluent in result.holds:
        lines.append(
            f"{fluent.probability:.9f}::holdsAt("
            f"{fluent_term(fluent.source_kind, fluent.name)},"
            f"{fluent.sample_index})."
        )
    for interval in result.intervals:
        lines.append(
            f"holdsFor("
            f"{fluent_term(interval.source_kind, interval.name)},"
            f"[({interval.start_sample_index},{interval.end_sample_index})])."
        )
    if not lines:
        return ""
    return "\n".join(lines) + "\n"
