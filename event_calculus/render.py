from __future__ import annotations

from .engine import EventCalculusResult


def render_event_calculus_facts(result: EventCalculusResult) -> str:
    lines = []
    for fluent in result.holds:
        lines.append(
            f'ec_holds("{_escape_string(fluent.source_kind)}", '
            f'"{_escape_string(fluent.name)}", '
            f"{fluent.sample_index}, "
            f"{fluent.probability:.9f})."
        )
    for interval in result.intervals:
        lines.append(
            f'ec_interval("{_escape_string(interval.source_kind)}", '
            f'"{_escape_string(interval.name)}", '
            f"{interval.start_sample_index}, "
            f"{interval.end_sample_index}, "
            f"{interval.min_probability:.9f}, "
            f"{interval.max_probability:.9f})."
        )
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _escape_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')
