from __future__ import annotations

from pathlib import Path

from stl.monitor import MonitorResult

from .converter import (
    ProbabilityConfig,
    ScallopFact,
    monitor_result_to_facts,
    render_event_calculus_events,
    render_scallop_facts,
)


class ScallopFactExporter:
    """Append converted STL predicate probability facts to a Scallop fact file."""

    def __init__(
        self,
        output_path: str | Path,
        *,
        config: ProbabilityConfig | None = None,
        include_groups: bool = True,
        reset_on_start: bool = True,
        export_format: str = "event_calculus",
    ):
        self.output_path = Path(output_path)
        self.config = config or ProbabilityConfig()
        self.include_groups = include_groups
        if export_format not in {"event_calculus", "legacy_scallop"}:
            raise ValueError("export_format must be `event_calculus` or `legacy_scallop`")
        self.export_format = export_format
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if reset_on_start:
            self.output_path.write_text("", encoding="utf-8")

    def export_result(self, result: MonitorResult) -> tuple[ScallopFact, ...]:
        facts = monitor_result_to_facts(
            result,
            config=self.config,
            include_groups=self.include_groups,
        )
        if facts:
            with self.output_path.open("a", encoding="utf-8") as handle:
                if self.export_format == "event_calculus":
                    handle.write(render_event_calculus_events(facts))
                else:
                    handle.write(render_scallop_facts(facts))
        return facts
