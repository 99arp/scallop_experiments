from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import html
import json
from pathlib import Path
import subprocess
from typing import Optional

from .monitor import CompiledRule, MonitorResult, RuleEvaluation


@dataclass(frozen=True)
class VisualizationArtifacts:
    svg_path: Path
    json_path: Path
    dot_path: Path


class STLRuntimeVisualizer:
    """Render the current STL rule graph with Graphviz dot."""

    def __init__(
        self,
        output_dir: str | Path | None = None,
        stem: str = "stl_live",
    ):
        self.output_dir = (
            Path(output_dir) if output_dir is not None else Path(__file__).with_name("stl_viz")
        )
        self.stem = stem
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts = VisualizationArtifacts(
            svg_path=self.output_dir / f"{stem}.svg",
            json_path=self.output_dir / f"{stem}.json",
            dot_path=self.output_dir / f"{stem}.dot",
        )
        self._write_json_snapshot(result=None, status_message="waiting for first render")

    def render(
        self,
        compiled_rules: tuple[CompiledRule, ...],
        combine_operator: str,
        result: Optional[MonitorResult],
        status_message: Optional[str] = None,
    ) -> VisualizationArtifacts:
        dot = self._build_dot(
            compiled_rules=compiled_rules,
            combine_operator=combine_operator,
            result=result,
            status_message=status_message,
        )
        self.artifacts.dot_path.write_text(dot, encoding="utf-8")
        self._render_svg(dot)
        self._write_json_snapshot(result=result, status_message=status_message)
        return self.artifacts

    def _render_svg(self, dot: str) -> None:
        try:
            completed = subprocess.run(
                ["dot", "-Tsvg"],
                input=dot.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError("Graphviz `dot` was not found on PATH.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"Graphviz `dot` failed: {stderr}") from exc
        self.artifacts.svg_path.write_bytes(completed.stdout)

    def _build_dot(
        self,
        *,
        compiled_rules: tuple[CompiledRule, ...],
        combine_operator: str,
        result: Optional[MonitorResult],
        status_message: Optional[str],
    ) -> str:
        eval_by_name = (
            {rule_eval.name: rule_eval for rule_eval in result.rule_evaluations}
            if result is not None
            else {}
        )
        lines = [
            "digraph stl_runtime {",
            '  graph [rankdir="TB", bgcolor="white", pad="0.35", nodesep="0.35", ranksep="0.55"];',
            '  node [shape="box", style="rounded,filled", color="#334155", fillcolor="#f8fafc", fontname="Helvetica", fontsize="11", margin="0.16,0.10"];',
            '  edge [color="#64748b", penwidth="1.3", arrowsize="0.7"];',
            f'  label="{self._escape_label(self._graph_label(result, status_message))}";',
            '  labelloc="t";',
            '  fontsize="18";',
            '  fontname="Helvetica-Bold";',
        ]

        for group_name, group_label, group_color in self._group_nodes(compiled_rules, result):
            lines.append(
                f'  "group_{group_name}" [label="{self._escape_label(group_label)}", fillcolor="{group_color}"];'
            )
        combine_label = self._escape_label("combine\n" + combine_operator)
        lines.append(
            f'  "combine" [label="{combine_label}", fillcolor="#ddd6fe"];'
        )

        for group_name, _, _ in self._group_nodes(compiled_rules, result):
            lines.append(f'  "group_{group_name}" -> "combine";')

        for entity_name in self._ordered_entities(compiled_rules):
            entity_label = self._escape_label("signal\n" + entity_name)
            lines.append(
                f'  "entity_{entity_name}" [label="{entity_label}", fillcolor="#dbeafe"];'
            )

        for compiled_rule in compiled_rules:
            rule_eval = eval_by_name.get(compiled_rule.name)
            metric_node_id = f"metric_{compiled_rule.name}"
            predicate_node_id = f"predicate_{compiled_rule.name}"
            temporal_node_id = f"temporal_{compiled_rule.name}"

            lines.append(
                f'  "{metric_node_id}" [label="{self._escape_label(self._metric_label(compiled_rule, rule_eval))}", fillcolor="{self._metric_color(rule_eval)}"];'
            )
            lines.append(
                f'  "{predicate_node_id}" [label="{self._escape_label(self._predicate_label(compiled_rule, rule_eval))}", fillcolor="{self._predicate_color(rule_eval)}"];'
            )
            lines.append(
                f'  "{temporal_node_id}" [label="{self._escape_label(self._temporal_label(compiled_rule, rule_eval))}", fillcolor="{self._temporal_color(rule_eval, result)}"];'
            )
            lines.append(f'  "entity_{compiled_rule.lhs}" -> "{metric_node_id}";')
            if compiled_rule.rhs is not None:
                lines.append(f'  "entity_{compiled_rule.rhs}" -> "{metric_node_id}";')
            lines.append(f'  "{metric_node_id}" -> "{predicate_node_id}";')
            lines.append(f'  "{predicate_node_id}" -> "{temporal_node_id}";')
            lines.append(f'  "{temporal_node_id}" -> "group_{compiled_rule.kind}";')

        lines.append("}")
        return "\n".join(lines) + "\n"

    def _write_json_snapshot(
        self,
        *,
        result: Optional[MonitorResult],
        status_message: Optional[str],
    ) -> None:
        snapshot = {
            "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "status": status_message or ("ready" if result and result.is_ready else "collecting"),
            "graph_svg": self.artifacts.svg_path.name,
            "sample_index": result.sample_index if result is not None else None,
            "timestamp_ns": result.timestamp_ns if result is not None else None,
            "is_ready": result.is_ready if result is not None else False,
            "ok_now": result.ok_now if result is not None else None,
            "ok_window": result.ok_window if result is not None else None,
            "rules": [
                {
                    "name": rule.name,
                    "label": rule.label,
                    "current_value": rule.current_value,
                    "instant_robustness": rule.instant_robustness,
                    "window_robustness": rule.window_robustness,
                    "ok_now": rule.ok_now,
                    "ok_window": rule.ok_window,
                }
                for rule in (result.rule_evaluations if result is not None else ())
            ],
            "groups": [
                {
                    "name": group.name,
                    "rule_count": group.rule_count,
                    "window_robustness": group.window_robustness,
                    "ok_now": group.ok_now,
                    "ok_window": group.ok_window,
                }
                for group in (result.group_evaluations if result is not None else ())
            ],
        }
        self.artifacts.json_path.write_text(
            json.dumps(snapshot, indent=2, sort_keys=False),
            encoding="utf-8",
        )

    @staticmethod
    def _ordered_entities(compiled_rules: tuple[CompiledRule, ...]) -> list[str]:
        seen = []
        for compiled_rule in compiled_rules:
            for entity_name in (compiled_rule.lhs, compiled_rule.rhs):
                if entity_name is not None and entity_name not in seen:
                    seen.append(entity_name)
        return seen

    @staticmethod
    def _graph_label(result: Optional[MonitorResult], status_message: Optional[str]) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if status_message:
            return f"STL Live Graph\n{status_message}\nupdated {timestamp}"
        if result is None:
            return f"STL Live Graph\nupdated {timestamp}"
        if not result.is_ready:
            return (
                "STL Live Graph\n"
                f"collecting samples {result.collected_samples}/{result.window_steps}\n"
                f"sample #{result.sample_index}\n"
                f"updated {timestamp}"
            )
        return f"STL Live Graph\nsample #{result.sample_index}\nupdated {timestamp}"

    @staticmethod
    def _metric_label(compiled_rule: CompiledRule, rule_eval: Optional[RuleEvaluation]) -> str:
        metric_name = STLRuntimeVisualizer._metric_name(compiled_rule)
        unit = STLRuntimeVisualizer._metric_unit(compiled_rule)
        if rule_eval is None:
            return f"{compiled_rule.kind}\n{metric_name}\nvalue=waiting"
        return f"{compiled_rule.kind}\n{metric_name}\nvalue={rule_eval.current_value:.2f} {unit}"

    @staticmethod
    def _predicate_label(compiled_rule: CompiledRule, rule_eval: Optional[RuleEvaluation]) -> str:
        if rule_eval is None:
            return f"{compiled_rule.op} {compiled_rule.threshold:.2f}\nrho_now=waiting"
        return (
            f"{compiled_rule.op} {compiled_rule.threshold:.2f}\n"
            f"rho_now={rule_eval.instant_robustness:.2f}\n"
            f"ok_now={rule_eval.ok_now}"
        )

    @staticmethod
    def _temporal_label(compiled_rule: CompiledRule, rule_eval: Optional[RuleEvaluation]) -> str:
        start, end = compiled_rule.interval
        if rule_eval is None:
            return f"{compiled_rule.temporal_op}[{start},{end}]\nrho_window=waiting"
        return (
            f"{compiled_rule.label}\n"
            f"{compiled_rule.temporal_op}[{start},{end}]\n"
            f"rho_window={rule_eval.window_robustness:.2f}\n"
            f"ok_window={rule_eval.ok_window}"
        )

    @staticmethod
    def _metric_color(rule_eval: Optional[RuleEvaluation]) -> str:
        if rule_eval is None:
            return "#e2e8f0"
        return "#bfdbfe" if rule_eval.instant_robustness > 0.0 else "#fecaca"

    @staticmethod
    def _predicate_color(rule_eval: Optional[RuleEvaluation]) -> str:
        if rule_eval is None:
            return "#e2e8f0"
        return "#fde68a" if rule_eval.ok_now else "#fca5a5"

    @staticmethod
    def _temporal_color(
        rule_eval: Optional[RuleEvaluation],
        result: Optional[MonitorResult],
    ) -> str:
        if result is None or not result.is_ready or rule_eval is None:
            return "#e2e8f0"
        return "#93c5fd" if rule_eval.ok_window else "#fca5a5"

    @staticmethod
    def _metric_name(compiled_rule: CompiledRule) -> str:
        if compiled_rule.kind == "distance":
            return f"{compiled_rule.lhs} <-> {compiled_rule.rhs}"
        if compiled_rule.kind == "height":
            return f"z({compiled_rule.lhs})"
        return compiled_rule.label

    @staticmethod
    def _metric_unit(compiled_rule: CompiledRule) -> str:
        if compiled_rule.kind in {"distance", "height"}:
            return "m"
        return "units"

    @staticmethod
    def _group_nodes(
        compiled_rules: tuple[CompiledRule, ...],
        result: Optional[MonitorResult],
    ) -> list[tuple[str, str, str]]:
        ordered_group_names = []
        for compiled_rule in compiled_rules:
            if compiled_rule.kind not in ordered_group_names:
                ordered_group_names.append(compiled_rule.kind)
        group_eval_by_name = (
            {group_eval.name: group_eval for group_eval in result.group_evaluations}
            if result is not None
            else {}
        )
        return [
            (
                group_name,
                STLRuntimeVisualizer._group_label(
                    group_name,
                    group_eval_by_name.get(group_name),
                    result,
                ),
                STLRuntimeVisualizer._group_color(group_eval_by_name.get(group_name), result),
            )
            for group_name in ordered_group_names
        ]

    @staticmethod
    def _group_label(group_name: str, group_eval, result: Optional[MonitorResult]) -> str:
        title = group_name.upper()
        if result is None:
            return f"{title}\nwaiting for data"
        if not result.is_ready or group_eval is None:
            return f"{title}\ncollecting {result.collected_samples}/{result.window_steps}"
        return (
            f"{title}\n"
            f"rules={group_eval.rule_count}\n"
            f"rho_window={group_eval.window_robustness:.2f}\n"
            f"ok_now={group_eval.ok_now}\n"
            f"ok_window={group_eval.ok_window}"
        )

    @staticmethod
    def _group_color(group_eval, result: Optional[MonitorResult]) -> str:
        if result is None or not result.is_ready or group_eval is None:
            return "#e2e8f0"
        return "#86efac" if group_eval.ok_window else "#fca5a5"

    @staticmethod
    def _escape_label(value: str) -> str:
        return html.escape(value, quote=True).replace("\n", r"\n")
