from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import operator
from functools import reduce
from pathlib import Path
from typing import Callable, Deque, Optional

import jax.numpy as jnp

from .formula import Always, Predicate

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None


ENTITY_SLICES = {
    "origin": slice(0, 3),
    "drone1": slice(3, 6),
    "drone2": slice(6, 9),
}


@dataclass(frozen=True)
class RuleEvaluation:
    name: str
    label: str
    threshold: float
    op: str
    temporal_op: str
    interval: tuple[int, int]
    current_value: float
    instant_robustness: float
    window_robustness: float
    ok_now: bool
    ok_window: bool


@dataclass(frozen=True)
class GroupEvaluation:
    name: str
    rule_count: int
    window_robustness: float
    ok_now: bool
    ok_window: bool


@dataclass(frozen=True)
class RuleBeliefSnapshot:
    sample_index: int
    timestamp_ns: int
    name: str
    label: str
    current_value: float
    instant_robustness: float
    window_robustness: float
    ok_now: bool
    ok_window: bool


@dataclass(frozen=True)
class GroupBeliefSnapshot:
    sample_index: int
    timestamp_ns: int
    name: str
    rule_count: int
    window_robustness: float
    ok_now: bool
    ok_window: bool


@dataclass(frozen=True)
class MonitorResult:
    is_ready: bool
    sample_index: int
    timestamp_ns: int
    collected_samples: int
    window_steps: int
    rule_evaluations: tuple[RuleEvaluation, ...] = ()
    group_evaluations: tuple[GroupEvaluation, ...] = ()
    ok_now: Optional[bool] = None
    ok_window: Optional[bool] = None


@dataclass(frozen=True)
class CompiledRule:
    name: str
    kind: str
    label: str
    lhs: str
    rhs: Optional[str]
    threshold: float
    op: str
    temporal_op: str
    interval: tuple[int, int]
    metric_function: Callable[[jnp.ndarray], jnp.ndarray]
    formula: object


class STLDistanceMonitor:
    """
    Pure STL monitor salvaged from command_agent.

    It loads drone rules from YAML, accepts pose samples for origin/drone1/drone2,
    evaluates the configured STL formulas, and keeps an in-memory belief history.
    """

    def __init__(
        self,
        threshold_m: float = 100.0,
        window_steps: int = 50,
        rules_path: Optional[str | Path] = None,
        history_limit: int = 10000,
    ):
        if window_steps < 2:
            raise ValueError("window_steps must be at least 2")
        if history_limit <= 0:
            raise ValueError("history_limit must be greater than 0")

        self.threshold_m = float(threshold_m)
        self.window_steps = int(window_steps)
        self.history_limit = int(history_limit)
        self.sample_index = -1
        self.trace_buffer: Deque[jnp.ndarray] = deque(maxlen=self.window_steps)
        self.rules_path = Path(rules_path) if rules_path is not None else self._default_rules_path()
        self.rules_text = self._load_rules_text()
        self.ruleset_hash = hashlib.sha256(self.rules_text.encode("utf-8")).hexdigest()
        self.combine_operator = "and"
        self.compiled_rules = self._load_and_compile_rules()
        self.grouped_rules = self._group_rules_by_kind()
        self.group_formulas = self._build_group_formulas()
        self.rule_belief_history = {
            rule.name: deque(maxlen=self.history_limit) for rule in self.compiled_rules
        }
        self.group_belief_history = {
            group_name: deque(maxlen=self.history_limit) for group_name in self.grouped_rules
        }

    def add_sample(
        self,
        origin: jnp.ndarray,
        drone1: jnp.ndarray,
        drone2: jnp.ndarray,
        sample_time_ns: Optional[int] = None,
    ) -> MonitorResult:
        self.sample_index += 1
        timestamp_ns = int(sample_time_ns) if sample_time_ns is not None else self.sample_index
        row = jnp.concatenate(
            [
                self._as_pose_vector(origin),
                self._as_pose_vector(drone1),
                self._as_pose_vector(drone2),
            ],
            axis=0,
        )
        self.trace_buffer.append(row)

        if len(self.trace_buffer) < self.window_steps:
            return MonitorResult(
                is_ready=False,
                sample_index=self.sample_index,
                timestamp_ns=timestamp_ns,
                collected_samples=len(self.trace_buffer),
                window_steps=self.window_steps,
            )

        trace = jnp.stack(list(self.trace_buffer), axis=0)
        rule_evaluations = []
        for rule in self.compiled_rules:
            signal = rule.metric_function(trace)
            current_value = float(signal[-1])
            instant_robustness = self._instant_robustness(
                current_value=current_value,
                threshold=rule.threshold,
                op=rule.op,
            )
            window_robustness = self._robustness_value(rule.formula, trace)
            rule_evaluations.append(
                RuleEvaluation(
                    name=rule.name,
                    label=rule.label,
                    threshold=rule.threshold,
                    op=rule.op,
                    temporal_op=rule.temporal_op,
                    interval=rule.interval,
                    current_value=current_value,
                    instant_robustness=instant_robustness,
                    window_robustness=window_robustness,
                    ok_now=instant_robustness > 0.0,
                    ok_window=window_robustness > 0.0,
                )
            )

        rule_eval_by_name = {rule.name: rule for rule in rule_evaluations}
        group_evaluations = []
        for group_name, grouped_rules in self.grouped_rules.items():
            group_robustness = self._robustness_value(self.group_formulas[group_name], trace)
            current_group_rules = [rule_eval_by_name[rule.name] for rule in grouped_rules]
            group_evaluations.append(
                GroupEvaluation(
                    name=group_name,
                    rule_count=len(grouped_rules),
                    window_robustness=group_robustness,
                    ok_now=all(rule.ok_now for rule in current_group_rules),
                    ok_window=group_robustness > 0.0,
                )
            )

        result = MonitorResult(
            is_ready=True,
            sample_index=self.sample_index,
            timestamp_ns=timestamp_ns,
            collected_samples=len(self.trace_buffer),
            window_steps=self.window_steps,
            rule_evaluations=tuple(rule_evaluations),
            group_evaluations=tuple(group_evaluations),
            ok_now=all(group.ok_now for group in group_evaluations),
            ok_window=all(group.ok_window for group in group_evaluations),
        )
        self._record_belief_history(result)
        return result

    def get_latest_belief(
        self,
        name: str,
    ) -> Optional[RuleBeliefSnapshot | GroupBeliefSnapshot]:
        history = self._get_named_history(name)
        if history is None or not history:
            return None
        return history[-1]

    def get_belief_at_step(
        self,
        name: str,
        sample_index: int,
    ) -> Optional[RuleBeliefSnapshot | GroupBeliefSnapshot]:
        history = self._get_named_history(name)
        if history is None:
            return None
        for snapshot in reversed(history):
            if snapshot.sample_index <= sample_index:
                return snapshot
        return None

    def get_belief_series_by_step(
        self,
        name: str,
        start_sample_index: Optional[int] = None,
        end_sample_index: Optional[int] = None,
    ) -> tuple[RuleBeliefSnapshot | GroupBeliefSnapshot, ...]:
        history = self._get_named_history(name)
        if history is None:
            return ()
        return tuple(
            snapshot
            for snapshot in history
            if (start_sample_index is None or snapshot.sample_index >= start_sample_index)
            and (end_sample_index is None or snapshot.sample_index <= end_sample_index)
        )

    def close(self) -> None:
        return None

    @staticmethod
    def _default_rules_path() -> Path:
        return Path(__file__).resolve().parents[1] / "drone_rules" / "drone_rules.yaml"

    def _load_rules_text(self) -> str:
        if not self.rules_path.is_file():
            raise FileNotFoundError(f"STL rules file not found: {self.rules_path}")
        return self.rules_path.read_text(encoding="utf-8")

    def _load_and_compile_rules(self) -> tuple[CompiledRule, ...]:
        config = self._load_rules_config()
        combine_operator = config.get("combined_operator", "and")
        if not isinstance(combine_operator, str) or not combine_operator:
            raise ValueError("`combined_operator` must be a non-empty string when provided.")
        self.combine_operator = combine_operator

        rules = config.get("rules")
        if not isinstance(rules, list) or not rules:
            raise ValueError("The STL rules YAML must contain a non-empty `rules` list.")

        defaults = config.get("defaults", {})
        if defaults is None:
            defaults = {}
        if not isinstance(defaults, dict):
            raise ValueError("`defaults` must be a mapping when provided.")

        return tuple(self._compile_rule(rule_config, defaults) for rule_config in rules)

    def _compile_rule(self, rule_config: object, defaults: dict) -> CompiledRule:
        if not isinstance(rule_config, dict):
            raise ValueError("Each STL rule must be a mapping.")

        name = self._require_string(rule_config, "name")
        kind = self._string_or_default(rule_config, "kind", "distance")
        if kind == "distance":
            lhs = self._require_string(rule_config, "lhs")
            rhs = self._require_string(rule_config, "rhs")
            self._validate_entity_name(lhs, name)
            self._validate_entity_name(rhs, name)
            metric_function = self._build_distance_metric(lhs=lhs, rhs=rhs)
            default_label = f"distance({lhs}, {rhs})"
        elif kind == "height":
            entity = self._require_string(rule_config, "entity")
            self._validate_entity_name(entity, name)
            lhs = entity
            rhs = None
            metric_function = self._build_height_metric(entity=entity)
            default_label = f"height({entity})"
        else:
            raise ValueError(f"Unsupported rule kind `{kind}` for rule `{name}`.")

        op = self._string_or_default(rule_config, "op", ">")
        if op not in {">", "<"}:
            raise ValueError(
                f"Unsupported operator `{op}` for rule `{name}`. Only `>` and `<` are supported."
            )

        threshold_ref = rule_config.get("threshold", defaults.get("threshold", "threshold_m"))
        threshold = self._resolve_numeric_reference(threshold_ref)
        temporal_config = self._merged_temporal_config(rule_config, defaults, name)
        temporal_op = str(temporal_config.get("op", "always"))
        if temporal_op != "always":
            raise ValueError(
                f"Unsupported temporal operator `{temporal_op}` for rule `{name}`."
            )
        interval = self._resolve_interval(
            temporal_config.get("interval", [0, "window_end"]),
            rule_name=name,
        )

        label = str(rule_config.get("label", f"{default_label} {op} {threshold:.2f}"))
        predicate = Predicate(name, predicate_function=metric_function)
        atomic_formula = predicate > threshold if op == ">" else predicate < threshold
        formula = Always(atomic_formula, interval=list(interval))

        return CompiledRule(
            name=name,
            kind=kind,
            label=label,
            lhs=lhs,
            rhs=rhs,
            threshold=threshold,
            op=op,
            temporal_op=temporal_op,
            interval=interval,
            metric_function=metric_function,
            formula=formula,
        )

    def _merged_temporal_config(
        self,
        rule_config: dict,
        defaults: dict,
        rule_name: str,
    ) -> dict:
        temporal_defaults = defaults.get("temporal", {}) or {}
        temporal_config = rule_config.get("temporal", {}) or {}
        if not isinstance(temporal_defaults, dict):
            raise ValueError("`defaults.temporal` must be a mapping when provided.")
        if not isinstance(temporal_config, dict):
            raise ValueError(f"`temporal` must be a mapping for rule `{rule_name}`.")
        return {**temporal_defaults, **temporal_config}

    def _group_rules_by_kind(self) -> dict[str, tuple[CompiledRule, ...]]:
        grouped_rules: dict[str, list[CompiledRule]] = {}
        for compiled_rule in self.compiled_rules:
            grouped_rules.setdefault(compiled_rule.kind, []).append(compiled_rule)
        return {kind: tuple(rules) for kind, rules in grouped_rules.items()}

    def _build_group_formulas(self) -> dict[str, object]:
        if self.combine_operator != "and":
            raise ValueError("Only `and` is supported for combining STL rules.")
        group_formulas = {}
        for group_name, grouped_rules in self.grouped_rules.items():
            if len(grouped_rules) == 1:
                group_formulas[group_name] = grouped_rules[0].formula
            else:
                group_formulas[group_name] = reduce(
                    operator.and_,
                    (rule.formula for rule in grouped_rules),
                )
        return group_formulas

    def _record_belief_history(self, result: MonitorResult) -> None:
        for rule_eval in result.rule_evaluations:
            self.rule_belief_history[rule_eval.name].append(
                RuleBeliefSnapshot(
                    sample_index=result.sample_index,
                    timestamp_ns=result.timestamp_ns,
                    name=rule_eval.name,
                    label=rule_eval.label,
                    current_value=rule_eval.current_value,
                    instant_robustness=rule_eval.instant_robustness,
                    window_robustness=rule_eval.window_robustness,
                    ok_now=rule_eval.ok_now,
                    ok_window=rule_eval.ok_window,
                )
            )
        for group_eval in result.group_evaluations:
            self.group_belief_history[group_eval.name].append(
                GroupBeliefSnapshot(
                    sample_index=result.sample_index,
                    timestamp_ns=result.timestamp_ns,
                    name=group_eval.name,
                    rule_count=group_eval.rule_count,
                    window_robustness=group_eval.window_robustness,
                    ok_now=group_eval.ok_now,
                    ok_window=group_eval.ok_window,
                )
            )

    def _get_named_history(
        self,
        name: str,
    ) -> Optional[Deque[RuleBeliefSnapshot] | Deque[GroupBeliefSnapshot]]:
        if name in self.rule_belief_history:
            return self.rule_belief_history[name]
        if name in self.group_belief_history:
            return self.group_belief_history[name]
        return None

    def _load_rules_config(self) -> dict:
        if yaml is not None:
            try:
                config = yaml.safe_load(self.rules_text)
            except Exception as exc:
                raise ValueError("Unable to parse the STL rules YAML.") from exc
            if not isinstance(config, dict):
                raise ValueError("The STL rules YAML must contain a top-level mapping.")
            return config
        return _parse_drone_rules_yaml(self.rules_text)

    @staticmethod
    def _require_string(config: dict, key: str) -> str:
        value = config.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"`{key}` is required and must be a non-empty string.")
        return value

    @staticmethod
    def _string_or_default(config: dict, key: str, default: str) -> str:
        value = config.get(key, default)
        if not isinstance(value, str) or not value:
            raise ValueError(f"`{key}` must be a non-empty string.")
        return value

    @staticmethod
    def _validate_entity_name(entity_name: str, rule_name: str) -> None:
        if entity_name not in ENTITY_SLICES:
            available_entities = ", ".join(sorted(ENTITY_SLICES))
            raise ValueError(
                f"Unknown entity `{entity_name}` in rule `{rule_name}`. "
                f"Available entities: {available_entities}."
            )

    def _resolve_numeric_reference(self, value: object) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if value == "threshold_m":
            return self.threshold_m
        raise ValueError(
            "Numeric fields in the STL rules YAML must be numbers or the `threshold_m` reference."
        )

    def _resolve_interval(
        self,
        interval_ref: object,
        rule_name: str,
    ) -> tuple[int, int]:
        if not isinstance(interval_ref, list) or len(interval_ref) != 2:
            raise ValueError(
                f"`temporal.interval` must be a two-element list for rule `{rule_name}`."
            )
        start = self._resolve_interval_value(interval_ref[0], rule_name=rule_name)
        end = self._resolve_interval_value(interval_ref[1], rule_name=rule_name)
        if start < 0 or end < start:
            raise ValueError(f"Invalid interval [{start}, {end}] for rule `{rule_name}`.")
        return start, end

    def _resolve_interval_value(self, value: object, rule_name: str) -> int:
        if isinstance(value, int):
            return value
        if value == "window_end":
            return self.window_steps - 1
        raise ValueError(
            f"Unsupported interval value `{value}` in rule `{rule_name}`. "
            "Use integers or `window_end`."
        )

    @staticmethod
    def _instant_robustness(current_value: float, threshold: float, op: str) -> float:
        if op == ">":
            return current_value - threshold
        if op == "<":
            return threshold - current_value
        raise ValueError(f"Unsupported operator `{op}`.")

    @staticmethod
    def _build_distance_metric(lhs: str, rhs: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
        lhs_slice = ENTITY_SLICES[lhs]
        rhs_slice = ENTITY_SLICES[rhs]

        def distance_metric(trace: jnp.ndarray) -> jnp.ndarray:
            lhs_points = trace[:, lhs_slice]
            rhs_points = trace[:, rhs_slice]
            return jnp.linalg.norm(rhs_points - lhs_points, axis=1)

        return distance_metric

    @staticmethod
    def _build_height_metric(entity: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
        entity_slice = ENTITY_SLICES[entity]
        z_index = entity_slice.stop - 1

        def height_metric(trace: jnp.ndarray) -> jnp.ndarray:
            return trace[:, z_index]

        return height_metric

    @staticmethod
    def _as_pose_vector(value: jnp.ndarray) -> jnp.ndarray:
        vector = jnp.asarray(value, dtype=jnp.float32)
        if vector.shape != (3,):
            raise ValueError(f"Expected a 3D pose vector with shape (3,), got {vector.shape}.")
        return vector

    @staticmethod
    def _robustness_value(formula, trace: jnp.ndarray) -> float:
        return float(formula.robustness(trace))


def _parse_drone_rules_yaml(text: str) -> dict:
    """
    Parse the salvaged drone_rules.yaml when PyYAML is unavailable.

    This intentionally supports only the subset used by drone_rules.yaml:
    scalar fields, nested mappings, list-of-scalars, and list-of-mappings.
    Install PyYAML if the rules file grows beyond that subset.
    """

    lines = [
        line.rstrip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    root: dict[str, object] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.startswith(" "):
            key, value = _split_key_value(line)
            if value is not None:
                root[key] = _parse_scalar(value)
                i += 1
                continue
            if key == "defaults":
                defaults, i = _parse_defaults(lines, i + 1)
                root[key] = defaults
                continue
            if key == "rules":
                rules, i = _parse_rules(lines, i + 1)
                root[key] = rules
                continue
        raise ValueError(f"Unsupported YAML line: {line}")
    return root


def _parse_defaults(lines: list[str], i: int) -> tuple[dict, int]:
    defaults: dict[str, object] = {}
    while i < len(lines):
        line = lines[i]
        if not line.startswith("  "):
            break
        stripped = line.strip()
        key, value = _split_key_value(stripped)
        if value is not None:
            defaults[key] = _parse_scalar(value)
            i += 1
            continue
        if key == "temporal":
            temporal, i = _parse_temporal(lines, i + 1, base_indent=4)
            defaults[key] = temporal
            continue
        raise ValueError(f"Unsupported defaults line: {line}")
    return defaults, i


def _parse_rules(lines: list[str], i: int) -> tuple[list[dict], int]:
    rules: list[dict] = []
    current: Optional[dict] = None
    while i < len(lines):
        line = lines[i]
        if not line.startswith("  "):
            break
        stripped = line.strip()
        if stripped.startswith("- "):
            current = {}
            rules.append(current)
            remainder = stripped[2:]
            if remainder:
                key, value = _split_key_value(remainder)
                current[key] = _parse_scalar(value or "")
            i += 1
            continue
        if current is None:
            raise ValueError(f"Rule field before rule item: {line}")
        key, value = _split_key_value(stripped)
        if value is not None:
            current[key] = _parse_scalar(value)
            i += 1
            continue
        if key == "temporal":
            temporal, i = _parse_temporal(lines, i + 1, base_indent=6)
            current[key] = temporal
            continue
        raise ValueError(f"Unsupported rule line: {line}")
    return rules, i


def _parse_temporal(lines: list[str], i: int, base_indent: int) -> tuple[dict, int]:
    temporal: dict[str, object] = {}
    prefix = " " * base_indent
    while i < len(lines):
        line = lines[i]
        if not line.startswith(prefix):
            break
        stripped = line.strip()
        key, value = _split_key_value(stripped)
        if key == "interval" and value is None:
            values: list[object] = []
            i += 1
            item_prefix = " " * (base_indent + 2)
            while i < len(lines) and lines[i].startswith(item_prefix):
                item = lines[i].strip()
                if not item.startswith("- "):
                    break
                values.append(_parse_scalar(item[2:]))
                i += 1
            temporal[key] = values
            continue
        if value is not None:
            temporal[key] = _parse_scalar(value)
            i += 1
            continue
        raise ValueError(f"Unsupported temporal line: {line}")
    return temporal, i


def _split_key_value(line: str) -> tuple[str, Optional[str]]:
    if ":" not in line:
        raise ValueError(f"Expected key/value line: {line}")
    key, value = line.split(":", 1)
    value = value.strip()
    return key.strip(), value if value else None


def _parse_scalar(value: str) -> object:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value
