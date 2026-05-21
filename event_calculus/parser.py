from __future__ import annotations

import csv
import io

from .models import ProbabilisticFact
from .syntax import parse_ec_atom, split_top_level_args


def parse_scallop_fact_line(
    line: str,
    *,
    expected_relation_name: str = "happensAt",
) -> ProbabilisticFact | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("//"):
        return None
    if not stripped.endswith("."):
        raise ValueError(f"Fact line must end with `.`: {line!r}")

    body = stripped[:-1].strip()
    if "::" in body:
        return _parse_probabilistic_happens_at(
            body,
            expected_relation_name=expected_relation_name,
        )

    open_paren = body.find("(")
    close_paren = body.rfind(")")
    if open_paren <= 0 or close_paren != len(body) - 1:
        raise ValueError(f"Invalid fact syntax: {line!r}")

    relation_name = body[:open_paren].strip()
    if relation_name != "stl_predicate_probability":
        return None

    args_text = body[open_paren + 1 : close_paren]
    args = next(csv.reader(io.StringIO(args_text), skipinitialspace=True))
    if len(args) != 6:
        raise ValueError(
            f"{expected_relation_name} expects 6 arguments, got {len(args)}: {line!r}"
        )

    source_kind = args[0]
    name = args[1]
    sample_index = int(args[2])
    value = _parse_bool(args[3])
    probability = float(args[4])
    rho = float(args[5])
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Fact probability must be in [0, 1]: {line!r}")

    return ProbabilisticFact(
        relation_name=relation_name,
        source_kind=source_kind,
        name=name,
        sample_index=sample_index,
        value=value,
        probability=probability,
        rho=rho,
    )


def parse_scallop_facts(
    text: str,
    *,
    expected_relation_name: str = "happensAt",
) -> tuple[ProbabilisticFact, ...]:
    facts = []
    for line in text.splitlines():
        fact = parse_scallop_fact_line(
            line,
            expected_relation_name=expected_relation_name,
        )
        if fact is not None:
            facts.append(fact)
    return tuple(facts)


def parse_event_calculus_facts(
    text: str,
    *,
    expected_relation_name: str = "happensAt",
) -> tuple[ProbabilisticFact, ...]:
    return parse_scallop_facts(
        text,
        expected_relation_name=expected_relation_name,
    )


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Expected boolean `true` or `false`, got {value!r}")


def _parse_probabilistic_happens_at(
    body: str,
    *,
    expected_relation_name: str,
) -> ProbabilisticFact | None:
    probability_text, event_text = body.split("::", 1)
    probability = float(probability_text.strip())
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Fact probability must be in [0, 1]: {body!r}")

    relation_name, args_text = _parse_call(event_text.strip())
    if relation_name != expected_relation_name:
        return None
    args = split_top_level_args(args_text)
    if len(args) != 2:
        raise ValueError(f"{expected_relation_name} expects event and time: {body!r}")

    event_name, event_args_text = _parse_call(args[0])
    if event_name != "stl_predicate":
        raise ValueError(f"Only stl_predicate events are supported: {body!r}")
    event_args = split_top_level_args(event_args_text)
    if len(event_args) != 3:
        raise ValueError(f"stl_predicate expects source, name, value: {body!r}")

    return ProbabilisticFact(
        relation_name=relation_name,
        source_kind=parse_ec_atom(event_args[0]),
        name=parse_ec_atom(event_args[1]),
        sample_index=int(args[1]),
        value=_parse_bool(event_args[2]),
        probability=probability,
        rho=0.0,
    )


def _parse_call(text: str) -> tuple[str, str]:
    open_paren = text.find("(")
    close_paren = text.rfind(")")
    if open_paren <= 0 or close_paren != len(text) - 1:
        raise ValueError(f"Invalid call syntax: {text!r}")
    return text[:open_paren].strip(), text[open_paren + 1 : close_paren]
