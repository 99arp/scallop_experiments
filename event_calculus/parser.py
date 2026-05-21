from __future__ import annotations

import csv
import io

from .models import ProbabilisticFact


def parse_scallop_fact_line(
    line: str,
    *,
    expected_relation_name: str = "stl_predicate_probability",
) -> ProbabilisticFact | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or stripped.startswith("//"):
        return None
    if not stripped.endswith("."):
        raise ValueError(f"Fact line must end with `.`: {line!r}")

    body = stripped[:-1].strip()
    open_paren = body.find("(")
    close_paren = body.rfind(")")
    if open_paren <= 0 or close_paren != len(body) - 1:
        raise ValueError(f"Invalid fact syntax: {line!r}")

    relation_name = body[:open_paren].strip()
    if relation_name != expected_relation_name:
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
    expected_relation_name: str = "stl_predicate_probability",
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


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"Expected boolean `true` or `false`, got {value!r}")
