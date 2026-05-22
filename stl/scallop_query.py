"""Scallop query server — reads a JSON request from stdin, returns JSON to stdout."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def _free_vars(call: str) -> list[str]:
    """Return lowercase free-variable names from the argument list of a Scallop call."""
    m = re.match(r"^\w+\s*\((.*)\)$", call.strip(), re.DOTALL)
    if not m:
        return []
    args_str = m.group(1)
    no_strings = re.sub(r'"[^"]*"', '""', args_str)
    skip = {"true", "false", "not", "and", "or", "if", "then", "else"}
    seen: set[str] = set()
    result: list[str] = []
    for c in re.findall(r"\b([a-z_]\w*)\b", no_strings):
        if c not in skip and c not in seen:
            seen.add(c)
            result.append(c)
    return result


def run_query(program: str, intervals: list[dict]) -> dict:
    import scallopy

    ctx = scallopy.ScallopContext(provenance="minmaxprob")
    ctx.add_relation("holdsFor", (str, str, int, int))
    if intervals:
        ctx.add_facts("holdsFor", [
            (float(iv["p_min"]), (iv["source_kind"], iv["name"], int(iv["start"]), int(iv["end"])))
            for iv in intervals
        ])

    # Fold continuation lines onto their keyword line
    stmts: list[str] = []
    current = ""
    for raw in program.replace("\r\n", "\n").split("\n"):
        line = raw.strip()
        if not line or line.startswith("%") or line.startswith("//"):
            continue
        if line.startswith(("rel ", "type ", "query ")):
            if current:
                stmts.append(current)
            current = line
        else:
            current = (current + " " + line).strip() if current else line
    if current:
        stmts.append(current)

    user_relations: set[str] = set()
    rules: list[str] = []
    query_call: str | None = None
    query_relation: str | None = None

    for stmt in stmts:
        stmt = stmt.rstrip(" ;,")
        if not stmt or stmt.startswith("type "):
            continue
        if stmt.startswith("rel "):
            body = stmt[4:].strip()
            m = re.match(r"(\w+)\s*\(", body)
            if m:
                user_relations.add(m.group(1))
            rules.append(body.replace(" = ", " :- ", 1))
        elif stmt.startswith("query "):
            query_call = stmt[6:].strip()
            m = re.match(r"^(\w+)", query_call)
            if m:
                query_relation = m.group(1)

    if not query_relation:
        return {"ok": False, "error": "No query statement found"}

    # Atomic query against a base relation — wrap in a helper
    if query_relation not in user_relations:
        free = _free_vars(query_call)
        sig = ", ".join(free)
        rules.append(f"qresult({sig}) :- {query_call}")
        query_relation = "qresult"

    for rule in rules:
        try:
            ctx.add_rule(rule)
        except Exception as exc:
            return {"ok": False, "error": f"Rule syntax error in {rule!r}: {exc}"}

    try:
        ctx.run()
    except Exception as exc:
        return {"ok": False, "error": f"Execution error: {exc}"}

    try:
        raw_results = list(ctx.relation(query_relation))
    except Exception as exc:
        return {"ok": False, "error": f"Cannot read relation '{query_relation}': {exc}"}

    answers = []
    for row in raw_results:
        if (
            isinstance(row, tuple)
            and len(row) == 2
            and isinstance(row[0], (int, float))
            and isinstance(row[1], tuple)
        ):
            prob, args = row
            answers.append({"probability": round(float(prob), 9), "tuple": list(args)})
        else:
            tup = list(row) if isinstance(row, tuple) else [row]
            answers.append({"probability": 1.0, "tuple": tup})

    return {"ok": True, "query": query_call, "answers": answers, "answer_count": len(answers)}


def main() -> None:
    try:
        body = json.loads(sys.stdin.read())
    except json.JSONDecodeError as exc:
        json.dump({"ok": False, "error": f"Invalid JSON input: {exc}"}, sys.stdout)
        return

    query_program = str(body.get("query", ""))
    json_path_str = str(body.get("json_path", ""))
    intervals: list[dict] = []
    if json_path_str:
        try:
            data = json.loads(Path(json_path_str).read_text(encoding="utf-8"))
            intervals = data.get("intervals", [])
        except Exception:
            pass

    json.dump(run_query(query_program, intervals), sys.stdout)


if __name__ == "__main__":
    main()
