from __future__ import annotations

import re


_ATOM_RE = re.compile(r"^[a-z][a-zA-Z0-9_]*$")


def stl_predicate_term(source_kind: str, name: str, value: bool | None = None) -> str:
    args = [ec_atom(source_kind), ec_atom(name)]
    if value is not None:
        args.append("true" if value else "false")
    return f"stl_predicate({','.join(args)})"


def fluent_term(source_kind: str, name: str) -> str:
    return f"{stl_predicate_term(source_kind, name)}=true"


def ec_atom(value: str) -> str:
    if _ATOM_RE.match(value):
        return value
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def parse_ec_atom(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == "'" and stripped[-1] == "'":
        return stripped[1:-1].replace("\\'", "'").replace("\\\\", "\\")
    return stripped


def split_top_level_args(text: str) -> list[str]:
    args = []
    start = 0
    depth = 0
    quote: str | None = None
    escaped = False
    for index, char in enumerate(text):
        if quote is not None:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            continue
        if char in {"'", '"'}:
            quote = char
            continue
        if char == "(":
            depth += 1
            continue
        if char == ")":
            depth -= 1
            continue
        if char == "," and depth == 0:
            args.append(text[start:index].strip())
            start = index + 1
    args.append(text[start:].strip())
    return args
