from __future__ import annotations

import argparse
from pathlib import Path

from .engine import infer_event_calculus_from_file
from .models import EventCalculusConfig
from .render import render_event_calculus_facts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run probabilistic Event Calculus over exported STL Scallop facts.",
    )
    parser.add_argument("facts_path", help="Input stl_predicate_probability fact file")
    parser.add_argument(
        "--output",
        "-o",
        help="Output EC fact file. Prints to stdout when omitted.",
    )
    parser.add_argument(
        "--hold-threshold",
        type=float,
        default=0.5,
        help="Probability threshold used for ec_interval extraction.",
    )
    args = parser.parse_args(argv)

    result = infer_event_calculus_from_file(
        args.facts_path,
        config=EventCalculusConfig(hold_threshold=args.hold_threshold),
    )
    rendered = render_event_calculus_facts(result)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
