"""Small CLI for the project scaffold."""

from __future__ import annotations

import argparse
from typing import Iterable

from .doctor import build_report, print_report
from .project import BENCHMARK_LINES, FILES_LINES, LOOP_LINES, PROJECT_NAME, TAGLINE, THESIS_LINES


def _format_block(title: str, lines: Iterable[str]) -> str:
    body = "\n".join(f"- {line}" for line in lines)
    return f"{title}\n{body}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROJECT_NAME,
        description=TAGLINE,
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("thesis", help="Show the project thesis.")
    subparsers.add_parser("loop", help="Show the autoresearch loop.")
    subparsers.add_parser("files", help="Show the in-scope files.")
    subparsers.add_parser("benchmark", help="Show the benchmark design.")
    doctor_parser = subparsers.add_parser("doctor", help="Run lightweight environment checks.")
    doctor_parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "thesis":
        print(_format_block("Project Thesis", THESIS_LINES))
        return
    if args.command == "loop":
        print(_format_block("Autoresearch Loop", LOOP_LINES))
        return
    if args.command == "files":
        print(_format_block("In-Scope Files", FILES_LINES))
        return
    if args.command == "benchmark":
        print(_format_block("Benchmark Design", BENCHMARK_LINES))
        return
    if args.command == "doctor":
        report = build_report()
        print_report(report, json_output=args.json)
        raise SystemExit(0 if report["ready_for_autoresearch"] else 1)

    parser.print_help()


if __name__ == "__main__":
    main()
