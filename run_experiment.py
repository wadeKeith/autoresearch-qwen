"""Run one fixed train+eval experiment while keeping train.py as the mutable file."""

from __future__ import annotations

import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Maximum wall-clock seconds before a step is killed.
TRAIN_TIMEOUT = 7200   # 2 hours
EVAL_TIMEOUT = 14400   # 4 hours


class TimeoutError(Exception):
    pass


def _run_script(args: list[str], timeout: int | None = None) -> str:
    """Run a Python script, streaming stderr to the terminal, capturing stdout."""
    try:
        proc = subprocess.run(
            [sys.executable, *args],
            cwd=ROOT,
            check=True,
            capture_output=False,
            text=True,
            stdout=subprocess.PIPE,
            timeout=timeout,
        )
        return proc.stdout
    except subprocess.TimeoutExpired:
        raise
    except subprocess.CalledProcessError as exc:
        # Print whatever stdout we got before the crash for debugging
        if exc.stdout:
            print(exc.stdout, file=sys.stderr)
        raise


def _extract_float(name: str, text: str) -> float:
    match = re.search(rf"^{re.escape(name)}:\s+([0-9.]+)", text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Could not find {name} in output:\n{text}")
    return float(match.group(1))


def main() -> None:
    started = time.perf_counter()
    try:
        train_output = _run_script(["train.py"], timeout=TRAIN_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"FAIL: training exceeded {TRAIN_TIMEOUT}s timeout", file=sys.stderr)
        sys.exit(1)
    try:
        eval_output = _run_script(["evaluate.py", "--split", "validation"], timeout=EVAL_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"FAIL: evaluation exceeded {EVAL_TIMEOUT}s timeout", file=sys.stderr)
        sys.exit(1)
    total_seconds = time.perf_counter() - started

    train_seconds = _extract_float("train_seconds", train_output)
    val_score = _extract_float("val_score", eval_output)

    print("---")
    print(f"val_score:      {val_score:.6f}")
    print(f"train_seconds:  {train_seconds:.2f}")
    print(f"total_seconds:  {total_seconds:.2f}")


if __name__ == "__main__":
    main()
