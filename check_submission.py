"""Validate a DocVQA test submission file against the official hidden-answer split."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch_qwen.config import TEST_SPLIT, TEST_SUBMISSION_PATH
from autoresearch_qwen.docvqa import ensure_snapshot_exists, load_split_question_ids
from autoresearch_qwen.submission import load_docvqa_submission, validate_docvqa_submission


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a DocVQA test submission JSON file.")
    parser.add_argument("--submission-file", type=Path, default=TEST_SUBMISSION_PATH)
    args = parser.parse_args()

    ensure_snapshot_exists()
    expected_question_ids = load_split_question_ids(TEST_SPLIT)
    submission = load_docvqa_submission(args.submission_file)
    errors = validate_docvqa_submission(submission, expected_question_ids=expected_question_ids)

    print("---")
    print(f"submission_file: {args.submission_file}")
    print(f"expected_rows:   {len(expected_question_ids)}")
    print(f"errors:          {len(errors)}")
    if errors:
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)
    print("status:          ok")


if __name__ == "__main__":
    main()
