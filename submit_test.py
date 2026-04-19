"""One-command DocVQA test export, local validation, and final packaging."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch_qwen.config import (
    TEST_PREDICTIONS_PATH,
    TEST_SPLIT,
    TEST_SUBMISSION_BUNDLE_PATH,
    TEST_SUBMISSION_PATH,
)
from autoresearch_qwen.docvqa import ensure_snapshot_exists, load_split_question_ids
from autoresearch_qwen.submission import (
    load_docvqa_submission,
    validate_docvqa_submission,
    write_submission_bundle,
)
from evaluate import run_eval


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export DocVQA test predictions, run local submission checks, and package the final submission."
    )
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--submission-output", type=Path, default=None)
    parser.add_argument("--bundle-output", type=Path, default=None)
    args = parser.parse_args()

    ensure_snapshot_exists()
    summary = run_eval(
        split=TEST_SPLIT,
        use_trained_artifact=not args.base_only,
        verbose=args.verbose,
        predictions_output=args.predictions_output,
        submission_output=args.submission_output,
    )
    submission_path = Path(str(summary["submission_path"]))
    prediction_path = Path(str(summary["prediction_path"]))
    expected_question_ids = load_split_question_ids(TEST_SPLIT)
    submission = load_docvqa_submission(submission_path)
    errors = validate_docvqa_submission(submission, expected_question_ids=expected_question_ids)
    if errors:
        raise ValueError("Submission validation failed:\n- " + "\n- ".join(errors))

    bundle_path = write_submission_bundle(
        submission_path=submission_path,
        predictions_path=prediction_path,
        bundle_path=args.bundle_output or TEST_SUBMISSION_BUNDLE_PATH,
        manifest={
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "split": TEST_SPLIT,
            "examples": summary["examples"],
            "artifact": summary["artifact"],
            "submission_ready": summary["submission_ready"],
            "submission_file": str(submission_path),
            "predictions_file": str(prediction_path),
        },
    )

    print("---")
    print(f"split:            {TEST_SPLIT}")
    print(f"examples:         {summary['examples']}")
    print(f"artifact:         {summary['artifact']}")
    print(f"prediction_path:  {prediction_path}")
    print(f"submission_path:  {submission_path}")
    print(f"bundle_path:      {bundle_path}")
    print(f"submission_ready: {summary['submission_ready']}")
    print("status:           ok")


if __name__ == "__main__":
    main()
