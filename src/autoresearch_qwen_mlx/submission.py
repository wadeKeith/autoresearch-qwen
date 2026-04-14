"""Build and validate DocVQA submission files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import zipfile

from .config import DOCVQA_SUBMISSION_DATASET_NAME, DOCVQA_SUBMISSION_VERSION, TEST_SPLIT


def build_docvqa_submission(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dataset_name": DOCVQA_SUBMISSION_DATASET_NAME,
        "dataset_split": TEST_SPLIT,
        "dataset_version": DOCVQA_SUBMISSION_VERSION,
        "data": [
            {
                "questionId": int(prediction["question_id"]),
                "answer": str(prediction["prediction"]).strip(),
            }
            for prediction in predictions
        ],
    }


def validate_docvqa_submission(
    submission: dict[str, Any],
    *,
    expected_question_ids: list[int],
) -> list[str]:
    errors: list[str] = []

    if submission.get("dataset_name") != DOCVQA_SUBMISSION_DATASET_NAME:
        errors.append(
            f"dataset_name must be {DOCVQA_SUBMISSION_DATASET_NAME!r}, got {submission.get('dataset_name')!r}."
        )
    if submission.get("dataset_split") != TEST_SPLIT:
        errors.append(f"dataset_split must be {TEST_SPLIT!r}, got {submission.get('dataset_split')!r}.")

    data = submission.get("data")
    if not isinstance(data, list):
        errors.append("data must be a list.")
        return errors

    seen: set[int] = set()
    predicted_question_ids: list[int] = []
    for index, row in enumerate(data):
        if not isinstance(row, dict):
            errors.append(f"data[{index}] must be an object.")
            continue
        if "questionId" not in row:
            errors.append(f"data[{index}] is missing questionId.")
            continue
        if "answer" not in row:
            errors.append(f"data[{index}] is missing answer.")
            continue

        question_id = row["questionId"]
        if not isinstance(question_id, int):
            errors.append(f"data[{index}].questionId must be an int.")
            continue
        predicted_question_ids.append(question_id)
        if question_id in seen:
            errors.append(f"Duplicate questionId detected: {question_id}.")
        seen.add(question_id)

        answer = row["answer"]
        if not isinstance(answer, str):
            errors.append(f"data[{index}].answer must be a string.")
            continue
        if not answer.strip():
            errors.append(f"data[{index}].answer must not be empty.")

    expected = set(expected_question_ids)
    predicted = set(predicted_question_ids)
    missing = sorted(expected - predicted)
    extras = sorted(predicted - expected)
    if missing:
        errors.append(f"Missing {len(missing)} questionId values from the expected split.")
    if extras:
        errors.append(f"Found {len(extras)} unexpected questionId values.")
    if len(data) != len(expected_question_ids):
        errors.append(
            f"Expected {len(expected_question_ids)} predictions, found {len(data)}."
        )
    return errors


def write_docvqa_submission(submission: dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(submission, ensure_ascii=True, indent=2), encoding="utf-8")
    return output_path


def load_docvqa_submission(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_submission_bundle(
    *,
    submission_path: Path,
    predictions_path: Path,
    bundle_path: Path,
    manifest: dict[str, Any],
) -> Path:
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(submission_path, arcname=submission_path.name)
        archive.write(predictions_path, arcname=predictions_path.name)
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=True, indent=2))
    return bundle_path
