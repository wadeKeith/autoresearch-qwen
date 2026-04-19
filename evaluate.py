"""Fixed evaluator for the autoresearch loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch_qwen.config import (
    ADAPTER_DIR,
    TEST_PREDICTIONS_PATH,
    TEST_SUBMISSION_PATH,
    TEST_SPLIT,
    TRAINED_MODEL_DIR,
    VAL_SPLIT,
)
from autoresearch_qwen.docvqa import DocVQASplitDataset, ensure_snapshot_exists, load_split_question_ids
from autoresearch_qwen.docvqa_eval import run_generation_eval, write_predictions
from autoresearch_qwen.hub import configure_hub_env, resolve_base_model_source
from autoresearch_qwen.submission import (
    build_docvqa_submission,
    validate_docvqa_submission,
    write_docvqa_submission,
)

configure_hub_env()

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor


def resolve_dtype() -> torch.dtype:
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_model_and_processor(*, use_trained_artifact: bool) -> tuple[torch.nn.Module, AutoProcessor, torch.device, str]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = resolve_dtype()

    artifact_label = "base"
    model_source = resolve_base_model_source()
    processor_source = resolve_base_model_source()

    if use_trained_artifact and TRAINED_MODEL_DIR.exists():
        model_source = str(TRAINED_MODEL_DIR)
        processor_source = str(TRAINED_MODEL_DIR)
        artifact_label = str(TRAINED_MODEL_DIR)

    processor = AutoProcessor.from_pretrained(processor_source)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "right"

    model = AutoModelForImageTextToText.from_pretrained(
        model_source,
        dtype=dtype,
    )
    if artifact_label == "base" and use_trained_artifact and ADAPTER_DIR.exists():
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
        artifact_label = str(ADAPTER_DIR)
    model.to(device)
    model.eval()
    return model, processor, device, artifact_label


def run_eval(
    *,
    split: str,
    use_trained_artifact: bool,
    verbose: bool = False,
    eval_batch_size: int = 4,
    predictions_output: Path | None = None,
    submission_output: Path | None = None,
) -> dict[str, float | int | str]:
    ensure_snapshot_exists()
    dataset = DocVQASplitDataset(split)
    model, processor, device, artifact_label = load_model_and_processor(
        use_trained_artifact=use_trained_artifact
    )
    summary = run_generation_eval(
        model=model,
        processor=processor,
        dataset=dataset,
        device=device,
        eval_batch_size=eval_batch_size,
        verbose=verbose,
    )

    output: dict[str, float | int | str] = {
        "split": split,
        "examples": summary["examples"],
        "total_seconds": summary["total_seconds"],
        "artifact": artifact_label,
    }
    if summary["score"] is not None:
        output["val_score"] = float(summary["score"])
        return output

    prediction_path = write_predictions(
        summary["predictions"],
        predictions_output or TEST_PREDICTIONS_PATH,
    )
    submission = build_docvqa_submission(summary["predictions"])
    submission_path = write_docvqa_submission(
        submission,
        submission_output or TEST_SUBMISSION_PATH,
    )
    expected_question_ids = load_split_question_ids(split)
    validation_errors = validate_docvqa_submission(
        submission,
        expected_question_ids=expected_question_ids,
    )
    if validation_errors:
        raise ValueError(
            "Submission preflight check failed:\n- " + "\n- ".join(validation_errors)
        )
    output["prediction_path"] = str(prediction_path)
    output["submission_path"] = str(submission_path)
    output["submission_ready"] = "yes"
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on official DocVQA validation or export blind test predictions.")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--split", choices=(VAL_SPLIT, TEST_SPLIT), default=VAL_SPLIT)
    parser.add_argument("--eval-batch-size", type=int, default=4,
                        help="Batch size for generation eval (default: 4). Use 1 to disable batching.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--predictions-output", type=Path, default=None)
    parser.add_argument("--submission-output", type=Path, default=None)
    args = parser.parse_args()

    summary = run_eval(
        split=args.split,
        use_trained_artifact=not args.base_only,
        verbose=args.verbose,
        eval_batch_size=args.eval_batch_size,
        predictions_output=args.predictions_output,
        submission_output=args.submission_output,
    )
    print("---")
    print(f"split:          {summary['split']}")
    if "val_score" in summary:
        print(f"val_score:      {summary['val_score']:.6f}")
    else:
        print("val_score:      unavailable")
        print(f"prediction_path:{summary['prediction_path']}")
        print(f"submission_path:{summary['submission_path']}")
        print(f"submit_ready:   {summary['submission_ready']}")
    print(f"examples:       {summary['examples']}")
    print(f"total_seconds:  {summary['total_seconds']:.2f}")
    print(f"artifact:       {summary['artifact']}")


if __name__ == "__main__":
    main()
