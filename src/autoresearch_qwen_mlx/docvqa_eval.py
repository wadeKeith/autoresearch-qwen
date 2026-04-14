"""Shared generation-based evaluation helpers for DocVQA."""

from __future__ import annotations

import json
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from transformers import AutoProcessor

from .config import EVAL_PROMPT_PREFIX, MAX_NEW_TOKENS
from .scoring import anls_score, canonicalize


def build_eval_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = "\n".join(
        [
            EVAL_PROMPT_PREFIX,
            f"Question: {example['question']}",
            "Return only the answer value.",
        ]
    )
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def run_generation_eval(
    *,
    model: torch.nn.Module,
    processor: AutoProcessor,
    dataset: Any,
    device: torch.device,
    verbose: bool = False,
) -> dict[str, Any]:
    scores: list[float] = []
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()

    for index in range(len(dataset)):
        example = dataset[index]
        inputs = processor.apply_chat_template(
            build_eval_messages(example),
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        prompt_length = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
            )

        prediction = canonicalize(
            processor.batch_decode(generated_ids[:, prompt_length:], skip_special_tokens=True)[0]
        )
        answers = example["answers"]
        prediction_record = {
            "example_id": example["example_id"],
            "question_id": example["question_id"],
            "question": example["question"],
            "prediction": prediction,
        }
        predictions.append(prediction_record)

        if answers:
            score = anls_score(prediction, answers)
            scores.append(score)
            if verbose:
                print(
                    f"{example['example_id']}\tgold={answers[0]}\tpred={prediction}\tscore={score:.3f}"
                )
        elif verbose:
            print(f"{example['example_id']}\tpred={prediction}")

        if device.type == "mps":
            torch.mps.empty_cache()

    elapsed = time.perf_counter() - started
    return {
        "score": mean(scores) if scores else None,
        "examples": len(dataset),
        "total_seconds": elapsed,
        "predictions": predictions,
    }


def write_predictions(predictions: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(json.dumps(prediction, ensure_ascii=True))
            handle.write("\n")
    return output_path
