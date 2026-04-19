"""Shared generation-based evaluation helpers for DocVQA."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

import torch
from transformers import AutoProcessor

from .config import (
    DEFAULT_PROMPT_PREFIX,
    DEFAULT_PROMPT_SUFFIX,
    MAX_NEW_TOKENS,
    PROMPT_CONFIG_PATH,
)
from .scoring import anls_score, canonicalize


def load_prompt_config(path: Path = PROMPT_CONFIG_PATH) -> dict[str, Any]:
    """Load prompt config written by train.py, falling back to defaults."""
    if path.exists():
        config = json.loads(path.read_text(encoding="utf-8"))
        return {
            "prompt_prefix": config.get("prompt_prefix", DEFAULT_PROMPT_PREFIX),
            "prompt_suffix": config.get("prompt_suffix", DEFAULT_PROMPT_SUFFIX),
            "include_question_type_hints": config.get("include_question_type_hints", False),
        }
    return {
        "prompt_prefix": DEFAULT_PROMPT_PREFIX,
        "prompt_suffix": DEFAULT_PROMPT_SUFFIX,
        "include_question_type_hints": False,
    }


def build_eval_messages(
    example: dict[str, Any],
    prompt_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if prompt_config is None:
        prompt_config = load_prompt_config()
    prompt_lines = [prompt_config["prompt_prefix"]]
    if prompt_config.get("include_question_type_hints") and example.get("question_types"):
        prompt_lines.append(f"Question types: {', '.join(example['question_types'])}")
    prompt_lines.append(f"Question: {example['question']}")
    prompt_lines.append(prompt_config["prompt_suffix"])
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": "\n".join(prompt_lines)},
            ],
        }
    ]


def _prepare_single(
    example: dict[str, Any],
    processor: AutoProcessor,
    prompt_config: dict[str, Any],
) -> dict[str, torch.Tensor]:
    return processor.apply_chat_template(
        build_eval_messages(example, prompt_config),
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )


def _batch_generate(
    *,
    model: torch.nn.Module,
    processor: AutoProcessor,
    examples: list[dict[str, Any]],
    device: torch.device,
    prompt_config: dict[str, Any],
) -> list[str]:
    """Run batched greedy generation and return canonicalized predictions."""
    all_inputs = [_prepare_single(ex, processor, prompt_config) for ex in examples]

    prompt_lengths = [inp["input_ids"].shape[1] for inp in all_inputs]
    max_prompt_len = max(prompt_lengths)

    pad_id = processor.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = processor.tokenizer.eos_token_id

    padded_input_ids = []
    padded_attention_mask = []
    padded_mm_token_type_ids = []
    all_pixel_values = []
    all_image_grid_thw = []
    has_mm_type = "mm_token_type_ids" in all_inputs[0]

    for inp in all_inputs:
        seq_len = inp["input_ids"].shape[1]
        pad_len = max_prompt_len - seq_len

        if pad_len > 0:
            padded_input_ids.append(torch.cat([
                torch.full((1, pad_len), pad_id, dtype=torch.long),
                inp["input_ids"],
            ], dim=1))
            padded_attention_mask.append(torch.cat([
                torch.zeros(1, pad_len, dtype=torch.long),
                inp["attention_mask"],
            ], dim=1))
            if has_mm_type:
                padded_mm_token_type_ids.append(torch.cat([
                    torch.zeros(1, pad_len, dtype=inp["mm_token_type_ids"].dtype),
                    inp["mm_token_type_ids"],
                ], dim=1))
        else:
            padded_input_ids.append(inp["input_ids"])
            padded_attention_mask.append(inp["attention_mask"])
            if has_mm_type:
                padded_mm_token_type_ids.append(inp["mm_token_type_ids"])

        all_pixel_values.append(inp["pixel_values"])
        all_image_grid_thw.append(inp["image_grid_thw"])

    batch = {
        "input_ids": torch.cat(padded_input_ids, dim=0).to(device),
        "attention_mask": torch.cat(padded_attention_mask, dim=0).to(device),
        "pixel_values": torch.cat(all_pixel_values, dim=0).to(device),
        "image_grid_thw": torch.cat(all_image_grid_thw, dim=0).to(device),
    }
    if has_mm_type and padded_mm_token_type_ids:
        batch["mm_token_type_ids"] = torch.cat(padded_mm_token_type_ids, dim=0).to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **batch,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    predictions = []
    for i in range(len(examples)):
        gen_tokens = generated_ids[i, max_prompt_len:]
        text = processor.batch_decode([gen_tokens], skip_special_tokens=True)[0]
        predictions.append(canonicalize(text))
    return predictions


def _single_generate(
    *,
    model: torch.nn.Module,
    processor: AutoProcessor,
    example: dict[str, Any],
    device: torch.device,
    prompt_config: dict[str, Any],
) -> str:
    """Fallback: generate for a single example."""
    inputs = _prepare_single(example, processor, prompt_config)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    prompt_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
    return canonicalize(
        processor.batch_decode(generated_ids[:, prompt_length:], skip_special_tokens=True)[0]
    )


def run_generation_eval(
    *,
    model: torch.nn.Module,
    processor: AutoProcessor,
    dataset: Any,
    device: torch.device,
    eval_batch_size: int = 1,
    verbose: bool = False,
) -> dict[str, Any]:
    total = len(dataset)
    prompt_config = load_prompt_config()

    prompt_src = "prompt_config.json" if PROMPT_CONFIG_PATH.exists() else "defaults"
    print(f"eval prompt source: {prompt_src}", file=sys.stderr)

    scores: list[float] = []
    predictions: list[dict[str, Any]] = []
    started = time.perf_counter()

    index = 0
    while index < total:
        batch_end = min(index + eval_batch_size, total)
        batch_examples = [dataset[i] for i in range(index, batch_end)]
        batch_size_actual = len(batch_examples)

        if batch_size_actual > 1:
            try:
                batch_preds = _batch_generate(
                    model=model,
                    processor=processor,
                    examples=batch_examples,
                    device=device,
                    prompt_config=prompt_config,
                )
            except (RuntimeError, torch.OutOfMemoryError):
                torch.cuda.empty_cache()
                batch_preds = [
                    _single_generate(
                        model=model,
                        processor=processor,
                        example=ex,
                        device=device,
                        prompt_config=prompt_config,
                    )
                    for ex in batch_examples
                ]
        else:
            batch_preds = [
                _single_generate(
                    model=model,
                    processor=processor,
                    example=batch_examples[0],
                    device=device,
                    prompt_config=prompt_config,
                )
            ]

        for j, (example, prediction) in enumerate(zip(batch_examples, batch_preds)):
            prediction_record = {
                "example_id": example["example_id"],
                "question_id": example["question_id"],
                "question": example["question"],
                "prediction": prediction,
            }
            predictions.append(prediction_record)

            answers = example["answers"]
            if answers:
                score = anls_score(prediction, answers)
                scores.append(score)
                if verbose:
                    print(
                        f"{example['example_id']}\tgold={answers[0]}\tpred={prediction}\tscore={score:.3f}"
                    )
            elif verbose:
                print(f"{example['example_id']}\tpred={prediction}")

        done = index + batch_size_actual
        if done % 50 < batch_size_actual or done == total:
            elapsed_so_far = time.perf_counter() - started
            rate = done / elapsed_so_far if elapsed_so_far > 0 else 0
            remaining = (total - done) / rate if rate > 0 else 0
            running_score = mean(scores) if scores else 0.0
            print(
                f"\reval {done}/{total} | "
                f"score: {running_score:.4f} | "
                f"rate: {rate:.1f} ex/s | "
                f"remaining: {remaining:.0f}s",
                end="", flush=True, file=sys.stderr,
            )

        if device.type == "cuda":
            torch.cuda.empty_cache()

        index = batch_end

    if total > 0:
        print(file=sys.stderr)

    elapsed = time.perf_counter() - started
    return {
        "score": mean(scores) if scores else None,
        "examples": total,
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
