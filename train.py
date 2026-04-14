"""Mutable training script for autoresearch-qwen-mlx.

Autoresearch should primarily edit this file.
"""

from __future__ import annotations

import argparse
import random
import shutil
import time
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch_qwen_mlx.config import (
    ADAPTER_DIR,
    IMAGE_MAX_PIXELS,
    IMAGE_MIN_PIXELS,
    TRAINED_MODEL_DIR,
    TRAINER_OUTPUT_DIR,
    TRAIN_SPLIT,
)
from autoresearch_qwen_mlx.docvqa import DocVQASplitDataset, ensure_snapshot_exists
from autoresearch_qwen_mlx.hub import configure_hub_env, resolve_base_model_source
from autoresearch_qwen_mlx.scoring import anls_score, canonicalize

configure_hub_env()

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

# Main editable surface for autoresearch.
FINETUNE_MODE = "lora"
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 100
LR_SCHEDULER_TYPE = "cosine"
NUM_TRAIN_EPOCHS = 10
MAX_STEPS = 2000
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
MAX_GRAD_NORM = 1.0
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 3
OPTIMIZER = "adamw_torch"

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

GRADIENT_CHECKPOINTING = True
INCLUDE_QUESTION_TYPE_HINTS = False

USER_PROMPT_PREFIX = "Read the document image and answer the question with only the exact value from the page."
ANSWER_STYLE = "Return only the answer value."


def resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    # On MPS, load in float32 for training to avoid dtype mismatches
    # when Trainer runs without AMP (fp16/bf16 both disabled).
    return torch.float32


def load_training_dataset() -> DocVQASplitDataset:
    ensure_snapshot_exists()
    return DocVQASplitDataset(TRAIN_SPLIT)


def select_training_answer(example: dict[str, Any]) -> str:
    """Randomly pick one reference answer so the model sees all valid phrasings across epochs."""
    return random.choice(example["answers"])


def build_train_messages(example: dict[str, Any]) -> list[dict[str, Any]]:
    prompt_lines = [USER_PROMPT_PREFIX]
    if INCLUDE_QUESTION_TYPE_HINTS and example["question_types"]:
        prompt_lines.append(f"Question types: {', '.join(example['question_types'])}")
    prompt_lines.append(f"Question: {example['question']}")
    prompt_lines.append(ANSWER_STYLE)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": example["image"]},
                {"type": "text", "text": "\n".join(prompt_lines)},
            ],
        }
    ]


class DocVQACollator:
    def __init__(self, processor: AutoProcessor) -> None:
        self.processor = processor
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
        self.processor.tokenizer.padding_side = "right"

    def _encode_prompt(self, messages: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    def _encode_full_turn(self, messages: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        return self.processor.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_ids_list = []
        attention_mask_list = []
        mm_token_type_ids_list = []
        labels_list = []
        pixel_values_list = []
        image_grid_thw_list = []
        answer_texts = []

        for example in examples:
            answer = select_training_answer(example)
            prompt_messages = build_train_messages(example)
            full_messages = prompt_messages + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                }
            ]

            prompt_inputs = self._encode_prompt(prompt_messages)
            full_inputs = self._encode_full_turn(full_messages)

            input_ids = full_inputs["input_ids"][0]
            attention_mask = full_inputs["attention_mask"][0]
            mm_token_type_ids = full_inputs["mm_token_type_ids"][0]
            labels = input_ids.clone()

            prompt_length = prompt_inputs["input_ids"].shape[1]
            labels[:prompt_length] = -100

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            mm_token_type_ids_list.append(mm_token_type_ids)
            labels_list.append(labels)
            pixel_values_list.append(full_inputs["pixel_values"])
            image_grid_thw_list.append(full_inputs["image_grid_thw"])
            answer_texts.append(answer)

        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        return {
            "input_ids": torch.nn.utils.rnn.pad_sequence(
                input_ids_list,
                batch_first=True,
                padding_value=self.processor.tokenizer.pad_token_id,
            ),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(
                attention_mask_list,
                batch_first=True,
                padding_value=0,
            ),
            "mm_token_type_ids": torch.nn.utils.rnn.pad_sequence(
                mm_token_type_ids_list,
                batch_first=True,
                padding_value=0,
            ),
            "pixel_values": torch.cat(pixel_values_list, dim=0),
            "image_grid_thw": torch.cat(image_grid_thw_list, dim=0),
            "labels": labels,
            "answer_texts": answer_texts,
        }


def compute_batch_anls(
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    answer_texts: list[str],
    processor: AutoProcessor,
) -> float | None:
    prediction_ids = logits.detach().argmax(dim=-1)
    scores: list[float] = []

    for row_index, gold_answer in enumerate(answer_texts):
        valid_positions = labels[row_index] != -100
        if not torch.any(valid_positions):
            continue
        predicted_text = processor.batch_decode(
            [prediction_ids[row_index][valid_positions].detach().cpu()],
            skip_special_tokens=True,
        )[0]
        scores.append(anls_score(canonicalize(predicted_text), (gold_answer,)))

    if not scores:
        return None
    return sum(scores) / len(scores)


class DocVQATrainer(Trainer):
    def __init__(self, *args, batch_metric_processor: AutoProcessor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.batch_metric_processor = batch_metric_processor

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        answer_texts = inputs.pop("answer_texts", None)
        labels = inputs.get("labels")
        outputs = model(**inputs)
        loss = outputs.loss

        if model.training and answer_texts is not None and labels is not None:
            with torch.no_grad():
                batch_anls = compute_batch_anls(
                    logits=outputs.logits,
                    labels=labels,
                    answer_texts=answer_texts,
                    processor=self.batch_metric_processor,
                )
            if batch_anls is not None:
                self.log({"train_batch_anls": batch_anls})

        if return_outputs:
            return loss, outputs
        return loss


def build_lora_config(model: torch.nn.Module) -> LoraConfig:
    found = {
        name.rsplit(".", 1)[-1]
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and name.rsplit(".", 1)[-1] in set(LORA_TARGET_MODULES)
    }
    if not found:
        raise ValueError("Could not find any target modules for LoRA.")
    return LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=sorted(found),
    )


def load_model_and_processor(*, finetune_mode: str) -> tuple[torch.nn.Module, AutoProcessor]:
    device = resolve_device()
    dtype = resolve_dtype(device)
    base_model_source = resolve_base_model_source()

    processor = AutoProcessor.from_pretrained(
        base_model_source,
        trust_remote_code=True,
        min_pixels=IMAGE_MIN_PIXELS,
        max_pixels=IMAGE_MAX_PIXELS,
    )
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_source,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False

    if finetune_mode == "lora":
        lora_config = build_lora_config(model)
        model = get_peft_model(model, lora_config)

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        if finetune_mode == "lora":
            model.enable_input_require_grads()

    return model, processor, device


def resolve_artifact_dir(finetune_mode: str) -> Path:
    if finetune_mode == "full":
        return TRAINED_MODEL_DIR
    if finetune_mode == "lora":
        return ADAPTER_DIR
    raise ValueError(f"Unsupported finetune mode: {finetune_mode}")


def reset_training_artifacts() -> None:
    for path in (ADAPTER_DIR, TRAINED_MODEL_DIR, TRAINER_OUTPUT_DIR):
        if path.exists():
            shutil.rmtree(path)


def build_training_arguments(device: torch.device) -> TrainingArguments:
    if TRAINER_OUTPUT_DIR.exists():
        shutil.rmtree(TRAINER_OUTPUT_DIR)
    TRAINER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    kwargs = {
        "output_dir": str(TRAINER_OUTPUT_DIR),
        "remove_unused_columns": False,
        "per_device_train_batch_size": PER_DEVICE_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_grad_norm": MAX_GRAD_NORM,
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "max_steps": MAX_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "lr_scheduler_type": LR_SCHEDULER_TYPE,
        "logging_steps": LOGGING_STEPS,
        "save_strategy": "no",
        "eval_strategy": "no",
        "report_to": [],
        "dataloader_num_workers": 0,
        "dataloader_pin_memory": False,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "save_total_limit": SAVE_TOTAL_LIMIT,
        "optim": OPTIMIZER,
    }
    if device.type == "cuda":
        kwargs["bf16"] = resolve_dtype(device) == torch.bfloat16
        kwargs["fp16"] = resolve_dtype(device) == torch.float16
    elif device.type == "mps":
        # MPS does not support the FP16 GradScaler workflow used by Trainer.
        # Keep fp16/bf16 disabled so training runs in float32 on MPS.
        # Model weights are loaded in float32 via resolve_dtype() to match.
        pass
    return TrainingArguments(**kwargs)


def run_training(*, finetune_mode: str = FINETUNE_MODE) -> dict[str, float | int | str]:
    artifact_dir = resolve_artifact_dir(finetune_mode)
    train_dataset = load_training_dataset()

    started = time.perf_counter()
    reset_training_artifacts()
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model, processor, device = load_model_and_processor(finetune_mode=finetune_mode)
    collator = DocVQACollator(processor)
    training_args = build_training_arguments(device)

    trainer = DocVQATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
        batch_metric_processor=processor,
    )
    trainer.train()
    trainer.save_model(artifact_dir)
    processor.save_pretrained(artifact_dir)

    elapsed = time.perf_counter() - started
    return {
        "train_seconds": elapsed,
        "examples": len(train_dataset),
        "artifact_dir": str(artifact_dir),
        "finetune_mode": finetune_mode,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Qwen3-VL-4B on the official DocVQA train split and evaluate on validation."
    )
    parser.add_argument("--finetune-mode", choices=("full", "lora"), default=FINETUNE_MODE)
    args = parser.parse_args()

    summary = run_training(finetune_mode=args.finetune_mode)
    print("---")
    print(f"train_seconds:      {summary['train_seconds']:.2f}")
    print(f"examples:           {summary['examples']}")
    print(f"artifact_dir:       {summary['artifact_dir']}")
    print(f"mode:               {summary['finetune_mode']}")


if __name__ == "__main__":
    main()
