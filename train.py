"""Mutable training script for autoresearch-qwen.

Autoresearch should primarily edit this file.
Supports single-GPU and multi-GPU training via torchrun / accelerate.
"""

from __future__ import annotations

import argparse
import json
import os
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

from autoresearch_qwen.config import (
    ADAPTER_DIR,
    TRAINED_MODEL_DIR,
    TRAINER_OUTPUT_DIR,
    TRAIN_SPLIT,
)
from autoresearch_qwen.docvqa import DocVQASplitDataset, ensure_snapshot_exists
from autoresearch_qwen.hub import configure_hub_env, resolve_base_model_source
from autoresearch_qwen.scoring import anls_score, canonicalize

configure_hub_env()

import torch
from torch.utils.data import Subset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments, TrainerCallback

# ---------------------------------------------------------------------------
# Training mode
# ---------------------------------------------------------------------------
FINETUNE_MODE = "lora"

# ---------------------------------------------------------------------------
# DeepSpeed — set to a config path to enable, or None to disable.
# Options: "configs/zero2.json", "configs/zero3.json",
#          "configs/zero2_offload.json", "configs/zero3_offload.json"
# ZeRO-2: shards optimizer states + gradients across GPUs (recommended start)
# ZeRO-3: also shards parameters — saves more memory but slower
# *_offload: additionally offloads to CPU RAM — for extreme memory pressure
# ---------------------------------------------------------------------------
DEEPSPEED_CONFIG: str | None = "configs/zero2.json"

# ---------------------------------------------------------------------------
# Time budget (seconds). Training stops after this wall-clock duration
# (excluding model loading). Set to 0 to disable and use MAX_STEPS instead.
# ---------------------------------------------------------------------------
TIME_BUDGET = 0

# ---------------------------------------------------------------------------
# Optimizer & schedule
# ---------------------------------------------------------------------------
LEARNING_RATE = 1e-4
LORA_LR_MULTIPLIER_B = 1.0       # multiplier for lora_B matrices vs lora_A
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 100
LR_SCHEDULER_TYPE = "cosine"
# When TIME_BUDGET <= 0, MAX_STEPS takes precedence over NUM_TRAIN_EPOCHS.
NUM_TRAIN_EPOCHS = 10
MAX_STEPS = 2000
PER_DEVICE_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 4
MAX_GRAD_NORM = 1.0
LOGGING_STEPS = 10
SAVE_TOTAL_LIMIT = 3
SAVE_STEPS = 500
OPTIMIZER = "adamw_torch_fused"
AUTO_FIND_BATCH_SIZE = True

# ---------------------------------------------------------------------------
# Eval during training — log eval loss every EVAL_STEPS optimizer steps.
# Set EVAL_STEPS = 0 to disable mid-training eval.
# ---------------------------------------------------------------------------
EVAL_STEPS = 500

# ---------------------------------------------------------------------------
# LoRA
# ---------------------------------------------------------------------------
LORA_RANK = 64
LORA_ALPHA = 128
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

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# Evaluation automatically reads these prompts via artifacts/prompt_config.json,
# so changing them here will also change eval prompts on the next run.
USER_PROMPT_PREFIX = "Read the document image and answer the question with only the exact value from the page."
ANSWER_STYLE = "Return only the answer value."


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def is_main_process() -> bool:
    return int(os.environ.get("LOCAL_RANK", "0")) == 0


def resolve_dtype() -> torch.dtype:
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


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


class TimeBudgetCallback(TrainerCallback):
    """Stop training after TIME_BUDGET wall-clock seconds of training."""

    def __init__(self, budget_seconds: float) -> None:
        self.budget_seconds = budget_seconds
        self.start_time: float | None = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is None:
            return
        elapsed = time.perf_counter() - self.start_time
        if elapsed >= self.budget_seconds:
            control.should_training_stop = True


class DocVQATrainer(Trainer):
    def __init__(self, *args, batch_metric_processor: AutoProcessor, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.batch_metric_processor = batch_metric_processor

    def create_optimizer(self):
        if LORA_LR_MULTIPLIER_B != 1.0 and FINETUNE_MODE == "lora":
            param_groups = _build_lora_param_groups(self.model, self.args.learning_rate)
            fused = OPTIMIZER == "adamw_torch_fused" and torch.cuda.is_available()
            self.optimizer = torch.optim.AdamW(
                param_groups,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=self.args.weight_decay,
                fused=fused,
            )
            return self.optimizer
        return super().create_optimizer()

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
    dtype = resolve_dtype()
    base_model_source = resolve_base_model_source()

    processor = AutoProcessor.from_pretrained(base_model_source)
    model = AutoModelForImageTextToText.from_pretrained(
        base_model_source,
        dtype=dtype,
    )
    model.config.use_cache = False

    if finetune_mode == "lora":
        lora_config = build_lora_config(model)
        model = get_peft_model(model, lora_config)

    if GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        if finetune_mode == "lora":
            model.enable_input_require_grads()

    return model, processor


def resolve_artifact_dir(finetune_mode: str) -> Path:
    if finetune_mode == "full":
        return TRAINED_MODEL_DIR
    if finetune_mode == "lora":
        return ADAPTER_DIR
    raise ValueError(f"Unsupported finetune mode: {finetune_mode}")


def reset_training_artifacts(*, keep_checkpoints: bool = False) -> None:
    """Remove mutable training outputs so each run starts fresh."""
    for path in (ADAPTER_DIR, TRAINED_MODEL_DIR):
        if path.exists():
            shutil.rmtree(path)
    if not keep_checkpoints and TRAINER_OUTPUT_DIR.exists():
        shutil.rmtree(TRAINER_OUTPUT_DIR)


def _find_latest_checkpoint() -> str | None:
    """Return the path to the latest training checkpoint, if any."""
    if not TRAINER_OUTPUT_DIR.exists():
        return None
    checkpoints = []
    for p in TRAINER_OUTPUT_DIR.glob("checkpoint-*"):
        if p.is_dir():
            try:
                step = int(p.name.split("-")[1])
                checkpoints.append((step, p))
            except (ValueError, IndexError):
                continue
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda x: x[0])
    return str(checkpoints[-1][1])


def _resolve_deepspeed_config() -> str | None:
    if DEEPSPEED_CONFIG is None:
        return None
    config_path = ROOT / DEEPSPEED_CONFIG
    if not config_path.exists():
        raise FileNotFoundError(f"DeepSpeed config not found: {config_path}")
    return str(config_path)


def build_training_arguments(*, has_eval_dataset: bool = False) -> TrainingArguments:
    if is_main_process():
        TRAINER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    eval_strategy = "no"
    eval_steps = None
    if has_eval_dataset and EVAL_STEPS > 0:
        eval_strategy = "steps"
        eval_steps = EVAL_STEPS

    effective_max_steps = MAX_STEPS if TIME_BUDGET <= 0 else -1
    ds_config = _resolve_deepspeed_config()

    dtype = resolve_dtype()
    return TrainingArguments(
        output_dir=str(TRAINER_OUTPUT_DIR),
        remove_unused_columns=False,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        auto_find_batch_size=AUTO_FIND_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        max_grad_norm=MAX_GRAD_NORM,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=effective_max_steps,
        warmup_steps=WARMUP_STEPS,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        report_to=[],
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        save_total_limit=SAVE_TOTAL_LIMIT,
        optim=OPTIMIZER,
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        bf16_full_eval=True,
        tf32=True,
        deepspeed=ds_config,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        seed=SEED,
    )


def load_eval_dataset() -> DocVQASplitDataset | Subset | None:
    """Load a random 1/N subset of validation for mid-training loss monitoring."""
    if EVAL_STEPS <= 0:
        return None
    from autoresearch_qwen.config import MID_TRAIN_EVAL_FRACTION, VAL_SPLIT
    ensure_snapshot_exists()
    full_dataset = DocVQASplitDataset(VAL_SPLIT)
    if MID_TRAIN_EVAL_FRACTION > 1:
        subset_size = max(1, len(full_dataset) // MID_TRAIN_EVAL_FRACTION)
        indices = random.sample(range(len(full_dataset)), subset_size)
        return Subset(full_dataset, indices)
    return full_dataset


def _build_lora_param_groups(model: torch.nn.Module, lr: float) -> list[dict[str, Any]]:
    """Split LoRA params into A/B groups with differential learning rates."""
    group_a, group_b, group_other = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "lora_A" in name:
            group_a.append(param)
        elif "lora_B" in name:
            group_b.append(param)
        else:
            group_other.append(param)

    groups = [{"params": group_a, "lr": lr}]
    if group_b:
        groups.append({"params": group_b, "lr": lr * LORA_LR_MULTIPLIER_B})
    if group_other:
        groups.append({"params": group_other, "lr": lr})
    return groups


def run_training(*, finetune_mode: str = FINETUNE_MODE, resume: bool = False) -> dict[str, float | int | str]:
    set_seed(SEED)

    artifact_dir = resolve_artifact_dir(finetune_mode)
    train_dataset = load_training_dataset()
    eval_dataset = load_eval_dataset()

    started = time.perf_counter()
    if is_main_process():
        reset_training_artifacts(keep_checkpoints=resume)
        artifact_dir.mkdir(parents=True, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    model, processor = load_model_and_processor(finetune_mode=finetune_mode)
    collator = DocVQACollator(processor)
    training_args = build_training_arguments(has_eval_dataset=eval_dataset is not None)

    callbacks = []
    if TIME_BUDGET > 0:
        callbacks.append(TimeBudgetCallback(TIME_BUDGET))

    trainer = DocVQATrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        batch_metric_processor=processor,
        callbacks=callbacks,
    )

    checkpoint = _find_latest_checkpoint() if resume else None
    if checkpoint and is_main_process():
        print(f"Resuming from checkpoint: {checkpoint}", file=sys.stderr)
    trainer.train(resume_from_checkpoint=checkpoint)

    trainer.save_model(artifact_dir)
    if trainer.is_world_process_zero():
        processor.save_pretrained(artifact_dir)
        prompt_config = {
            "prompt_prefix": USER_PROMPT_PREFIX,
            "prompt_suffix": ANSWER_STYLE,
            "include_question_type_hints": INCLUDE_QUESTION_TYPE_HINTS,
        }
        prompt_config_path = ROOT / "artifacts" / "prompt_config.json"
        prompt_config_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_config_path.write_text(
            json.dumps(prompt_config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        for ckpt_dir in TRAINER_OUTPUT_DIR.glob("checkpoint-*"):
            shutil.rmtree(ckpt_dir)

    elapsed = time.perf_counter() - started
    peak_vram_mb = 0.0
    if torch.cuda.is_available():
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    return {
        "train_seconds": elapsed,
        "examples": len(train_dataset),
        "artifact_dir": str(artifact_dir),
        "finetune_mode": finetune_mode,
        "peak_vram_mb": peak_vram_mb,
        "num_steps": trainer.state.global_step,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Qwen3-VL-4B on the official DocVQA train split (supports multi-GPU via torchrun)."
    )
    parser.add_argument("--finetune-mode", choices=("full", "lora"), default=FINETUNE_MODE)
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the latest checkpoint if available.")
    args = parser.parse_args()

    summary = run_training(finetune_mode=args.finetune_mode, resume=args.resume)
    if is_main_process():
        print("---")
        print(f"train_seconds:      {summary['train_seconds']:.2f}")
        print(f"peak_vram_mb:       {summary['peak_vram_mb']:.1f}")
        print(f"num_steps:          {summary['num_steps']}")
        print(f"examples:           {summary['examples']}")
        print(f"artifact_dir:       {summary['artifact_dir']}")
        print(f"mode:               {summary['finetune_mode']}")


if __name__ == "__main__":
    main()
