"""Fixed paths and benchmark constants for autoresearch-qwen."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = ROOT / "artifacts"
LOCAL_MODEL_DIR = ROOT / "Qwen" / "Qwen3-VL-4B-Instruct"
DATASET_SNAPSHOT_DIR = ARTIFACTS_DIR / "documentvqa_snapshot"
ADAPTER_DIR = ARTIFACTS_DIR / "adapter"
TRAINED_MODEL_DIR = ARTIFACTS_DIR / "trained_model"
TRAINER_OUTPUT_DIR = ARTIFACTS_DIR / "trainer_output"
PROMPT_CONFIG_PATH = ARTIFACTS_DIR / "prompt_config.json"
TEST_PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.jsonl"
TEST_SUBMISSION_PATH = ARTIFACTS_DIR / "docvqa_test_submission.json"
TEST_SUBMISSION_BUNDLE_PATH = ARTIFACTS_DIR / "docvqa_test_submission.bundle.zip"
RESULTS_PATH = ROOT / "results.tsv"

HF_ENDPOINT = "https://hf-mirror.com"
DOCVQA_DATASET = "HuggingFaceM4/DocumentVQA"
TRAIN_SPLIT = "train"
VAL_SPLIT = "validation"
TEST_SPLIT = "test"
DOCVQA_OFFICIAL_SPLITS = (TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT)

MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
MAX_NEW_TOKENS = 1024
MID_TRAIN_EVAL_FRACTION = 10
DOCVQA_SUBMISSION_DATASET_NAME = "docvqa"
DOCVQA_SUBMISSION_VERSION = "0.1"

DEFAULT_PROMPT_PREFIX = (
    "Read the document image and answer the question with only the exact value from the page."
)
DEFAULT_PROMPT_SUFFIX = "Return only the answer value."
