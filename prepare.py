"""Download the full official DocVQA snapshot used by training and evaluation."""

from __future__ import annotations

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from autoresearch_qwen.config import (
    DOCVQA_DATASET,
    DOCVQA_OFFICIAL_SPLITS,
    HF_ENDPOINT,
    LOCAL_MODEL_DIR,
    MODEL_NAME,
    RESULTS_PATH,
    TEST_SPLIT,
)
from autoresearch_qwen.docvqa import load_docvqa_splits, snapshot_dataset
from autoresearch_qwen.hub import local_model_is_ready

from huggingface_hub import snapshot_download


def ensure_results_tsv() -> None:
    if RESULTS_PATH.exists():
        return
    RESULTS_PATH.write_text(
        "commit\tval_score\ttrain_seconds\tmemory_gb\tstatus\tdescription\n",
        encoding="utf-8",
    )


def snapshot_model() -> Path:
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=MODEL_NAME,
        endpoint=HF_ENDPOINT,
        local_dir=LOCAL_MODEL_DIR,
        token=os.environ.get("HF_TOKEN") or None,
    )
    return LOCAL_MODEL_DIR


def prepare_snapshot() -> tuple[Path, dict[str, int], bool, Path]:
    snapshot_dir = snapshot_dataset()
    model_dir = snapshot_model()
    dataset_dict = load_docvqa_splits(snapshot_dir=snapshot_dir)
    split_sizes = {split: len(dataset_dict[split]) for split in DOCVQA_OFFICIAL_SPLITS}
    test_has_answers = bool(dataset_dict[TEST_SPLIT][0]["answers"])
    ensure_results_tsv()
    return snapshot_dir, split_sizes, test_has_answers, model_dir


def main() -> None:
    snapshot_dir, split_sizes, test_has_answers, model_dir = prepare_snapshot()
    print("DocVQA snapshot ready")
    print(f"- dataset: {DOCVQA_DATASET}")
    print(f"- snapshot_dir: {snapshot_dir}")
    for split in DOCVQA_OFFICIAL_SPLITS:
        print(f"- {split}_examples: {split_sizes[split]}")
    print(f"- test_has_answers: {test_has_answers}")
    print(f"- model: {MODEL_NAME}")
    print(f"- local_model_dir: {model_dir}")
    print(f"- local_model_ready: {local_model_is_ready()}")


if __name__ == "__main__":
    main()
