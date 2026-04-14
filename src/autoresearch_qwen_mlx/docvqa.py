"""Helpers for loading the official DocVQA dataset splits from a local snapshot."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from .config import DATASET_SNAPSHOT_DIR, DOCVQA_DATASET, DOCVQA_OFFICIAL_SPLITS, HF_ENDPOINT
from .hub import configure_hub_env

configure_hub_env()

from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset
from huggingface_hub import snapshot_download


def snapshot_dataset(*, snapshot_dir: Path | None = None) -> Path:
    target_dir = snapshot_dir or DATASET_SNAPSHOT_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=DOCVQA_DATASET,
        repo_type="dataset",
        endpoint=HF_ENDPOINT,
        local_dir=target_dir,
        token=os.environ.get("HF_TOKEN") or None,
    )
    return target_dir


def ensure_snapshot_exists(*, snapshot_dir: Path | None = None) -> Path:
    target_dir = snapshot_dir or DATASET_SNAPSHOT_DIR
    missing = [split for split in DOCVQA_OFFICIAL_SPLITS if not list(target_dir.glob(f"data/{split}-*.parquet"))]
    if missing:
        raise FileNotFoundError(
            f"Missing local DocVQA snapshot data for splits {missing} under {target_dir}. "
            "Run `uv run python prepare.py` first."
        )
    return target_dir


def _parquet_files(snapshot_dir: Path, split: str) -> list[str]:
    files = sorted(snapshot_dir.glob(f"data/{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"Could not find parquet files for split {split!r} under {snapshot_dir}.")
    return [str(path) for path in files]


def load_docvqa_splits(*, snapshot_dir: Path | None = None) -> DatasetDict:
    target_dir = ensure_snapshot_exists(snapshot_dir=snapshot_dir)
    return load_dataset(
        "parquet",
        data_files={split: _parquet_files(target_dir, split) for split in DOCVQA_OFFICIAL_SPLITS},
    )


def load_docvqa_split(
    split: str,
    *,
    snapshot_dir: Path | None = None,
) -> HFDataset:
    target_dir = ensure_snapshot_exists(snapshot_dir=snapshot_dir)
    dataset = load_dataset(
        "parquet",
        data_files={split: _parquet_files(target_dir, split)},
    )[split]
    return dataset


def load_split_question_ids(
    split: str,
    *,
    snapshot_dir: Path | None = None,
) -> list[int]:
    dataset = load_docvqa_split(split, snapshot_dir=snapshot_dir)
    return [int(question_id) for question_id in dataset["questionId"]]


def row_answers(row: dict[str, Any]) -> tuple[str, ...]:
    answers = row.get("answers")
    if answers is None:
        return ()
    return tuple(str(answer) for answer in answers)


def row_question_types(row: dict[str, Any]) -> tuple[str, ...]:
    question_types = row.get("question_types")
    if question_types is None:
        return ()
    return tuple(str(question_type) for question_type in question_types)


class DocVQASplitDataset(Dataset):
    def __init__(
        self,
        split: str,
        *,
        snapshot_dir: Path | None = None,
    ) -> None:
        self.dataset = load_docvqa_split(split, snapshot_dir=snapshot_dir)
        self.split = split

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataset[index]
        image = row["image"].convert("RGB")
        return {
            "example_id": f"{self.split}_{index:06d}",
            "image": image,
            "question": row["question"],
            "answers": row_answers(row),
            "question_types": row_question_types(row),
            "source_split": self.split,
            "question_id": int(row["questionId"]),
            "doc_id": int(row["docId"]),
            "ucsf_document_id": str(row["ucsf_document_id"]),
            "ucsf_document_page_no": str(row["ucsf_document_page_no"]),
        }
