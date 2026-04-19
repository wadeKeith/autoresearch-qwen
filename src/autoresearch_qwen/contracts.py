"""Fixed data contracts shared by prepare, train, and evaluate."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkRecord:
    example_id: str
    image_path: str
    question: str
    answers: tuple[str, ...]
    question_types: tuple[str, ...]
    source_split: str
    question_id: int
    doc_id: int
    ucsf_document_id: str
    ucsf_document_page_no: str

    @staticmethod
    def from_json(line: str) -> "BenchmarkRecord":
        payload = json.loads(line)
        payload["answers"] = tuple(payload["answers"])
        payload["question_types"] = tuple(payload["question_types"])
        return BenchmarkRecord(**payload)

    @staticmethod
    def to_json(record: "BenchmarkRecord") -> str:
        payload = asdict(record)
        payload["answers"] = list(record.answers)
        payload["question_types"] = list(record.question_types)
        return json.dumps(payload, ensure_ascii=True)


def load_records(path: Path) -> list[BenchmarkRecord]:
    with path.open("r", encoding="utf-8") as handle:
        return [BenchmarkRecord.from_json(line) for line in handle if line.strip()]
