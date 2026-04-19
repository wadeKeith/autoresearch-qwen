"""Lightweight preflight checks for autoresearch-qwen."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

from .config import DATASET_SNAPSHOT_DIR, DOCVQA_OFFICIAL_SPLITS, HF_ENDPOINT, LOCAL_MODEL_DIR, RESULTS_PATH
from .hub import local_model_is_ready

SUPPORTED_PYTHON = ">=3.11,<3.14"


def _python_supported() -> bool:
    return (3, 11) <= sys.version_info[:2] < (3, 14)


def _dataset_snapshot_report() -> tuple[bool, dict[str, int]]:
    shard_counts: dict[str, int] = {}
    ready = True
    for split in DOCVQA_OFFICIAL_SPLITS:
        count = len(list(DATASET_SNAPSHOT_DIR.glob(f"data/{split}-*.parquet")))
        shard_counts[split] = count
        if count == 0:
            ready = False
    return ready, shard_counts


def build_report() -> dict[str, Any]:
    dataset_ready, shard_counts = _dataset_snapshot_report()
    model_ready = local_model_is_ready()

    torch_info: dict[str, Any] = {
        "import_ok": False,
        "cuda_available": False,
        "cuda_device_count": 0,
        "bf16_supported": False,
        "error": None,
    }
    try:
        import torch

        torch_info["import_ok"] = True
        torch_info["cuda_available"] = torch.cuda.is_available()
        torch_info["cuda_device_count"] = torch.cuda.device_count() if torch_info["cuda_available"] else 0
        torch_info["bf16_supported"] = bool(
            torch_info["cuda_available"] and torch.cuda.is_bf16_supported()
        )
    except Exception as exc:  # pragma: no cover - best-effort diagnostics
        torch_info["error"] = f"{type(exc).__name__}: {exc}"

    blockers: list[str] = []
    if not _python_supported():
        blockers.append(f"Python {sys.version.split()[0]} is unsupported; expected {SUPPORTED_PYTHON}.")
    if shutil.which("uv") is None:
        blockers.append("`uv` is not installed or not on PATH.")
    if shutil.which("git") is None:
        blockers.append("`git` is not installed or not on PATH.")
    if not torch_info["import_ok"]:
        blockers.append(f"PyTorch import failed: {torch_info['error']}.")
    elif not torch_info["cuda_available"]:
        blockers.append("No CUDA-capable GPU is visible to PyTorch.")
    if not dataset_ready:
        blockers.append("DocVQA snapshot is incomplete. Run `uv run python prepare.py`.")
    if not model_ready:
        blockers.append("Local Qwen model snapshot is missing. Run `uv run python prepare.py`.")

    next_steps: list[str] = []
    if shutil.which("uv") is None:
        next_steps.append("Install uv first, then run `uv sync`.")
    else:
        next_steps.append("Run `uv sync` if dependencies are missing or stale.")
    if not dataset_ready or not model_ready:
        next_steps.append("Run `uv run python prepare.py` to download the dataset and model.")
    if not RESULTS_PATH.exists():
        next_steps.append(
            "Create `results.tsv` with the standard header, or re-run `uv run python prepare.py` if setup is incomplete."
        )
    if not blockers:
        next_steps.append("Run `uv run python evaluate.py --base-only --split validation` for a baseline.")
        next_steps.append("Run `./run_experiment.sh` for one train+eval iteration.")

    return {
        "ready_for_autoresearch": not blockers,
        "python": {
            "version": sys.version.split()[0],
            "supported": _python_supported(),
            "required": SUPPORTED_PYTHON,
        },
        "tools": {
            "uv_available": shutil.which("uv") is not None,
            "git_available": shutil.which("git") is not None,
        },
        "torch": torch_info,
        "dataset": {
            "ready": dataset_ready,
            "snapshot_dir": str(DATASET_SNAPSHOT_DIR),
            "parquet_shards": shard_counts,
        },
        "model": {
            "ready": model_ready,
            "local_model_dir": str(LOCAL_MODEL_DIR),
        },
        "results": {
            "results_tsv_exists": RESULTS_PATH.exists(),
            "results_tsv_path": str(RESULTS_PATH),
        },
        "env": {
            "hf_endpoint": os.environ.get("HF_ENDPOINT", HF_ENDPOINT),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "blockers": blockers,
        "next_steps": next_steps,
    }


def _format_bool(value: bool) -> str:
    return "yes" if value else "no"


def print_report(report: dict[str, Any], *, json_output: bool = False) -> None:
    if json_output:
        print(json.dumps(report, indent=2, ensure_ascii=True))
        return

    print("Autoresearch Doctor")
    print(f"- ready_for_autoresearch: {_format_bool(report['ready_for_autoresearch'])}")
    print(
        f"- python: {report['python']['version']} "
        f"(supported={_format_bool(report['python']['supported'])}, required={report['python']['required']})"
    )
    print(
        f"- tools: uv={_format_bool(report['tools']['uv_available'])}, "
        f"git={_format_bool(report['tools']['git_available'])}"
    )

    torch_info = report["torch"]
    if torch_info["import_ok"]:
        print(
            f"- torch: import_ok=yes, cuda={_format_bool(torch_info['cuda_available'])}, "
            f"visible_gpus={torch_info['cuda_device_count']}, bf16={_format_bool(torch_info['bf16_supported'])}"
        )
    else:
        print(f"- torch: import_ok=no ({torch_info['error']})")

    shard_counts = report["dataset"]["parquet_shards"]
    print(
        f"- dataset: ready={_format_bool(report['dataset']['ready'])}, "
        f"train={shard_counts['train']}, validation={shard_counts['validation']}, test={shard_counts['test']} shards"
    )
    print(
        f"- model: ready={_format_bool(report['model']['ready'])}, "
        f"path={report['model']['local_model_dir']}"
    )
    print(
        f"- results.tsv: present={_format_bool(report['results']['results_tsv_exists'])}, "
        f"path={report['results']['results_tsv_path']}"
    )
    print(f"- HF_ENDPOINT: {report['env']['hf_endpoint']}")
    print(f"- CUDA_VISIBLE_DEVICES: {report['env']['cuda_visible_devices']}")

    if report["blockers"]:
        print("Blockers:")
        for blocker in report["blockers"]:
            print(f"- {blocker}")
    else:
        print("Blockers: none")

    if report["next_steps"]:
        print("Next steps:")
        for step in report["next_steps"]:
            print(f"- {step}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run lightweight preflight checks for autoresearch-qwen.")
    parser.add_argument("--json", action="store_true", help="Print the report as JSON.")
    args = parser.parse_args(argv)

    report = build_report()
    print_report(report, json_output=args.json)
    return 0 if report["ready_for_autoresearch"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
