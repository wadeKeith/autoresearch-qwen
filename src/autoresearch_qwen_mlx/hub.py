"""Shared Hugging Face hub runtime settings."""

from __future__ import annotations

import os
from pathlib import Path

from .config import HF_ENDPOINT, LOCAL_MODEL_DIR, MODEL_NAME


def configure_hub_env() -> None:
    os.environ.setdefault("HF_ENDPOINT", HF_ENDPOINT)
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "1200")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "120")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def resolve_base_model_source() -> str:
    if local_model_is_ready():
        return str(LOCAL_MODEL_DIR)
    return MODEL_NAME


def local_model_is_ready() -> bool:
    if not (LOCAL_MODEL_DIR / "config.json").exists():
        return False
    if (LOCAL_MODEL_DIR / "model.safetensors.index.json").exists():
        return True
    if (LOCAL_MODEL_DIR / "model.safetensors").exists():
        return True
    return any(LOCAL_MODEL_DIR.glob("*.safetensors"))
