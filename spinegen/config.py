from __future__ import annotations

import os
from dataclasses import dataclass


DEFAULT_BASE_URL = "https://ellm.nrp-nautilus.io/v1"
DEFAULT_MODEL = "qwen3"


@dataclass(frozen=True)
class LLMSettings:
    model: str = DEFAULT_MODEL
    max_tokens: int = 16000
    temperature: float = 0.2
    top_p: float = 0.95
    enable_thinking: bool = True
    preserve_thinking: bool = False
    timeout_seconds: float = 180.0

    @classmethod
    def from_env(cls, model: str | None = None) -> "LLMSettings":
        return cls(
            model=model or os.getenv("NRP_MODEL", DEFAULT_MODEL),
            max_tokens=_int_env("NRP_MAX_TOKENS", 16000),
            temperature=_float_env("NRP_TEMPERATURE", 0.2),
            top_p=_float_env("NRP_TOP_P", 0.95),
            enable_thinking=_bool_env("NRP_ENABLE_THINKING", True),
            preserve_thinking=_bool_env("NRP_PRESERVE_THINKING", False),
            timeout_seconds=_float_env("NRP_TIMEOUT_SECONDS", 180.0),
        )


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default

