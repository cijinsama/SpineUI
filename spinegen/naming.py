from __future__ import annotations

import re
import unicodedata


_SAFE_RE = re.compile(r"[^a-zA-Z0-9_./-]+")


def slugify(value: str, fallback: str = "item") -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    ascii_value = _SAFE_RE.sub("_", ascii_value).strip("._-/")
    ascii_value = re.sub(r"_+", "_", ascii_value)
    return ascii_value or fallback


def unique_name(base: str, used: set[str]) -> str:
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    used.add(candidate)
    return candidate

