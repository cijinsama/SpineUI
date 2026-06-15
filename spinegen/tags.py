from __future__ import annotations

import re


TAG_RE = re.compile(r"\[([a-zA-Z0-9_-]+)(?::([^\]]+))?\]")


def parse_tags(name: str) -> dict[str, str | bool]:
    tags: dict[str, str | bool] = {}
    for match in TAG_RE.finditer(name):
        key = match.group(1).strip().lower()
        value = match.group(2)
        tags[key] = value.strip() if value else True
    return tags


def strip_tags(name: str) -> str:
    return TAG_RE.sub("", name).strip() or name.strip()

