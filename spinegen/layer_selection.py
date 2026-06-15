from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from spinegen.models import CanvasInfo, LayerArtifact
from spinegen.naming import slugify


ROOT_GROUP_ID = "__root__"


@dataclass(frozen=True)
class LayerGroup:
    id: str
    label: str
    layers: list[LayerArtifact]
    bbox: tuple[int, int, int, int]
    canonical_parts: set[str]
    tokens: set[str]

    @property
    def layer_count(self) -> int:
        return len(self.layers)


@dataclass(frozen=True)
class LayerSelection:
    layers: list[LayerArtifact]
    groups: list[LayerGroup]
    alternative_groups: list[LayerGroup]
    selected_group_id: str | None
    messages: list[str]

    @property
    def has_alternatives(self) -> bool:
        return bool(self.alternative_groups)


def select_setup_layers(
    layers: list[LayerArtifact],
    canvas: CanvasInfo,
    prompt: str,
    preferred_group_id: str | None = None,
) -> LayerSelection:
    groups = build_layer_groups(layers)
    alternative_groups = _find_alternative_groups(groups, canvas)
    messages: list[str] = []

    if len(groups) <= 1:
        return LayerSelection(layers=layers, groups=groups, alternative_groups=[], selected_group_id=None, messages=messages)

    group_summary = ", ".join(f"{group.label}={group.layer_count}" for group in groups)
    messages.append(f"检测到 top-level groups：{group_summary}。")

    if not alternative_groups:
        messages.append("未检测到互斥 pose/variant group，保留所有可见图层。")
        return LayerSelection(layers=layers, groups=groups, alternative_groups=[], selected_group_id=None, messages=messages)

    selected = _choose_group(alternative_groups, prompt, preferred_group_id)
    selected_ids = {selected.id}
    alternative_ids = {group.id for group in alternative_groups}
    selected_layers = [
        layer for layer in layers if _top_group_id(layer) in selected_ids or _top_group_id(layer) not in alternative_ids
    ]
    skipped = sum(group.layer_count for group in alternative_groups if group.id != selected.id)
    messages.append(
        f"检测到 {len(alternative_groups)} 个互斥 pose/variant group，选择 `{selected.label}`，隐藏其他 {skipped} 个图层。"
    )
    return LayerSelection(
        layers=selected_layers,
        groups=groups,
        alternative_groups=alternative_groups,
        selected_group_id=selected.id,
        messages=messages,
    )


def build_layer_groups(layers: list[LayerArtifact]) -> list[LayerGroup]:
    grouped: dict[str, list[LayerArtifact]] = {}
    labels: dict[str, str] = {}
    for layer in layers:
        group_label = _top_group_label(layer)
        group_id = slugify(group_label, fallback=ROOT_GROUP_ID)
        grouped.setdefault(group_id, []).append(layer)
        labels[group_id] = group_label

    groups = [
        LayerGroup(
            id=group_id,
            label=labels[group_id],
            layers=group_layers,
            bbox=_union_bbox(group_layers),
            canonical_parts={_canonical_part_name(layer) for layer in group_layers},
            tokens=_group_tokens(labels[group_id], group_layers),
        )
        for group_id, group_layers in grouped.items()
    ]
    return sorted(groups, key=lambda group: min(layer.draw_order for layer in group.layers))


def group_summaries(groups: list[LayerGroup]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for group in groups:
        summaries.append(
            {
                "id": group.id,
                "label": group.label,
                "layer_count": group.layer_count,
                "bbox": list(group.bbox),
                "tokens": sorted(group.tokens)[:40],
                "canonical_parts": sorted(group.canonical_parts)[:80],
            }
        )
    return summaries


def _find_alternative_groups(groups: list[LayerGroup], canvas: CanvasInfo) -> list[LayerGroup]:
    candidates = [
        group
        for group in groups
        if group.id != ROOT_GROUP_ID
        and group.layer_count >= 3
        and _bbox_area(group.bbox) >= canvas.width * canvas.height * 0.03
    ]
    if len(candidates) < 2:
        return []

    related_ids: set[str] = set()
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            if _groups_look_alternative(left, right):
                related_ids.add(left.id)
                related_ids.add(right.id)

    return [group for group in candidates if group.id in related_ids]


def _groups_look_alternative(left: LayerGroup, right: LayerGroup) -> bool:
    part_similarity = _part_similarity(left.canonical_parts, right.canonical_parts)
    overlap = _bbox_overlap(left.bbox, right.bbox)
    count_ratio = min(left.layer_count, right.layer_count) / max(left.layer_count, right.layer_count)
    return part_similarity >= 0.35 and overlap >= 0.15 and count_ratio >= 0.35


def _choose_group(groups: list[LayerGroup], prompt: str, preferred_group_id: str | None) -> LayerGroup:
    group_by_id = {group.id: group for group in groups}
    if preferred_group_id and preferred_group_id in group_by_id:
        return group_by_id[preferred_group_id]

    prompt_tokens = _tokens(prompt)
    scored = []
    for index, group in enumerate(groups):
        token_hits = len(prompt_tokens & group.tokens)
        label_hits = len(prompt_tokens & _tokens(group.label))
        scored.append((token_hits * 3 + label_hits * 2, group.layer_count, -index, group))
    return max(scored, key=lambda item: item[:3])[3]


def _top_group_label(layer: LayerArtifact) -> str:
    if "/" not in layer.source_path:
        return ROOT_GROUP_ID
    return layer.source_path.split("/", 1)[0]


def _top_group_id(layer: LayerArtifact) -> str:
    return slugify(_top_group_label(layer), fallback=ROOT_GROUP_ID)


def _canonical_part_name(layer: LayerArtifact) -> str:
    parts = layer.source_path.split("/")
    leaf = parts[-1] if parts else layer.name
    group_tokens = _tokens(parts[0]) if len(parts) > 1 else set()
    part_tokens = [token for token in _tokens(leaf) if token not in group_tokens]
    if not part_tokens:
        part_tokens = list(_tokens(Path(layer.asset_name).stem))
    return "_".join(sorted(part_tokens)) or layer.asset_name


def _group_tokens(label: str, layers: list[LayerArtifact]) -> set[str]:
    tokens = set(_tokens(label))
    for layer in layers:
        tokens.update(_tokens(layer.source_path))
        tokens.update(_tokens(layer.name))
    return tokens


def _tokens(value: str) -> set[str]:
    slug = slugify(value.replace("/", "_").replace("-", "_"), fallback="")
    return {token for token in slug.lower().split("_") if len(token) >= 2}


def _union_bbox(layers: list[LayerArtifact]) -> tuple[int, int, int, int]:
    left = min(layer.bbox[0] for layer in layers)
    top = min(layer.bbox[1] for layer in layers)
    right = max(layer.bbox[2] for layer in layers)
    bottom = max(layer.bbox[3] for layer in layers)
    return (left, top, right, bottom)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _bbox_overlap(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = _bbox_area((x1, y1, x2, y2))
    smaller = min(_bbox_area(left), _bbox_area(right))
    if smaller <= 0:
        return 0.0
    return intersection / smaller


def _part_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / min(len(left), len(right))

