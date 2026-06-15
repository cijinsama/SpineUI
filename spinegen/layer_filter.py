from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spinegen.models import LayerArtifact
from spinegen.naming import slugify


@dataclass(frozen=True)
class LayerOverlapHint:
    first_layer_id: str
    second_layer_id: str
    reason: str
    overlap: float
    area_ratio: float


@dataclass(frozen=True)
class LayerFilterResult:
    layers: list[LayerArtifact]
    hidden_layer_ids: list[str]
    overlap_hints: list[LayerOverlapHint]
    messages: list[str]


def resolve_redundant_layers(
    layers: list[LayerArtifact],
    hidden_layer_ids: set[str] | None = None,
) -> LayerFilterResult:
    overlap_hints = find_layer_overlap_hints(layers)
    valid_ids = {layer.id for layer in layers}
    hidden_ids = {layer_id for layer_id in hidden_layer_ids or set() if layer_id in valid_ids}

    filtered_layers = [layer for layer in layers if layer.id not in hidden_ids]
    messages: list[str] = []
    if hidden_ids:
        messages.append(f"LLM 可见性计划隐藏 {len(hidden_ids)} 个图层。")
    return LayerFilterResult(
        layers=filtered_layers,
        hidden_layer_ids=sorted(hidden_ids),
        overlap_hints=overlap_hints,
        messages=messages,
    )


def find_layer_overlap_hints(layers: list[LayerArtifact]) -> list[LayerOverlapHint]:
    hints: dict[tuple[str, str], LayerOverlapHint] = {}
    for index, left in enumerate(layers):
        for right in layers[index + 1 :]:
            hint = _overlap_hint_for_pair(left, right)
            if hint is not None:
                hints[(hint.first_layer_id, hint.second_layer_id)] = hint
    return sorted(hints.values(), key=lambda hint: (hint.first_layer_id, hint.second_layer_id))


def layer_summaries(layers: list[LayerArtifact]) -> list[dict[str, Any]]:
    return [_layer_summary(layer) for layer in layers]


def layer_overlap_hint_summaries(hints: list[LayerOverlapHint], layers: list[LayerArtifact]) -> list[dict[str, Any]]:
    layer_by_id = {layer.id: layer for layer in layers}
    summaries: list[dict[str, Any]] = []
    for hint in hints:
        first = layer_by_id.get(hint.first_layer_id)
        second = layer_by_id.get(hint.second_layer_id)
        if first is None or second is None:
            continue
        first_tokens = _semantic_tokens(first)
        second_tokens = _semantic_tokens(second)
        summaries.append(
            {
                "layer_ids": [hint.first_layer_id, hint.second_layer_id],
                "reason": hint.reason,
                "overlap": round(hint.overlap, 4),
                "area_ratio": round(hint.area_ratio, 4),
                "shared_tokens": sorted(first_tokens & second_tokens),
                "different_tokens": sorted(first_tokens ^ second_tokens),
                "token_relation": _token_relation(first_tokens, second_tokens),
                "layers": [_layer_summary(first), _layer_summary(second)],
            }
        )
    return summaries


def _overlap_hint_for_pair(left: LayerArtifact, right: LayerArtifact) -> LayerOverlapHint | None:
    overlap = _bbox_overlap(left.bbox, right.bbox)
    area_ratio = _area_ratio(left.bbox, right.bbox)
    if overlap < 0.9 or area_ratio < 0.8:
        return None

    left_tokens = _semantic_tokens(left)
    right_tokens = _semantic_tokens(right)
    if not left_tokens or not right_tokens:
        return None

    if not (left_tokens & right_tokens or left_tokens < right_tokens or right_tokens < left_tokens):
        return None

    return LayerOverlapHint(
        first_layer_id=left.id,
        second_layer_id=right.id,
        reason="highly overlapping bounds with related layer names",
        overlap=overlap,
        area_ratio=area_ratio,
    )


def _layer_summary(layer: LayerArtifact) -> dict[str, Any]:
    return {
        "id": layer.id,
        "name": layer.name,
        "path": layer.source_path,
        "asset": layer.asset_name,
        "bbox": list(layer.bbox),
        "size": [layer.width, layer.height],
        "tokens": sorted(_semantic_tokens(layer)),
        "draw_order": layer.draw_order,
    }


def _semantic_tokens(layer: LayerArtifact) -> set[str]:
    tokens = _tokens(layer.source_path)
    tokens.update(_tokens(layer.name))
    tokens.update(_tokens(layer.asset_name))
    return {
        token
        for token in tokens
        if token
        not in {
            "pose",
            "base",
            "default",
            "variant",
            "layer",
            "left",
            "right",
        }
    }


def _tokens(value: str) -> set[str]:
    slug = slugify(value.replace("/", "_").replace("-", "_"), fallback="")
    return {token for token in slug.lower().split("_") if len(token) >= 2}


def _token_relation(first_tokens: set[str], second_tokens: set[str]) -> str:
    if first_tokens < second_tokens:
        return "first_tokens_subset_of_second"
    if second_tokens < first_tokens:
        return "second_tokens_subset_of_first"
    if first_tokens == second_tokens:
        return "same_tokens"
    if first_tokens & second_tokens:
        return "partial_overlap"
    return "different_tokens"


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


def _area_ratio(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    left_area = _bbox_area(left)
    right_area = _bbox_area(right)
    larger = max(left_area, right_area)
    if larger <= 0:
        return 0.0
    return min(left_area, right_area) / larger
