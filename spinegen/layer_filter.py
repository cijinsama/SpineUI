from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from spinegen.models import LayerArtifact
from spinegen.naming import slugify


@dataclass(frozen=True)
class RedundantLayerCandidate:
    hide_layer_id: str
    keep_layer_id: str
    reason: str
    overlap: float
    area_ratio: float


@dataclass(frozen=True)
class LayerFilterResult:
    layers: list[LayerArtifact]
    hidden_layer_ids: list[str]
    candidates: list[RedundantLayerCandidate]
    messages: list[str]


def resolve_redundant_layers(
    layers: list[LayerArtifact],
    extra_hidden_layer_ids: set[str] | None = None,
) -> LayerFilterResult:
    candidates = find_redundant_layer_candidates(layers)
    hidden_ids = {candidate.hide_layer_id for candidate in candidates}
    if extra_hidden_layer_ids:
        valid_ids = {layer.id for layer in layers}
        hidden_ids.update(layer_id for layer_id in extra_hidden_layer_ids if layer_id in valid_ids)

    filtered_layers = [layer for layer in layers if layer.id not in hidden_ids]
    messages: list[str] = []
    if hidden_ids:
        messages.append(f"检测到同组内冗余/组合图层，隐藏 {len(hidden_ids)} 个图层。")
    return LayerFilterResult(
        layers=filtered_layers,
        hidden_layer_ids=sorted(hidden_ids),
        candidates=candidates,
        messages=messages,
    )


def find_redundant_layer_candidates(layers: list[LayerArtifact]) -> list[RedundantLayerCandidate]:
    candidates: dict[str, RedundantLayerCandidate] = {}
    for index, left in enumerate(layers):
        for right in layers[index + 1 :]:
            candidate = _candidate_for_pair(left, right)
            if candidate is not None:
                candidates[candidate.hide_layer_id] = candidate
    return sorted(candidates.values(), key=lambda candidate: candidate.hide_layer_id)


def redundant_candidate_summaries(candidates: list[RedundantLayerCandidate], layers: list[LayerArtifact]) -> list[dict[str, Any]]:
    layer_by_id = {layer.id: layer for layer in layers}
    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        hidden = layer_by_id.get(candidate.hide_layer_id)
        kept = layer_by_id.get(candidate.keep_layer_id)
        if hidden is None or kept is None:
            continue
        summaries.append(
            {
                "hide_layer_id": candidate.hide_layer_id,
                "keep_layer_id": candidate.keep_layer_id,
                "reason": candidate.reason,
                "overlap": round(candidate.overlap, 4),
                "area_ratio": round(candidate.area_ratio, 4),
                "hide_layer": _layer_summary(hidden),
                "keep_layer": _layer_summary(kept),
            }
        )
    return summaries


def _candidate_for_pair(left: LayerArtifact, right: LayerArtifact) -> RedundantLayerCandidate | None:
    overlap = _bbox_overlap(left.bbox, right.bbox)
    area_ratio = _area_ratio(left.bbox, right.bbox)
    if overlap < 0.9 or area_ratio < 0.8:
        return None

    left_tokens = _semantic_tokens(left)
    right_tokens = _semantic_tokens(right)
    if not left_tokens or not right_tokens:
        return None

    if right_tokens < left_tokens:
        return RedundantLayerCandidate(
            hide_layer_id=right.id,
            keep_layer_id=left.id,
            reason="component layer tokens are contained in a larger composite layer with matching bounds",
            overlap=overlap,
            area_ratio=area_ratio,
        )
    if left_tokens < right_tokens:
        return RedundantLayerCandidate(
            hide_layer_id=left.id,
            keep_layer_id=right.id,
            reason="component layer tokens are contained in a larger composite layer with matching bounds",
            overlap=overlap,
            area_ratio=area_ratio,
        )
    return None


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
            "attack",
            "idle",
            "walk",
            "run",
        }
    }


def _tokens(value: str) -> set[str]:
    slug = slugify(value.replace("/", "_").replace("-", "_"), fallback="")
    return {token for token in slug.lower().split("_") if len(token) >= 2}


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

