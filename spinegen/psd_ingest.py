from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
from psd_tools import PSDImage

from spinegen.models import CanvasInfo, LayerArtifact
from spinegen.naming import slugify, unique_name
from spinegen.tags import parse_tags, strip_tags


def export_psd_layers(psd_path: Path, output_dir: Path) -> tuple[CanvasInfo, list[LayerArtifact], Path, dict[str, Any]]:
    psd = PSDImage.open(psd_path)
    raw_layers: list[tuple[list[str], Any]] = []
    _collect_leaf_layers(psd, [], raw_layers)

    origin_x, origin_y = _find_origin(raw_layers, psd.width, psd.height)
    canvas = CanvasInfo(
        width=int(psd.width),
        height=int(psd.height),
        origin_x=origin_x,
        origin_y=origin_y,
    )

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    composite_path = output_dir / "composite.png"
    composite = psd.composite()
    if composite is not None:
        composite.save(composite_path)

    used_assets: set[str] = set()
    used_ids: set[str] = set()
    exported: list[LayerArtifact] = []
    metadata_layers: list[dict[str, Any]] = []

    for index, (path_parts, layer) in enumerate(raw_layers):
        raw_layer_name = str(layer.name or f"Layer {index + 1}")
        tags = parse_tags(raw_layer_name)
        if "ignore" in tags or "origin" in tags:
            continue

        if not _is_visible(layer):
            continue

        bbox = tuple(int(value) for value in layer.bbox)
        left, top, right, bottom = bbox
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            continue

        image = layer.composite()
        if image is None:
            continue

        if image.mode != "RGBA":
            image = image.convert("RGBA")

        if not _has_visible_pixels(image):
            continue

        layer_name = strip_tags(raw_layer_name)
        clean_path_parts = [strip_tags(part) for part in path_parts if strip_tags(part)]
        source_path = "/".join([*clean_path_parts, layer_name])
        layer_id = unique_name(f"layer_{index + 1:03d}", used_ids)
        asset_name = unique_name(slugify(source_path.replace("/", "_"), fallback=layer_id), used_assets)
        image_path = images_dir / f"{asset_name}.png"
        image.save(image_path)

        artifact = LayerArtifact(
            id=layer_id,
            name=layer_name,
            source_path=source_path,
            asset_name=asset_name,
            image_path=image_path,
            bbox=bbox,
            width=image.width,
            height=image.height,
            opacity=float(getattr(layer, "opacity", 255)) / 255,
            blend_mode=str(getattr(layer, "blend_mode", "normal")),
            draw_order=index,
            tags=tags,
        )
        exported.append(artifact)
        metadata_layers.append(_layer_to_metadata(artifact, canvas))

    ir = {
        "source": str(psd_path.name),
        "canvas": {
            "width": canvas.width,
            "height": canvas.height,
            "origin": [canvas.origin_x, canvas.origin_y],
        },
        "layers": metadata_layers,
    }

    return canvas, exported, composite_path, ir


def write_ir(ir: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(ir, ensure_ascii=False, indent=2), encoding="utf-8")


def _collect_leaf_layers(node: Any, path_parts: list[str], output: list[tuple[list[str], Any]]) -> None:
    for child in node:
        child_name = str(getattr(child, "name", "") or "")
        if getattr(child, "is_group", lambda: False)():
            _collect_leaf_layers(child, [*path_parts, child_name], output)
        else:
            output.append((path_parts, child))


def _find_origin(raw_layers: list[tuple[list[str], Any]], width: int, height: int) -> tuple[float, float]:
    for _, layer in raw_layers:
        if "origin" not in parse_tags(str(getattr(layer, "name", "") or "")):
            continue
        bbox = tuple(int(value) for value in layer.bbox)
        left, top, right, bottom = bbox
        if right > left and bottom > top:
            return ((left + right) / 2, (top + bottom) / 2)
    return (float(width) / 2, float(height) / 2)


def _is_visible(layer: Any) -> bool:
    is_visible = getattr(layer, "is_visible", True)
    return bool(is_visible() if callable(is_visible) else is_visible)


def _has_visible_pixels(image: Image.Image) -> bool:
    alpha = image.getchannel("A")
    return alpha.getbbox() is not None


def _layer_to_metadata(layer: LayerArtifact, canvas: CanvasInfo) -> dict[str, Any]:
    return {
        "id": layer.id,
        "name": layer.name,
        "path": layer.source_path,
        "asset": layer.asset_name,
        "bbox": list(layer.bbox),
        "size": [layer.width, layer.height],
        "spine_center": [round(layer.spine_x(canvas), 3), round(layer.spine_y(canvas), 3)],
        "opacity": round(layer.opacity, 4),
        "blend_mode": layer.blend_mode,
        "draw_order": layer.draw_order,
        "tags": layer.tags,
    }
