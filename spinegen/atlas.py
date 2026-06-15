from __future__ import annotations

from pathlib import Path

from PIL import Image

from spinegen.models import AtlasRegion, AtlasResult, LayerArtifact


def pack_atlas(
    layers: list[LayerArtifact],
    output_dir: Path,
    atlas_name: str,
    max_width: int = 2048,
    padding: int = 2,
) -> AtlasResult:
    if not layers:
        raise ValueError("PSD 中没有可导出的可见像素图层。")

    placements = _shelf_pack(layers, max_width=max_width, padding=padding)
    width = max(region.x + region.width + padding for region in placements.values())
    height = max(region.y + region.height + padding for region in placements.values())

    atlas_image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    for layer in layers:
        region = placements[layer.asset_name]
        image = Image.open(layer.image_path).convert("RGBA")
        atlas_image.alpha_composite(image, (region.x, region.y))

    image_path = output_dir / f"{atlas_name}.png"
    atlas_path = output_dir / f"{atlas_name}.atlas"
    atlas_image.save(image_path)
    atlas_path.write_text(
        _write_atlas_text(image_path.name, width, height, placements),
        encoding="utf-8",
    )

    return AtlasResult(
        image_path=image_path,
        atlas_path=atlas_path,
        regions=placements,
        width=width,
        height=height,
    )


def _shelf_pack(
    layers: list[LayerArtifact],
    max_width: int,
    padding: int,
) -> dict[str, AtlasRegion]:
    atlas_width = max(64, max_width)
    x = padding
    y = padding
    row_height = 0
    regions: dict[str, AtlasRegion] = {}

    for layer in layers:
        if layer.width + padding * 2 > atlas_width:
            atlas_width = layer.width + padding * 2

    for layer in sorted(layers, key=lambda item: (-item.height, item.draw_order)):
        if x + layer.width + padding > atlas_width:
            x = padding
            y += row_height + padding
            row_height = 0

        regions[layer.asset_name] = AtlasRegion(
            name=layer.asset_name,
            x=x,
            y=y,
            width=layer.width,
            height=layer.height,
            original_width=layer.width,
            original_height=layer.height,
        )
        x += layer.width + padding
        row_height = max(row_height, layer.height)

    return regions


def _write_atlas_text(
    page_name: str,
    width: int,
    height: int,
    regions: dict[str, AtlasRegion],
) -> str:
    lines = [
        page_name,
        f"size: {width}, {height}",
        "format: RGBA8888",
        "filter: Linear, Linear",
        "repeat: none",
    ]

    for name in sorted(regions):
        region = regions[name]
        lines.extend(
            [
                name,
                "  rotate: false",
                f"  xy: {region.x}, {region.y}",
                f"  size: {region.width}, {region.height}",
                f"  orig: {region.original_width}, {region.original_height}",
                f"  offset: {region.offset_x}, {region.offset_y}",
                "  index: -1",
            ]
        )

    return "\n".join(lines) + "\n"

