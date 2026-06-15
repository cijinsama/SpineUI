from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CanvasInfo:
    width: int
    height: int
    origin_x: float
    origin_y: float


@dataclass(frozen=True)
class LayerArtifact:
    id: str
    name: str
    source_path: str
    asset_name: str
    image_path: Path
    bbox: tuple[int, int, int, int]
    width: int
    height: int
    opacity: float
    blend_mode: str
    draw_order: int
    tags: dict[str, str | bool]

    @property
    def center_x(self) -> float:
        return self.bbox[0] + self.width / 2

    @property
    def center_y(self) -> float:
        return self.bbox[1] + self.height / 2

    def spine_x(self, canvas: CanvasInfo) -> float:
        return self.center_x - canvas.origin_x

    def spine_y(self, canvas: CanvasInfo) -> float:
        return canvas.origin_y - self.center_y


@dataclass(frozen=True)
class AtlasRegion:
    name: str
    x: int
    y: int
    width: int
    height: int
    original_width: int
    original_height: int
    offset_x: int = 0
    offset_y: int = 0
    rotate: bool = False


@dataclass(frozen=True)
class AtlasResult:
    image_path: Path
    atlas_path: Path
    regions: dict[str, AtlasRegion]
    width: int
    height: int


@dataclass
class RigPlan:
    skeleton_name: str
    bones: list[dict[str, Any]] = field(default_factory=list)
    slots: list[dict[str, Any]] = field(default_factory=list)
    animations: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    source: str = "fallback"

    def to_dict(self) -> dict[str, Any]:
        return {
            "skeleton_name": self.skeleton_name,
            "bones": self.bones,
            "slots": self.slots,
            "animations": self.animations,
            "notes": self.notes,
            "source": self.source,
        }


@dataclass(frozen=True)
class ConversionResult:
    output_dir: Path
    zip_path: Path
    download_files: list[Path]
    preview_iframe_html: str
    spine_json: dict[str, Any]
    rig_plan: dict[str, Any]
    messages: list[str]
