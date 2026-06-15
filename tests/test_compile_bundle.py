from pathlib import Path

from PIL import Image

from spinegen.atlas import pack_atlas
from spinegen.llm import build_fallback_rig
from spinegen.models import CanvasInfo, LayerArtifact
from spinegen.spine_json import compile_spine_json
from spinegen.validate import validate_spine_bundle


def test_atlas_and_spine_json_compile(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    body_path = images_dir / "body.png"
    head_path = images_dir / "head.png"
    Image.new("RGBA", (40, 60), (180, 60, 60, 255)).save(body_path)
    Image.new("RGBA", (32, 28), (60, 60, 180, 255)).save(head_path)

    canvas = CanvasInfo(width=128, height=128, origin_x=64, origin_y=96)
    layers = [
        LayerArtifact(
            id="layer_001",
            name="head",
            source_path="head",
            asset_name="head",
            image_path=head_path,
            bbox=(48, 16, 80, 44),
            width=32,
            height=28,
            opacity=1,
            blend_mode="normal",
            draw_order=0,
            tags={"bone": True},
        ),
        LayerArtifact(
            id="layer_002",
            name="body",
            source_path="body",
            asset_name="body",
            image_path=body_path,
            bbox=(44, 44, 84, 104),
            width=40,
            height=60,
            opacity=1,
            blend_mode="normal",
            draw_order=1,
            tags={"bone": True},
        ),
    ]

    atlas = pack_atlas(layers, tmp_path, "hero", max_width=256)
    rig = build_fallback_rig("hero", layers, canvas)
    spine_json = compile_spine_json("hero", canvas, layers, rig)

    assert atlas.image_path.exists()
    assert atlas.atlas_path.exists()
    assert spine_json["skins"][0]["attachments"]
    assert validate_spine_bundle(spine_json, atlas) == []

