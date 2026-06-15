from pathlib import Path

from PIL import Image

from spinegen.atlas import pack_atlas
from spinegen.animations import ensure_prompted_animations
from spinegen.llm import build_fallback_rig
from spinegen.models import CanvasInfo, LayerArtifact, RigPlan
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


def test_spine_42_rotate_timelines_use_value(tmp_path: Path) -> None:
    image_path = tmp_path / "weapon.png"
    Image.new("RGBA", (20, 80), (180, 60, 60, 255)).save(image_path)
    canvas = CanvasInfo(width=128, height=128, origin_x=64, origin_y=96)
    layer = LayerArtifact(
        id="layer_001",
        name="weapon",
        source_path="weapon",
        asset_name="weapon",
        image_path=image_path,
        bbox=(64, 20, 84, 100),
        width=20,
        height=80,
        opacity=1,
        blend_mode="normal",
        draw_order=0,
        tags={},
    )
    rig = RigPlan(
        skeleton_name="hero",
        bones=[
            {"name": "root", "parent": None},
            {"name": "weapon", "parent": "root", "pivot_layer_id": "layer_001"},
        ],
        slots=[{"name": "weapon", "bone": "weapon", "layer_id": "layer_001", "attachment": "weapon"}],
        animations=[
            {
                "name": "attack",
                "duration": 0.8,
                "bone_timelines": [
                    {
                        "bone": "weapon",
                        "rotate": [
                            {"time": 0.0, "angle": 0.0},
                            {"time": 0.4, "angle": 35.0},
                        ],
                    }
                ],
            }
        ],
    )

    spine_json = compile_spine_json("hero", canvas, [layer], rig)
    rotate_frame = spine_json["animations"]["attack"]["bones"]["weapon"]["rotate"][1]

    assert rotate_frame["value"] == 35.0
    assert "angle" not in rotate_frame


def test_attack_prompt_adds_visible_motion_to_fallback(tmp_path: Path) -> None:
    image_path = tmp_path / "body.png"
    Image.new("RGBA", (40, 60), (180, 60, 60, 255)).save(image_path)
    canvas = CanvasInfo(width=128, height=128, origin_x=64, origin_y=96)
    layer = LayerArtifact(
        id="layer_001",
        name="body",
        source_path="body",
        asset_name="body",
        image_path=image_path,
        bbox=(44, 44, 84, 104),
        width=40,
        height=60,
        opacity=1,
        blend_mode="normal",
        draw_order=0,
        tags={},
    )
    rig = build_fallback_rig("hero", [layer], canvas)

    ensure_prompted_animations(rig, "这是一个 Q 版角色，生成 attack 的基础骨骼和动画。")
    attack = next(animation for animation in rig.animations if animation["name"] == "attack")
    root = next(timeline for timeline in attack["bone_timelines"] if timeline["bone"] == "root")

    assert root["translate"][2]["x"] == 42.0
