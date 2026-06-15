from __future__ import annotations

from typing import Any

from spinegen.models import CanvasInfo, LayerArtifact, RigPlan


TARGET_SPINE_VERSION = "4.2.0"


def compile_spine_json(
    skeleton_name: str,
    canvas: CanvasInfo,
    layers: list[LayerArtifact],
    rig_plan: RigPlan,
) -> dict[str, Any]:
    layer_by_id = {layer.id: layer for layer in layers}
    bone_positions = _compute_bone_positions(rig_plan, layer_by_id, canvas)
    bones = _compile_bones(rig_plan, bone_positions)
    slots = _compile_slots(rig_plan, layer_by_id)
    skins = _compile_skins(rig_plan, layer_by_id, canvas, bone_positions)
    animations = _compile_animations(rig_plan)

    return {
        "skeleton": {
            "hash": "",
            "spine": TARGET_SPINE_VERSION,
            "x": -canvas.origin_x,
            "y": -canvas.origin_y,
            "width": canvas.width,
            "height": canvas.height,
            "images": "./",
            "audio": "",
        },
        "bones": bones,
        "slots": slots,
        "skins": skins,
        "animations": animations,
    }


def _compute_bone_positions(
    rig_plan: RigPlan,
    layer_by_id: dict[str, LayerArtifact],
    canvas: CanvasInfo,
) -> dict[str, tuple[float, float]]:
    world_positions: dict[str, tuple[float, float]] = {"root": (0.0, 0.0)}
    for bone in rig_plan.bones:
        name = str(bone.get("name") or "root")
        if name == "root":
            world_positions[name] = (0.0, 0.0)
            continue
        pivot_layer_id = bone.get("pivot_layer_id")
        layer = layer_by_id.get(str(pivot_layer_id)) if pivot_layer_id else None
        if layer is not None:
            world_positions[name] = (layer.spine_x(canvas), layer.spine_y(canvas))
        else:
            world_positions[name] = (0.0, 0.0)
    return world_positions


def _compile_bones(
    rig_plan: RigPlan,
    bone_positions: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    compiled: list[dict[str, Any]] = []
    for bone in rig_plan.bones:
        name = str(bone.get("name") or "root")
        parent = bone.get("parent")
        world_x, world_y = bone_positions.get(name, (0.0, 0.0))
        if not parent:
            compiled.append({"name": name})
            continue
        parent_x, parent_y = bone_positions.get(str(parent), (0.0, 0.0))
        compiled.append(
            {
                "name": name,
                "parent": str(parent),
                "x": round(world_x - parent_x, 3),
                "y": round(world_y - parent_y, 3),
            }
        )
    return compiled


def _compile_slots(
    rig_plan: RigPlan,
    layer_by_id: dict[str, LayerArtifact],
) -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    for slot in rig_plan.slots:
        layer = layer_by_id.get(str(slot.get("layer_id")))
        if layer is None:
            continue
        attachment_name = str(slot.get("attachment") or layer.asset_name)
        slots.append(
            {
                "name": str(slot.get("name") or layer.asset_name),
                "bone": str(slot.get("bone") or "root"),
                "attachment": attachment_name,
            }
        )
    return slots


def _compile_skins(
    rig_plan: RigPlan,
    layer_by_id: dict[str, LayerArtifact],
    canvas: CanvasInfo,
    bone_positions: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    attachments: dict[str, dict[str, Any]] = {}
    for slot in rig_plan.slots:
        layer = layer_by_id.get(str(slot.get("layer_id")))
        if layer is None:
            continue
        slot_name = str(slot.get("name") or layer.asset_name)
        bone_name = str(slot.get("bone") or "root")
        bone_x, bone_y = bone_positions.get(bone_name, (0.0, 0.0))
        attachment_name = str(slot.get("attachment") or layer.asset_name)
        attachments[slot_name] = {
            attachment_name: {
                "type": "region",
                "path": layer.asset_name,
                "x": round(layer.spine_x(canvas) - bone_x, 3),
                "y": round(layer.spine_y(canvas) - bone_y, 3),
                "scaleX": 1,
                "scaleY": 1,
                "rotation": 0,
                "width": layer.width,
                "height": layer.height,
                "color": _opacity_color(layer.opacity),
            }
        }
    return [{"name": "default", "attachments": attachments}]


def _compile_animations(rig_plan: RigPlan) -> dict[str, Any]:
    animations: dict[str, Any] = {}
    for animation in rig_plan.animations:
        name = str(animation.get("name") or "animation")
        bone_timelines: dict[str, Any] = {}
        for timeline in animation.get("bone_timelines", []):
            if not isinstance(timeline, dict):
                continue
            bone_name = str(timeline.get("bone") or "")
            if not bone_name:
                continue
            compiled_timeline: dict[str, Any] = {}
            if isinstance(timeline.get("rotate"), list):
                compiled_timeline["rotate"] = [
                    {
                        "time": float(frame.get("time", 0.0)),
                        "angle": float(frame.get("angle", 0.0)),
                    }
                    for frame in timeline["rotate"]
                    if isinstance(frame, dict)
                ]
            if isinstance(timeline.get("translate"), list):
                compiled_timeline["translate"] = [
                    {
                        "time": float(frame.get("time", 0.0)),
                        "x": float(frame.get("x", 0.0)),
                        "y": float(frame.get("y", 0.0)),
                    }
                    for frame in timeline["translate"]
                    if isinstance(frame, dict)
                ]
            if isinstance(timeline.get("scale"), list):
                compiled_timeline["scale"] = [
                    {
                        "time": float(frame.get("time", 0.0)),
                        "x": float(frame.get("x", 1.0)),
                        "y": float(frame.get("y", 1.0)),
                    }
                    for frame in timeline["scale"]
                    if isinstance(frame, dict)
                ]
            if compiled_timeline:
                bone_timelines[bone_name] = compiled_timeline
        animations[name] = {"bones": bone_timelines}
    return animations


def _opacity_color(opacity: float) -> str:
    alpha = max(0, min(255, round(opacity * 255)))
    return f"ffffff{alpha:02x}"

