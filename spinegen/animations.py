from __future__ import annotations

from typing import Any

from spinegen.models import RigPlan


ATTACK_KEYWORDS = ("attack", "attacking", "strike", "slash", "hit", "攻击", "打击", "挥砍", "挥击")


def ensure_prompted_animations(rig_plan: RigPlan, prompt: str) -> RigPlan:
    if _mentions(prompt, ATTACK_KEYWORDS):
        _ensure_attack_animation(rig_plan)
    return rig_plan


def _ensure_attack_animation(rig_plan: RigPlan) -> None:
    animation = _find_animation(rig_plan.animations, "attack")
    if animation is None:
        animation = {"name": "attack", "duration": 0.8, "bone_timelines": []}
        rig_plan.animations.append(animation)

    animation["name"] = "attack"
    animation["duration"] = max(float(animation.get("duration") or 0.0), 0.8)
    timelines = animation.setdefault("bone_timelines", [])
    if not isinstance(timelines, list):
        timelines = []
        animation["bone_timelines"] = timelines

    bone_names = [str(bone.get("name")) for bone in rig_plan.bones if isinstance(bone, dict) and bone.get("name")]
    if not bone_names:
        return

    root_bone = "root" if "root" in bone_names else bone_names[0]
    body_bone = _find_bone(bone_names, ("body", "torso", "chest", "spine")) or root_bone
    weapon_bone = _find_bone(bone_names, ("weapon", "staff", "sword", "wand", "right_arm", "right_hand", "hand")) or body_bone

    root_timeline = _get_timeline(timelines, root_bone)
    if not _has_nonzero_frames(root_timeline.get("translate"), ("x", "y")):
        root_timeline["translate"] = [
            {"time": 0.0, "x": 0.0, "y": 0.0},
            {"time": 0.16, "x": -14.0, "y": 0.0},
            {"time": 0.34, "x": 42.0, "y": 2.0},
            {"time": 0.62, "x": 0.0, "y": 0.0},
            {"time": 0.8, "x": 0.0, "y": 0.0},
        ]

    body_timeline = _get_timeline(timelines, body_bone)
    if not _has_nonzero_frames(body_timeline.get("rotate"), ("angle",)):
        body_timeline["rotate"] = [
            {"time": 0.0, "angle": 0.0},
            {"time": 0.16, "angle": -5.0},
            {"time": 0.34, "angle": 8.0},
            {"time": 0.62, "angle": 0.0},
            {"time": 0.8, "angle": 0.0},
        ]

    weapon_timeline = _get_timeline(timelines, weapon_bone)
    if not _has_nonzero_frames(weapon_timeline.get("rotate"), ("angle",)):
        weapon_timeline["rotate"] = [
            {"time": 0.0, "angle": 0.0},
            {"time": 0.16, "angle": -22.0},
            {"time": 0.34, "angle": 42.0},
            {"time": 0.62, "angle": 0.0},
            {"time": 0.8, "angle": 0.0},
        ]

    _stretch_short_animation(timelines, minimum_duration=0.8)


def _find_animation(animations: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for animation in animations:
        if isinstance(animation, dict) and str(animation.get("name", "")).lower() == name:
            return animation
    return None


def _get_timeline(timelines: list[Any], bone: str) -> dict[str, Any]:
    for timeline in timelines:
        if isinstance(timeline, dict) and timeline.get("bone") == bone:
            return timeline
    timeline = {"bone": bone}
    timelines.append(timeline)
    return timeline


def _find_bone(bone_names: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = [(bone_name, bone_name.lower()) for bone_name in bone_names]
    for candidate in candidates:
        for bone_name, lowered_name in lowered:
            if candidate in lowered_name:
                return bone_name
    return None


def _stretch_short_animation(timelines: list[Any], minimum_duration: float) -> None:
    max_time = 0.0
    for timeline in timelines:
        if not isinstance(timeline, dict):
            continue
        for key in ("rotate", "translate", "scale"):
            frames = timeline.get(key)
            if not isinstance(frames, list):
                continue
            for frame in frames:
                if isinstance(frame, dict):
                    max_time = max(max_time, float(frame.get("time") or 0.0))

    if max_time <= 0.0 or max_time >= minimum_duration:
        return

    scale = minimum_duration / max_time
    for timeline in timelines:
        if not isinstance(timeline, dict):
            continue
        for key in ("rotate", "translate", "scale"):
            frames = timeline.get(key)
            if not isinstance(frames, list):
                continue
            for frame in frames:
                if isinstance(frame, dict):
                    frame["time"] = round(float(frame.get("time") or 0.0) * scale, 4)


def _has_nonzero_frames(frames: Any, value_keys: tuple[str, ...]) -> bool:
    if not isinstance(frames, list):
        return False
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        for key in value_keys:
            if abs(float(frame.get(key) or 0.0)) > 0.001:
                return True
    return False


def _mentions(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)

