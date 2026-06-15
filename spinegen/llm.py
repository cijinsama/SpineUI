from __future__ import annotations

import base64
import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai import OpenAIError

from spinegen.config import DEFAULT_BASE_URL, LLMSettings
from spinegen.layer_selection import group_summaries
from spinegen.models import CanvasInfo, LayerArtifact, RigPlan
from spinegen.naming import slugify, unique_name


def build_fallback_rig(
    skeleton_name: str,
    layers: list[LayerArtifact],
    canvas: CanvasInfo,
    notes: list[str] | None = None,
) -> RigPlan:
    slots: list[dict[str, Any]] = []
    bones: list[dict[str, Any]] = [{"name": "root", "parent": None}]
    used_bones = {"root"}

    for layer in _bottom_to_top(layers):
        bone_name = unique_name(slugify(layer.asset_name, fallback=layer.id), used_bones)
        bones.append(
            {
                "name": bone_name,
                "parent": "root",
                "pivot_layer_id": layer.id,
                "x": round(layer.spine_x(canvas), 3),
                "y": round(layer.spine_y(canvas), 3),
            }
        )
        slots.append(
            {
                "name": layer.asset_name,
                "bone": bone_name,
                "layer_id": layer.id,
                "attachment": layer.asset_name,
            }
        )

    return RigPlan(
        skeleton_name=skeleton_name,
        bones=bones,
        slots=slots,
        animations=[_default_idle_animation()],
        notes=notes or ["Fallback rig generated without LLM."],
        source="fallback",
    )


def request_rig_plan(
    skeleton_name: str,
    layers: list[LayerArtifact],
    canvas: CanvasInfo,
    prompt: str,
    settings: LLMSettings | None = None,
) -> RigPlan:
    load_dotenv()
    settings = settings or LLMSettings.from_env()
    api_key = os.getenv("NRP_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 NRP_API_KEY 或 OPENAI_API_KEY。请在 .env 中配置。")

    base_url = os.getenv("NRP_BASE_URL", DEFAULT_BASE_URL)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=settings.timeout_seconds)

    layer_summary = [_layer_prompt_record(layer, canvas) for layer in layers]
    messages = [
        {
            "role": "system",
            "content": (
                "You create constrained RigPlan JSON for a PSD-to-Spine compiler. "
                "Return only valid JSON. Do not invent layer_id values. "
                "Use existing layer_id values exactly. Keep coordinates out unless they are simple pivots. "
                "Schema: {skeleton_name:string,bones:[{name:string,parent:string|null,pivot_layer_id:string|null}],"
                "slots:[{name:string,bone:string,layer_id:string,attachment:string}],"
                "animations:[{name:string,duration:number,bone_timelines:[{bone:string,rotate?:[{time:number,angle:number}],"
                "translate?:[{time:number,x:number,y:number}],scale?:[{time:number,x:number,y:number}]}]}],notes:[string]}."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": "Generate a practical Spine setup-pose rig plan from the layer metadata and prompt.",
                    "prompt": prompt,
                    "skeleton_name": skeleton_name,
                    "canvas": {
                        "width": canvas.width,
                        "height": canvas.height,
                        "origin": [canvas.origin_x, canvas.origin_y],
                    },
                    "layers_bottom_to_top": layer_summary,
                    "rules": [
                        "Every slot must reference an existing layer_id.",
                        "Every slot bone must exist in bones.",
                        "Prefer root/body/head/limb semantic bones when layer names imply them.",
                        "If unsure, bind the layer to root or a simple part bone.",
                        "Keep animations subtle and short. Use rotate timelines first.",
                        "Use only ASCII names in bones, slots, and animations.",
                        "Do not include comments, markdown, or explanatory text outside the JSON object.",
                    ],
                },
                ensure_ascii=False,
            ),
        },
    ]

    extra_body: dict[str, Any] = {}
    cache_salt = os.getenv("NRP_CACHE_SALT")
    if cache_salt:
        extra_body["cache_salt"] = cache_salt
    elif os.getenv("NRP_DISABLE_RANDOM_CACHE_SALT", "").lower() not in {"1", "true", "yes"}:
        extra_body["cache_salt"] = base64.b64encode(os.urandom(32)).decode("ascii")
    if settings.enable_thinking:
        extra_body["chat_template_kwargs"] = {
            "enable_thinking": True,
            "thinking": True,
            "preserve_thinking": settings.preserve_thinking,
            "clear_thinking": not settings.preserve_thinking,
        }

    kwargs: dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
        "response_format": {"type": "json_object"},
    }
    if extra_body:
        kwargs["extra_body"] = extra_body

    completion = _create_chat_completion(client, kwargs)
    content = completion.choices[0].message.content or "{}"
    parsed = _parse_json_object(content)
    return _sanitize_rig_plan(parsed, skeleton_name, layers, canvas)


def request_setup_group_choice(
    prompt: str,
    groups: list[Any],
    settings: LLMSettings | None = None,
) -> str | None:
    load_dotenv()
    settings = settings or LLMSettings.from_env()
    api_key = os.getenv("NRP_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 NRP_API_KEY 或 OPENAI_API_KEY。请在 .env 中配置。")

    base_url = os.getenv("NRP_BASE_URL", DEFAULT_BASE_URL)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=settings.timeout_seconds)
    messages = [
        {
            "role": "system",
            "content": (
                "You select one active setup group from PSD top-level groups. "
                "The groups may be mutually exclusive pose or variant groups. "
                "Return only JSON: {\"selected_group_id\": string|null, \"reason\": string}. "
                "Use only ids from the provided group list. If the groups are not alternatives, return null."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "prompt": prompt,
                    "groups": group_summaries(groups),
                    "rules": [
                        "Do not invent group ids.",
                        "Prefer the group whose label or layer parts match the requested action or pose.",
                        "If the prompt does not indicate a pose, choose the most neutral setup group.",
                    ],
                },
                ensure_ascii=False,
            ),
        },
    ]
    kwargs: dict[str, Any] = {
        "model": settings.model,
        "messages": messages,
        "max_tokens": min(settings.max_tokens, 2048),
        "temperature": 0.0,
        "top_p": settings.top_p,
        "response_format": {"type": "json_object"},
    }
    completion = _create_chat_completion(client, kwargs)
    parsed = _parse_json_object(completion.choices[0].message.content or "{}")
    selected = parsed.get("selected_group_id")
    return str(selected) if selected else None


def _create_chat_completion(client: OpenAI, kwargs: dict[str, Any]) -> Any:
    try:
        return client.chat.completions.create(**kwargs)
    except OpenAIError:
        retry_kwargs = dict(kwargs)
        retry_kwargs.pop("response_format", None)
        return client.chat.completions.create(**retry_kwargs)


def _sanitize_rig_plan(
    raw: dict[str, Any],
    skeleton_name: str,
    layers: list[LayerArtifact],
    canvas: CanvasInfo,
) -> RigPlan:
    layer_by_id = {layer.id: layer for layer in layers}
    layer_by_asset = {layer.asset_name: layer for layer in layers}

    used_bones: set[str] = set()
    bones: list[dict[str, Any]] = []
    raw_bones = raw.get("bones") if isinstance(raw.get("bones"), list) else []
    for item in raw_bones:
        if not isinstance(item, dict):
            continue
        name = slugify(str(item.get("name") or ""), fallback="")
        if not name:
            continue
        name = unique_name(name, used_bones)
        parent = item.get("parent")
        parent_name = slugify(str(parent), fallback="root") if parent else None
        pivot_layer_id = item.get("pivot_layer_id")
        if pivot_layer_id not in layer_by_id:
            pivot_layer_id = None
        bones.append({"name": name, "parent": parent_name, "pivot_layer_id": pivot_layer_id})

    if "root" not in used_bones:
        bones.insert(0, {"name": "root", "parent": None, "pivot_layer_id": None})
        used_bones.add("root")

    known_bones = {bone["name"] for bone in bones}
    for bone in bones:
        if bone["name"] == "root":
            bone["parent"] = None
        elif bone.get("parent") not in known_bones:
            bone["parent"] = "root"

    slots: list[dict[str, Any]] = []
    used_slots: set[str] = set()
    covered_layers: set[str] = set()
    raw_slots = raw.get("slots") if isinstance(raw.get("slots"), list) else []

    for item in raw_slots:
        if not isinstance(item, dict):
            continue
        layer_id = str(item.get("layer_id") or "")
        layer = layer_by_id.get(layer_id) or layer_by_asset.get(layer_id)
        if layer is None:
            continue
        bone = slugify(str(item.get("bone") or "root"), fallback="root")
        if bone not in known_bones:
            bone = "root"
        slot_name = unique_name(slugify(str(item.get("name") or layer.asset_name), fallback=layer.asset_name), used_slots)
        slots.append(
            {
                "name": slot_name,
                "bone": bone,
                "layer_id": layer.id,
                "attachment": layer.asset_name,
            }
        )
        covered_layers.add(layer.id)

    for layer in _bottom_to_top(layers):
        if layer.id in covered_layers:
            continue
        slot_name = unique_name(layer.asset_name, used_slots)
        slots.append(
            {
                "name": slot_name,
                "bone": "root",
                "layer_id": layer.id,
                "attachment": layer.asset_name,
            }
        )

    layer_order = {layer.id: index for index, layer in enumerate(_bottom_to_top(layers))}
    slots.sort(key=lambda slot: layer_order.get(str(slot.get("layer_id")), len(layer_order)))

    animations = _sanitize_animations(raw.get("animations"), known_bones)
    if not animations:
        animations = [_default_idle_animation()]

    notes = raw.get("notes") if isinstance(raw.get("notes"), list) else []
    text_notes = [str(note) for note in notes[:10]]
    return RigPlan(
        skeleton_name=slugify(str(raw.get("skeleton_name") or skeleton_name), fallback=skeleton_name),
        bones=_order_bones(bones),
        slots=slots,
        animations=animations,
        notes=text_notes,
        source="llm",
    )


def _order_bones(bones: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_name = {bone["name"]: bone for bone in bones}
    ordered: list[dict[str, Any]] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(name: str) -> None:
        if name in visited or name in visiting:
            return
        visiting.add(name)
        bone = by_name[name]
        parent = bone.get("parent")
        if parent and parent in by_name:
            visit(parent)
        visiting.remove(name)
        visited.add(name)
        ordered.append(bone)

    if "root" in by_name:
        visit("root")
    for name in list(by_name):
        visit(name)
    return ordered


def _sanitize_animations(raw_animations: Any, known_bones: set[str]) -> list[dict[str, Any]]:
    if not isinstance(raw_animations, list):
        return []

    animations: list[dict[str, Any]] = []
    for item in raw_animations[:8]:
        if not isinstance(item, dict):
            continue
        name = slugify(str(item.get("name") or "animation"), fallback="animation")
        duration = _positive_float(item.get("duration"), default=1.0)
        raw_timelines = item.get("bone_timelines")
        timelines: list[dict[str, Any]] = []
        if isinstance(raw_timelines, list):
            for raw_timeline in raw_timelines[:32]:
                timeline = _sanitize_timeline(raw_timeline, known_bones, duration)
                if timeline:
                    timelines.append(timeline)
        animations.append({"name": name, "duration": duration, "bone_timelines": timelines})
    return animations


def _sanitize_timeline(raw: Any, known_bones: set[str], duration: float) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        return None
    bone = slugify(str(raw.get("bone") or ""), fallback="")
    if bone not in known_bones:
        return None

    timeline: dict[str, Any] = {"bone": bone}
    for key in ("rotate", "translate", "scale"):
        values = raw.get(key)
        if not isinstance(values, list):
            continue
        frames = []
        for frame in values[:16]:
            if not isinstance(frame, dict):
                continue
            time_value = min(max(_positive_float(frame.get("time"), default=0.0), 0.0), duration)
            if key == "rotate":
                frames.append({"time": time_value, "angle": float(frame.get("angle", 0.0))})
            elif key == "translate":
                frames.append(
                    {
                        "time": time_value,
                        "x": float(frame.get("x", 0.0)),
                        "y": float(frame.get("y", 0.0)),
                    }
                )
            elif key == "scale":
                frames.append(
                    {
                        "time": time_value,
                        "x": float(frame.get("x", 1.0)),
                        "y": float(frame.get("y", 1.0)),
                    }
                )
        if frames:
            timeline[key] = sorted(frames, key=lambda value: value["time"])
    return timeline if len(timeline) > 1 else None


def _default_idle_animation() -> dict[str, Any]:
    return {
        "name": "idle",
        "duration": 1.0,
        "bone_timelines": [
            {
                "bone": "root",
                "translate": [
                    {"time": 0.0, "x": 0.0, "y": 0.0},
                    {"time": 0.5, "x": 0.0, "y": 2.0},
                    {"time": 1.0, "x": 0.0, "y": 0.0},
                ],
            }
        ],
    }


def _layer_prompt_record(layer: LayerArtifact, canvas: CanvasInfo) -> dict[str, Any]:
    return {
        "id": layer.id,
        "name": layer.name,
        "path": layer.source_path,
        "asset": layer.asset_name,
        "bbox": list(layer.bbox),
        "size": [layer.width, layer.height],
        "spine_center": [round(layer.spine_x(canvas), 3), round(layer.spine_y(canvas), 3)],
        "draw_order": layer.draw_order,
        "tags": layer.tags,
    }


def _bottom_to_top(layers: list[LayerArtifact]) -> list[LayerArtifact]:
    return sorted(layers, key=lambda layer: layer.draw_order)


def _parse_json_object(content: str) -> dict[str, Any]:
    stripped = _strip_thinking(content.strip())
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        parsed = json.loads(stripped[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("LLM 没有返回可解析的 JSON 对象。")


def _strip_thinking(content: str) -> str:
    return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE).strip()


def _positive_float(value: Any, default: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number >= 0 else default
