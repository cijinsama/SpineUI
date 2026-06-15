from __future__ import annotations

from typing import Any

from spinegen.models import AtlasResult


def validate_spine_bundle(spine_json: dict[str, Any], atlas: AtlasResult) -> list[str]:
    errors: list[str] = []
    bones = {str(bone.get("name")) for bone in spine_json.get("bones", []) if isinstance(bone, dict)}
    if "root" not in bones:
        errors.append("缺少 root bone。")

    slot_names: set[str] = set()
    for slot in spine_json.get("slots", []):
        if not isinstance(slot, dict):
            errors.append("slots 中包含非对象条目。")
            continue
        name = str(slot.get("name") or "")
        bone = str(slot.get("bone") or "")
        if not name:
            errors.append("slot 缺少 name。")
        elif name in slot_names:
            errors.append(f"slot 重名：{name}")
        slot_names.add(name)
        if bone not in bones:
            errors.append(f"slot {name} 引用了不存在的 bone：{bone}")

    atlas_regions = set(atlas.regions)
    skins = spine_json.get("skins", [])
    if not isinstance(skins, list) or not skins:
        errors.append("缺少 skins。")
        return errors

    referenced_paths: set[str] = set()
    for skin in skins:
        if not isinstance(skin, dict):
            errors.append("skins 中包含非对象条目。")
            continue
        attachments = skin.get("attachments", {})
        if not isinstance(attachments, dict):
            errors.append(f"skin {skin.get('name', '<unnamed>')} attachments 不是对象。")
            continue
        for slot_name, slot_attachments in attachments.items():
            if slot_name not in slot_names:
                errors.append(f"skin 引用了不存在的 slot：{slot_name}")
            if not isinstance(slot_attachments, dict):
                errors.append(f"slot {slot_name} 的 attachment 集合不是对象。")
                continue
            for attachment_name, attachment in slot_attachments.items():
                if not isinstance(attachment, dict):
                    errors.append(f"attachment {attachment_name} 不是对象。")
                    continue
                path = str(attachment.get("path") or attachment_name)
                referenced_paths.add(path)
                if path not in atlas_regions:
                    errors.append(f"attachment path 没有对应 atlas region：{path}")

    missing_attachments = atlas_regions - referenced_paths
    if missing_attachments:
        preview = ", ".join(sorted(missing_attachments)[:8])
        errors.append(f"atlas 中有未被 skin 引用的 region：{preview}")
    return errors

