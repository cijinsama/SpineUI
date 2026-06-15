from __future__ import annotations

import json
import shutil
import time
import zipfile
from pathlib import Path

from spinegen.atlas import pack_atlas
from spinegen.config import LLMSettings
from spinegen.llm import build_fallback_rig, request_rig_plan
from spinegen.models import ConversionResult
from spinegen.naming import slugify
from spinegen.preview import iframe_for_preview, write_preview_html
from spinegen.psd_ingest import export_psd_layers, write_ir
from spinegen.spine_json import compile_spine_json
from spinegen.validate import validate_spine_bundle


OUTPUT_ROOT = Path("outputs")


def run_conversion(
    psd_path: Path,
    prompt: str,
    llm_settings: LLMSettings | None,
    use_llm: bool,
    atlas_width: int,
) -> ConversionResult:
    if psd_path.suffix.lower() not in {".psd", ".psb"}:
        raise ValueError("只支持 .psd 或 .psb 文件。")

    skeleton_name = slugify(psd_path.stem, fallback="skeleton")
    output_dir = _new_output_dir(skeleton_name)
    messages: list[str] = [f"输出目录：{output_dir.resolve()}"]

    canvas, layers, composite_path, ir = export_psd_layers(psd_path, output_dir)
    if not layers:
        raise ValueError("没有找到可导出的可见像素图层。")

    ir_path = output_dir / f"{skeleton_name}.ir.json"
    write_ir(ir, ir_path)
    messages.append(f"已导出 {len(layers)} 个可见图层。")

    atlas = pack_atlas(layers, output_dir, skeleton_name, max_width=atlas_width)
    messages.append(f"已生成 atlas：{atlas.atlas_path.name} ({atlas.width}x{atlas.height})。")

    rig_messages: list[str] = []
    if use_llm:
        try:
            settings = llm_settings or LLMSettings.from_env()
            rig = request_rig_plan(
                skeleton_name=skeleton_name,
                layers=layers,
                canvas=canvas,
                prompt=prompt,
                settings=settings,
            )
            rig_messages.append(
                f"LLM RigPlan 已生成：model={settings.model}, max_tokens={settings.max_tokens}, thinking={settings.enable_thinking}。"
            )
        except Exception as exc:  # noqa: BLE001 - conversion should still produce deterministic files.
            rig = build_fallback_rig(
                skeleton_name=skeleton_name,
                layers=layers,
                canvas=canvas,
                notes=[f"LLM RigPlan failed, used fallback: {exc}"],
            )
            rig_messages.append(f"LLM RigPlan 失败，已使用规则回退：{exc}")
    else:
        rig = build_fallback_rig(skeleton_name=skeleton_name, layers=layers, canvas=canvas)
        rig_messages.append("已跳过 LLM，使用规则 RigPlan。")
    messages.extend(rig_messages)

    rig_path = output_dir / f"{skeleton_name}.rigplan.json"
    rig_path.write_text(json.dumps(rig.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    spine_json = compile_spine_json(skeleton_name, canvas, layers, rig)
    validation_errors = validate_spine_bundle(spine_json, atlas)
    if validation_errors:
        joined = "；".join(validation_errors)
        raise ValueError(f"生成结果未通过校验：{joined}")

    json_path = output_dir / f"{skeleton_name}.json"
    json_path.write_text(json.dumps(spine_json, ensure_ascii=False, indent=2), encoding="utf-8")
    messages.append(f"已生成 Spine JSON：{json_path.name}。")

    preview_path = output_dir / "preview.html"
    animation_name = next(iter(spine_json.get("animations", {}) or {}), None)
    preview_doc = write_preview_html(
        output_path=preview_path,
        skeleton_name=skeleton_name,
        json_path=json_path,
        atlas_path=atlas.atlas_path,
        image_path=atlas.image_path,
        animation_name=animation_name,
    )

    download_files = [
        json_path,
        atlas.atlas_path,
        atlas.image_path,
        ir_path,
        rig_path,
        preview_path,
    ]
    if composite_path.exists():
        download_files.append(composite_path)

    zip_path = output_dir / f"{skeleton_name}.zip"
    _write_zip(zip_path, output_dir, download_files)
    messages.append(f"已打包下载文件：{zip_path.name}。")

    return ConversionResult(
        output_dir=output_dir,
        zip_path=zip_path,
        download_files=download_files,
        preview_iframe_html=iframe_for_preview(preview_doc),
        spine_json=spine_json,
        rig_plan=rig.to_dict(),
        messages=messages,
    )


def _new_output_dir(skeleton_name: str) -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_dir = OUTPUT_ROOT / f"{timestamp}-{skeleton_name}"
    counter = 2
    while output_dir.exists():
        output_dir = OUTPUT_ROOT / f"{timestamp}-{skeleton_name}-{counter}"
        counter += 1
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def _write_zip(zip_path: Path, output_dir: Path, download_files: list[Path]) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in download_files:
            if path.exists():
                archive.write(path, path.relative_to(output_dir))
        images_dir = output_dir / "images"
        if images_dir.exists():
            for image_path in sorted(images_dir.glob("*.png")):
                archive.write(image_path, image_path.relative_to(output_dir))

    stale_copy = output_dir / zip_path.name
    if stale_copy.exists() and stale_copy != zip_path:
        stale_copy.unlink()


def clean_outputs(max_age_seconds: int = 60 * 60 * 24) -> None:
    now = time.time()
    if not OUTPUT_ROOT.exists():
        return
    for path in OUTPUT_ROOT.iterdir():
        if path.is_dir() and now - path.stat().st_mtime > max_age_seconds:
            shutil.rmtree(path, ignore_errors=True)
