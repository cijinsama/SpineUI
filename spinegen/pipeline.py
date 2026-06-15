from __future__ import annotations

import json
import logging
import shutil
import time
import zipfile
from collections.abc import Callable
from pathlib import Path

from spinegen.animations import ensure_prompted_animations
from spinegen.atlas import pack_atlas
from spinegen.config import LLMSettings
from spinegen.layer_filter import resolve_redundant_layers
from spinegen.layer_selection import select_setup_layers
from spinegen.llm import build_fallback_rig, request_layer_visibility_plan, request_rig_plan, request_setup_group_choice
from spinegen.models import ConversionResult
from spinegen.naming import slugify
from spinegen.preview import iframe_for_preview, write_preview_html
from spinegen.prompts import build_prompt_context
from spinegen.psd_ingest import export_psd_layers, write_ir
from spinegen.spine_json import compile_spine_json
from spinegen.validate import validate_spine_bundle


OUTPUT_ROOT = Path("outputs")
StageLogger = Callable[[str], None]
logger = logging.getLogger(__name__)


def run_conversion(
    psd_path: Path,
    prompt: str,
    llm_settings: LLMSettings | None,
    use_llm: bool,
    atlas_width: int,
    stage_logger: StageLogger | None = None,
) -> ConversionResult:
    prompt_context = build_prompt_context(prompt)
    messages: list[str] = []
    _stage(messages, stage_logger, "检查输入文件...")
    if psd_path.suffix.lower() not in {".psd", ".psb"}:
        raise ValueError("只支持 .psd 或 .psb 文件。")

    skeleton_name = slugify(psd_path.stem, fallback="skeleton")
    _stage(messages, stage_logger, "创建输出目录...")
    output_dir = _new_output_dir(skeleton_name)
    _stage(messages, stage_logger, f"输出目录：{output_dir.resolve()}")

    _stage(messages, stage_logger, "读取 PSD 并导出可见图层...")
    canvas, layers, composite_path, ir = export_psd_layers(psd_path, output_dir)
    if not layers:
        raise ValueError("没有找到可导出的可见像素图层。")

    settings = llm_settings or LLMSettings.from_env()
    selection = select_setup_layers(layers, canvas, prompt_context.user_prompt)
    for message in selection.messages:
        _stage(messages, stage_logger, message)
    if use_llm and selection.has_alternatives:
        try:
            _stage(messages, stage_logger, "调用 LLM 选择 active pose/variant group...")
            preferred_group_id = request_setup_group_choice(prompt_context, selection.alternative_groups, settings=settings)
            selection = select_setup_layers(
                layers,
                canvas,
                prompt_context.user_prompt,
                preferred_group_id=preferred_group_id,
            )
            if preferred_group_id and selection.selected_group_id == preferred_group_id:
                _stage(messages, stage_logger, f"LLM 选择 active group：{preferred_group_id}。")
            elif preferred_group_id:
                _stage(messages, stage_logger, f"LLM 返回的 group 不可用，已使用保守选择：{selection.selected_group_id}。")
            for message in selection.messages:
                _stage(messages, stage_logger, message)
        except Exception as exc:  # noqa: BLE001 - conversion can continue with the conservative group choice.
            _stage(messages, stage_logger, f"LLM active group 选择失败，使用保守选择：{exc}")
    layers = selection.layers
    layer_filter = resolve_redundant_layers(layers)
    if use_llm:
        try:
            _stage(messages, stage_logger, "调用 LLM 生成 active group 图层可见性计划...")
            llm_hidden_ids = request_layer_visibility_plan(prompt_context, layers, settings=settings)
            layer_filter = resolve_redundant_layers(layers, hidden_layer_ids=llm_hidden_ids)
            if llm_hidden_ids:
                _stage(messages, stage_logger, f"LLM 建议隐藏 {len(llm_hidden_ids)} 个图层，已校验并应用。")
        except Exception as exc:  # noqa: BLE001 - conversion can continue with all active layers.
            _stage(messages, stage_logger, f"LLM 图层可见性计划失败，保留 active group 图层：{exc}")
    for message in layer_filter.messages:
        _stage(messages, stage_logger, message)
    layers = layer_filter.layers
    ir["selection"] = {
        "selected_group_id": selection.selected_group_id,
        "alternative_group_ids": [group.id for group in selection.alternative_groups],
        "exported_layer_count": len(layers),
        "hidden_redundant_layer_ids": layer_filter.hidden_layer_ids,
    }

    _stage(messages, stage_logger, "写入 IR 元数据...")
    ir_path = output_dir / f"{skeleton_name}.ir.json"
    write_ir(ir, ir_path)
    _stage(messages, stage_logger, f"已导出 {len(layers)} 个可见图层。")

    _stage(messages, stage_logger, f"打包 atlas，最大宽度 {atlas_width}px...")
    atlas = pack_atlas(layers, output_dir, skeleton_name, max_width=atlas_width)
    _stage(messages, stage_logger, f"已生成 atlas：{atlas.atlas_path.name} ({atlas.width}x{atlas.height})。")

    if use_llm:
        try:
            _stage(
                messages,
                stage_logger,
                f"调用 LLM 生成 RigPlan：max_tokens={settings.max_tokens}, thinking={settings.enable_thinking}...",
            )
            rig = request_rig_plan(
                skeleton_name=skeleton_name,
                layers=layers,
                canvas=canvas,
                prompt_context=prompt_context,
                settings=settings,
            )
            _stage(
                messages,
                stage_logger,
                f"LLM RigPlan 已生成：max_tokens={settings.max_tokens}, thinking={settings.enable_thinking}。",
            )
        except Exception as exc:  # noqa: BLE001 - conversion should still produce fallback files.
            rig = build_fallback_rig(
                skeleton_name=skeleton_name,
                layers=layers,
                canvas=canvas,
                notes=[f"LLM RigPlan failed, used fallback: {exc}"],
            )
            _stage(messages, stage_logger, f"LLM RigPlan 失败，已使用基础 fallback：{exc}")
    else:
        _stage(messages, stage_logger, "跳过 LLM，生成基础 fallback RigPlan...")
        rig = build_fallback_rig(skeleton_name=skeleton_name, layers=layers, canvas=canvas)
        _stage(messages, stage_logger, "已跳过 LLM，使用基础 fallback RigPlan。")

    if rig.source != "llm":
        _stage(messages, stage_logger, "fallback 模式根据 prompt 补齐基础动作动画...")
        rig = ensure_prompted_animations(rig, prompt_context.user_prompt)

    _stage(messages, stage_logger, "写入 RigPlan JSON...")
    rig_path = output_dir / f"{skeleton_name}.rigplan.json"
    rig_path.write_text(json.dumps(rig.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    _stage(messages, stage_logger, "编译 Spine JSON...")
    spine_json = compile_spine_json(skeleton_name, canvas, layers, rig)
    _stage(messages, stage_logger, "校验 Spine JSON 与 atlas 引用...")
    validation_errors = validate_spine_bundle(spine_json, atlas)
    if validation_errors:
        joined = "；".join(validation_errors)
        raise ValueError(f"生成结果未通过校验：{joined}")

    json_path = output_dir / f"{skeleton_name}.json"
    json_path.write_text(json.dumps(spine_json, ensure_ascii=False, indent=2), encoding="utf-8")
    _stage(messages, stage_logger, f"已生成 Spine JSON：{json_path.name}。")

    _stage(messages, stage_logger, "生成 Spine Web Player 预览 HTML...")
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

    _stage(messages, stage_logger, "打包下载 zip...")
    zip_path = output_dir / f"{skeleton_name}.zip"
    _write_zip(zip_path, output_dir, download_files)
    _stage(messages, stage_logger, f"已打包下载文件：{zip_path.name}。")
    _stage(messages, stage_logger, "完成。")

    return ConversionResult(
        output_dir=output_dir,
        zip_path=zip_path,
        download_files=download_files,
        preview_iframe_html=iframe_for_preview(preview_doc),
        spine_json=spine_json,
        rig_plan=rig.to_dict(),
        messages=messages,
    )


def _stage(messages: list[str], stage_logger: StageLogger | None, message: str) -> None:
    messages.append(message)
    logger.info(message)
    if stage_logger is not None:
        stage_logger(message)


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
