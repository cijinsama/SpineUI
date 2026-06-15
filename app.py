from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Any, Iterator

import gradio as gr
from dotenv import load_dotenv

from spinegen.config import LLMSettings
from spinegen.pipeline import run_conversion


load_dotenv()
logger = logging.getLogger(__name__)

ConvertOutput = tuple[str, str, str | None, list[str], str, str]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _launch_kwargs() -> dict[str, object]:
    kwargs: dict[str, object] = {
        "server_name": os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        "share": _env_bool("GRADIO_SHARE", True),
    }
    server_port = os.getenv("GRADIO_SERVER_PORT")
    if server_port:
        kwargs["server_port"] = int(server_port)
    return kwargs


def convert_psd(
    psd_file: str | None,
    prompt: str,
    model: str,
    use_llm: bool,
    atlas_width: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
    preserve_thinking: bool,
) -> Iterator[ConvertOutput]:
    if not psd_file:
        yield (
            "请先上传 PSD 或 PSB 文件。",
            "",
            None,
            [],
            "",
            "",
        )
        return

    events: Queue[tuple[str, Any]] = Queue()

    def stage_logger(message: str) -> None:
        events.put(("stage", message))

    def worker() -> None:
        try:
            env_settings = LLMSettings.from_env()
            llm_settings = LLMSettings(
                model=model.strip() or env_settings.model,
                max_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                enable_thinking=bool(enable_thinking),
                preserve_thinking=bool(preserve_thinking),
                timeout_seconds=env_settings.timeout_seconds,
            )
            result = run_conversion(
                psd_path=Path(psd_file),
                prompt=prompt or "",
                llm_settings=llm_settings,
                use_llm=use_llm,
                atlas_width=int(atlas_width),
                stage_logger=stage_logger,
            )
            events.put(("result", result))
        except Exception as exc:  # noqa: BLE001 - surface conversion failures in Gradio.
            logger.exception("PSD to Spine conversion failed")
            events.put(("error", exc))

    Thread(target=worker, daemon=True).start()

    log_lines: list[str] = []
    while True:
        event_type, payload = events.get()
        if event_type == "stage":
            log_lines.append(str(payload))
            yield _empty_output("\n".join(log_lines))
        elif event_type == "error":
            log_lines.append(f"生成失败：{payload}")
            yield _empty_output("\n".join(log_lines))
            return
        elif event_type == "result":
            result = payload
            status = "\n".join(result.messages)
            json_preview = json.dumps(result.spine_json, ensure_ascii=False, indent=2)
            rig_preview = json.dumps(result.rig_plan, ensure_ascii=False, indent=2)
            yield (
                status,
                result.preview_iframe_html,
                str(result.zip_path),
                [str(path) for path in result.download_files],
                json_preview,
                rig_preview,
            )
            return


def _empty_output(status: str) -> ConvertOutput:
    return (status, "", None, [], "", "")


with gr.Blocks(title="PSD to Spine") as demo:
    gr.Markdown("## PSD to Spine")
    with gr.Row():
        with gr.Column(scale=1):
            psd_input = gr.File(
                label="PSD / PSB",
                file_types=[".psd", ".psb"],
                type="filepath",
            )
            prompt_input = gr.Textbox(
                label="Prompt",
                lines=7,
                placeholder="例如：这是一个 Q 版骑士角色，生成 idle / walk / attack 的基础骨骼和轻微待机动画。",
            )
            with gr.Row():
                model_input = gr.Dropdown(
                    label="NRP model",
                    value=LLMSettings.from_env().model,
                    choices=["qwen3", "gpt-oss", "kimi", "glm-5", "minimax-m2", "glm-4.7", "llama3-sdsc"],
                    allow_custom_value=True,
                    scale=2,
                )
                atlas_width_input = gr.Dropdown(
                    label="Atlas width",
                    value=2048,
                    choices=[512, 1024, 2048, 4096],
                    scale=1,
                )
            use_llm_input = gr.Checkbox(label="Use LLM RigPlan", value=True)
            with gr.Accordion("LLM parameters", open=False):
                max_tokens_input = gr.Slider(
                    label="max_tokens",
                    minimum=1024,
                    maximum=32768,
                    step=1024,
                    value=LLMSettings.from_env().max_tokens,
                )
                temperature_input = gr.Slider(
                    label="temperature",
                    minimum=0,
                    maximum=1.5,
                    step=0.05,
                    value=LLMSettings.from_env().temperature,
                )
                top_p_input = gr.Slider(
                    label="top_p",
                    minimum=0.05,
                    maximum=1,
                    step=0.05,
                    value=LLMSettings.from_env().top_p,
                )
                enable_thinking_input = gr.Checkbox(
                    label="Enable thinking",
                    value=LLMSettings.from_env().enable_thinking,
                )
                preserve_thinking_input = gr.Checkbox(
                    label="Preserve thinking in provider response",
                    value=LLMSettings.from_env().preserve_thinking,
                )
            generate_button = gr.Button("Generate", variant="primary")

        with gr.Column(scale=2):
            status_output = gr.Textbox(label="Status", lines=8)
            preview_output = gr.HTML(label="Spine preview")

    with gr.Row():
        zip_output = gr.File(label="Download zip")
        files_output = gr.File(label="Generated files", file_count="multiple")

    with gr.Accordion("Generated Spine JSON", open=False):
        json_output = gr.Code(language="json", lines=24)

    with gr.Accordion("RigPlan", open=False):
        rig_output = gr.Code(language="json", lines=18)

    generate_button.click(
        convert_psd,
        inputs=[
            psd_input,
            prompt_input,
            model_input,
            use_llm_input,
            atlas_width_input,
            max_tokens_input,
            temperature_input,
            top_p_input,
            enable_thinking_input,
            preserve_thinking_input,
        ],
        outputs=[
            status_output,
            preview_output,
            zip_output,
            files_output,
            json_output,
            rig_output,
        ],
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    demo.launch(**_launch_kwargs())
