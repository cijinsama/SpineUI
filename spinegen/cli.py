from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from spinegen.config import LLMSettings
from spinegen.pipeline import run_conversion


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Convert PSD/PSB layers to a Spine JSON + atlas bundle.")
    parser.add_argument("psd", type=Path, help="Input .psd or .psb file")
    parser.add_argument("--prompt", default="", help="Prompt for LLM RigPlan generation")
    parser.add_argument("--model", default=None, help="NRP model id")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM and use deterministic fallback rig")
    parser.add_argument("--atlas-width", type=int, default=2048, help="Maximum atlas shelf width")
    parser.add_argument("--max-tokens", type=int, default=None, help="LLM max_tokens")
    parser.add_argument("--temperature", type=float, default=None, help="LLM temperature")
    parser.add_argument("--top-p", type=float, default=None, help="LLM top_p")
    parser.add_argument("--no-thinking", action="store_true", help="Disable NRP chat_template thinking flags")
    args = parser.parse_args()

    settings = LLMSettings.from_env(model=args.model)
    if args.max_tokens is not None or args.temperature is not None or args.top_p is not None or args.no_thinking:
        settings = LLMSettings(
            model=settings.model,
            max_tokens=args.max_tokens if args.max_tokens is not None else settings.max_tokens,
            temperature=args.temperature if args.temperature is not None else settings.temperature,
            top_p=args.top_p if args.top_p is not None else settings.top_p,
            enable_thinking=not args.no_thinking,
            preserve_thinking=settings.preserve_thinking,
            timeout_seconds=settings.timeout_seconds,
        )

    result = run_conversion(
        psd_path=args.psd,
        prompt=args.prompt,
        llm_settings=settings,
        use_llm=not args.no_llm,
        atlas_width=args.atlas_width,
    )
    print("\n".join(result.messages))
    print(result.zip_path.resolve())


if __name__ == "__main__":
    main()

