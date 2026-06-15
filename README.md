# PSD to Spine Gradio MVP

This app converts a PSD/PSB into layer PNGs, a Spine-compatible `.atlas`, a setup-pose Spine JSON file, and a downloadable zip. An optional LLM step uses the NRP OpenAI-compatible API to create a constrained rig plan from the PSD layer metadata and the user prompt.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Put your NRP token in `.env`:

```bash
NRP_API_KEY=...
NRP_BASE_URL=https://ellm.nrp-nautilus.io/v1
NRP_MODEL=qwen3
NRP_MAX_TOKENS=16000
NRP_TEMPERATURE=0.2
NRP_TOP_P=0.95
NRP_ENABLE_THINKING=true
NRP_PRESERVE_THINKING=false
NRP_TIMEOUT_SECONDS=180
```

`.env` is gitignored.

## Run

```bash
python app.py
```

Open the Gradio URL printed in the terminal, upload a PSD/PSB, enter a prompt, and click generate. The generated zip includes:

- `images/*.png`
- `<name>.png`
- `<name>.atlas`
- `<name>.json`
- `<name>.ir.json`
- `<name>.rigplan.json`
- `preview.html`

The embedded preview uses Spine Web Player from a CDN, so previewing needs browser network access.

## CLI

```bash
python -m spinegen.cli character.psd \
  --prompt "Q版骑士，生成 idle/walk/attack 的基础 RigPlan" \
  --model qwen3 \
  --max-tokens 16000
```

Use `--no-llm` for an offline fallback setup-pose JSON output.

## PSD Layer Tags

Layer names can include lightweight tags:

- `[ignore]`: skip this layer.
- `[origin]`: use this layer's center as Spine `(0, 0)` and do not export it.
- `[bone]` / `[slot]`: kept in metadata for the LLM planner.

The compiler never lets the LLM invent coordinates or atlas entries. It only accepts a constrained RigPlan that references existing `layer_id` values.

## Pose Groups

If a PSD contains multiple top-level groups that look like mutually exclusive full-character poses or variants, the app detects candidate groups before atlas packing. When LLM mode is enabled, the LLM chooses among those detected groups from the user prompt; code only validates the returned group id before using it. If the LLM is disabled or unavailable, the app uses a conservative fallback group choice.

This prevents multiple complete poses from being displayed at the same time while avoiding hardcoded project-specific group names.

## LLM Visibility Planning

After the active group is selected, the app asks the configured LLM backend to produce a visibility plan for the active layers. The model can hide mutually exclusive layers, composite/component duplicates, optional equipment states, and expression variants according to the user prompt. The compiler only applies layer ids that exist in the active layer set, and it rejects plans that would hide every layer.

The Gradio `User prompt` field and the CLI `--prompt` flag are only for the user's request, for example `这是一个 Q 版角色，生成 attack 的基础骨骼和动画。`. The hidden default quality prompt is added in backend code and is not shown in the app.

Prompt routing is split by LLM task:

- Setup group choice: `SETUP_GROUP_SYSTEM_PROMPT` plus the user request, hidden quality prompt, and candidate group summaries.
- Layer visibility planning: `LAYER_VISIBILITY_SYSTEM_PROMPT` plus the user request, hidden quality prompt, active layer metadata, and unordered overlap hints.
- Rig and animation generation: `RIG_SYSTEM_PROMPT` plus the user request, hidden quality prompt, canvas metadata, active layers, and RigPlan schema constraints.

User text is treated as the explicit request and can override the hidden quality defaults when it asks for a specific state.
