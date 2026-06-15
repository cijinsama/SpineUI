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

Use `--no-llm` for deterministic PSD to setup-pose JSON output.

## PSD Layer Tags

Layer names can include lightweight tags:

- `[ignore]`: skip this layer.
- `[origin]`: use this layer's center as Spine `(0, 0)` and do not export it.
- `[bone]` / `[slot]`: kept in metadata for the LLM planner.

The compiler never lets the LLM invent coordinates or atlas entries. It only accepts a constrained RigPlan that references existing `layer_id` values.
