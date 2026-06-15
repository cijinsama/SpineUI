from __future__ import annotations

import base64
import html
import json
from pathlib import Path


SPINE_PLAYER_CSS = "https://unpkg.com/@esotericsoftware/spine-player@4.2.*/dist/spine-player.css"
SPINE_PLAYER_JS = "https://unpkg.com/@esotericsoftware/spine-player@4.2.*/dist/iife/spine-player.js"


def write_preview_html(
    output_path: Path,
    skeleton_name: str,
    json_path: Path,
    atlas_path: Path,
    image_path: Path,
    animation_name: str | None = "idle",
) -> str:
    json_uri = _data_uri("application/json", json_path.read_bytes())
    atlas_uri = _data_uri("text/plain", atlas_path.read_bytes())
    image_uri = _data_uri("image/png", image_path.read_bytes())
    raw_data = {
        json_path.name: json_uri,
        atlas_path.name: atlas_uri,
        image_path.name: image_uri,
    }
    config = {
        "jsonUrl": json_path.name,
        "atlasUrl": atlas_path.name,
        "rawDataURIs": raw_data,
        "showControls": True,
        "alpha": True,
        "backgroundColor": "#f7f7f2",
        "viewport": {"debugRender": False},
    }
    if animation_name:
        config["animation"] = animation_name

    document = _preview_document(skeleton_name, config)
    output_path.write_text(document, encoding="utf-8")
    return document


def iframe_for_preview(preview_document: str, height: int = 560) -> str:
    escaped = html.escape(preview_document, quote=True)
    return (
        f'<iframe title="Spine preview" srcdoc="{escaped}" '
        f'width="100%" height="{height}" '
        'style="border:1px solid #d9d9d0;border-radius:8px;background:#f7f7f2;"></iframe>'
    )


def _preview_document(skeleton_name: str, config: dict[str, object]) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(skeleton_name)} preview</title>
  <link rel="stylesheet" href="{SPINE_PLAYER_CSS}">
  <style>
    html, body {{
      margin: 0;
      width: 100%;
      height: 100%;
      background: #f7f7f2;
      overflow: hidden;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    #player {{
      width: 100vw;
      height: 100vh;
    }}
    .fallback {{
      box-sizing: border-box;
      padding: 16px;
      color: #242421;
      font-size: 14px;
      line-height: 1.45;
    }}
  </style>
</head>
<body>
  <div id="player"></div>
  <script src="{SPINE_PLAYER_JS}"></script>
  <script>
    const config = {json.dumps(config)};
    window.addEventListener("load", () => {{
      if (!window.spine || !window.spine.SpinePlayer) {{
        document.getElementById("player").innerHTML =
          '<div class="fallback">Spine Player CDN 没有加载成功。可以下载 preview.html 后在能访问网络的浏览器中打开。</div>';
        return;
      }}
      new spine.SpinePlayer("player", config);
    }});
  </script>
</body>
</html>
"""


def _data_uri(mime_type: str, data: bytes) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"

