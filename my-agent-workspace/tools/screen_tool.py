"""
tools/screen_tool.py
────────────────────
Skill 5 — Screenshot → text description via Anthropic vision model.

Install: pip install anthropic Pillow
"""

import base64
import os
import subprocess
import uuid
from datetime import datetime

try:
    import anthropic as _anthropic_mod
except ImportError:
    _anthropic_mod = None
try:
    from upsonic.tools import tool
except ImportError:
    def tool(fn): return fn

from config import ANTHROPIC_API_KEY, VISION_MODEL


def _capture_screenshot(save_path: str | None = None) -> str | None:
    """Capture screen via macOS screencapture. Returns PNG path or None."""
    if save_path is None:
        save_path = f"/tmp/_hapando_screen_{uuid.uuid4().hex[:8]}.png"
    try:
        result = subprocess.run(
            ["screencapture", "-x", "-t", "png", save_path],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0 or not os.path.exists(save_path):
            return None
        return save_path
    except Exception as e:
        print(f"[screen] Capture error: {e}")
        return None


def _image_to_base64(path: str) -> tuple[str, str]:
    """Read image file → (base64_string, media_type)."""
    with open(path, "rb") as f:
        raw = f.read()
    b64 = base64.standard_b64encode(raw).decode("utf-8")
    media_type = "image/png" if raw[:4] == b"\x89PNG" else "image/jpeg"
    return b64, media_type


def read_screen(
    question: str = "Describe everything you see on screen in detail.",
    save_screenshot: bool = False,
) -> str:
    """Capture screen and ask vision model a question about it."""
    tmp_path    = f"/tmp/_hapando_screen_{uuid.uuid4().hex[:8]}.png"
    screen_path = _capture_screenshot(tmp_path)
    if screen_path is None:
        return "[screen_tool] Failed to capture screenshot."
    try:
        b64_data, media_type = _image_to_base64(screen_path)
        client   = _anthropic_mod.Anthropic(api_key=ANTHROPIC_API_KEY)
        response = client.messages.create(
            model=VISION_MODEL,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64", "media_type": media_type, "data": b64_data,
                    }},
                    {"type": "text", "text": (
                        "You are Hapando's vision system. "
                        "Hao is playing a video game. "
                        "Look at this screenshot carefully and answer:\n\n"
                        f"{question}\n\n"
                        "Be specific. Mention health/mana bars, enemy names and positions, "
                        "buffs/debuffs, quest text, minimap, cooldowns, and any warnings."
                    )},
                ],
            }],
        )
        description = response.content[0].text.strip()
        if save_screenshot:
            from config import MEMORY_DIR
            screenshots_dir = os.path.join(MEMORY_DIR, "screenshots")
            os.makedirs(screenshots_dir, exist_ok=True)
            timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_path = os.path.join(screenshots_dir, f"screen_{timestamp}.png")
            os.rename(screen_path, saved_path)
            screen_path = None
        return description
    except Exception as e:
        return f"[screen_tool] Vision API error: {e}"
    finally:
        if screen_path and os.path.exists(screen_path):
            os.remove(screen_path)


@tool
def screen_tool(
    question: str = "Describe everything you see on screen in detail.",
    save_screenshot: bool = False,
) -> str:
    """
    Take a screenshot and describe what the vision model sees.
    Use when Hao wants Hapando to observe the current game state.

    Args:
        question:        What to ask about the screen.
        save_screenshot: True = save PNG to memory/screenshots/.
    Returns:
        Plain-English description of the screen.
    """
    return read_screen(question=question, save_screenshot=save_screenshot)
