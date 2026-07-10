"""Two-stage Gemini (VLM -> VLA) inference wrapper for Miles, via Google AI Studio.

Stage 1 (perception): image + vlm_prompt.txt -> plain-text scene description.
Stage 2 (decision):   text-only + vla_prompt.txt -> move/arm/mem JSON.

Pure autonomous exploration: no human-interaction I/O (no speech in/out). The
robot only observes, decides, and remembers.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_VLA_MODEL, GEMINI_VLM_MODEL

SAFE_DEFAULT = {"move": "STOP", "arm": "NONE", "mem": None}

_VLM_PROMPT = Path(__file__).with_name("vlm_prompt.txt").read_text(encoding="utf-8")
_VLA_PROMPT = Path(__file__).with_name("vla_prompt.txt").read_text(encoding="utf-8")

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Run `export GEMINI_API_KEY=...` "
                "(get a key at https://aistudio.google.com) before starting Miles."
            )
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _normalize_response(text: str) -> dict[str, Any]:
    try:
        # Gemini sometimes wraps JSON in ```json fences even when told not to.
        cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            return SAFE_DEFAULT.copy()

        mem_value = parsed.get("mem")
        if mem_value is None or (isinstance(mem_value, str) and mem_value.strip().lower() in {"", "null", "none"}):
            normalized_mem = None
        else:
            normalized_mem = mem_value

        return {
            "move": parsed.get("move", "STOP"),
            "arm": parsed.get("arm", "NONE"),
            "mem": normalized_mem,
        }
    except Exception:
        return SAFE_DEFAULT.copy()


def describe_scene(frame_b64: str, memory_str: str) -> str:
    """Stage 1 (VLM): image in, one short plain-text description out."""
    if not frame_b64:
        return "No camera frame available."

    client = _get_client()
    prompt_text = _VLM_PROMPT.replace("{M}", memory_str or "(none yet)")
    image_bytes = base64.b64decode(frame_b64)

    response = client.models.generate_content(
        model=GEMINI_VLM_MODEL,
        contents=[
            prompt_text,
            types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
        ],
    )
    return (response.text or "").strip()


def decide_action(scene_description: str, memory_str: str) -> dict[str, Any]:
    """Stage 2 (VLA): text-only in, move/arm/mem JSON out."""
    client = _get_client()
    prompt_text = (
        _VLA_PROMPT.replace("{S}", scene_description or "")
        .replace("{M}", memory_str or "(none yet)")
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_VLA_MODEL,
            contents=prompt_text,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        raw_text = response.text or ""
        result = _normalize_response(raw_text)
        result["_raw"] = raw_text
        return result
    except Exception as exc:
        fallback = SAFE_DEFAULT.copy()
        fallback["_raw"] = json.dumps({"error": str(exc)})
        return fallback


def run_inference(frame_b64: str, memory_str: str) -> dict[str, Any]:
    """Full two-stage cycle: VLM describes, VLA decides. Matches inference.py's shape."""
    scene = describe_scene(frame_b64, memory_str)
    result = decide_action(scene, memory_str)
    result["_scene"] = scene
    return result
