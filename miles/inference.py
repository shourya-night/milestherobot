"""Ollama inference wrapper with prompt injection and safe fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from config import OLLAMA_MODEL, OLLAMA_URL

SAFE_DEFAULT = {"move": "STOP", "arm": "NONE", "say": "...", "mem": None}
_PROMPT_TEMPLATE = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")


def _normalize_response(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return SAFE_DEFAULT.copy()
        return {
            "move": parsed.get("move", "STOP"),
            "arm": parsed.get("arm", "NONE"),
            "say": parsed.get("say", "..."),
            "mem": parsed.get("mem"),
        }
    except Exception:
        return SAFE_DEFAULT.copy()


def run_inference(frame_b64: str, memory_str: str, human_speech: str | None) -> dict[str, Any]:
    system_prompt = _PROMPT_TEMPLATE.replace("{M}", memory_str or "")
    system_prompt = system_prompt.replace("{H}", human_speech or "")

    payload = {
        "model": OLLAMA_MODEL,
        "system": system_prompt,
        "prompt": "Analyze the scene and respond with JSON only.",
        "images": [frame_b64] if frame_b64 else [],
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=2)
        response.raise_for_status()
        data = response.json()
        raw_text = data.get("response", "")
        result = _normalize_response(raw_text)
        result["_raw"] = raw_text
        return result
    except Exception:
        fallback = SAFE_DEFAULT.copy()
        fallback["_raw"] = json.dumps(fallback)
        return fallback
