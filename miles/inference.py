"""Ollama inference wrapper with prompt injection and safe fallback."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from config import OLLAMA_MODEL, OLLAMA_URL

SAFE_DEFAULT = {"move": "STOP", "arm": "NONE", "say": "...", "mem": None}
_PROMPT_TEMPLATE = Path(__file__).with_name("prompt.txt").read_text(encoding="utf-8")
REQUEST_TIMEOUT_SECONDS = 10


def _normalize_response(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, dict):
            return SAFE_DEFAULT.copy()
        mem_value = parsed.get("mem")
        if mem_value is None:
            normalized_mem = None
        elif isinstance(mem_value, str) and mem_value.strip().lower() in {"", "null", "none"}:
            normalized_mem = None
        else:
            normalized_mem = mem_value

        return {
            "move": parsed.get("move", "STOP"),
            "arm": parsed.get("arm", "NONE"),
            "say": parsed.get("say", "..."),
            "mem": normalized_mem,
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
        response = requests.post(OLLAMA_URL, json=payload, timeout=REQUEST_TIMEOUT_SECONDS)
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
