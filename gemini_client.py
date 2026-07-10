"""Two-stage Gemini (VLM -> VLA) inference wrapper for Miles, via Google AI Studio.

Stage 1 (perception): image + vlm_prompt.txt -> structured object detections
                       (label + normalized bounding box per object) + a short
                       scene description. Position (left/center/right) and
                       distance (near/far) are then computed in plain Python
                       from the bounding box, NOT guessed by the model — this
                       is more reliable and costs nothing extra.
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

SAFE_DEFAULT = {"move": "STOP", "arm": {"A": 0.0, "B": 0.0, "C": 0.0}, "mem": None}
SAFE_PERCEPTION = {"description": "", "objects": []}

# Per-cycle safety clamp: no matter what the model says, a single cycle can
# never command a servo to move more than this many degrees. Keeps the arm
# gentle (per GENTLENESS principle) even if the model outputs something wild.
_MAX_ARM_STEP_DEG = {"A": 20.0, "B": 20.0, "C": 30.0}

_VLM_PROMPT = Path(__file__).with_name("vlm_prompt.txt").read_text(encoding="utf-8")
_VLA_PROMPT = Path(__file__).with_name("vla_prompt.txt").read_text(encoding="utf-8")

_client: genai.Client | None = None

# Heuristic thresholds for turning a normalized box_2d into near/far. Area is
# out of 1,000,000 (1000x1000 normalized grid). Tune these once you see real
# detections from your camera's actual field of view / mounting distance.
_NEAR_AREA_THRESHOLD = 120_000
_FAR_AREA_THRESHOLD = 30_000


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


def _strip_json_fences(text: str) -> str:
    return text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()


def _position_from_box(box_2d: list[float]) -> tuple[str, str]:
    """Compute (horizontal, depth) labels from a normalized [ymin,xmin,ymax,xmax] box."""
    try:
        ymin, xmin, ymax, xmax = box_2d
    except (ValueError, TypeError):
        return "unknown", "unknown"

    center_x = (xmin + xmax) / 2
    if center_x < 333:
        horizontal = "left"
    elif center_x < 667:
        horizontal = "center"
    else:
        horizontal = "right"

    area = max(0.0, ymax - ymin) * max(0.0, xmax - xmin)
    if area >= _NEAR_AREA_THRESHOLD:
        depth = "near"
    elif area <= _FAR_AREA_THRESHOLD:
        depth = "far"
    else:
        depth = "mid-range"

    return horizontal, depth


def _normalize_perception(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(_strip_json_fences(text))
        if not isinstance(parsed, dict):
            return SAFE_PERCEPTION.copy()

        description = str(parsed.get("description", "")).strip()
        raw_objects = parsed.get("objects", [])
        if not isinstance(raw_objects, list):
            raw_objects = []

        objects: list[dict[str, Any]] = []
        for obj in raw_objects:
            if not isinstance(obj, dict):
                continue
            label = str(obj.get("label", "")).strip().lower()
            box_2d = obj.get("box_2d")
            if not label or not isinstance(box_2d, list) or len(box_2d) != 4:
                continue
            horizontal, depth = _position_from_box(box_2d)
            objects.append({
                "label": label,
                "box_2d": box_2d,
                "position": horizontal,
                "distance": depth,
            })

        return {"description": description, "objects": objects}
    except Exception:
        return SAFE_PERCEPTION.copy()


def _clamp(value: float, limit: float) -> float:
    if value > limit:
        return limit
    if value < -limit:
        return -limit
    return value


def _normalize_arm(raw_arm: Any) -> dict[str, float]:
    """Coerce whatever the model returned for 'arm' into a safe {A,B,C} degree-delta dict.

    Any missing/invalid/out-of-type servo value defaults to 0.0 (no movement), and every
    value is clamped to _MAX_ARM_STEP_DEG regardless of what the model asked for.
    """
    result = {"A": 0.0, "B": 0.0, "C": 0.0}
    if not isinstance(raw_arm, dict):
        return result

    for servo in ("A", "B", "C"):
        try:
            value = float(raw_arm.get(servo, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        result[servo] = _clamp(value, _MAX_ARM_STEP_DEG[servo])

    return result


def _normalize_action(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(_strip_json_fences(text))
        if not isinstance(parsed, dict):
            return SAFE_DEFAULT.copy()

        mem_value = parsed.get("mem")
        if mem_value is None or (isinstance(mem_value, str) and mem_value.strip().lower() in {"", "null", "none"}):
            normalized_mem = None
        else:
            normalized_mem = mem_value

        return {
            "move": parsed.get("move", "STOP"),
            "arm": _normalize_arm(parsed.get("arm")),
            "mem": normalized_mem,
        }
    except Exception:
        return SAFE_DEFAULT.copy()


def perceive_scene(frame_b64: str) -> dict[str, Any]:
    """Stage 1 (VLM): image in, {description, objects[{label,box_2d,position,distance}]} out.

    Deliberately does NOT take memory as input — detection should be grounded
    only in the current frame, not biased toward re-detecting things memory
    says were there before.
    """
    if not frame_b64:
        result = SAFE_PERCEPTION.copy()
        result["_raw"] = ""
        return result

    client = _get_client()
    image_bytes = base64.b64decode(frame_b64)

    try:
        response = client.models.generate_content(
            model=GEMINI_VLM_MODEL,
            contents=[
                _VLM_PROMPT,
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        raw_text = response.text or ""
        result = _normalize_perception(raw_text)
        result["_raw"] = raw_text
        return result
    except Exception as exc:
        result = SAFE_PERCEPTION.copy()
        result["_raw"] = json.dumps({"error": str(exc)})
        return result


def format_scene_for_vla(perception: dict[str, Any], novel_labels: set[str] | None = None) -> str:
    """Turn perceive_scene()'s structured output into the {S} text vla_prompt.txt expects."""
    novel_labels = novel_labels or set()
    description = perception.get("description", "")
    objects = perception.get("objects", [])

    if not objects:
        object_lines = "No distinct objects detected."
    else:
        parts = []
        for obj in objects:
            tag = "NEW" if obj["label"] in novel_labels else "known"
            parts.append(f"{obj['label']} ({obj['position']}, {obj['distance']}, {tag})")
        object_lines = "Objects: " + "; ".join(parts)

    return f"{description}\n{object_lines}"


def decide_action(scene_text: str, memory_str: str) -> dict[str, Any]:
    """Stage 2 (VLA): text-only in, move/arm/mem JSON out."""
    client = _get_client()
    prompt_text = (
        _VLA_PROMPT.replace("{S}", scene_text or "")
        .replace("{M}", memory_str or "(none yet)")
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_VLA_MODEL,
            contents=prompt_text,
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
        raw_text = response.text or ""
        result = _normalize_action(raw_text)
        result["_raw"] = raw_text
        return result
    except Exception as exc:
        fallback = SAFE_DEFAULT.copy()
        fallback["_raw"] = json.dumps({"error": str(exc)})
        return fallback


def run_inference(frame_b64: str, memory_str: str, novel_labels: set[str] | None = None) -> dict[str, Any]:
    """Full two-stage cycle: VLM detects, VLA decides."""
    perception = perceive_scene(frame_b64)
    scene_text = format_scene_for_vla(perception, novel_labels)
    result = decide_action(scene_text, memory_str)
    result["_perception"] = perception
    result["_scene_text"] = scene_text
    return result
