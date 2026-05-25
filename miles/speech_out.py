"""Offline text-to-speech helper."""

from __future__ import annotations

import threading
from typing import Optional

import pyttsx3

from config import TTS_RATE, TTS_VOLUME

_engine: Optional[pyttsx3.Engine] = None
_engine_lock = threading.Lock()
_tts_ok = False


try:
    _engine = pyttsx3.init()
    _engine.setProperty("rate", TTS_RATE)
    _engine.setProperty("volume", TTS_VOLUME)

    voices = _engine.getProperty("voices") or []
    preferred = next((v for v in voices if "natural" in v.name.lower()), None)
    if preferred:
        _engine.setProperty("voice", preferred.id)

    _tts_ok = True
except Exception as exc:
    print(f"[speech_out] Warning: TTS unavailable ({exc})")


def _do_speak(text: str) -> None:
    if not _tts_ok or _engine is None or not text:
        return

    with _engine_lock:
        try:
            _engine.say(text)
            _engine.runAndWait()
        except Exception as exc:
            print(f"[speech_out] Warning: speak failed ({exc})")


def speak(text: str) -> None:
    """Speak text in a non-blocking background thread."""
    if not _tts_ok:
        return

    threading.Thread(target=_do_speak, args=(text,), daemon=True).start()


def tts_status() -> bool:
    return _tts_ok
