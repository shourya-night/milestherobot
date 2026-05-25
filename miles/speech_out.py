"""Offline text-to-speech helper."""

from __future__ import annotations

import threading
import queue
from typing import Optional

import pyttsx3

from config import TTS_RATE, TTS_VOLUME

_engine: Optional[pyttsx3.Engine] = None
_tts_ok = False
_speech_queue: "queue.Queue[str]" = queue.Queue(maxsize=100)
_stop_event = threading.Event()
_worker_thread: Optional[threading.Thread] = None


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


def _speech_worker() -> None:
    """Single-threaded engine worker; pyttsx3 is accessed only here."""
    while not _stop_event.is_set():
        try:
            text = _speech_queue.get(timeout=0.25)
        except queue.Empty:
            continue

        if not text:
            _speech_queue.task_done()
            continue

        try:
            if _engine is not None:
                _engine.say(text)
                _engine.runAndWait()
        except Exception as exc:
            print(f"[speech_out] Warning: speak failed ({exc})")
        finally:
            _speech_queue.task_done()


def _ensure_worker() -> None:
    global _worker_thread
    if not _tts_ok:
        return
    if _worker_thread is None or not _worker_thread.is_alive():
        _worker_thread = threading.Thread(target=_speech_worker, daemon=True)
        _worker_thread.start()


def speak(text: str) -> None:
    """Speak text in a non-blocking background thread."""
    if not _tts_ok:
        return
    _ensure_worker()
    try:
        _speech_queue.put_nowait(text)
    except queue.Full:
        # Drop oldest requests under pressure to protect control-loop latency.
        try:
            _speech_queue.get_nowait()
            _speech_queue.task_done()
        except queue.Empty:
            pass
        try:
            _speech_queue.put_nowait(text)
        except queue.Full:
            pass


def tts_status() -> bool:
    return _tts_ok


def shutdown_tts() -> None:
    """Graceful shutdown for speech worker thread."""
    _stop_event.set()
    if _worker_thread and _worker_thread.is_alive():
        _worker_thread.join(timeout=1.0)
