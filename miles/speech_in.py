"""Background speech capture and offline transcription."""

from __future__ import annotations

import tempfile
import threading
import os
from typing import Optional

import speech_recognition as sr
import whisper

from config import SPEECH_ENERGY_THRESHOLD, STT_MODEL

_latest_speech: Optional[str] = None
_speech_lock = threading.Lock()
_stop_event = threading.Event()
_thread: Optional[threading.Thread] = None
_whisper_model = None
_mic_available = False


def _set_latest_speech(text: str) -> None:
    global _latest_speech
    with _speech_lock:
        _latest_speech = text


def _get_model():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = whisper.load_model(STT_MODEL)
    return _whisper_model


def preload_model() -> bool:
    """Preload Whisper model during startup to avoid first-use runtime stalls."""
    try:
        _get_model()
        return True
    except Exception as exc:
        print(f"[speech_in] Warning: Whisper model failed to load ({exc})")
        return False


def _listen_loop() -> None:
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = SPEECH_ENERGY_THRESHOLD
    recognizer.dynamic_energy_threshold = True

    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            while not _stop_event.is_set():
                try:
                    audio = recognizer.listen(source, timeout=1.0, phrase_time_limit=6.0)
                    temp_path = None
                    try:
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            tmp.write(audio.get_wav_data())
                            tmp.flush()
                            temp_path = tmp.name

                        text = _get_model().transcribe(temp_path).get("text", "").strip()
                        if text:
                            _set_latest_speech(text)
                    finally:
                        if temp_path and os.path.exists(temp_path):
                            try:
                                os.remove(temp_path)
                            except Exception:
                                pass
                except sr.WaitTimeoutError:
                    continue
                except Exception as exc:  # Keep loop alive despite transcription/mic hiccups.
                    print(f"[speech_in] Warning: {exc}")
    except Exception as exc:
        print(f"[speech_in] Microphone unavailable: {exc}")


def start_listening() -> bool:
    """Start background listener thread. Returns True if mic appears available."""
    global _thread, _mic_available
    _stop_event.clear()

    model_ok = preload_model()
    if not model_ok:
        _mic_available = False
        return False

    try:
        with sr.Microphone():
            _mic_available = True
    except Exception:
        _mic_available = False
        return False

    if _thread is None or not _thread.is_alive():
        _thread = threading.Thread(target=_listen_loop, daemon=True)
        _thread.start()

    return True


def get_latest_speech() -> Optional[str]:
    """Return the latest transcript and clear it. Returns None if no new speech exists."""
    global _latest_speech
    with _speech_lock:
        value = _latest_speech
        _latest_speech = None
    return value


def stop_listening() -> None:
    """Signal listener thread to stop."""
    _stop_event.set()


def microphone_status() -> bool:
    """Expose current microphone availability for startup summaries."""
    return _mic_available
