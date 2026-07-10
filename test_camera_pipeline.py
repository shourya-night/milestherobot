"""Camera capture utilities for Miles.

Runs camera polling in a background thread so the newest frame is always available
without blocking the real-time control loop.
"""

from __future__ import annotations

import base64
import threading
import time
from typing import Optional

import cv2

from config import CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH

_camera: Optional[cv2.VideoCapture] = None
_camera_lock = threading.Lock()
_frame_lock = threading.Lock()
_latest_frame = None
_stop_event = threading.Event()
_thread: Optional[threading.Thread] = None


def _capture_loop() -> None:
    """Read frames continuously to prevent camera buffer backlog."""
    while not _stop_event.is_set():
        with _camera_lock:
            camera = _camera
        if camera is None or not camera.isOpened():
            time.sleep(0.05)
            continue

        ok, frame = camera.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        with _frame_lock:
            global _latest_frame
            _latest_frame = frame


def start_camera() -> bool:
    """Initialize camera and start background capture thread."""
    global _camera, _thread
    with _camera_lock:
        if _camera is None:
            _camera = cv2.VideoCapture(CAMERA_INDEX)

        if _camera is None or not _camera.isOpened():
            return False

    _stop_event.clear()
    if _thread is None or not _thread.is_alive():
        _thread = threading.Thread(target=_capture_loop, daemon=True)
        _thread.start()
    return True


def get_frame_base64() -> str:
    """Return the latest captured frame as base64 JPEG without blocking capture."""
    with _frame_lock:
        frame = None if _latest_frame is None else _latest_frame.copy()

    if frame is None:
        return ""

    resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", resized)
    if not ok:
        return ""
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def release_camera() -> None:
    """Stop background capture and release hardware resources."""
    global _camera
    _stop_event.set()
    if _thread is not None and _thread.is_alive():
        _thread.join(timeout=1.0)

    with _camera_lock:
        if _camera is not None:
            _camera.release()
            _camera = None
