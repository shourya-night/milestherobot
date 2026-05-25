"""Camera capture utilities for Miles."""

from __future__ import annotations

import base64
import threading
from typing import Optional

import cv2

from config import CAMERA_INDEX

_camera_lock = threading.Lock()
_camera: Optional[cv2.VideoCapture] = None


def _get_camera() -> cv2.VideoCapture:
    """Lazily create and return a shared camera handle."""
    global _camera
    with _camera_lock:
        if _camera is None:
            _camera = cv2.VideoCapture(CAMERA_INDEX)
        return _camera


def get_frame_base64() -> str:
    """Capture a frame, resize to 320x240, and return a base64 JPEG string."""
    camera = _get_camera()
    if not camera.isOpened():
        return ""

    ok, frame = camera.read()
    if not ok or frame is None:
        return ""

    resized = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
    ok, encoded = cv2.imencode(".jpg", resized)
    if not ok:
        return ""

    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def release_camera() -> None:
    """Release camera resources cleanly."""
    global _camera
    with _camera_lock:
        if _camera is not None:
            _camera.release()
            _camera = None
