"""Serial communication helpers for wheels and arm Arduinos."""

from __future__ import annotations

from typing import Optional

import serial

from config import SERIAL_ARM_PORT, SERIAL_BAUD, SERIAL_WHEELS_PORT

_VALID_MOVE = {"FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"}

_wheels: Optional[serial.Serial] = None
_arm: Optional[serial.Serial] = None


def init_serial() -> tuple[bool, bool]:
    global _wheels, _arm

    wheels_ok = False
    arm_ok = False

    try:
        _wheels = serial.Serial(SERIAL_WHEELS_PORT, SERIAL_BAUD, timeout=0.2)
        wheels_ok = True
    except Exception as exc:
        print(f"[serial] Warning: wheels port unavailable ({exc})")
        _wheels = None

    try:
        _arm = serial.Serial(SERIAL_ARM_PORT, SERIAL_BAUD, timeout=0.2)
        arm_ok = True
    except Exception as exc:
        print(f"[serial] Warning: arm port unavailable ({exc})")
        _arm = None

    return wheels_ok, arm_ok


def _send(port: Optional[serial.Serial], command: str, label: str) -> None:
    if port is None:
        return
    try:
        port.write(f"{command}\n".encode("ascii", errors="ignore"))
    except Exception as exc:
        print(f"[serial] Warning: failed to send {label} command ({exc})")


def send_move(command: str) -> None:
    cmd = command if command in _VALID_MOVE else "STOP"
    _send(_wheels, cmd, "move")


def send_arm(deltas: dict) -> None:
    """Send per-servo degree deltas, e.g. {"A": 10.0, "B": -5.0, "C": 0.0}.

    Wire format is a compact "A:<deg>,B:<deg>,C:<deg>\n" line — cheap to parse on an
    Arduino with strtok()/String.indexOf() without needing a JSON library on-device.
    Any missing/invalid servo value is sent as 0.0 (no movement) rather than dropped,
    so the receiving firmware always gets all three fields.
    """
    if not isinstance(deltas, dict):
        deltas = {}
    try:
        a = float(deltas.get("A", 0.0))
        b = float(deltas.get("B", 0.0))
        c = float(deltas.get("C", 0.0))
    except (TypeError, ValueError):
        a = b = c = 0.0
    cmd = f"A:{a:.1f},B:{b:.1f},C:{c:.1f}"
    _send(_arm, cmd, "arm")


def close_serial() -> None:
    for port in (_wheels, _arm):
        if port is not None:
            try:
                port.close()
            except Exception:
                pass
