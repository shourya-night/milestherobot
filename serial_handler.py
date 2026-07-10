"""Serial communication helpers for wheels and arm Arduinos."""

from __future__ import annotations

from typing import Optional

import serial

from config import SERIAL_ARM_PORT, SERIAL_BAUD, SERIAL_WHEELS_PORT

_VALID_MOVE = {"FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"}
_VALID_ARM = {"REACH", "RETRACT", "GRAB", "RELEASE", "POINT", "PUSH", "TAP", "NONE"}

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


def send_arm(command: str) -> None:
    cmd = command if command in _VALID_ARM else "NONE"
    _send(_arm, cmd, "arm")


def close_serial() -> None:
    for port in (_wheels, _arm):
        if port is not None:
            try:
                port.close()
            except Exception:
                pass
