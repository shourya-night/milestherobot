"""Standalone local test of the camera -> Gemini (VLM -> VLA) pipeline.

Run this on the Desktop Station with a USB webcam plugged in, BEFORE wiring
anything into ROS. It reuses vision.py's capture loop untouched, so if this
works, the camera half of the pipeline is proven and the only remaining step
is swapping main.py's image source from cv2.VideoCapture to the /camera/image_raw
subscriber.

Usage:
    export GEMINI_API_KEY="your-key-here"
    python test_camera_pipeline.py
"""

from __future__ import annotations

import base64
import sys
import time

import vision
from gemini_client import decide_action, describe_scene


def main() -> None:
    print("Starting camera...")
    if not vision.start_camera():
        print("ERROR: could not open camera at CAMERA_INDEX (see config.py). "
              "Check that a webcam is connected and not in use by another app.")
        sys.exit(1)

    # Give the capture thread a moment to grab its first frame.
    time.sleep(1.0)

    frame_b64 = vision.get_frame_base64()
    if not frame_b64:
        print("ERROR: camera opened but no frame was captured yet. Try again in a second.")
        vision.release_camera()
        sys.exit(1)

    # Save the exact frame being sent to Gemini, so you can visually confirm
    # framing/exposure/focus before trusting the model's description of it.
    with open("last_frame.jpg", "wb") as f:
        f.write(base64.b64decode(frame_b64))
    print("Saved last_frame.jpg — open it and check it looks like what you expect.")

    print("\n--- Stage 1: VLM (perception) ---")
    scene = describe_scene(frame_b64, memory_str="(no memory yet, first run)")
    print(f"Scene description: {scene}")

    print("\n--- Stage 2: VLA (decision) ---")
    decision = decide_action(scene, memory_str="(no memory yet, first run)")
    print(f"move: {decision['move']}")
    print(f"arm:  {decision['arm']}")
    print(f"mem:  {decision['mem']}")
    print(f"\nraw model output: {decision.get('_raw')}")

    vision.release_camera()
    print("\nDone. Camera released.")


if __name__ == "__main__":
    main()
