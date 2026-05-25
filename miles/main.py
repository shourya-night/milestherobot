"""Main runtime loop for the Miles embodied AI pipeline."""

from __future__ import annotations

import time

from config import CAMERA_INDEX, DB_PATH, FPS, MEMORY_MAX_ENTRIES, OLLAMA_MODEL
from inference import run_inference
from logger import init_logger, log_cycle
from memory import add_memory, close_db, count_memories, get_recent_memory, init_db
from serial_handler import close_serial, init_serial, send_arm, send_move
from speech_in import get_latest_speech, microphone_status, start_listening, stop_listening
from speech_out import shutdown_tts, speak, tts_status
from vision import get_frame_base64, release_camera


def _startup_summary(wheels_ok: bool, arm_ok: bool, mic_ok: bool) -> None:
    print("\n=== Miles Startup Summary ===")
    print(f"Camera index : {CAMERA_INDEX}")
    print(f"Mic status   : {'OK' if mic_ok else 'UNAVAILABLE'}")
    print(f"TTS status   : {'OK' if tts_status() else 'UNAVAILABLE'}")
    print(f"Wheels port  : {'OK' if wheels_ok else 'DRY-RUN'}")
    print(f"Arm port     : {'OK' if arm_ok else 'DRY-RUN'}")
    print(f"Ollama model : {OLLAMA_MODEL}")
    print(f"DB path      : {DB_PATH}")
    print("=============================\n")


def main() -> None:
    init_db()
    init_logger()
    wheels_ok, arm_ok = init_serial()
    mic_ok = start_listening()
    _startup_summary(wheels_ok, arm_ok, mic_ok and microphone_status())

    cycle = 0
    loop_interval = 1.0 / FPS

    try:
        while True:
            cycle_start = time.time()
            cycle += 1

            frame_b64 = get_frame_base64()
            human_speech = get_latest_speech()
            memory_str = get_recent_memory(MEMORY_MAX_ENTRIES)

            result = run_inference(frame_b64, memory_str, human_speech)
            move = str(result.get("move", "STOP"))
            arm = str(result.get("arm", "NONE"))
            say = str(result.get("say", "..."))
            mem = result.get("mem")
            raw = str(result.get("_raw", ""))

            send_move(move)
            send_arm(arm)
            speak(say)

            if human_speech:
                print(f"[Human] {human_speech}")
            print(f"[Miles] {say}")

            if mem is not None:
                add_memory(str(mem))

            log_cycle(cycle, human_speech, move, arm, say, mem, raw)

            elapsed = time.time() - cycle_start
            time.sleep(max(0, loop_interval - elapsed))
    except KeyboardInterrupt:
        print("\nShutting down Miles...")
    finally:
        stop_listening()
        shutdown_tts()
        close_serial()
        release_camera()
        print(f"Saved memories: {count_memories()}")
        close_db()


if __name__ == "__main__":
    main()
