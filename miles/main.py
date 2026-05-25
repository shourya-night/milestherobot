"""Main runtime loop for the Miles embodied AI pipeline."""

from __future__ import annotations

import time
import threading
import queue
import json

from config import ACTION_STALE_TIMEOUT_SEC, CAMERA_INDEX, DB_PATH, FPS, MEMORY_MAX_ENTRIES, OLLAMA_MODEL
from inference import run_inference
from logger import init_logger, log_cycle
from memory import add_memory, close_db, count_memories, get_recent_memory, init_db
from serial_handler import close_serial, init_serial, send_arm, send_move
from speech_in import get_latest_speech, microphone_status, start_listening, stop_listening
from speech_out import shutdown_tts, speak, tts_status
from vision import get_frame_base64, release_camera, start_camera


class _InferenceManager:
    """Runs VLM inference off the control thread and stores latest result safely."""

    def __init__(self) -> None:
        self._task_queue: "queue.Queue[tuple[str, str, str | None]]" = queue.Queue(maxsize=1)
        self._result_lock = threading.Lock()
        self._latest_result = {"move": "STOP", "arm": "NONE", "say": "...", "mem": None, "_raw": ""}
        self._latest_result_time = 0.0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def _worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame_b64, memory_str, human_speech = self._task_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                result = run_inference(frame_b64, memory_str, human_speech)
                with self._result_lock:
                    self._latest_result = result
                    self._latest_result_time = time.time()
            finally:
                self._task_queue.task_done()

    def submit(self, frame_b64: str, memory_str: str, human_speech: str | None) -> bool:
        """Submit newest state snapshot; drop stale queued work if needed."""
        if self._stop_event.is_set():
            return False
        try:
            self._task_queue.put_nowait((frame_b64, memory_str, human_speech))
            return True
        except queue.Full:
            # Replace queued stale job with the latest snapshot.
            try:
                self._task_queue.get_nowait()
                self._task_queue.task_done()
            except queue.Empty:
                pass
            try:
                self._task_queue.put_nowait((frame_b64, memory_str, human_speech))
                return True
            except queue.Full:
                return False

    def get_latest(self) -> tuple[dict, float]:
        with self._result_lock:
            return dict(self._latest_result), self._latest_result_time

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)


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
    camera_ok = start_camera()
    _startup_summary(wheels_ok, arm_ok, mic_ok and microphone_status())
    if not camera_ok:
        print("[vision] Warning: camera unavailable; inference will receive empty frames.")

    inference_mgr = _InferenceManager()
    inference_mgr.start()

    cycle = 0
    loop_interval = 1.0 / FPS
    last_say = "..."
    last_move = "STOP"
    last_arm = "NONE"
    last_inference_submit = 0.0

    try:
        while True:
            cycle_start = time.time()
            cycle += 1

            frame_b64 = get_frame_base64()
            human_speech = get_latest_speech()
            # Keep inference asynchronous; submit snapshots at most once per control cycle.
            if time.time() - last_inference_submit >= loop_interval:
                memory_str = get_recent_memory(MEMORY_MAX_ENTRIES)
                inference_mgr.submit(frame_b64, memory_str, human_speech)
                last_inference_submit = time.time()

            result, result_time = inference_mgr.get_latest()
            move = str(result.get("move", "STOP"))
            arm = str(result.get("arm", "NONE"))
            say = str(result.get("say", "..."))
            mem = result.get("mem")
            raw = str(result.get("_raw", json.dumps(result)))

            # Safety fallback: stop if the action command has gone stale.
            if result_time <= 0 or (time.time() - result_time) > ACTION_STALE_TIMEOUT_SEC:
                move = "STOP"
                arm = "NONE"
                say = "..."

            send_move(move)
            send_arm(arm)
            if say != last_say:
                speak(say)
                last_say = say
            last_move, last_arm = move, arm

            if human_speech:
                print(f"[Human] {human_speech}")
            print(f"[Miles] {say}")

            if mem is not None:
                add_memory(str(mem))

            log_cycle(cycle, human_speech, last_move, last_arm, say, mem, raw)

            elapsed = time.time() - cycle_start
            time.sleep(max(0, loop_interval - elapsed))
    except KeyboardInterrupt:
        print("\nShutting down Miles...")
    finally:
        inference_mgr.stop()
        stop_listening()
        shutdown_tts()
        close_serial()
        release_camera()
        print(f"Saved memories: {count_memories()}")
        close_db()


if __name__ == "__main__":
    main()
