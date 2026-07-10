"""Central configuration constants for the Miles robotics pipeline."""

import os

CAMERA_INDEX = 0
FPS = 3
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
OLLAMA_MODEL = "llava"  # or "llava"
OLLAMA_URL = "http://localhost:11434/api/generate"

# --- Google AI Studio / Gemini config ---
# Never hardcode the key here — set it with `export GEMINI_API_KEY=...` on the
# machine running this code. The SDK also picks up GEMINI_API_KEY automatically,
# but we read it explicitly so we can fail loudly if it's missing.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_VLM_MODEL = "gemini-3.5-flash"       # Stage 1: perception (needs vision)
GEMINI_VLA_MODEL = "gemini-3.1-flash-lite"  # Stage 2: decision (text-only, cheaper/faster)
# RPM is shared per-model per-project; 2 calls/cycle (VLM + VLA) means the
# cycle period must be >= 60 / min(VLM_RPM, VLA_RPM) seconds, with margin.
# Check your project's live limits: AI Studio -> API keys -> Billing Tier column.
#   - Free tier:            ~8-10s/cycle is safe.
#   - Tier 1 (billing on):  ~1-2s/cycle is safe; can match a snappier ROS timer.
# Google AI Pro (the consumer subscription) does NOT by itself change this —
# it's the Cloud project's billing tier that matters. Set BRAIN_CYCLE_SECONDS
# to match whichever tier your API key's project is actually on.
BRAIN_CYCLE_SECONDS = 8.0

SERIAL_WHEELS_PORT = "/dev/ttyUSB0"
SERIAL_ARM_PORT = "/dev/ttyUSB1"
SERIAL_BAUD = 9600
DB_PATH = "miles_memory.db"
LOG_PATH = "miles_log.csv"
MEMORY_MAX_ENTRIES = 20
ACTION_STALE_TIMEOUT_SEC = 2.5  # Stop motors if no fresh inference command in this window.

# ROS 2 topic configuration for the Miles brain node.
IMAGE_TOPIC = "/camera/image_raw"
WHEELS_CMD_TOPIC = "/cmd_vel"
ARM_CMD_TOPIC = "/arm/commands"
