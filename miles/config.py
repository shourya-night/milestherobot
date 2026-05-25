"""Central configuration constants for the Miles robotics pipeline."""

CAMERA_INDEX = 0
FPS = 3
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
OLLAMA_MODEL = "moondream2"  # or "llava"
OLLAMA_URL = "http://localhost:11434/api/generate"
SERIAL_WHEELS_PORT = "/dev/ttyUSB0"
SERIAL_ARM_PORT = "/dev/ttyUSB1"
SERIAL_BAUD = 9600
DB_PATH = "miles_memory.db"
LOG_PATH = "miles_log.csv"
MEMORY_MAX_ENTRIES = 20
STT_MODEL = "base"  # Whisper model size: tiny/base/small
TTS_RATE = 175  # Speech rate for pyttsx3
TTS_VOLUME = 1.0
SPEECH_ENERGY_THRESHOLD = 300  # Mic sensitivity for speech detection
