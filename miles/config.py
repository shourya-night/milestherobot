"""Central configuration constants for the Miles robotics pipeline."""

CAMERA_INDEX = 0
FPS = 3
FRAME_WIDTH = 320
FRAME_HEIGHT = 240
OLLAMA_MODEL = "llava"  # or "llava"
OLLAMA_URL = "http://localhost:11434/api/generate"
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
