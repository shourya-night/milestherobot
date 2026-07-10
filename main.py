import base64
import json
import threading

import cv2
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from config import ARM_CMD_TOPIC, IMAGE_TOPIC, MEMORY_MAX_ENTRIES, WHEELS_CMD_TOPIC
from inference import run_inference
from logger import init_logger, log_cycle
from memory import add_memory, close_db, get_recent_memory, init_db


class MilesBrainNode(Node):
    def __init__(self):
        super().__init__("miles_brain_node")
        init_db()
        init_logger()

        self.bridge = CvBridge()
        self._state_lock = threading.Lock()
        self.latest_frame_b64 = ""
        self.cycle = 0

        self.create_subscription(Image, IMAGE_TOPIC, self.image_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, WHEELS_CMD_TOPIC, 10)
        self.arm_pub = self.create_publisher(String, ARM_CMD_TOPIC, 10)
        self.timer = self.create_timer(0.5, self.execute_brain_cycle)

        self.get_logger().info("MilesBrainNode initialized.")

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            resized = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", resized)
            if not ok:
                return
            frame_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
            with self._state_lock:
                self.latest_frame_b64 = frame_b64
        except Exception as exc:
            self.get_logger().warning(f"Image conversion failed: {exc}")

    def _move_to_twist(self, move: str) -> Twist:
        twist = Twist()
        if move == "FORWARD":
            twist.linear.x = 0.2
        elif move == "BACKWARD":
            twist.linear.x = -0.2
        elif move == "LEFT":
            twist.angular.z = 0.8
        elif move == "RIGHT":
            twist.angular.z = -0.8
        return twist

    def execute_brain_cycle(self):
        with self._state_lock:
            frame_b64 = self.latest_frame_b64

        if not frame_b64:
            self.get_logger().debug("No camera frame yet; skipping cycle.")
            return

        memory_str = get_recent_memory(MEMORY_MAX_ENTRIES)
        result = run_inference(frame_b64, memory_str)

        move = str(result.get("move", "STOP"))
        arm = str(result.get("arm", "NONE"))
        mem = result.get("mem")
        raw = str(result.get("_raw", json.dumps(result)))

        self.cmd_pub.publish(self._move_to_twist(move))
        self.arm_pub.publish(String(data=arm))

        if mem is not None:
            add_memory(str(mem))

        self.cycle += 1
        log_cycle(self.cycle, move, arm, mem, raw)


def main(args=None):
    rclpy.init(args=args)
    node = MilesBrainNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        close_db()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
