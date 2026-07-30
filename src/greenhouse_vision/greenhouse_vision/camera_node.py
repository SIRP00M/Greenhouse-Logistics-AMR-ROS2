#!/usr/bin/env python3

import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


GST_PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "image/jpeg,width=640,height=480,framerate=30/1 ! "
    "jpegdec ! "
    "videoconvert ! "
    "video/x-raw,format=BGR ! "
    "appsink drop=true max-buffers=1 sync=false"
)


class CameraNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_node")

        self.publisher = self.create_publisher(
            Image,
            "/camera/image_raw",
            qos_profile_sensor_data,
        )

        self.bridge = CvBridge()

        self.capture = cv2.VideoCapture(
            GST_PIPELINE,
            cv2.CAP_GSTREAMER,
        )

        if not self.capture.isOpened():
            raise RuntimeError(
                "Failed to open /dev/video0 using the GStreamer pipeline."
            )

        self.frame_count = 0
        self.fps_start_time = time.perf_counter()
        self.last_warning_time = 0.0

        # Run continuously. Camera frame rate is controlled by GStreamer.
        self.timer = self.create_timer(
            0.001,
            self.capture_and_publish,
        )

        self.get_logger().info("Camera opened successfully.")
        self.get_logger().info("Publishing: /camera/image_raw")
        self.get_logger().info("Resolution: 640x480 @ 30 FPS")

    def capture_and_publish(self) -> None:
        success, frame = self.capture.read()

        if not success or frame is None:
            current_time = time.perf_counter()

            if current_time - self.last_warning_time >= 1.0:
                self.get_logger().warning("Failed to capture camera frame.")
                self.last_warning_time = current_time

            return

        message = self.bridge.cv2_to_imgmsg(
            frame,
            encoding="bgr8",
        )

        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "camera_optical_frame"

        self.publisher.publish(message)

        self.frame_count += 1
        current_time = time.perf_counter()
        elapsed_time = current_time - self.fps_start_time

        if elapsed_time >= 1.0:
            camera_fps = self.frame_count / elapsed_time
            self.get_logger().info(f"Camera FPS: {camera_fps:.2f}")

            self.frame_count = 0
            self.fps_start_time = current_time

    def destroy_node(self) -> None:
        if hasattr(self, "capture") and self.capture.isOpened():
            self.capture.release()

        self.get_logger().info("Camera released.")
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)

    node = None

    try:
        node = CameraNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as error:
        if node is not None:
            node.get_logger().fatal(str(error))
        else:
            print(f"[camera_node] Fatal error: {error}")

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
