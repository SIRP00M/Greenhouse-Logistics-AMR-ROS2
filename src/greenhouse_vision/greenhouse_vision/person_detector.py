#!/usr/bin/env python3

import time
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from ultralytics import YOLO

from greenhouse_vision.msg import PersonDetection


class PersonDetector(Node):
    def __init__(self) -> None:
        super().__init__("person_detector")

        self.declare_parameter("model_path", "yolo11n.pt")
        self.declare_parameter("confidence_threshold", 0.40)
        self.declare_parameter("image_size", 640)

        model_path = (
            self.get_parameter("model_path")
            .get_parameter_value()
            .string_value
        )

        self.confidence_threshold = (
            self.get_parameter("confidence_threshold")
            .get_parameter_value()
            .double_value
        )

        self.image_size = (
            self.get_parameter("image_size")
            .get_parameter_value()
            .integer_value
        )

        self.bridge = CvBridge()

        self.debug_publisher = self.create_publisher(
            Image,
            "/vision/debug_image",
            qos_profile_sensor_data,
        )

        self.detection_publisher = self.create_publisher(
            PersonDetection,
            "/person_detection",
            10,
        )

        self.image_subscription = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.get_logger().info(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)

        self.inference_frame_count = 0
        self.inference_start_time = time.perf_counter()

        self.get_logger().info("YOLO model loaded successfully.")
        self.get_logger().info("Subscribing: /camera/image_raw")
        self.get_logger().info("Publishing: /vision/debug_image")
        self.get_logger().info("Publishing: /person_detection")

    def image_callback(self, image_message: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(
                image_message,
                desired_encoding="bgr8",
            )
        except Exception as error:
            self.get_logger().error(
                f"Failed to convert ROS image: {error}"
            )
            return

        inference_begin = time.perf_counter()

        try:
            results = self.model.predict(
                source=frame,
                classes=[0],
                conf=self.confidence_threshold,
                imgsz=self.image_size,
                verbose=False,
            )
        except Exception as error:
            self.get_logger().error(f"YOLO inference failed: {error}")
            return

        inference_elapsed = time.perf_counter() - inference_begin

        debug_frame = frame.copy()
        detection_message = PersonDetection()

        detection_message.detected = False
        detection_message.confidence = 0.0
        detection_message.center_x = 0
        detection_message.center_y = 0
        detection_message.width = 0
        detection_message.height = 0

        best_confidence = -1.0
        best_detection: Optional[tuple[int, int, int, int, float]] = None

        if results and results[0].boxes is not None:
            boxes = results[0].boxes

            for box in boxes:
                coordinates = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().item())

                x1, y1, x2, y2 = coordinates.astype(np.int32)

                x1 = max(0, min(x1, frame.shape[1] - 1))
                y1 = max(0, min(y1, frame.shape[0] - 1))
                x2 = max(0, min(x2, frame.shape[1] - 1))
                y2 = max(0, min(y2, frame.shape[0] - 1))

                width = max(0, x2 - x1)
                height = max(0, y2 - y1)

                center_x = x1 + width // 2
                center_y = y1 + height // 2

                cv2.rectangle(
                    debug_frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.circle(
                    debug_frame,
                    (center_x, center_y),
                    5,
                    (0, 0, 255),
                    -1,
                )

                label = f"person {confidence:.2f}"

                text_size, _ = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    2,
                )

                text_width, text_height = text_size
                label_y = max(y1, text_height + 10)

                cv2.rectangle(
                    debug_frame,
                    (x1, label_y - text_height - 10),
                    (x1 + text_width + 10, label_y),
                    (0, 255, 0),
                    -1,
                )

                cv2.putText(
                    debug_frame,
                    label,
                    (x1 + 5, label_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

                if confidence > best_confidence:
                    best_confidence = confidence
                    best_detection = (
                        center_x,
                        center_y,
                        width,
                        height,
                        confidence,
                    )

        if best_detection is not None:
            center_x, center_y, width, height, confidence = best_detection

            detection_message.detected = True
            detection_message.confidence = confidence
            detection_message.center_x = center_x
            detection_message.center_y = center_y
            detection_message.width = width
            detection_message.height = height

        self.detection_publisher.publish(detection_message)

        debug_message = self.bridge.cv2_to_imgmsg(
            debug_frame,
            encoding="bgr8",
        )
        debug_message.header = image_message.header
        self.debug_publisher.publish(debug_message)

        self.inference_frame_count += 1
        fps_elapsed = time.perf_counter() - self.inference_start_time

        if fps_elapsed >= 1.0:
            inference_fps = self.inference_frame_count / fps_elapsed
            inference_time_ms = inference_elapsed * 1000.0

            self.get_logger().info(
                f"Inference FPS: {inference_fps:.2f} | "
                f"Last inference: {inference_time_ms:.1f} ms"
            )

            self.inference_frame_count = 0
            self.inference_start_time = time.perf_counter()


def main(args=None) -> None:
    rclpy.init(args=args)

    node = None

    try:
        node = PersonDetector()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as error:
        if node is not None:
            node.get_logger().fatal(str(error))
        else:
            print(f"[person_detector] Fatal error: {error}")

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
