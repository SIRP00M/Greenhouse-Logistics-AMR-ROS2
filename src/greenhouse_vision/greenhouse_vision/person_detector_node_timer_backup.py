#!/usr/bin/env python3

# ต้องกำหนดก่อนโหลด OpenCV และ NCNN
import os

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("GOMP_CPU_AFFINITY", "1 2 3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from ultralytics import YOLO

from greenhouse_vision_msgs.msg import PersonDetection

from .camera_stream import LatestFrameCamera


cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)


class PersonDetectorNode(Node):

    def __init__(self) -> None:
        super().__init__("person_detector")

        self.declare_parameter(
            "model_path",
            "/home/poom/Desktop/"
            "Greenhouse-Logistics-AMR-ROS2/"
            "yolo11n_ncnn_model",
        )
        self.declare_parameter("device", "/dev/video0")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("camera_fps", 30)

        self.declare_parameter("imgsz", 320)
        self.declare_parameter("confidence", 0.35)
        self.declare_parameter("target_fps", 10.0)
        self.declare_parameter("warmup_runs", 60)

        self.declare_parameter(
            "target_strategy",
            "largest",
        )
        self.declare_parameter(
            "log_interval_sec",
            2.0,
        )

        self.model_path = str(
            self.get_parameter("model_path").value
        )
        self.device = str(
            self.get_parameter("device").value
        )
        self.width = int(
            self.get_parameter("width").value
        )
        self.height = int(
            self.get_parameter("height").value
        )
        self.camera_fps = int(
            self.get_parameter("camera_fps").value
        )

        self.imgsz = int(
            self.get_parameter("imgsz").value
        )
        self.confidence = float(
            self.get_parameter("confidence").value
        )
        self.target_fps = float(
            self.get_parameter("target_fps").value
        )
        self.warmup_runs = int(
            self.get_parameter("warmup_runs").value
        )

        self.target_strategy = str(
            self.get_parameter("target_strategy").value
        )
        self.log_interval_sec = float(
            self.get_parameter("log_interval_sec").value
        )

        if self.target_fps <= 0:
            raise ValueError("target_fps must be greater than zero")

        if self.target_strategy not in {
            "largest",
            "highest_confidence",
        }:
            raise ValueError(
                "target_strategy must be 'largest' "
                "or 'highest_confidence'"
            )

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model not found: {self.model_path}"
            )

        self.publisher = self.create_publisher(
            PersonDetection,
            "/person_detection",
            10,
        )

        self.camera = LatestFrameCamera(
            device=self.device,
            width=self.width,
            height=self.height,
            fps=self.camera_fps,
        )

        self.get_logger().info(
            "Starting latest-frame camera..."
        )

        self.camera.start()

        _, _, first_frame = self.camera.wait_for_first_frame(
            timeout=5.0
        )

        actual_height, actual_width = first_frame.shape[:2]

        self.get_logger().info(
            f"First camera frame: "
            f"{actual_width}x{actual_height}"
        )

        self.get_logger().info(
            f"Loading NCNN model: {self.model_path}"
        )

        self.model = YOLO(
            self.model_path,
            task="detect",
        )

        self.get_logger().info(
            f"Warming up model for {self.warmup_runs} runs..."
        )

        for _ in range(self.warmup_runs):
            self.model.predict(
                source=first_frame,
                imgsz=self.imgsz,
                classes=[0],
                conf=self.confidence,
                verbose=False,
            )

        self.get_logger().info(
            "Model warm-up completed"
        )

        self.last_processed_sequence = -1
        self.detector_timestamps = deque(maxlen=50)

        self.last_log_time = time.perf_counter()

        timer_period = 1.0 / self.target_fps

        self.timer = self.create_timer(
            timer_period,
            self.detect_once,
        )

        self.get_logger().info(
            "Person detector started: "
            f"camera={self.width}x{self.height}"
            f"@{self.camera_fps}FPS, "
            f"YOLO={self.imgsz}, "
            f"target={self.target_fps:.1f}FPS"
        )

    def _calculate_detector_fps(self) -> float:
        if len(self.detector_timestamps) < 2:
            return 0.0

        elapsed = (
            self.detector_timestamps[-1]
            - self.detector_timestamps[0]
        )

        if elapsed <= 0:
            return 0.0

        return (
            len(self.detector_timestamps) - 1
        ) / elapsed

    def _select_target(
        self,
        xyxy: np.ndarray,
        confidences: np.ndarray,
    ) -> int:
        if self.target_strategy == "highest_confidence":
            return int(np.argmax(confidences))

        widths = xyxy[:, 2] - xyxy[:, 0]
        heights = xyxy[:, 3] - xyxy[:, 1]
        areas = widths * heights

        return int(np.argmax(areas))

    def detect_once(self) -> None:
        latest = self.camera.get_latest()

        if latest is None:
            return

        sequence, capture_timestamp, frame = latest

        if sequence == self.last_processed_sequence:
            return

        self.last_processed_sequence = sequence

        inference_start = time.perf_counter()

        frame_age_ms = (
            inference_start - capture_timestamp
        ) * 1000.0

        result = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            classes=[0],
            conf=self.confidence,
            verbose=False,
        )[0]

        inference_end = time.perf_counter()

        inference_ms = (
            inference_end - inference_start
        ) * 1000.0

        end_to_end_ms = (
            inference_end - capture_timestamp
        ) * 1000.0

        self.detector_timestamps.append(inference_end)
        detector_fps = self._calculate_detector_fps()

        frame_height, frame_width = frame.shape[:2]

        message = PersonDetection()

        message.stamp = self.get_clock().now().to_msg()
        message.detected = False
        message.person_count = 0

        message.confidence = 0.0

        message.x_min = 0
        message.y_min = 0
        message.x_max = 0
        message.y_max = 0

        message.center_x = 0.0
        message.center_y = 0.0

        message.horizontal_error = 0.0
        message.area_ratio = 0.0

        message.frame_width = frame_width
        message.frame_height = frame_height

        message.detector_fps = float(detector_fps)
        message.frame_age_ms = float(frame_age_ms)
        message.inference_ms = float(inference_ms)
        message.end_to_end_ms = float(end_to_end_ms)

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            message.person_count = int(len(xyxy))

            target_index = self._select_target(
                xyxy,
                confidences,
            )

            x_min, y_min, x_max, y_max = xyxy[
                target_index
            ]

            x_min = int(max(0, round(x_min)))
            y_min = int(max(0, round(y_min)))
            x_max = int(min(frame_width - 1, round(x_max)))
            y_max = int(min(frame_height - 1, round(y_max)))

            center_x = (x_min + x_max) / 2.0
            center_y = (y_min + y_max) / 2.0

            bbox_width = max(0, x_max - x_min)
            bbox_height = max(0, y_max - y_min)
            bbox_area = bbox_width * bbox_height

            frame_area = frame_width * frame_height

            horizontal_error = (
                center_x - (frame_width / 2.0)
            ) / (frame_width / 2.0)

            area_ratio = (
                bbox_area / frame_area
                if frame_area > 0
                else 0.0
            )

            message.detected = True
            message.confidence = float(
                confidences[target_index]
            )

            message.x_min = x_min
            message.y_min = y_min
            message.x_max = x_max
            message.y_max = y_max

            message.center_x = float(center_x)
            message.center_y = float(center_y)

            message.horizontal_error = float(
                horizontal_error
            )
            message.area_ratio = float(area_ratio)

        self.publisher.publish(message)

        now = time.perf_counter()

        if now - self.last_log_time >= self.log_interval_sec:
            self.get_logger().info(
                f"camera={self.camera.get_capture_fps():.2f} FPS | "
                f"detect={detector_fps:.2f} FPS | "
                f"persons={message.person_count} | "
                f"infer={inference_ms:.1f} ms | "
                f"age={frame_age_ms:.1f} ms | "
                f"e2e={end_to_end_ms:.1f} ms"
            )

            self.last_log_time = now

    def destroy_node(self):
        if hasattr(self, "camera") and self.camera is not None:
            self.camera.stop()

        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)

    node = None

    try:
        node = PersonDetectorNode()
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    except Exception as error:
        if node is not None:
            node.get_logger().error(str(error))
        else:
            print(f"[ERROR] {error}")

        raise

    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
