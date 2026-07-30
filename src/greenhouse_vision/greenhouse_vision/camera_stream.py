#!/usr/bin/env python3

import subprocess
import threading
import time
from collections import deque

import cv2


class LatestFrameCamera:
    """
    อ่านภาพจากกล้องใน Thread แยก และเก็บเฉพาะเฟรมล่าสุด

    ไม่มีการสะสมเฟรมใน Queue ฝั่ง Python
    เฟรมใหม่จะเขียนทับเฟรมเก่าเสมอ
    """

    def __init__(
        self,
        device: str = "/dev/video0",
        width: int = 640,
        height: int = 480,
        fps: int = 30,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = (
            f"v4l2src device={self.device} ! "
            f"image/jpeg,width={self.width},height={self.height},"
            f"framerate={self.fps}/1 ! "
            "queue max-size-buffers=2 max-size-bytes=0 "
            "max-size-time=0 leaky=downstream ! "
            "jpegdec ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "queue max-size-buffers=1 max-size-bytes=0 "
            "max-size-time=0 leaky=downstream ! "
            "appsink drop=true max-buffers=1 sync=false"
        )

        self.cap = None
        self.thread = None

        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        self.latest_frame = None
        self.latest_timestamp = 0.0
        self.latest_sequence = 0

        self.capture_timestamps = deque(maxlen=120)

        self.read_failures = 0
        self.error = None

    def _disable_dynamic_framerate(self) -> None:
        """
        ป้องกันกล้องลดตัวเองจาก 30 FPS เหลือ 15 FPS
        เมื่อ Auto Exposure ทำงาน
        """

        command = [
            "v4l2-ctl",
            f"--device={self.device}",
            "--set-ctrl=exposure_dynamic_framerate=0",
        ]

        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=3,
                check=False,
            )

            if result.returncode != 0:
                print(
                    "[CAMERA WARNING] Cannot disable dynamic framerate:",
                    result.stderr.strip(),
                )

        except (FileNotFoundError, subprocess.TimeoutExpired) as error:
            print(
                "[CAMERA WARNING] v4l2 control failed:",
                error,
            )

    def start(self) -> None:
        self._disable_dynamic_framerate()

        print("[CAMERA] Opening GStreamer pipeline:")
        print(self.pipeline)

        self.cap = cv2.VideoCapture(
            self.pipeline,
            cv2.CAP_GSTREAMER,
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {self.device} using GStreamer"
            )

        self.thread = threading.Thread(
            target=self._capture_loop,
            name="greenhouse-camera-capture",
            daemon=True,
        )

        self.thread.start()

    def _capture_loop(self) -> None:
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            timestamp = time.perf_counter()

            if not ok or frame is None:
                self.read_failures += 1

                if self.read_failures >= 30:
                    self.error = RuntimeError(
                        "Camera failed to provide 30 consecutive frames"
                    )
                    self.stop_event.set()

                time.sleep(0.005)
                continue

            self.read_failures = 0

            with self.lock:
                self.latest_frame = frame
                self.latest_timestamp = timestamp
                self.latest_sequence += 1
                self.capture_timestamps.append(timestamp)

    def get_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return None

            return (
                self.latest_sequence,
                self.latest_timestamp,
                self.latest_frame,
            )

    def wait_for_first_frame(self, timeout: float = 5.0):
        deadline = time.perf_counter() + timeout

        while time.perf_counter() < deadline:
            latest = self.get_latest()

            if latest is not None:
                return latest

            if self.error is not None:
                raise self.error

            time.sleep(0.01)

        raise TimeoutError(
            "Timed out waiting for first camera frame"
        )

    def get_capture_fps(self) -> float:
        with self.lock:
            timestamps = list(self.capture_timestamps)

        if len(timestamps) < 2:
            return 0.0

        elapsed = timestamps[-1] - timestamps[0]

        if elapsed <= 0:
            return 0.0

        return (len(timestamps) - 1) / elapsed

    def stop(self) -> None:
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=2.0)

        if self.cap is not None:
            self.cap.release()
