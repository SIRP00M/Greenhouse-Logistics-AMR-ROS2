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

    Backend:
    1. ลอง OpenCV + GStreamer ก่อน
    2. ถ้าเปิดไม่ได้ fallback ไป V4L2
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

        # ====================================================
        # GStreamer pipeline
        # ====================================================

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

        self.backend = None

    # ========================================================
    # Camera controls
    # ========================================================

    def _disable_dynamic_framerate(self) -> None:
        """
        ป้องกันกล้องลดตัวเองจาก 30 FPS เหลือ 15 FPS
        เมื่อ Auto Exposure ทำงาน

        ถ้ากล้องไม่รองรับ control นี้ จะ warning แล้วทำงานต่อ
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
                    "[CAMERA WARNING] "
                    "Cannot disable dynamic framerate:",
                    result.stderr.strip(),
                )

        except (FileNotFoundError, subprocess.TimeoutExpired) as error:

            print(
                "[CAMERA WARNING] "
                "v4l2 control failed:",
                error,
            )

    # ========================================================
    # Open camera
    # ========================================================

    def start(self) -> None:

        self._disable_dynamic_framerate()

        # ====================================================
        # Try GStreamer first
        # ====================================================

        print("[CAMERA] Trying GStreamer backend...")
        print("[CAMERA] Pipeline:")
        print(self.pipeline)

        self.cap = cv2.VideoCapture(
            self.pipeline,
            cv2.CAP_GSTREAMER,
        )

        if self.cap.isOpened():

            self.backend = "GStreamer"

            print(
                "[CAMERA] Opened successfully "
                "using GStreamer"
            )

        else:

            print(
                "[CAMERA WARNING] "
                "OpenCV GStreamer backend unavailable "
                "or failed to open camera."
            )

            if self.cap is not None:
                self.cap.release()

            # =================================================
            # Fallback: V4L2
            # =================================================

            print(
                "[CAMERA] Falling back to V4L2..."
            )

            self.cap = cv2.VideoCapture(
                self.device,
                cv2.CAP_V4L2,
            )

            if not self.cap.isOpened():

                raise RuntimeError(
                    f"Cannot open camera {self.device} "
                    "using GStreamer or V4L2"
                )

            # Request MJPEG
            fourcc = cv2.VideoWriter_fourcc(
                "M",
                "J",
                "P",
                "G",
            )

            self.cap.set(
                cv2.CAP_PROP_FOURCC,
                fourcc,
            )

            self.cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                self.width,
            )

            self.cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                self.height,
            )

            self.cap.set(
                cv2.CAP_PROP_FPS,
                self.fps,
            )

            # ลด latency ถ้า backend รองรับ
            self.cap.set(
                cv2.CAP_PROP_BUFFERSIZE,
                1,
            )

            self.backend = "V4L2"

            # =================================================
            # Report actual camera settings
            # =================================================

            actual_width = int(
                self.cap.get(
                    cv2.CAP_PROP_FRAME_WIDTH
                )
            )

            actual_height = int(
                self.cap.get(
                    cv2.CAP_PROP_FRAME_HEIGHT
                )
            )

            actual_fps = self.cap.get(
                cv2.CAP_PROP_FPS
            )

            actual_fourcc = int(
                self.cap.get(
                    cv2.CAP_PROP_FOURCC
                )
            )

            fourcc_text = "".join(
                [
                    chr(
                        (actual_fourcc >> (8 * i))
                        & 0xFF
                    )
                    for i in range(4)
                ]
            )

            print(
                "[CAMERA] V4L2 opened:"
            )

            print(
                f"[CAMERA] device={self.device}"
            )

            print(
                f"[CAMERA] resolution="
                f"{actual_width}x{actual_height}"
            )

            print(
                f"[CAMERA] fps={actual_fps:.2f}"
            )

            print(
                f"[CAMERA] FOURCC={fourcc_text}"
            )

        # ====================================================
        # Start latest-frame thread
        # ====================================================

        self.stop_event.clear()
        self.error = None
        self.read_failures = 0

        self.thread = threading.Thread(
            target=self._capture_loop,
            name="greenhouse-camera-capture",
            daemon=True,
        )

        self.thread.start()

        print(
            f"[CAMERA] Capture thread started "
            f"using {self.backend}"
        )

    # ========================================================
    # Capture loop
    # ========================================================

    def _capture_loop(self) -> None:

        while not self.stop_event.is_set():

            ok, frame = self.cap.read()

            timestamp = time.perf_counter()

            if not ok or frame is None:

                self.read_failures += 1

                if self.read_failures >= 30:

                    self.error = RuntimeError(
                        "Camera failed to provide "
                        "30 consecutive frames"
                    )

                    self.stop_event.set()

                time.sleep(0.005)

                continue

            self.read_failures = 0

            with self.lock:

                self.latest_frame = frame

                self.latest_timestamp = timestamp

                self.latest_sequence += 1

                self.capture_timestamps.append(
                    timestamp
                )

    # ========================================================
    # Latest frame
    # ========================================================

    def get_latest(self):

        with self.lock:

            if self.latest_frame is None:
                return None

            return (
                self.latest_sequence,
                self.latest_timestamp,
                self.latest_frame,
            )

    # ========================================================
    # Wait for camera
    # ========================================================

    def wait_for_first_frame(
        self,
        timeout: float = 5.0,
    ):

        deadline = (
            time.perf_counter()
            + timeout
        )

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

    # ========================================================
    # Capture FPS
    # ========================================================

    def get_capture_fps(self) -> float:

        with self.lock:
            timestamps = list(
                self.capture_timestamps
            )

        if len(timestamps) < 2:
            return 0.0

        elapsed = (
            timestamps[-1]
            - timestamps[0]
        )

        if elapsed <= 0:
            return 0.0

        return (
            len(timestamps) - 1
        ) / elapsed

    # ========================================================
    # Shutdown
    # ========================================================

    def stop(self) -> None:

        self.stop_event.set()

        if self.thread is not None:

            self.thread.join(
                timeout=2.0
            )

            self.thread = None

        if self.cap is not None:

            self.cap.release()

            self.cap = None

        print("[CAMERA] Stopped")
