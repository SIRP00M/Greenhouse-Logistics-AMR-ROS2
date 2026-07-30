#!/usr/bin/env python3

import argparse
import statistics
import threading
import time
from collections import deque
from pathlib import Path

import cv2
from ultralytics import YOLO


class LatestFrameCamera:
    """
    อ่านกล้องใน Thread แยก และเก็บเฉพาะเฟรมล่าสุดหนึ่งเฟรม

    ไม่มี Queue:
    - เฟรมใหม่เขียนทับเฟรมเก่า
    - Detector จะไม่ประมวลผลภาพย้อนหลัง
    """

    def __init__(
        self,
        device: str,
        width: int,
        height: int,
        fps: int,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps

        self.pipeline = (
            f"v4l2src device={device} ! "
            f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
            "jpegdec ! "
            "videoconvert ! "
            "video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )

        self.cap = None
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        self.latest_frame = None
        self.latest_timestamp = 0.0
        self.latest_sequence = 0

        self.total_frames = 0
        self.read_failures = 0
        self.error = None

    def start(self) -> None:
        print("[CAMERA] Opening GStreamer pipeline:")
        print(self.pipeline)

        self.cap = cv2.VideoCapture(
            self.pipeline,
            cv2.CAP_GSTREAMER,
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {self.device} with GStreamer"
            )

        self.thread = threading.Thread(
            target=self._capture_loop,
            name="camera-capture",
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
                self.total_frames += 1

    def get_latest(self):
        with self.lock:
            if self.latest_frame is None:
                return None

            return (
                self.latest_sequence,
                self.latest_timestamp,
                self.latest_frame,
            )

    def stop(self) -> None:
        self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=2.0)

        if self.cap is not None:
            self.cap.release()


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0

    ordered = sorted(values)
    index = round((len(ordered) - 1) * ratio)
    return ordered[index]


def wait_for_first_frame(
    camera: LatestFrameCamera,
    timeout: float = 5.0,
):
    deadline = time.perf_counter() + timeout

    while time.perf_counter() < deadline:
        latest = camera.get_latest()

        if latest is not None:
            return latest

        if camera.error is not None:
            raise camera.error

        time.sleep(0.01)

    raise TimeoutError("Timed out waiting for the first camera frame")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Live YOLO11n NCNN latest-frame benchmark"
    )

    parser.add_argument(
        "--model",
        default="yolo11n_ncnn_model",
    )
    parser.add_argument(
        "--device",
        default="/dev/video0",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
    )
    parser.add_argument(
        "--camera-fps",
        type=int,
        default=30,
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=0.0,
        help="0 = run detector as fast as possible",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=30.0,
        help="Test duration in seconds. Use 0 to run indefinitely.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show debug image. This may reduce FPS.",
    )

    args = parser.parse_args()

    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    camera = LatestFrameCamera(
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.camera_fps,
    )

    camera.start()

    try:
        _, _, first_frame = wait_for_first_frame(camera)

        frame_height, frame_width = first_frame.shape[:2]

        print(
            f"[CAMERA] First frame received: "
            f"{frame_width}x{frame_height}"
        )

        print(f"[MODEL] Loading {model_path}")
        model = YOLO(str(model_path))

        print(
            f"[MODEL] Warm-up: {args.warmup} runs "
            f"at imgsz={args.imgsz}"
        )

        for _ in range(args.warmup):
            model.predict(
                source=first_frame,
                imgsz=args.imgsz,
                classes=[0],
                conf=args.conf,
                verbose=False,
            )

        target_period = (
            1.0 / args.target_fps
            if args.target_fps > 0
            else 0.0
        )

        test_start = time.perf_counter()
        next_detection_time = test_start
        last_report_time = test_start

        last_camera_total = camera.total_frames
        last_detector_total = 0
        last_processed_sequence = -1

        detector_total = 0
        duplicate_skips = 0

        report_wall_times = []
        report_frame_ages = []
        report_e2e_times = []

        all_wall_times = []
        all_frame_ages = []
        all_e2e_times = []

        print()
        print("=" * 72)
        print("LIVE PERSON DETECTION STARTED")
        print("=" * 72)
        print(f"Camera       : {args.width}x{args.height} @ {args.camera_fps} FPS")
        print(f"Inference    : {args.imgsz}x{args.imgsz}")
        print(f"Confidence   : {args.conf}")
        print(
            f"Detector cap : "
            f"{args.target_fps:.1f} FPS"
            if args.target_fps > 0
            else "Detector cap : Unlimited"
        )
        print("=" * 72)

        while not camera.stop_event.is_set():
            now = time.perf_counter()

            if args.duration > 0:
                if now - test_start >= args.duration:
                    break

            if target_period > 0 and now < next_detection_time:
                time.sleep(min(next_detection_time - now, 0.002))
                continue

            latest = camera.get_latest()

            if latest is None:
                time.sleep(0.001)
                continue

            sequence, capture_timestamp, frame = latest

            if sequence == last_processed_sequence:
                duplicate_skips += 1
                time.sleep(0.001)
                continue

            last_processed_sequence = sequence

            inference_start = time.perf_counter()

            frame_age_ms = (
                inference_start - capture_timestamp
            ) * 1000.0

            results = model.predict(
                source=frame,
                imgsz=args.imgsz,
                classes=[0],
                conf=args.conf,
                verbose=False,
            )

            inference_end = time.perf_counter()

            wall_time_ms = (
                inference_end - inference_start
            ) * 1000.0

            end_to_end_ms = (
                inference_end - capture_timestamp
            ) * 1000.0

            detector_total += 1

            report_wall_times.append(wall_time_ms)
            report_frame_ages.append(frame_age_ms)
            report_e2e_times.append(end_to_end_ms)

            all_wall_times.append(wall_time_ms)
            all_frame_ages.append(frame_age_ms)
            all_e2e_times.append(end_to_end_ms)

            result = results[0]
            person_count = (
                0
                if result.boxes is None
                else len(result.boxes)
            )

            if args.display:
                annotated = result.plot()

                cv2.putText(
                    annotated,
                    f"Persons: {person_count}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    annotated,
                    f"Latency: {end_to_end_ms:.1f} ms",
                    (10, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.imshow("YOLO11n NCNN Person Detection", annotated)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    break

            if target_period > 0:
                next_detection_time += target_period

                # ถ้าหลุดตารางมากเกินหนึ่งรอบ ให้เริ่มจากเวลาปัจจุบัน
                if next_detection_time < inference_end:
                    next_detection_time = inference_end

            report_now = time.perf_counter()
            report_elapsed = report_now - last_report_time

            if report_elapsed >= 2.0:
                camera_total = camera.total_frames

                capture_fps = (
                    camera_total - last_camera_total
                ) / report_elapsed

                detector_fps = (
                    detector_total - last_detector_total
                ) / report_elapsed

                average_wall = statistics.mean(report_wall_times)
                average_age = statistics.mean(report_frame_ages)
                average_e2e = statistics.mean(report_e2e_times)
                p95_e2e = percentile(report_e2e_times, 0.95)

                print(
                    f"[LIVE] "
                    f"camera={capture_fps:5.2f} FPS | "
                    f"detect={detector_fps:5.2f} FPS | "
                    f"infer={average_wall:6.1f} ms | "
                    f"age={average_age:5.1f} ms | "
                    f"e2e={average_e2e:6.1f} ms | "
                    f"p95={p95_e2e:6.1f} ms | "
                    f"persons={person_count}"
                )

                last_camera_total = camera_total
                last_detector_total = detector_total
                last_report_time = report_now

                report_wall_times.clear()
                report_frame_ages.clear()
                report_e2e_times.clear()

        total_elapsed = time.perf_counter() - test_start

        average_detector_fps = (
            detector_total / total_elapsed
            if total_elapsed > 0
            else 0.0
        )

        average_capture_fps = (
            camera.total_frames / total_elapsed
            if total_elapsed > 0
            else 0.0
        )

        print()
        print("=" * 72)
        print("LIVE BENCHMARK RESULT")
        print("=" * 72)
        print(f"Test duration          : {total_elapsed:8.2f} seconds")
        print(f"Captured frames        : {camera.total_frames:8d}")
        print(f"Detector runs          : {detector_total:8d}")
        print(f"Duplicate skips        : {duplicate_skips:8d}")
        print("-" * 72)
        print(f"Average capture FPS    : {average_capture_fps:8.2f}")
        print(f"Average detector FPS   : {average_detector_fps:8.2f}")

        if all_wall_times:
            print(f"Average inference      : {statistics.mean(all_wall_times):8.2f} ms")
            print(f"Median inference       : {statistics.median(all_wall_times):8.2f} ms")
            print(f"P95 inference          : {percentile(all_wall_times, 0.95):8.2f} ms")
            print(f"Average frame age      : {statistics.mean(all_frame_ages):8.2f} ms")
            print(f"Average end-to-end     : {statistics.mean(all_e2e_times):8.2f} ms")
            print(f"P95 end-to-end         : {percentile(all_e2e_times, 0.95):8.2f} ms")

        print("=" * 72)

        if average_detector_fps >= 10.0:
            print("[PASS] Live detection reaches the 10 FPS target.")
        else:
            print("[FAIL] Live detection is below the 10 FPS target.")

        return 0

    finally:
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
