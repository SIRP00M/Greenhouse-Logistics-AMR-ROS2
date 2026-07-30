#!/usr/bin/env python3

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


def capture_one_frame(device: str) -> object:
    pipeline = (
        f"v4l2src device={device} ! "
        "image/jpeg,width=640,height=480,framerate=30/1 ! "
        "jpegdec ! "
        "videoconvert ! "
        "video/x-raw,format=BGR ! "
        "appsink drop=true max-buffers=1 sync=false"
    )

    print("\n[INFO] Opening camera...")
    print(f"[INFO] Pipeline: {pipeline}")

    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

    if not cap.isOpened():
        raise RuntimeError(
            "Cannot open camera with GStreamer. "
            "Check the camera device and pipeline."
        )

    frame = None

    # อ่านทิ้งเล็กน้อยเพื่อให้กล้องปรับ exposure
    for _ in range(15):
        ok, candidate = cap.read()
        if ok and candidate is not None:
            frame = candidate

    cap.release()

    if frame is None:
        raise RuntimeError("Camera opened, but no valid frame was received.")

    height, width = frame.shape[:2]
    print(f"[INFO] Captured frame: {width}x{height}")

    return frame


def percentile(values: list[float], percent: float) -> float:
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * percent))
    return ordered[index]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark YOLO11n NCNN using one camera frame."
    )

    parser.add_argument(
        "--model",
        default="yolo11n_ncnn_model",
        help="Path to NCNN model directory.",
    )

    parser.add_argument(
        "--device",
        default="/dev/video0",
        help="Camera device.",
    )

    parser.add_argument(
        "--imgsz",
        type=int,
        default=320,
        help="YOLO inference image size.",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warm-up runs.",
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Number of measured runs.",
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Person confidence threshold.",
    )

    args = parser.parse_args()

    model_path = Path(args.model)

    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}", file=sys.stderr)
        return 1

    frame = capture_one_frame(args.device)

    cv2.imwrite("benchmark_frame.jpg", frame)
    print("[INFO] Saved benchmark_frame.jpg")

    print(f"\n[INFO] Loading model: {model_path}")
    model = YOLO(str(model_path))

    print(
        f"[INFO] Benchmark settings: imgsz={args.imgsz}, "
        f"warmup={args.warmup}, runs={args.runs}"
    )

    print("\n[INFO] Warming up model...")

    for _ in range(args.warmup):
        model.predict(
            source=frame,
            imgsz=args.imgsz,
            classes=[0],
            conf=args.conf,
            verbose=False,
        )

    wall_times_ms: list[float] = []
    preprocess_times_ms: list[float] = []
    inference_times_ms: list[float] = []
    postprocess_times_ms: list[float] = []
    detection_counts: list[int] = []

    print("[INFO] Running measured benchmark...")

    benchmark_start = time.perf_counter()

    for run_number in range(1, args.runs + 1):
        start = time.perf_counter()

        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            classes=[0],
            conf=args.conf,
            verbose=False,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        wall_times_ms.append(elapsed_ms)

        result = results[0]
        speed = result.speed

        preprocess_times_ms.append(float(speed.get("preprocess", 0.0)))
        inference_times_ms.append(float(speed.get("inference", 0.0)))
        postprocess_times_ms.append(float(speed.get("postprocess", 0.0)))

        box_count = 0 if result.boxes is None else len(result.boxes)
        detection_counts.append(box_count)

        if run_number % 10 == 0:
            current_average = statistics.mean(wall_times_ms)
            current_fps = 1000.0 / current_average

            print(
                f"[RUN {run_number:3d}/{args.runs}] "
                f"last={elapsed_ms:7.2f} ms | "
                f"average={current_average:7.2f} ms | "
                f"FPS={current_fps:5.2f}"
            )

    total_elapsed = time.perf_counter() - benchmark_start

    average_wall = statistics.mean(wall_times_ms)
    median_wall = statistics.median(wall_times_ms)
    p95_wall = percentile(wall_times_ms, 0.95)
    minimum_wall = min(wall_times_ms)
    maximum_wall = max(wall_times_ms)

    average_preprocess = statistics.mean(preprocess_times_ms)
    average_inference = statistics.mean(inference_times_ms)
    average_postprocess = statistics.mean(postprocess_times_ms)

    wall_fps = 1000.0 / average_wall
    throughput_fps = args.runs / total_elapsed

    print("\n" + "=" * 56)
    print("YOLO11n NCNN BENCHMARK RESULT")
    print("=" * 56)
    print(f"Model                 : {model_path}")
    print(f"Input image            : 640x480 MJPEG camera frame")
    print(f"Inference imgsz        : {args.imgsz}")
    print(f"Measured runs          : {args.runs}")
    print("-" * 56)
    print(f"Average wall time      : {average_wall:8.2f} ms")
    print(f"Median wall time       : {median_wall:8.2f} ms")
    print(f"P95 wall time          : {p95_wall:8.2f} ms")
    print(f"Minimum wall time      : {minimum_wall:8.2f} ms")
    print(f"Maximum wall time      : {maximum_wall:8.2f} ms")
    print("-" * 56)
    print(f"Average preprocess     : {average_preprocess:8.2f} ms")
    print(f"Average inference      : {average_inference:8.2f} ms")
    print(f"Average postprocess    : {average_postprocess:8.2f} ms")
    print("-" * 56)
    print(f"Wall-time FPS          : {wall_fps:8.2f} FPS")
    print(f"Total throughput       : {throughput_fps:8.2f} FPS")
    print(f"Person detections      : {detection_counts[-1]}")
    print("=" * 56)

    if wall_fps >= 10.0:
        print("[PASS] NCNN model reaches the 10 FPS target.")
    elif wall_fps >= 8.0:
        print("[BORDERLINE] Close to 10 FPS; pipeline optimization may be enough.")
    else:
        print("[FAIL] Model inference itself is below the 10 FPS target.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
