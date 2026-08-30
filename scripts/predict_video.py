"""
Video & Real-Time Webcam Inference Script for Sea Debris Detection.
Streams frames sequentially to preserve memory, overlays detections and real-time HUD,
and encodes output video stream or live display.

Usage:
    # Process video file
    python scripts/predict_video.py --model models/best.pt --source underwater_video.mp4 --output outputs/predictions/output.mp4

    # Real-time webcam stream
    python scripts/predict_video.py --model models/best.pt --source 0 --show
"""

import argparse
import sys
import os
import time
import cv2
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.device_utils import select_device
from utils.visualization import draw_detections, draw_fps_and_stats


def main():
    parser = argparse.ArgumentParser(description="Run YOLO11 Marine Debris detection on video or live camera.")
    parser.add_argument("--model", type=str, required=True, help="Path to trained model weights (.pt)")
    parser.add_argument("--source", type=str, required=True, help="Path to video file or webcam index (e.g. '0')")
    parser.add_argument("--output", type=str, default=None, help="Path to save output video (e.g. outputs/predictions/annotated.mp4)")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference resolution")
    parser.add_argument("--device", type=str, default="auto", help="Device ('auto', 'cpu', '0')")
    parser.add_argument("--show", action="store_true", help="Display live video window")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional maximum number of frames to process")
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        print(f"[Error] Model checkpoint not found: {model_path}")
        sys.exit(1)

    # Determine input source
    is_webcam = False
    if args.source.isdigit():
        source = int(args.source)
        is_webcam = True
    else:
        source_path = Path(args.source).resolve()
        if not source_path.exists():
            print(f"[Error] Video source not found: {source_path}")
            sys.exit(1)
        source = str(source_path)

    selected_device = select_device(args.device)

    from ultralytics import YOLO
    model = YOLO(str(model_path))
    class_names = model.names

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[Error] Could not open video source: {args.source}")
        sys.exit(1)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else -1

    # Setup VideoWriter if output requested or for non-webcam source
    out_writer = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(str(out_path), fourcc, input_fps, (width, height))
    elif not is_webcam and not args.show:
        default_out = Path("outputs/predictions") / f"annotated_{Path(args.source).name}"
        default_out.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(str(default_out), fourcc, input_fps, (width, height))
        args.output = str(default_out)

    print("\n" + "=" * 70)
    print(" 🎥 YOLO11 Sea Debris Video Stream Processor")
    print("=" * 70)
    print(f"📦 Model:  {model_path.name}")
    print(f"📹 Source: {args.source} ({width}x{height} @ {input_fps:.1f} FPS, Total: {total_frames if total_frames > 0 else 'Live Stream'} frames)")
    print(f"💾 Output: {args.output or 'None (Display only)'}")
    print("Press 'q' or 'ESC' in the display window to exit anytime.")
    print("=" * 70 + "\n")

    frame_idx = 0
    fps_history = []

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                print(f"[Info] Reached maximum requested frames ({args.max_frames}). Stopping.")
                break

            start_t = time.perf_counter()

            # Run YOLO inference
            results = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=selected_device,
                verbose=False
            )[0]

            boxes_xyxy = []
            scores = []
            class_ids = []

            for box in results.boxes:
                boxes_xyxy.append(box.xyxy[0].cpu().numpy().tolist())
                scores.append(float(box.conf[0]))
                class_ids.append(int(box.cls[0]))

            annotated_frame = draw_detections(frame, boxes_xyxy, scores, class_ids, class_names)

            elapsed = time.perf_counter() - start_t
            curr_fps = 1.0 / elapsed if elapsed > 0 else 0.0
            fps_history.append(curr_fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            annotated_frame = draw_fps_and_stats(
                annotated_frame,
                fps=avg_fps,
                detection_count=len(boxes_xyxy),
                model_name=f"YOLO11 ({model_path.stem})"
            )

            if out_writer is not None:
                out_writer.write(annotated_frame)

            if args.show:
                cv2.imshow("YOLO11 Marine Debris Live Stream", annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("\n[User Interrupted] Exiting video stream.")
                    break

            if frame_idx % 30 == 0:
                progress_str = f"Frame {frame_idx}/{total_frames}" if total_frames > 0 else f"Frame {frame_idx}"
                print(f"[{progress_str}] Processing at {avg_fps:.1f} FPS | Detected Debris: {len(boxes_xyxy)}")

    finally:
        cap.release()
        if out_writer is not None:
            out_writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print("\n" + "=" * 70)
    print(f"✅ Video processing complete. Processed {frame_idx} frames.")
    if args.output:
        print(f"🎬 Output video saved to: {Path(args.output).resolve()}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
