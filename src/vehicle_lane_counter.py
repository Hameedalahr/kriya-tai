from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


COCO_VEHICLE_CLASSES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    7: "truck",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lane based vehicle detection and counting using YOLOv8n")
    p.add_argument("--model", default="models/yolov8n.pt", help="Path to YOLOv8n .pt file")
    p.add_argument("--video_in", default="data/video/raw/Traffic_video_demo.mp4", help="Input video path")
    p.add_argument("--video_out", default="data/video/processed/vehicle_count.mp4", help="Output video path")

    p.add_argument("--lane", choices=["left", "right"], default="left", help="Lane ROI to count vehicles in")
    p.add_argument("--count_mode", choices=["line_crossing", "unique_in_lane"], default="line_crossing",
                   help="line_crossing is recommended for correct unique counts")
    p.add_argument("--line_y_ratio", type=float, default=0.60, help="Horizontal counting line position as ratio of height")
    p.add_argument("--direction", choices=["down", "up", "both"], default="down",
                   help="Which direction counts for line crossing")

    p.add_argument("--conf", type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--iou", type=float, default=0.50, help="IOU threshold")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    model_path = Path(args.model)
    video_in = Path(args.video_in)
    video_out = Path(args.video_out)
    video_out.parent.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not video_in.exists():
        raise FileNotFoundError(f"Video not found: {video_in}")

    model = YOLO(str(model_path))

    cap = cv2.VideoCapture(str(video_in))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {video_in}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps) if fps and fps > 0 else 30

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_out), fourcc, fps, (width, height))
    if not out.isOpened():
        raise RuntimeError(f"Could not open output writer: {video_out}")

    class_ids = list(COCO_VEHICLE_CLASSES.keys())

    counts = {name: 0 for name in COCO_VEHICLE_CLASSES.values()}
    counted_ids = {name: set() for name in COCO_VEHICLE_CLASSES.values()}
    prev_center = {}  # track_id -> (cx, cy)

    count_line_y = int(height * args.line_y_ratio)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if args.lane == "left":
            roi_x1, roi_x2 = 0, width // 2
        else:
            roi_x1, roi_x2 = width // 2, width

        cv2.line(frame, (width // 2, 0), (width // 2, height), (0, 255, 255), 2)

        if args.count_mode == "line_crossing":
            cv2.line(frame, (roi_x1, count_line_y), (roi_x2, count_line_y), (255, 0, 0), 2)

        results = model.track(
            frame,
            persist=True,
            verbose=False,
            classes=class_ids,
            conf=args.conf,
            iou=args.iou,
        )

        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes

            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                vehicle_type = COCO_VEHICLE_CLASSES.get(cls_id)
                if vehicle_type is None:
                    continue

                x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if not (roi_x1 <= cx <= roi_x2):
                    continue

                track_id = None
                if boxes.id is not None:
                    track_id = int(boxes.id[i].item())

                if track_id is not None:
                    if args.count_mode == "unique_in_lane":
                        if track_id not in counted_ids[vehicle_type]:
                            counted_ids[vehicle_type].add(track_id)
                            counts[vehicle_type] += 1

                    else:
                        prev = prev_center.get(track_id)
                        prev_center[track_id] = (cx, cy)

                        if prev is not None and track_id not in counted_ids[vehicle_type]:
                            _, prev_cy = prev

                            crossed_down = prev_cy < count_line_y and cy >= count_line_y
                            crossed_up = prev_cy > count_line_y and cy <= count_line_y

                            should_count = False
                            if args.direction == "down" and crossed_down:
                                should_count = True
                            elif args.direction == "up" and crossed_up:
                                should_count = True
                            elif args.direction == "both" and (crossed_down or crossed_up):
                                should_count = True

                            if should_count:
                                counted_ids[vehicle_type].add(track_id)
                                counts[vehicle_type] += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{vehicle_type}" if track_id is None else f"{vehicle_type} id:{track_id}"
                cv2.putText(frame, label, (x1, max(20, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        y = 30
        cv2.putText(frame, f"Lane: {args.lane}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y += 30
        cv2.putText(frame, f"Mode: {args.count_mode}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y += 40

        for k, v in counts.items():
            cv2.putText(frame, f"{k}: {v}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            y += 30

        out.write(frame)

    cap.release()
    out.release()

    print("===== FINAL RESULTS =====")
    print("Selected Lane:", args.lane)
    print("Count Mode:", args.count_mode)
    print("Total Vehicles:", sum(counts.values()))
    print("Motorcycles:", counts["motorcycle"])
    print("Bicycles:", counts["bicycle"])
    print("Cars:", counts["car"])
    print("Trucks:", counts["truck"])
    print("Output saved at:", str(video_out))


if __name__ == "__main__":
    main()