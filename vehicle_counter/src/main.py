import cv2
from ultralytics import YOLO

VIDEO_PATH = "input/traffic.mp4"
OUTPUT_PATH = "output/output.mp4"

def main():
    model = YOLO("models/yolov8n.pt")

    cap = cv2.VideoCapture(VIDEO_PATH)

    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(
        OUTPUT_PATH,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    line_y = height // 2
    counted_ids = set()
    total_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(frame, persist=True, classes=[2,3,5,7])

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy()

            for box, track_id in zip(boxes, ids):
                x1, y1, x2, y2 = map(int, box)
                center_y = (y1 + y2) // 2

                if center_y > line_y and track_id not in counted_ids:
                    counted_ids.add(track_id)
                    total_count += 1

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"ID {int(track_id)}",
                            (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0,255,0),
                            2)

        cv2.line(frame, (0, line_y), (width, line_y), (0,0,255), 2)

        cv2.putText(frame, f"Total Count: {total_count}",
                    (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0,0,255),
                    3)

        out.write(frame)
        cv2.imshow("Vehicle Counter", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
