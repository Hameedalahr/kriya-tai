# 🚦 Dual-Lane Vehicle Detection and Counting System

## 📌 Project Overview

This project is a smart traffic monitoring system built using:

- YOLOv8 (Ultralytics)
- OpenCV
- Python

It detects vehicles in a road video and counts them separately in:

- LEFT lane
- RIGHT lane

The system tracks vehicles using ByteTrack and displays:

- Bounding boxes
- Vehicle class
- Track ID
- Lane information
- Real-time FPS
- Lane-wise vehicle counts
- Total vehicle count

---

# 📁 Project Folder Structure

```
traffic-vehicle-counter/
│
├── models/
│   └── yolov8n.pt
│
├── data/
│   └── video/
│       ├── raw/
│       │   └── traffic_video_demo.mp4
│       │
│       └── processed/
│           └── output_lane_count.mp4
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 🛠️ Installation Guide

## 1️⃣ Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

## 2️⃣ Install Dependencies

```bash
pip install ultralytics opencv-python numpy
```

OR run the automatic installer in the script.

---

# ⚙️ Configuration Parameters

Inside CONFIG:

```python
CONFIG = {
    'model_name': 'yolov8n.pt',
    'model_path': str(MODEL_PATH),
    'confidence_threshold': 0.5,
    'tracker': 'bytetrack.yaml',
    'device': 0,
    'target_classes': ['motorcycle', 'bicycle', 'car', 'truck'],
    'video_input': str(VIDEO_INPUT),
    'video_output': str(VIDEO_OUTPUT),
    'display_fps': True,
    'display_roi': True,
    'draw_track_ids': True,
    'left_count_line_y': 350,
    'right_count_line_y': 400,
}
```

### Explanation:

| Parameter | Purpose |
|------------|----------|
| model_name | YOLO model name |
| model_path | Path where model is stored |
| confidence_threshold | Minimum detection confidence |
| tracker | Tracking algorithm |
| device | 0 = GPU, cpu = CPU |
| target_classes | Vehicles to detect |
| video_input | Input video path |
| video_output | Output video path |
| display_fps | Show FPS on screen |
| display_roi | Show lane split line |
| draw_track_ids | Show tracking IDs |
| left_count_line_y | Left lane counting line |
| right_count_line_y | Right lane counting line |

---

# 🧠 How The System Works (Step-by-Step)

## 1️⃣ Path Setup

```python
PROJECT_ROOT = Path.cwd().parent
```

Defines root directory.

Creates:
- models folder
- output folder if not existing

---

## 2️⃣ VehicleDetector Class

This is the core logic of the system.

---

## 🔹 CLASS_NAMES

Maps COCO dataset class IDs to readable vehicle names.

Only includes:
- bicycle
- car
- motorcycle
- truck

---

## 🔹 __init__()

Initializes:

- YOLO model
- Tracking sets
- Lane counters
- Config parameters

Loads model using:

```python
self.model = YOLO(config['model_path'])
```

---

## 🔹 get_lane_for_bbox()

Determines lane by:

- Finding bounding box center X
- Comparing with frame midpoint

If:
- center < midpoint → LEFT
- center > midpoint → RIGHT

---

## 🔹 get_vehicle_class()

Checks if detected object is in target classes.

Filters unwanted classes like:
- person
- dog
- chair

---

## 🔹 process_frame()

This is the detection pipeline:

1. Run YOLOv8 tracking:
   ```python
   results = self.model.track(...)
   ```

2. Extract:
   - class ID
   - confidence
   - tracking ID
   - bounding box

3. Filter target vehicle classes

4. Determine lane

5. Count vehicle (once per ID)

6. Draw bounding boxes

Returns annotated frame.

---

## 🔹 _draw_detections()

Draws:

- Vertical lane split line
- Bounding boxes
- Labels
- Track IDs
- Lane indicators

---

## 🔹 draw_counts()

Displays:

- LEFT lane count
- RIGHT lane count
- Category-wise counts
- Total vehicles
- FPS

Uses transparent overlay for professional UI.

---

## 🔹 reset_counts()

Clears:

- Tracking memory
- Vehicle counters

Used before processing new video.

---

# 🎥 process_video() Function

This function controls video processing.

Steps:

1. Open video
2. Read resolution
3. Initialize video writer
4. Loop frame-by-frame
5. Calculate FPS
6. Call:
   - detector.process_frame()
   - detector.draw_counts()
7. Write output frame
8. Display progress
9. Release resources
10. Print final statistics

---

# 📊 Final Output Includes

After processing:

- LEFT lane vehicle count
- RIGHT lane vehicle count
- Category-wise counts
- Total vehicles
- Average FPS
- Output video file location

---

# 🔍 FPS Calculation

FPS is calculated using:

```python
cv2.getTickCount()
```

Formula:

```
FPS = 1 / time_difference
```

Higher FPS = better performance.

---

# 🚗 Vehicle Classes Detected

The system detects:

- Motorcycle
- Bicycle
- Car
- Truck

Other classes are ignored.

---

# 📈 Sample Output

```
LEFT LANE: 42 vehicles
  Cars: 25
  Trucks: 5
  Motorcycles: 8
  Bicycles: 4

RIGHT LANE: 37 vehicles
  Cars: 22
  Trucks: 3
  Motorcycles: 9
  Bicycles: 3

TOTAL: 79 vehicles
Average FPS: 23.5
```

---

# 🚀 How To Run

```bash
python main.py
```

Output video will be saved to:

```
data/video/processed/output_lane_count.mp4
```

---

# 🔥 System Features

✔ Real-time vehicle detection  
✔ ByteTrack multi-object tracking  
✔ Dual-lane separation  
✔ Separate vehicle counts per lane  
✔ Category-wise counting  
✔ FPS monitoring  
✔ Automatic folder creation  
✔ Clean configuration system  

---

# 📌 Future Improvements

- Angled lane split line
- Line crossing-based counting
- Speed estimation
- Emergency vehicle detection
- Auto-rickshaw custom training
- Traffic density estimation
- Web dashboard integration

---

# 👨‍💻 Technologies Used

- Python
- OpenCV
- Ultralytics YOLOv8
- NumPy
- ByteTrack

---

# 🎯 Conclusion

This project demonstrates:

- Real-time computer vision
- Object detection
- Multi-object tracking
- Lane-wise traffic analysis
- Practical smart traffic system design

It can be extended into:

- Smart traffic signal control
- Emergency vehicle priority system
- Urban traffic analytics dashboard
- AI-based congestion monitoring system

---

# 📜 License

Educational / Research Use

---

# ✨ Author

Traffic Monitoring System using YOLOv8