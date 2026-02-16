# Kriya-TAI: Computer Vision Detection Project

## Project Overview

Kriya-TAI is a computer vision project focused on object and vehicle detection using YOLOv8 (You Only Look Once v8) deep learning models. The system processes video data through detection pipelines to identify and annotate objects and vehicles in traffic scenarios.

## Project Structure

```
kriya-tai/
├── README.md                        # Project documentation
├── notebooks/
│   ├── object_detection.ipynb      # General object detection pipeline
│   └── vehicle_detection.ipynb     # Vehicle-specific detection pipeline
├── data/
│   ├── image/
│   │   ├── raw/                    # Reserved for raw image data
│   │   └── processed/              # Reserved for processed images
│   └── video/
│       ├── raw/                    # Input video files
│       │   ├── 4K Video of Highway Traffic! - Nicholas Abraham-Raegan Martinez (720p, h264).mp4
│       │   └── Traffic_video_demo.mp4
│       └── processed/              # Output videos after detection
│           ├── object_detection.mp4
│           ├── vehicle_detection.mp4
│           └── vehicle_count.mp4
├── models/                         # YOLO model weights (referenced)
├── myvenv/                         # Python virtual environment
└── .git/                           # Version control repository
```

## Environment Setup

- **Python Version**: 3.13
- **Virtual Environment**: `myvenv/`
- **Activation Command**: `.\myvenv\Scripts\activate`
- **Package Manager**: pip

## Key Dependencies Installed

- **Object Detection**: ultralytics (YOLO models)
- **Computer Vision**: opencv-python (cv2), torch, torchvision
- **Data Processing**: numpy, scipy, pandas, polars
- **Visualization**: matplotlib, Pillow
- **Scientific Computing**: scikit-learn, networkx, sympy
- **Other Utilities**: requests, pyyaml, psutil, filelock

## Project Notebooks

### 1. object_detection.ipynb

Detects all objects in video frames using YOLOv8n model.

**Workflow**:

- Loads pretrained YOLOv8n model (`../models/yolov8n.pt`)
- Reads input video from `../data/video/raw/Traffic_video_demo.mp4`
- Extracts video properties (width, height, FPS)
- Processes each frame with YOLO detection
- Draws bounding boxes and labels on detected objects
- Writes annotated frames to `../data/video/processed/object_detection.mp4`

**Status**: Fully implemented and executed (14 code cells executed)

### 2. vehicle_detection.ipynb

Detects and filters only vehicle classes from video frames.

**Workflow**:

- Loads YOLOv8n model and extracts class names
- Filters for vehicle classes: bike, truck, bus, bicycle, motorcycle, car
- Reads same input video
- Extracts video properties
- Processes frames with YOLO, applying class filter
- Writes vehicle-only detections to `../data/video/processed/vehicle_detection.mp4`

**Status**: Fully implemented and executed (12 code cells executed)

### 3. vehicle_count.ipynb

Detects vehicles and counts each type per frame, displaying cumulative counts on video overlay.

**Workflow**:

- Loads YOLOv8n model and extracts class names
- Filters for vehicle classes: bike, truck, bus, bicycle, motorcycle, car
- Reads same input video with 0.4 confidence threshold
- Extracts video properties
- For each frame:
  - Detects vehicles using filtered classes
  - Counts occurrences of each vehicle type
  - Displays live count for: bicycle, car, motorcycle, bus, truck
  - Renders counts as text overlay on frame (green text, top-left position)
- Writes annotated frames to `../data/video/processed/vehicle_count.mp4`

**Status**: Fully implemented and executed (13 code cells executed, 1 empty cell)

## Data Status

- **Raw Videos**: 2 files (traffic video demos in 720p H.264 format)
- **Processed Videos**: 3 files
  - object_detection.mp4 (from object_detection.ipynb)
  - vehicle_detection.mp4 (from vehicle_detection.ipynb)
  - vehicle_count.mp4 (from vehicle_count.ipynb with live vehicle counts overlaid)
- **Images**: No data yet (directories reserved for future use)

## Model Information

- **Model Used**: YOLOv8n (Nano version)
- **Model Location**: `../models/yolov8n.pt` (referenced from notebooks)
- **Input**: Traffic video footage (720p resolution)
- **Output Format**: MP4 video with annotated detections

## Getting Started

1. Start Windows Command Prompt or PowerShell
2. Navigate to project directory: `cd "d:\4-2 Projects\kriya-tai"`
3. Activate virtual environment: `.\myvenv\Scripts\activate`
4. Launch Jupyter: `jupyter notebook notebooks/`
5. Open `object_detection.ipynb` or `vehicle_detection.ipynb`
6. Run cells sequentially to process videos
7. Check `data/video/processed/` for output videos

## Notes

- Video processing can be time-consuming depending on video length and system resources
- YOLOv8n is the nano model (lightweight, fast inference)
- All paths in notebooks are relative to the notebooks/ directory
- Output videos maintain the same FPS as input videos
