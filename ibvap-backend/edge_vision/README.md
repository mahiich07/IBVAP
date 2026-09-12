# IBVAP Edge Computer Vision Inference Pipeline

Production-ready computer vision pipeline built for **NVIDIA Jetson (Orin Nano / AGX Orin / Xavier NX)** edge compute nodes processing multi-camera RTSP surveillance streams.

---

## 1. Architecture Overview

```
                      ┌─────────────────────────────────────────┐
                      │        RTSP Camera Stream (H.264)       │
                      └────────────────────┬────────────────────┘
                                           │
                        [GStreamer HW Decode (nvv4l2decoder)]
                                           │
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │    Low-Light Enhancement Engine         │
                      │    (MSRCR / Zero-DCE Curve / CLAHE)     │
                      └────────────────────┬────────────────────┘
                                           │ Enhanced Frame
                                           ▼
                      ┌─────────────────────────────────────────┐
                      │    YOLOv8 / YOLOv11 Object Detector     │
                      │    (TensorRT / ONNX / OpenCV DNN)       │
                      └─────────────┬─────────────┬─────────────┘
                                    │             │
                    [Persons/Vehicles]           [Vehicles Only]
                                    │             │
                                    ▼             ▼
                      ┌──────────────────┐  ┌──────────────────┐
                      │ ByteTrack MOT    │  │ ANPR Pipeline    │
                      │ (Kalman Filters, │  │ (Morphology,     │
                      │  Persistent IDs) │  │  Segmentation,   │
                      │                  │  │  OCR & Watchlist)│
                      └────────┬─────────┘  └────────┬─────────┘
                               │                     │
                               └──────────┬──────────┘
                                          │
                                          ▼
                      ┌─────────────────────────────────────────┐
                      │  YuNet Face Detection & Landmarks       │
                      └───────────────────┬─────────────────────┘
                                          │
                                          ▼
                      ┌─────────────────────────────────────────┐
                      │ Tactical HUD & WebSocket Telemetry JSON │
                      └─────────────────────────────────────────┘
```

---

## 2. Core Modules

### A. RTSP Stream Ingestion (`stream_handler.py`)
- Multi-threaded decoupled frame grabber loop with zero-copy ring buffer to eliminate RTSP network buffering lag.
- Jetson hardware accelerated GStreamer pipeline string:
  ```bash
  rtspsrc location=rtsp://camera.local:554/live latency=100 ! \
    rtph264depay ! h264parse ! \
    nvv4l2decoder ! nvvidconv ! \
    video/x-raw, format=BGRx ! videoconvert ! \
    video/x-raw, format=BGR ! appsink drop=1 sync=false max-buffers=1
  ```
- Automatic reconnection with exponential backoff on frame drops or network jitter.

### B. Object Detection & ByteTrack (`object_tracker.py`)
- **YOLOv8 / YOLOv11 Support**: Loads ONNX or TensorRT models with Letterbox preprocessing and multi-class NMS.
- **ByteTrack MOT**:
  - Kalman Filter state estimation ($[x, y, s, r, \dot{x}, \dot{y}, \dot{s}]$) for accurate trajectory prediction.
  - Two-stage association: high-confidence detections first, followed by low-confidence detections using the Hungarian algorithm (`linear_sum_assignment`).
  - Generates persistent Track IDs (`TRK-XXXX`), velocity vectors, and breadcrumb trajectory history.

### C. Face Detection & ANPR (`face_anpr.py`)
- **Face Detection**: Uses OpenCV's native `cv2.FaceDetectorYN` (YuNet) for sub-millisecond face localization on edge devices with 5 facial landmarks (eyes, nose, mouth corners).
- **ANPR (Automatic Number Plate Recognition)**:
  - Plate boundary extraction using vertical Sobel gradient edges and morphological aspect-ratio filtering.
  - Character segmentation: Adaptive thresholding, contrast normalization, and contour-based character isolation.
  - OCR Engine: Pluggable OCR interface supporting EasyOCR / PaddleOCR, with a high-accuracy OpenCV pattern recognition fallback.
  - Automated watchlist matching (stolen vehicles, smuggling alerts).

### D. Low-Light Image Enhancement (`low_light_enhancement.py`)
- **Multi-Scale Retinex with Color Restoration (MSRCR)**:
  $$\log(R) = \log(I) - \sum_{k} w_k \log(I * G_k)$$
  Restores visibility in high-contrast dark environments using multi-scale Gaussian filters ($\sigma \in \{15, 80, 250\}$).
- **Zero-DCE (Zero-Reference Deep Curve Estimation)**:
  $$LE_n(x) = LE_{n-1}(x) + \mathcal{A}_n(x) \cdot LE_{n-1}(x) \cdot (1 - LE_{n-1}(x))$$
  Iterative quadratic curve enhancement brightening dark shadow areas without highlight over-saturation.
- **Adaptive CLAHE**: Real-time LAB L-channel equalization for low-compute edge profiles (<2ms latency).

### E. Edge Model Optimization (`optimization/`)
- **ONNX Model Export (`export_onnx.py`)**: Converts PyTorch `.pt` checkpoints to ONNX with dynamic shapes and FP16 half-precision.
- **TensorRT Compilation (`trt_compiler.py`)**:
  - Compiles ONNX graphs into `.engine` binaries.
  - Precision modes: FP16 (`--fp16`) and INT8 (`--int8`).
  - Jetson DLA Core scheduling: offloads detection to DLA 0 / DLA 1 while GPU processes video frames.
  - Generates optimized `trtexec` shell compilation scripts.
- **Benchmarking & Profiling (`benchmark.py`)**:
  - Measures latency distribution: Preprocessing, Inference, NMS, ByteTrack.
  - Computes P50, P95, P99 latency and sustained FPS throughput.

---

## 3. Jetson TensorRT Optimization Guide

### Step 1: Export Trained Model to ONNX
```bash
python3 ibvap-backend/edge_vision/optimization/export_onnx.py \
  --weights models/custom_border_yolo.pt \
  --output models/custom_border_yolo.onnx \
  --imgsz 640 \
  --dynamic \
  --half
```

### Step 2: Compile to TensorRT Engine on Jetson
Generate the bash script or run directly with `trtexec`:
```bash
trtexec \
  --onnx=models/custom_border_yolo.onnx \
  --saveEngine=models/custom_border_yolo_fp16.engine \
  --memPoolSize=workspace:4096MiB \
  --fp16 \
  --useDLACore=0 \
  --allowGPUFallback \
  --minShapes=images:1x3x640x640 \
  --optShapes=images:1x3x640x640 \
  --maxShapes=images:4x3x640x640
```

---

## 4. Running Verification Tests

```bash
# Run computer vision pipeline test suite
python3 ibvap-backend/edge_vision/tests/test_vision_pipeline.py

# Run API & streaming server test suite
python3 ibvap-backend/tests/test_api.py
python3 ibvap-backend/tests/test_websocket.py
```
