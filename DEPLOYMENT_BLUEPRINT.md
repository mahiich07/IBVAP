# IBVAP — Edge Hardware Deployment Blueprint & Container Orchestration

**Production Deployment, Jetson Hardware Acceleration, and Edge-First Sync Architecture**

---

## 1. System Topology & Multi-Tier Architecture

IBVAP is designed as a hierarchical, edge-resilient surveillance grid operating across three operational tiers:

```
                            ┌─────────────────────────────────────────┐
                            │    CENTRAL COMMAND / MINISTRY CLOUD     │
                            │    - Macro Fleet Analytics & Archive    │
                            │    - Global Watchlist Synchronization   │
                            └────────────────────▲────────────────────┘
                                                 │
                                     [Encrypted SATCOM Uplink]
                                                 │
                            ┌────────────────────┴────────────────────┐
                            │      BATTALION HQ OPERATIONAL NODE      │
                            │      - Multi-BOP Sector Intelligence    │
                            │      - Tactical Escalation Dispatch     │
                            └────────────────────▲────────────────────┘
                                                 │
                                [Radio / Fiber Mesh: MTLS WireGuard]
                                                 │
 ┌───────────────────────────────────────────────┴───────────────────────────────────────────────┐
 │                       BORDER OUT POST (BOP) EDGE NODE — NVIDIA JETSON                         │
 │                                                                                               │
 │  ┌────────────────────────┐      ┌─────────────────────────┐      ┌────────────────────────┐  │
 │  │  RTSP CCTV Feeds       │      │  NVIDIA Jetson Unit     │      │  Tactical Terminal     │  │
 │  │  - Fixed Thermal Cams  │─────►│  - Hardware Decoders    │─────►│  - React UI (Port 3000)│  │
 │  │  - PTZ Optical Cams    │      │  - DLA & TensorRT Cores │      │  - WebSocket Telemetry │  │
 │  │  - ANPR Bullet Cams    │      │  - SQLite WAL Ledger    │      │  - Zero-Cloud Needed   │  │
 │  └────────────────────────┘      └─────────────────────────┘      └────────────────────────┘  │
 └───────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Core Value Proposition (USP)
- **Zero Hardware Replacement**: Interfaces directly with existing legacy IP/analog camera feeds over standard RTSP/H.264 streams.
- **Autonomous Edge Survivability**: Even when border connectivity is severed by enemy jamming or severe mountain weather, local edge nodes perform uninterrupted threat detection, track moving targets, and log cryptographically immutable audit blocks.

---

## 2. Containerized Multi-Service Orchestration

The platform orchestrates three decoupled microservices via `docker-compose.yml`:

```
                           ┌─────────────────────────────────────────┐
                           │      Nginx Reverse Proxy & Frontend     │
                           │      Container: ibvap-frontend (Port 3000│
                           └────────────────────┬────────────────────┘
                                                │
                          REST /api/v1          │         WS /ws/stream
                                ┌───────────────┴───────────────┐
                                ▼                               ▼
                   ┌─────────────────────────────────────────────────────────┐
                   │        FastAPI REST API & WebSocket Gateway             │
                   │        Container: ibvap-backend (Port 8000)             │
                   └────────────────────────────┬────────────────────────────┘
                                                │
                                    Mounts: /app/data/ibvap.db
                                                │
                                                ▼
     ┌──────────────────────────────────────────────────────────────────────────────────────┐
     │                       Shared Persistent Volume: ibvap-data                           │
     │                       - SQLite WAL Database (Events, Alerts, Cameras, BOPs)          │
     │                       - Tamper-Evident SHA-256 Hash Chain Ledger                     │
     │                       - Edge Sync Queue (Buffered Offline Logs)                      │
     └──────────────────────────────────────────┬───────────────────────────────────────────┘
                                                │
                                    Mounts: /app/data/ibvap.db
                                                │
                                                ▼
                   ┌─────────────────────────────────────────────────────────┐
                   │        Edge Computer Vision Inference Worker            │
                   │        Container: ibvap-edge-worker (NVIDIA Jetson)     │
                   │        - Zero-DCE / MSRCR Low-Light Enhancer            │
                   │        - YOLOv8/v11 TensorRT Object Detector            │
                   │        - ByteTrack Multi-Target Kalman Tracker          │
                   │        - YuNet Edge Face Detector & Landmarks           │
                   │        - ANPR License Plate Character Segmentation & OCR│
                   └─────────────────────────────────────────────────────────┘
```

---

## 3. NVIDIA Jetson Edge Hardware Blueprint

### Target Hardware Profiles

| Hardware Spec | Jetson Orin Nano (8GB) | Jetson Xavier NX (16GB) | Jetson AGX Orin (64GB) |
|:---|:---|:---|:---|
| **Deployment Role** | Isolated Outpost (1-4 Cams) | Remote BOP (4-8 Cams) | Battalion Sector Node (16-32 Cams) |
| **AI Compute** | 40 TOPS | 21 TOPS | 275 TOPS |
| **GPU Architecture** | 1024-core Ampere | 384-core Volta | 2048-core Ampere |
| **DLA Cores** | None (GPU only) | 2x NVDLA | 2x NVDLA v2.0 |
| **Precision Mode** | FP16 / INT8 | FP16 / INT8 | FP16 / INT8 / Sparse |
| **Power Target** | 7W – 15W | 10W – 20W | 15W – 60W |

---

### Step-by-Step Jetson Provisioning

#### Step 1: Maximize Compute Performance & Clocks
Prevent dynamic frequency throttling during high-workload multi-camera inference spikes:
```bash
# Set power profile to MAXN (Maximum compute mode)
sudo nvpmodel -m 0

# Lock CPU/GPU/EMC clocks to maximum frequency and activate cooling fans
sudo jetson_clocks
```

#### Step 2: Provision NVIDIA Container Runtime
Run the automated provisioning script provided in `scripts/setup_jetson_runtime.sh`:
```bash
sudo chmod +x scripts/setup_jetson_runtime.sh
sudo ./scripts/setup_jetson_runtime.sh
```
This script:
1. Adds the NVIDIA container repository.
2. Installs `nvidia-container-toolkit`.
3. Sets `"default-runtime": "nvidia"` in `/etc/docker/daemon.json`.
4. Restarts the Docker daemon.

#### Step 3: Hardware-Accelerated GStreamer Pipeline
The edge worker leverages hardware decoders (`nvv4l2decoder` and `nvvidconv`), bypassing the CPU for RTSP stream decoding:
```
rtspsrc location=rtsp://cam.local:554/live latency=100 ! \
  rtph264depay ! h264parse ! \
  nvv4l2decoder ! nvvidconv ! \
  video/x-raw, format=BGRx ! videoconvert ! \
  video/x-raw, format=BGR ! appsink drop=1 sync=false max-buffers=1
```

#### Step 4: DLA (Deep Learning Accelerator) Core Offloading
On Jetson Xavier NX and AGX Orin units, offload the primary YOLO detection model to DLA Core 0 to leave 100% of the GPU free for ANPR, Face, and Low-Light Enhancement:
```bash
trtexec \
  --onnx=models/custom_border_yolo.onnx \
  --saveEngine=models/custom_border_yolo_dla0.engine \
  --useDLACore=0 \
  --allowGPUFallback \
  --fp16
```

---

## 4. Local-First SQLite Offline Buffering & Sync Protocol

### State Transition Machine

```
      ┌────────────────────────────────────────────────────────┐
      │                   ONLINE (Link >= 50%)                 │
      │   - Real-time WebSocket telemetry to Central Command   │
      │   - Direct cloud sync of all alerts and hash blocks    │
      └───────────────────────────┬────────────────────────────┘
                                  │
                       Link Degrades / Drops
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │             DEGRADED / OFFLINE (Buffer Active)         │
      │   - Detections stored locally in SQLite WAL database   │
      │   - Cryptographic SHA-256 block chain continues intact │
      │   - Events queued in 'edge_sync_buffer' with status:   │
      │     'PENDING', synced_upstream = 0                     │
      │   - Zero frame loss; zero telemetry loss               │
      └───────────────────────────┬────────────────────────────┘
                                  │
                       Link Restored (Link >= 50%)
                                  │
                                  ▼
      ┌────────────────────────────────────────────────────────┐
      │             UPSTREAM DRAIN & SYNCHRONIZATION           │
      │   - Sync engine automatically detects link recovery    │
      │   - Batches of 50 buffered events pushed upstream      │
      │   - Records marked 'SYNCED' with remote receipt        │
      │   - Cryptographic integrity verified across connection │
      └────────────────────────────────────────────────────────┘
```

### Guarantees
1. **Zero Data Loss**: Every event is committed to local persistent storage before network dispatch is attempted.
2. **Hash Continuity**: The SHA-256 block chain links every event back to the previous event (`prev_hash`). When syncing 100 offline events to Central Command, the upstream server validates the cryptographic chain seamlessly without gaps.

---

## 5. Single-Command Deployment Runbook

### Launching on Edge Hardware (Jetson or Linux)
```bash
# 1. Clone repository
git clone <repo_url>
cd Reactproject

# 2. Launch edge stack (auto-detects Jetson hardware)
chmod +x scripts/*.sh
./scripts/run_edge_node.sh
```

### Verifying Active Services
```bash
# Check running containers
docker compose ps

# Inspect edge inference worker logs
docker compose logs -f ibvap-edge-worker

# Trigger manual upstream sync
curl -X POST http://localhost:8000/api/v1/system/sync

# Run cryptographic ledger integrity audit
curl -X POST http://localhost:8000/api/v1/events/verify
```

### Air-Gapped Deployment (Offline Military Outposts)
For deployment to classified border outposts without internet access:
```bash
# On Internet-connected build machine:
docker compose build
docker save -o ibvap_images.tar \
  ibvap-frontend:latest \
  ibvap-backend:latest \
  ibvap-edge-worker:latest

# Transfer ibvap_images.tar via encrypted hardware token to Jetson outpost:
docker load -i ibvap_images.tar
docker compose up -d
```
