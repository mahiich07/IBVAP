# IBVAP Backend — Intelligent Border Video Analytics Platform

A production-ready, military-grade, edge-resilient backend API and real-time streaming server for border surveillance and video analytics using **FastAPI, WebSockets, Python, and SQLite/PostgreSQL**.

---

## Architecture Overview

```
                          ┌────────────────────────┐
                          │   Battalion HQ /       │
                          │   Central Command      │
                          └───────────▲────────────┘
                                      │
                         [Edge Sync Protocol: MTLS]
                                      │
      ┌───────────────────────────────┴───────────────────────────────┐
      │                   BORDER OUT POST (BOP EDGE NODE)             │
      │                                                               │
      │  ┌────────────────────┐            ┌───────────────────────┐  │
      │  │ RTSP Stream        │            │ AI Inference Pipeline │  │
      │  │ Manager            │──Frames───►│ (Human, Vehicle, ANPR,│  │
      │  │ (Proxy & MJPEG)    │            │  Virtual Fence, Night)│  │
      │  └────────────────────┘            └───────────┬───────────┘  │
      │                                                │ Detections   │
      │                                                ▼              │
      │  ┌────────────────────┐            ┌───────────────────────┐  │
      │  │ Tamper-Evident     │◄──Audit───│ Rules & Alert Engine  │  │
      │  │ Hash Chain         │   Events   │ (CRITICAL, HIGH,      │  │
      │  │ (SHA-256 Ledger)   │            │  MEDIUM, LOW)         │  │
      │  └─────────┬──────────┘            └───────────┬───────────┘  │
      │            │                                   │ Alerts       │
      │            ▼                                   ▼              │
      │  ┌────────────────────┐            ┌───────────────────────┐  │
      │  │ Edge-First Sync    │            │ WebSocket Gateway     │  │
      │  │ Buffer (Offline)   │            │ (/ws/stream Telemetry)│  │
      │  └────────────────────┘            └───────────┬───────────┘  │
      └────────────────────────────────────────────────┼──────────────┘
                                                       │ Live Push
                                                       ▼
                                            ┌─────────────────────┐
                                            │ React Frontend      │
                                            │ Command Dashboard   │
                                            └─────────────────────┘
```

---

## Core Modules

### 1. RTSP Stream Manager (`services/rtsp_manager.py`)
- Ingests, proxies, and manages camera streams from edge outposts.
- Provides real-time tactical frame rendering with military HUD overlay (timestamps, crosshairs, camera ID, BOP sector, target bounding boxes).
- Supports live multipart MJPEG video streaming at `/api/v1/cameras/{id}/stream`.
- Continuous camera FPS tracking and health status monitoring (`ONLINE`, `DEGRADED`, `OFFLINE`).

### 2. AI Inference Pipeline Service (`services/ai_pipeline.py`)
- Simulates and integrates edge AI detection pipelines running locally on edge nodes:
  - **Human Detection**: Normalized bounding box `[x, y, w, h]`, confidence `90-99%`, speed vector.
  - **Vehicle Detection**: Bounding box, vehicle category (Heavy Utility, Civilian, Armored).
  - **ANPR (Automatic Number Plate Recognition)**: Plate string extraction (e.g. `DL01AB1234`, `JK02BC9871`) and watchlist matching.
  - **Virtual Fence Intrusion**: Bounding-box intersection with defined boundary polylines (Sector 3 fence line).
  - **Night-Movement**: Thermal signature centroid tracking in low-light environments.

### 3. Rules & Alert Engine (`services/rules_engine.py`)
- Evaluates detection telemetry against geofenced border zones.
- Real-time severity classification:
  - `CRITICAL`: Virtual fence breach / boundary line penetration.
  - `HIGH`: Unrecognized vehicle near access gate or thermal night movement.
  - `MEDIUM`: Suspicious loitering in buffer zone (>3 mins).
  - `LOW`: Authorized biometric / patrol face scan.
- Dispatches alert events atomically to:
  1. Tamper-evident hash chain logger.
  2. Edge sync buffer (if link degraded).
  3. WebSocket gateway broadcaster.

### 4. WebSocket Gateway (`services/websocket_gateway.py` & `routers/websocket.py`)
- Connects to frontend clients via `WS /ws/stream`.
- Real-time broadcasts:
  - `alert_telemetry`: Threat alerts with severity, zone, and hash linkage.
  - `detection_frame`: Live bounding boxes, track IDs, confidence, and camera FPS updates.
  - `system_status`: Node health, connectivity metrics, and sync heartbeat.
- Client ping/pong heartbeat and connection recovery.

### 5. Tamper-Evident Event Logging (`services/hash_chain.py`)
- Secures all recorded event metadata using cryptographic SHA-256 block chaining:
  - `GENESIS_HASH` = `"0000000000000000000000000000000000000000000000000000000000000000"`
  - `event_hash = SHA-256(event_id|timestamp|type|camera_id|bop_id|severity|payload|prev_hash)`
- Provides automated verification (`POST /api/v1/events/verify`) that audits the full chain and detects any tampering attempt down to the exact block ID.

### 6. Edge-First Sync Engine (`services/edge_sync.py`)
- Operates independently on edge nodes with local SQLite storage.
- Automatically buffers events in `edge_sync_buffer` during low-bandwidth or offline periods (`synced_upstream = 0`).
- Automatically drains the buffer and syncs upstream to Battalion/Central Command when connectivity is restored to `ONLINE`.
- Allows testing link conditions via `POST /api/v1/system/simulate-link`.

---

## Directory Structure

```
ibvap-backend/
├── main.py                     # FastAPI app with CORS, lifespan events, background simulator, routers
├── config.py                   # App configuration, security settings, JWT secret, database paths
├── database.py                 # SQLite connection manager, schema initialization, seed data
├── models/
│   ├── __init__.py
│   ├── domain.py               # Domain enums & data structures (Severity, AlertStatus, CameraStatus)
│   └── schemas.py              # Pydantic v2 schemas for API requests and responses
├── services/
│   ├── __init__.py
│   ├── auth_service.py         # Operator JWT generation, verification, and password hashing
│   ├── rtsp_manager.py         # RTSP/IP stream ingestion, simulated camera frame proxy & health monitor
│   ├── ai_pipeline.py          # Edge AI detection pipeline (Human, Vehicle, ANPR, Virtual Fence, Night Vision)
│   ├── rules_engine.py         # Geofencing evaluator, rule engine, alert generator (CRITICAL, HIGH, MEDIUM, LOW)
│   ├── hash_chain.py           # Cryptographic SHA-256 tamper-evident event block chain & audit verification
│   ├── edge_sync.py            # Edge-first offline buffering, sync protocol, link degradation simulator
│   └── websocket_gateway.py    # Multi-client WebSocket connection manager, telemetry & detection broadcaster
├── routers/
│   ├── __init__.py
│   ├── auth.py                 # POST /api/v1/auth/login, GET /api/v1/auth/me
│   ├── cameras.py              # GET /api/v1/cameras, GET /api/v1/cameras/{id}, GET /api/v1/cameras/{id}/snapshot, /stream
│   ├── alerts.py               # GET /api/v1/alerts, POST /api/v1/alerts/{id}/acknowledge
│   ├── events.py               # GET /api/v1/events, POST /api/v1/events/verify
│   ├── system.py               # GET /api/v1/system/health, POST /api/v1/system/sync, POST /api/v1/system/simulate-link
│   └── websocket.py            # WS /ws/stream real-time broadcast endpoint
├── tests/
│   ├── __init__.py
│   ├── test_api.py             # Comprehensive automated test suite verifying all 6 modules and endpoints
│   └── test_websocket.py       # WebSocket Gateway unit test
├── requirements.txt            # Python dependencies
└── README.md                   # Platform documentation
```

---

## API Endpoints Reference

### 1. Authentication
- `POST /api/v1/auth/login`
  - **Body**: `{"operator_id": "CMD-OP-8042-IN", "password": "defense2026"}`
  - **Response**: JWT access token + operator profile.
- `GET /api/v1/auth/me`
  - **Headers**: `Authorization: Bearer <token>`
  - **Response**: Current authenticated operator details.

### 2. Cameras Fleet & Streaming
- `GET /api/v1/cameras`
  - **Response**: All 8+ cameras across border outposts with live FPS, resolution, AI mode, and recent detections.
- `GET /api/v1/cameras/{id}`
  - **Response**: Specific camera telemetry.
- `GET /api/v1/cameras/{id}/snapshot`
  - **Response**: Tactical JPEG snapshot with HUD overlay (military coordinates, crosshairs, AI bounding boxes).
- `GET /api/v1/cameras/{id}/stream`
  - **Response**: Multipart live MJPEG stream preview.

### 3. Threat Alerts
- `GET /api/v1/alerts?severity=CRITICAL&status=Active&limit=50&offset=0`
  - **Query Params**: `severity`, `status`, `search`, `limit`, `offset`.
  - **Response**: Paginated and filtered alert records with SHA-256 hash chains.
- `POST /api/v1/alerts/{id}/acknowledge`
  - **Body**: `{"action": "Acknowledged"}` (or `"Escalated"`, `"Resolved"`)
  - **Response**: Status update confirmation with timestamp and operator audit log.

### 4. Tamper-Evident Events Ledger
- `GET /api/v1/events?limit=50`
  - **Response**: Audit log of security detections with SHA-256 block hashes and previous block linkages.
- `POST /api/v1/events/verify`
  - **Response**: Cryptographic integrity report (`is_valid: true`, total blocks checked, latest hash).

### 5. System Health & Edge Sync
- `GET /api/v1/system/health`
  - **Response**: Microservices latency, edge node connectivity %, active threats, online cameras, and sync buffer status.
- `POST /api/v1/system/sync`
  - **Response**: Flushes buffered offline events upstream to Central Command.
- `POST /api/v1/system/simulate-link`
  - **Body**: `{"bop_id": "BOP-03", "status": "DEGRADED", "connectivity_percent": 30, "latency_ms": "310ms"}`
  - **Response**: Simulates link degradation and tests offline event buffering.

### 6. Real-Time WebSocket Gateway
- `WS /ws/stream`
  - **Protocol**: Real-time bidirectional WebSocket stream.
  - **Broadcasted Events**:
    - `alert_telemetry`: Real-time threat violations.
    - `detection_frame`: Live bounding boxes, FPS telemetry, track IDs.
    - `system_status`: Edge node link updates.

---

## Running the Backend

### Start Server
```bash
cd ibvap-backend
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Interactive OpenAPI documentation is available at:
`http://localhost:8000/docs`

### Run Automated Tests
```bash
python3 ibvap-backend/tests/test_api.py
python3 ibvap-backend/tests/test_websocket.py
```
