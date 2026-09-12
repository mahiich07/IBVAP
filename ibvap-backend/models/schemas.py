from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from models.domain import Severity, AlertStatus, BOPStatus, CameraStatus, AIMode

# --- Auth Schemas ---
class LoginRequest(BaseModel):
    operator_id: str = Field(..., example="CMD-OP-8042-IN")
    password: str = Field(..., example="defense2026")

class OperatorInfo(BaseModel):
    operator_id: str
    full_name: str
    role: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    operator: OperatorInfo

# --- Camera & BOP Schemas ---
class CameraOut(BaseModel):
    id: str
    name: str
    bop: str
    bop_id: str
    status: str
    fps: int
    res: str
    ai: str
    type: str
    lastSeen: str
    detections: List[str] = []

class BOPOut(BaseModel):
    id: str
    name: str
    sector: str
    status: str
    cameras: int
    activeCams: int
    alertsCount: int
    connectivity: int
    edgeNode: str
    latency: str
    lat: float
    lng: float

# --- Alert Schemas ---
class AlertOut(BaseModel):
    id: str
    severity: str
    type: str
    camera: str
    bop: str
    timestamp: str
    status: str
    confidence: str
    trackingId: str
    zone: str
    description: str
    hash: str
    prevHash: str
    acknowledgedBy: Optional[str] = None
    resolvedAt: Optional[str] = None

class AlertAcknowledgeRequest(BaseModel):
    action: str = Field("Acknowledged", description="Acknowledged, Escalated, or Resolved")
    notes: Optional[str] = None

class AlertAcknowledgeResponse(BaseModel):
    success: bool
    alert_id: str
    status: str
    message: str
    updated_at: str

# --- Event & Hash Chain Schemas ---
class EventOut(BaseModel):
    id: str
    type: str
    camera: str
    bop: str
    timestamp: str
    severity: str
    integrity: str
    hash: str
    prevHash: str
    payload: Optional[Dict[str, Any]] = None

class ChainVerificationResponse(BaseModel):
    is_valid: bool
    total_blocks_checked: int
    genesis_hash: str
    latest_hash: str
    tampered_block_id: Optional[str] = None
    message: str
    verified_at: str

# --- System & Sync Schemas ---
class ServiceHealth(BaseModel):
    name: str
    status: str
    latency: str

class SystemHealthResponse(BaseModel):
    overall_status: str
    edge_node_id: str
    link_status: str
    bandwidth_usage_kbps: float
    total_cameras_online: int
    active_threats: int
    services: List[ServiceHealth]
    bops: List[BOPOut]
    unsynced_events_buffered: int
    tamper_chain_status: str

class SimulateLinkRequest(BaseModel):
    bop_id: str
    status: BOPStatus
    connectivity_percent: int = Field(..., ge=0, le=100)
    latency_ms: str

class SyncTriggerResponse(BaseModel):
    synced_count: int
    remaining_in_buffer: int
    status: str
    message: str

# --- Detection & WebSocket Schemas ---
class BoundingBox(BaseModel):
    x: float
    y: float
    w: float
    h: float
    label: str
    confidence: float
    track_id: str

class DetectionFrame(BaseModel):
    camera_id: str
    bop_id: str
    timestamp: str
    fps: int
    boxes: List[BoundingBox] = []
    detections: List[str] = []

class WebSocketMessage(BaseModel):
    event_type: str  # alert_telemetry, detection_frame, system_status, camera_update
    data: Dict[str, Any]
    timestamp: str
