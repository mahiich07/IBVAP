from fastapi import APIRouter, HTTPException, Query, status
from typing import List, Optional
from datetime import datetime, timezone
from models.schemas import AlertOut, AlertAcknowledgeRequest, AlertAcknowledgeResponse
from database import get_db
from services.websocket_gateway import ws_gateway

router = APIRouter(prefix="/alerts", tags=["Alerts"])

@router.get("", response_model=List[AlertOut])
def get_alerts(
    severity: Optional[str] = Query(None, description="Filter by CRITICAL, HIGH, MEDIUM, LOW"),
    alert_status: Optional[str] = Query(None, alias="status", description="Filter by Active, Acknowledged, Escalated, Resolved"),
    search: Optional[str] = Query(None, description="Search query across camera, bop, type, description"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0)
):
    """
    Fetch, filter, and search alerts (with pagination and severity filters).
    """
    with get_db() as conn:
        cursor = conn.cursor()
        query = """
            SELECT id, severity, type, camera_id as camera, bop_id as bop, timestamp, status, confidence, tracking_id as trackingId, zone, description, hash, prev_hash as prevHash, acknowledged_by as acknowledgedBy, resolved_at as resolvedAt
            FROM alerts
            WHERE 1=1
        """
        params = []
        
        if severity and severity.upper() != "ALL":
            query += " AND severity = ?"
            params.append(severity.upper())
            
        if alert_status and alert_status.upper() != "ALL":
            query += " AND status = ?"
            params.append(alert_status)
            
        if search:
            query += " AND (type LIKE ? OR camera_id LIKE ? OR bop_id LIKE ? OR description LIKE ?)"
            s_param = f"%{search}%"
            params.extend([s_param, s_param, s_param, s_param])
            
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(r) for r in rows]

@router.post("/{alert_id}/acknowledge", response_model=AlertAcknowledgeResponse)
async def acknowledge_alert(alert_id: str, body: AlertAcknowledgeRequest):
    """
    Acknowledge, escalate, or resolve an active threat alert.
    """
    action = body.action.capitalize()
    if action not in ("Acknowledged", "Escalated", "Resolved"):
        action = "Acknowledged"

    now_str = datetime.now(timezone.utc).isoformat()
    resolved_at = now_str if action == "Resolved" else None
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, bop_id FROM alerts WHERE id = ?", (alert_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found")
        
        bop_id = row["bop_id"]
        
        cursor.execute("""
            UPDATE alerts
            SET status = ?, acknowledged_by = 'CMD-OP-8042-IN', resolved_at = ?
            WHERE id = ?
        """, (action, resolved_at, alert_id))
        
        # If resolved, reduce active alerts count for that BOP
        if action == "Resolved":
            cursor.execute("UPDATE bops SET alerts_count = MAX(0, alerts_count - 1) WHERE name = ? OR id = ?", (bop_id, bop_id))

    # Broadcast alert resolution/acknowledgment to WebSocket gateway
    await ws_gateway.broadcast("alert_status_update", {
        "alert_id": alert_id,
        "status": action,
        "updated_at": now_str
    })

    return AlertAcknowledgeResponse(
        success=True,
        alert_id=alert_id,
        status=action,
        message=f"Threat alert {alert_id} marked as {action}",
        updated_at=now_str
    )
