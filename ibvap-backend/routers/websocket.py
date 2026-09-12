import json
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from services.websocket_gateway import ws_gateway

logger = logging.getLogger("ibvap.ws_router")
router = APIRouter(tags=["WebSocket"])

@router.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket):
    """
    Real-time WebSocket connection broadcasting live alert telemetry,
    camera FPS updates, and bounding-box detections.
    """
    await ws_gateway.connect(websocket)
    
    # Send initial connection acknowledgment
    try:
        await websocket.send_text(json.dumps({
            "event_type": "connection_established",
            "data": {
                "message": "Connected to IBVAP Real-Time Defense Telemetry Gateway",
                "channel": "/ws/stream",
                "encryption": "TLS_AES_256_GCM_SHA384"
            }
        }))
        
        while True:
            # Keep connection open and handle incoming client commands (e.g. ping or filter subscriptions)
            text_data = await websocket.receive_text()
            try:
                msg = json.loads(text_data)
                if msg.get("action") == "ping":
                    await websocket.send_text(json.dumps({"event_type": "pong", "timestamp": msg.get("timestamp")}))
            except Exception:
                pass
                
    except WebSocketDisconnect:
        await ws_gateway.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket endpoint exception: {e}")
        await ws_gateway.disconnect(websocket)
