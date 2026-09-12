import asyncio
import json
import logging
from typing import List, Dict, Any
from fastapi import WebSocket

logger = logging.getLogger("ibvap.websocket")

class WebSocketGateway:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        async with self._lock:
            self.active_connections.append(websocket)
        logger.info(f"WebSocket client connected. Total active: {len(self.active_connections)}")

    async def disconnect(self, websocket: WebSocket):
        async with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
        logger.info(f"WebSocket client disconnected. Total active: {len(self.active_connections)}")

    async def broadcast(self, event_type: str, data: Dict[str, Any]):
        message = {
            "event_type": event_type,
            "data": data,
            "timestamp": asyncio.get_event_loop().time()
        }
        json_payload = json.dumps(message)
        
        async with self._lock:
            disconnected = []
            for connection in self.active_connections:
                try:
                    await connection.send_text(json_payload)
                except Exception as e:
                    logger.warning(f"Error sending to WebSocket client: {e}")
                    disconnected.append(connection)
            
            for conn in disconnected:
                if conn in self.active_connections:
                    self.active_connections.remove(conn)

    @property
    def connection_count(self) -> int:
        return len(self.active_connections)

ws_gateway = WebSocketGateway()
