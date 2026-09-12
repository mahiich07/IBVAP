import logging
from datetime import datetime, timezone
from typing import Dict, Any, List
from database import get_db
from services.websocket_gateway import ws_gateway

logger = logging.getLogger("ibvap.edgesync")

class EdgeSyncEngine:
    def __init__(self):
        self.simulated_link_status = "ONLINE"
        self.is_syncing = False

    def is_link_operational(self, bop_id: str = "BOP-01") -> bool:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status, connectivity FROM bops WHERE id = ?", (bop_id,))
            row = cursor.fetchone()
            if row:
                status, conn_pct = row["status"], row["connectivity"]
                return status == "ONLINE" and conn_pct >= 50
            return True

    def get_pending_sync_count(self) -> int:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM edge_sync_buffer WHERE status = 'PENDING'")
            return cursor.fetchone()[0]

    async def sync_upstream(self, batch_size: int = 50) -> Dict[str, Any]:
        if self.is_syncing:
            return {"status": "IN_PROGRESS", "message": "Sync batch already in progress", "synced_count": 0, "remaining_in_buffer": self.get_pending_sync_count()}
        
        self.is_syncing = True
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT b.id as buffer_id, b.event_id, e.type, e.camera_id, e.bop_id, e.timestamp, e.severity, e.payload, e.hash, e.prev_hash
                    FROM edge_sync_buffer b
                    JOIN events e ON b.event_id = e.id
                    WHERE b.status = 'PENDING'
                    ORDER BY b.buffered_at ASC
                    LIMIT ?
                """, (batch_size,))
                rows = cursor.fetchall()
                
                synced_count = 0
                now_str = datetime.now(timezone.utc).isoformat()
                
                for row in rows:
                    buffer_id = row["buffer_id"]
                    event_id = row["event_id"]
                    
                    # In a multi-tier deployment, this payload is POSTed upstream via mutual-TLS/WireGuard
                    # Here we simulate the upstream Central Command ingestion receipt:
                    cursor.execute("""
                        UPDATE edge_sync_buffer 
                        SET status = 'SYNCED', synced_at = ? 
                        WHERE id = ?
                    """, (now_str, buffer_id))
                    
                    cursor.execute("""
                        UPDATE events 
                        SET synced_upstream = 1 
                        WHERE id = ?
                    """, (event_id,))
                    synced_count += 1
                
                remaining = self.get_pending_sync_count()
                
                # Notify connected clients via WebSocket about sync completion
                if synced_count > 0:
                    await ws_gateway.broadcast("sync_completed", {
                        "synced_count": synced_count,
                        "remaining_in_buffer": remaining,
                        "timestamp": now_str
                    })
                
                return {
                    "status": "COMPLETED",
                    "synced_count": synced_count,
                    "remaining_in_buffer": remaining,
                    "message": f"Successfully synchronized {synced_count} event records upstream to Central Command."
                }
        finally:
            self.is_syncing = False

    async def update_bop_link(self, bop_id: str, status: str, connectivity: int, latency: str) -> Dict[str, Any]:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE bops 
                SET status = ?, connectivity = ?, latency = ?,
                    edge_node = CASE 
                        WHEN ? = 'ONLINE' THEN 'Operational'
                        WHEN ? = 'DEGRADED' THEN 'Local AI Active (Link Degraded)'
                        ELSE 'Isolated Edge Mode (Offline Buffer)'
                    END
                WHERE id = ?
            """, (status, connectivity, latency, status, status, bop_id))
            
            cursor.execute("SELECT id, name, status, connectivity, edge_node, latency FROM bops WHERE id = ?", (bop_id,))
            updated_bop = dict(cursor.fetchone())

        # If restored to ONLINE with good connectivity, automatically trigger upstream sync
        synced_info = None
        if status == "ONLINE" and connectivity >= 50:
            synced_info = await self.sync_upstream()

        await ws_gateway.broadcast("system_status", {
            "bop_updated": updated_bop,
            "auto_synced": synced_info
        })

        return {
            "bop": updated_bop,
            "auto_synced": synced_info
        }

edge_sync = EdgeSyncEngine()
