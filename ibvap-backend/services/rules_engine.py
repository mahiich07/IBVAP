import random
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from database import get_db
from models.domain import Severity, AlertStatus
from services.hash_chain import hash_chain
from services.websocket_gateway import ws_gateway
from services.edge_sync import edge_sync

class RulesAndAlertEngine:
    async def evaluate_and_dispatch(
        self,
        camera_id: str,
        bop_id: str,
        bop_name: str,
        inference_result: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates detection telemetry against geofenced rules, determines severity,
        records into immutable cryptographic hash chain, updates DB, and broadcasts.
        """
        now = datetime.now(timezone.utc)
        ts = now.strftime("%H:%M:%S")
        
        fence = inference_result.get("fence_eval", {})
        anpr = inference_result.get("anpr")
        night = inference_result.get("night_eval")
        humans = inference_result.get("humans", [])
        
        alert_candidate = None

        # Rule 1: Virtual Fence Breach -> CRITICAL
        if fence.get("breached"):
            human = humans[0] if humans else {}
            conf = f"{round(human.get('confidence', 0.96) * 100, 1)}%"
            tracking_id = human.get("track_id", f"TRK-{random.randint(1000, 9999)}")
            zone = fence.get("zone", "Perimeter Sector")
            alert_candidate = {
                "severity": Severity.CRITICAL.value,
                "type": "Virtual Fence Intrusion",
                "confidence": conf,
                "tracking_id": tracking_id,
                "zone": zone,
                "description": f"Unauthorized human movement detected crossing virtual boundary in {zone}.",
                "payload": {"rule": "VIRTUAL_FENCE_BREACH", "tracking_id": tracking_id, "bbox": human.get("bbox")}
            }

        # Rule 2: Flagged Vehicle via ANPR -> HIGH
        elif anpr and anpr.get("is_flagged"):
            tracking_id = f"TRK-{random.randint(1000, 9999)}"
            conf = f"{round(anpr.get('confidence', 0.93) * 100, 1)}%"
            zone = "Access Gate Perimeter"
            alert_candidate = {
                "severity": Severity.HIGH.value,
                "type": "Unrecognized Vehicle",
                "confidence": conf,
                "tracking_id": tracking_id,
                "zone": zone,
                "description": f"Flagged vehicle [{anpr['plate']}] identified at barrier. Reason: {anpr.get('flag_reason')}",
                "payload": {"rule": "ANPR_WATCHLIST_MATCH", "plate": anpr["plate"], "confidence": conf}
            }

        # Rule 3: Night Movement Detected -> HIGH
        elif night and night.get("heat_signature_detected") and random.random() < 0.25:
            tracking_id = f"TRK-{random.randint(1000, 9999)}"
            conf = f"{round(night.get('confidence', 0.91) * 100, 1)}%"
            zone = "Thermal Ridge Buffer"
            alert_candidate = {
                "severity": Severity.HIGH.value,
                "type": "Night Movement Detected",
                "confidence": conf,
                "tracking_id": tracking_id,
                "zone": zone,
                "description": f"Thermal signature ({night.get('body_temp_c')}°C) tracked moving through restricted night sector.",
                "payload": {"rule": "THERMAL_NIGHT_MOVEMENT", "temp_c": night.get("body_temp_c")}
            }

        if not alert_candidate:
            return None

        # Determine edge link status
        is_link_up = edge_sync.is_link_operational(bop_id)
        
        # 1. Log to Cryptographic SHA-256 Hash Chain
        ev = hash_chain.append_event(
            event_type=alert_candidate["type"],
            camera_id=camera_id,
            bop_id=bop_id,
            severity=alert_candidate["severity"],
            payload=alert_candidate["payload"],
            timestamp=ts,
            synced_upstream=is_link_up
        )

        # 2. Insert into Alerts Table
        alert_id = f"ALT-{random.randint(1000, 9999)}"
        created_at = now.isoformat()
        
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO alerts (id, severity, type, camera_id, bop_id, timestamp, status, confidence, tracking_id, zone, description, hash, prev_hash, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                alert_id,
                alert_candidate["severity"],
                alert_candidate["type"],
                camera_id,
                bop_name,
                ts,
                AlertStatus.ACTIVE.value,
                alert_candidate["confidence"],
                alert_candidate["tracking_id"],
                alert_candidate["zone"],
                alert_candidate["description"],
                ev["hash"][:12] + "..." + ev["hash"][-6:],
                ev["prevHash"][:12] + "..." + ev["prevHash"][-6:],
                created_at
            ))
            
            # Increment BOP active alerts count
            cursor.execute("UPDATE bops SET alerts_count = alerts_count + 1 WHERE id = ?", (bop_id,))

        alert_record = {
            "id": alert_id,
            "severity": alert_candidate["severity"],
            "type": alert_candidate["type"],
            "camera": camera_id,
            "bop": bop_name,
            "timestamp": ts,
            "status": AlertStatus.ACTIVE.value,
            "confidence": alert_candidate["confidence"],
            "trackingId": alert_candidate["tracking_id"],
            "zone": alert_candidate["zone"],
            "description": alert_candidate["description"],
            "hash": ev["hash"][:12] + "..." + ev["hash"][-6:],
            "prevHash": ev["prevHash"][:12] + "..." + ev["prevHash"][-6:],
            "event_id": ev["id"],
            "synced_upstream": is_link_up
        }

        # 3. Broadcast Alert to WebSocket Gateway
        await ws_gateway.broadcast("alert_telemetry", alert_record)

        return alert_record

rules_engine = RulesAndAlertEngine()
