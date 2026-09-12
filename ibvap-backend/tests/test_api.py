import unittest
import os
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from config import settings
# Use a separate test database file
settings.DATABASE_PATH = str(backend_dir / "test_ibvap.db")

from database import init_db, get_db
from services.auth_service import hash_password, verify_password, create_access_token, decode_access_token
from services.rtsp_manager import rtsp_manager
from services.ai_pipeline import ai_pipeline
from services.hash_chain import hash_chain
from services.edge_sync import edge_sync
from services.rules_engine import rules_engine
from models.schemas import LoginRequest, AlertAcknowledgeRequest
from routers.auth import login
from routers.cameras import list_cameras, get_camera, get_camera_snapshot
from routers.alerts import get_alerts, acknowledge_alert
from routers.events import get_events, verify_hash_chain
from routers.system import get_system_health

class TestIBVAPBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Remove old test database if exists
        if os.path.exists(settings.DATABASE_PATH):
            try:
                os.remove(settings.DATABASE_PATH)
            except Exception:
                pass
        init_db()

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(settings.DATABASE_PATH):
            try:
                os.remove(settings.DATABASE_PATH)
            except Exception:
                pass

    # =========================================================================
    # 1. AUTHENTICATION & JWT SECURITY
    # =========================================================================
    def test_01_password_hashing(self):
        pw = "classifiedDefense123"
        h, salt = hash_password(pw)
        self.assertTrue(verify_password(pw, h, salt))
        self.assertFalse(verify_password("wrongpassword", h, salt))

    def test_02_jwt_token_generation_and_decode(self):
        payload = {"sub": "CMD-OP-8042-IN", "role": "Commander"}
        token = create_access_token(payload, expires_delta_seconds=3600)
        self.assertIsInstance(token, str)
        decoded = decode_access_token(token)
        self.assertEqual(decoded["sub"], "CMD-OP-8042-IN")
        self.assertEqual(decoded["role"], "Commander")

    def test_03_api_login_endpoint(self):
        login_req = LoginRequest(operator_id="CMD-OP-8042-IN", password="defense2026")
        token_resp = login(login_req)
        self.assertIsNotNone(token_resp.access_token)
        self.assertEqual(token_resp.operator.operator_id, "CMD-OP-8042-IN")

    # =========================================================================
    # 2. RTSP STREAM MANAGER
    # =========================================================================
    def test_04_camera_fleet_listing(self):
        cameras = list_cameras()
        self.assertGreaterEqual(len(cameras), 8)
        cam_ids = [c["id"] for c in cameras]
        self.assertIn("CAM-07", cam_ids)
        self.assertIn("CAM-04", cam_ids)

    def test_05_camera_synthetic_snapshot(self):
        resp = get_camera_snapshot("CAM-07")
        self.assertEqual(resp.media_type, "image/jpeg")
        self.assertGreater(len(resp.body), 1000)

    # =========================================================================
    # 3. AI INFERENCE PIPELINE
    # =========================================================================
    def test_06_ai_inference_pipeline_execution(self):
        res = ai_pipeline.run_full_inference("CAM-07")
        self.assertEqual(res["camera_id"], "CAM-07")
        self.assertIn("humans", res)
        self.assertGreater(len(res["humans"]), 0)
        human = res["humans"][0]
        self.assertEqual(human["label"], "PERSON")
        self.assertGreaterEqual(human["confidence"], 0.85)

    def test_07_anpr_watchlist_detection(self):
        anpr = ai_pipeline.detect_anpr("CAM-04")
        self.assertIn("plate", anpr)
        self.assertIn("is_flagged", anpr)

    # =========================================================================
    # 4. RULES & ALERT ENGINE
    # =========================================================================
    def test_08_rules_engine_breach_detection(self):
        import asyncio
        inference = ai_pipeline.run_full_inference("CAM-07")
        inference["fence_eval"]["breached"] = True
        
        loop = asyncio.new_event_loop()
        alert = loop.run_until_complete(
            rules_engine.evaluate_and_dispatch("CAM-07", "BOP-01", "BOP Alpha", inference)
        )
        loop.close()
        
        self.assertIsNotNone(alert)
        self.assertEqual(alert["severity"], "CRITICAL")
        self.assertEqual(alert["type"], "Virtual Fence Intrusion")
        self.assertIn("hash", alert)

    # =========================================================================
    # 5. TAMPER-EVIDENT HASH CHAIN AUDIT & LEDGER VERIFICATION
    # =========================================================================
    def test_09_hash_chain_integrity_verification(self):
        audit = hash_chain.verify_chain()
        self.assertTrue(audit["is_valid"])
        self.assertIsNone(audit["tampered_block_id"])
        self.assertGreater(audit["total_blocks_checked"], 0)

    def test_10_hash_chain_tamper_detection(self):
        """
        Simulate an adversary modifying a recorded event in the database directly.
        Verify that the cryptographic hash chain catches the anomaly!
        """
        # Append a verified event first
        ev = hash_chain.append_event(
            event_type="Test Event",
            camera_id="CAM-01",
            bop_id="BOP-01",
            severity="LOW",
            payload={"action": "test_audit"}
        )
        
        # Verify valid before tampering
        self.assertTrue(hash_chain.verify_chain()["is_valid"])
        
        # Tamper with the event payload directly in SQLite
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE events SET payload = '{\"action\": \"TAMPERED_PAYLOAD\"}' WHERE id = ?", (ev["id"],))

        # Run verification - MUST detect tampering!
        tampered_audit = hash_chain.verify_chain()
        self.assertFalse(tampered_audit["is_valid"])
        self.assertEqual(tampered_audit["tampered_block_id"], ev["id"])

        # Restore original payload to leave DB clean
        with get_db() as conn:
            cursor = conn.cursor()
            import json
            cursor.execute("UPDATE events SET payload = ? WHERE id = ?", (json.dumps({"action": "test_audit"}, sort_keys=True), ev["id"]))
            
        # Re-verify restored
        self.assertTrue(hash_chain.verify_chain()["is_valid"])

    # =========================================================================
    # 6. EDGE-FIRST SYNC ENGINE
    # =========================================================================
    def test_11_edge_sync_buffering_and_upstream_push(self):
        import asyncio
        # Add event buffered for offline mode
        ev = hash_chain.append_event(
            event_type="Border Trail Sighting",
            camera_id="CAM-11",
            bop_id="BOP-03",
            severity="MEDIUM",
            payload={"zone": "Gorge Trail"},
            synced_upstream=False
        )
        self.assertGreaterEqual(edge_sync.get_pending_sync_count(), 1)
        
        # Run sync upstream
        loop = asyncio.new_event_loop()
        res = loop.run_until_complete(edge_sync.sync_upstream())
        loop.close()
        
        self.assertEqual(res["status"], "COMPLETED")
        self.assertGreaterEqual(res["synced_count"], 1)

    def test_12_alerts_filtering_and_acknowledgment(self):
        import asyncio
        # 1. Fetch alerts
        alerts = get_alerts(severity="CRITICAL", alert_status=None, search=None, limit=50, offset=0)
        self.assertGreater(len(alerts), 0)
        self.assertTrue(all(a["severity"] == "CRITICAL" for a in alerts))
        
        # 2. Acknowledge alert
        first_alert_id = alerts[0]["id"]
        loop = asyncio.new_event_loop()
        ack_res = loop.run_until_complete(acknowledge_alert(first_alert_id, AlertAcknowledgeRequest(action="Acknowledged")))
        loop.close()
        
        self.assertTrue(ack_res.success)
        self.assertEqual(ack_res.status, "Acknowledged")

    def test_13_events_endpoint(self):
        events = get_events(limit=10, offset=0)
        self.assertGreater(len(events), 0)
        self.assertIn("hash", events[0])
        self.assertIn("prevHash", events[0])
        self.assertEqual(events[0]["integrity"], "VERIFIED")

    def test_14_system_health_endpoint(self):
        data = get_system_health()
        self.assertEqual(data.overall_status, "OPERATIONAL")
        self.assertGreaterEqual(len(data.bops), 4)
        self.assertEqual(data.tamper_chain_status, "VERIFIED")

if __name__ == "__main__":
    unittest.main()
