import hashlib
import json
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional
from config import settings
from database import get_db, hash_event_block

class HashChainService:
    @staticmethod
    def get_latest_hash() -> str:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT hash FROM events ORDER BY rowid DESC LIMIT 1")
            row = cursor.fetchone()
            if row and row[0]:
                return row[0]
            return settings.GENESIS_HASH

    @staticmethod
    def append_event(
        event_type: str,
        camera_id: str,
        bop_id: str,
        severity: str,
        payload: Dict[str, Any],
        custom_id: Optional[str] = None,
        timestamp: Optional[str] = None,
        synced_upstream: bool = True
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        ts = timestamp or now.strftime("%H:%M:%S")
        created_at = now.isoformat()
        
        with get_db() as conn:
            cursor = conn.cursor()
            
            # 1. Determine Event ID
            if custom_id:
                event_id = custom_id
            else:
                date_str = now.strftime("%Y%m%d")
                cursor.execute("SELECT COUNT(*) FROM events")
                cnt = cursor.fetchone()[0] + 1
                event_id = f"EVT-{date_str}-{cnt:05d}"
            
            # 2. Get latest hash as prev_hash
            cursor.execute("SELECT hash FROM events ORDER BY rowid DESC LIMIT 1")
            row = cursor.fetchone()
            prev_hash = row[0] if row and row[0] else settings.GENESIS_HASH
            
            # 3. Compute Cryptographic SHA-256 Block Hash
            payload_str = json.dumps(payload, sort_keys=True)
            event_hash = hash_event_block(event_id, ts, event_type, camera_id, bop_id, severity, payload_str, prev_hash)
            
            # 4. Insert into events table
            cursor.execute("""
                INSERT INTO events (id, type, camera_id, bop_id, timestamp, severity, payload, hash, prev_hash, integrity_status, synced_upstream, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'VERIFIED', ?, ?)
            """, (event_id, event_type, camera_id, bop_id, ts, severity, payload_str, event_hash, prev_hash, 1 if synced_upstream else 0, created_at))
            
            # 5. If not synced upstream (e.g. edge node in offline/degraded mode), buffer it
            if not synced_upstream:
                cursor.execute("""
                    INSERT INTO edge_sync_buffer (event_id, status, retry_count, buffered_at)
                    VALUES (?, 'PENDING', 0, ?)
                """, (event_id, created_at))
            
            return {
                "id": event_id,
                "type": event_type,
                "camera": camera_id,
                "bop": bop_id,
                "timestamp": ts,
                "severity": severity,
                "payload": payload,
                "hash": event_hash,
                "prevHash": prev_hash,
                "integrity": "VERIFIED",
                "synced_upstream": synced_upstream
            }

    @staticmethod
    def verify_chain() -> Dict[str, Any]:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, type, camera_id, bop_id, timestamp, severity, payload, hash, prev_hash FROM events ORDER BY rowid ASC")
            events = cursor.fetchall()
            
            if not events:
                return {
                    "is_valid": True,
                    "total_blocks_checked": 0,
                    "genesis_hash": settings.GENESIS_HASH,
                    "latest_hash": settings.GENESIS_HASH,
                    "tampered_block_id": None,
                    "message": "Empty ledger. Genesis verified.",
                    "verified_at": datetime.now(timezone.utc).isoformat()
                }
            
            expected_prev_hash = settings.GENESIS_HASH
            for idx, ev in enumerate(events):
                ev_id = ev["id"]
                ev_type = ev["type"]
                cam_id = ev["camera_id"]
                bop_id = ev["bop_id"]
                ts = ev["timestamp"]
                sev = ev["severity"]
                payload_str = ev["payload"]
                actual_hash = ev["hash"]
                actual_prev_hash = ev["prev_hash"]
                
                # Check link to previous hash
                if actual_prev_hash != expected_prev_hash:
                    return {
                        "is_valid": False,
                        "total_blocks_checked": idx,
                        "genesis_hash": settings.GENESIS_HASH,
                        "latest_hash": actual_hash,
                        "tampered_block_id": ev_id,
                        "message": f"Broken chain link at event {ev_id}. Expected prev_hash {expected_prev_hash}, got {actual_prev_hash}",
                        "verified_at": datetime.now(timezone.utc).isoformat()
                    }
                
                # Recompute SHA-256 block hash
                # Standardize payload serialization if it's valid JSON
                try:
                    p_obj = json.loads(payload_str)
                    norm_payload = json.dumps(p_obj, sort_keys=True)
                except Exception:
                    norm_payload = payload_str

                recalc_hash = hash_event_block(ev_id, ts, ev_type, cam_id, bop_id, sev, norm_payload, actual_prev_hash)
                
                # Also test unnormalized if it matches
                alt_hash = hash_event_block(ev_id, ts, ev_type, cam_id, bop_id, sev, payload_str, actual_prev_hash)
                
                if actual_hash not in (recalc_hash, alt_hash):
                    return {
                        "is_valid": False,
                        "total_blocks_checked": idx,
                        "genesis_hash": settings.GENESIS_HASH,
                        "latest_hash": actual_hash,
                        "tampered_block_id": ev_id,
                        "message": f"Hash signature mismatch at block {ev_id}. Evidence data was tampered or corrupted!",
                        "verified_at": datetime.now(timezone.utc).isoformat()
                    }
                
                expected_prev_hash = actual_hash

            return {
                "is_valid": True,
                "total_blocks_checked": len(events),
                "genesis_hash": settings.GENESIS_HASH,
                "latest_hash": expected_prev_hash,
                "tampered_block_id": None,
                "message": "Cryptographic ledger intact. All block hashes and previous links verified with 100% integrity.",
                "verified_at": datetime.now(timezone.utc).isoformat()
            }

hash_chain = HashChainService()
