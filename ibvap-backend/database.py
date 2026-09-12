import sqlite3
import hashlib
import json
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Generator
from config import settings

def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(settings.DATABASE_PATH, timeout=20.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn

@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    conn = get_db_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def hash_event_block(event_id: str, ts: str, ev_type: str, camera_id: str, bop_id: str, severity: str, payload: str, prev_hash: str) -> str:
    raw = f"{event_id}|{ts}|{ev_type}|{camera_id}|{bop_id}|{severity}|{payload}|{prev_hash}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def init_db():
    with get_db() as conn:
        cursor = conn.cursor()
        
        # 1. Operators & Users
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            operator_id TEXT UNIQUE NOT NULL,
            full_name TEXT NOT NULL,
            role TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            salt TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """)

        # 2. Border Out Posts (BOPs)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS bops (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            sector TEXT NOT NULL,
            status TEXT NOT NULL,
            cameras INTEGER NOT NULL,
            active_cams INTEGER NOT NULL,
            alerts_count INTEGER NOT NULL DEFAULT 0,
            connectivity INTEGER NOT NULL,
            edge_node TEXT NOT NULL,
            latency TEXT NOT NULL,
            lat REAL NOT NULL,
            lng REAL NOT NULL
        );
        """)

        # 3. Cameras
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS cameras (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            bop_id TEXT NOT NULL,
            status TEXT NOT NULL,
            fps INTEGER NOT NULL,
            res TEXT NOT NULL,
            ai TEXT NOT NULL,
            type TEXT NOT NULL,
            rtsp_url TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            FOREIGN KEY (bop_id) REFERENCES bops(id)
        );
        """)

        # 4. Alerts
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id TEXT PRIMARY KEY,
            severity TEXT NOT NULL,
            type TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            bop_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            status TEXT NOT NULL,
            confidence TEXT NOT NULL,
            tracking_id TEXT NOT NULL,
            zone TEXT NOT NULL,
            description TEXT NOT NULL,
            hash TEXT NOT NULL,
            prev_hash TEXT NOT NULL,
            acknowledged_by TEXT,
            resolved_at TEXT,
            created_at TEXT NOT NULL
        );
        """)

        # 5. Tamper-Evident Events Hash Chain
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            bop_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            severity TEXT NOT NULL,
            payload TEXT NOT NULL,
            hash TEXT NOT NULL,
            prev_hash TEXT NOT NULL,
            integrity_status TEXT NOT NULL DEFAULT 'VERIFIED',
            synced_upstream INTEGER NOT NULL DEFAULT 1,
            created_at TEXT NOT NULL
        );
        """)

        # 6. Edge Sync Buffer (for offline resilience)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS edge_sync_buffer (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING',
            retry_count INTEGER NOT NULL DEFAULT 0,
            buffered_at TEXT NOT NULL,
            synced_at TEXT,
            FOREIGN KEY (event_id) REFERENCES events(id)
        );
        """)

        # Seed initial data if database is new
        _seed_initial_data(cursor)

def _seed_initial_data(cursor: sqlite3.Cursor):
    # Check if users exist
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        import hashlib, secrets
        # Seed default operator CMD-OP-8042-IN
        salt = secrets.token_hex(16)
        pw_raw = "defense2026"
        pw_hash = hashlib.pbkdf2_hmac("sha256", pw_raw.encode(), salt.encode(), 100000).hex()
        
        cursor.execute("""
            INSERT INTO users (operator_id, full_name, role, password_hash, salt, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            "CMD-OP-8042-IN",
            "Commander Vikram Rathore",
            "Sector 3 Defense Commander",
            pw_hash,
            salt,
            datetime.now(timezone.utc).isoformat()
        ))

    # Check if BOPs exist
    cursor.execute("SELECT COUNT(*) FROM bops")
    if cursor.fetchone()[0] == 0:
        bops_data = [
            ("BOP-01", "BOP Alpha", "Sector 3 — Northern Ridge", "ONLINE", 8, 8, 2, 96, "Operational", "19ms", 34.12, 74.80),
            ("BOP-02", "BOP Bravo", "Sector 4 — Valley Pass", "ONLINE", 9, 9, 1, 89, "Operational", "38ms", 34.25, 74.95),
            ("BOP-03", "BOP Charlie", "Sector 5 — River Gorge", "DEGRADED", 6, 5, 3, 31, "Local AI Active (Link Degraded)", "310ms", 33.95, 75.10),
            ("BOP-04", "BOP Delta", "Sector 6 — High Altitude Post", "ONLINE", 4, 4, 0, 99, "Operational", "14ms", 34.40, 74.65)
        ]
        cursor.executemany("""
            INSERT INTO bops (id, name, sector, status, cameras, active_cams, alerts_count, connectivity, edge_node, latency, lat, lng)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, bops_data)

    # Check if Cameras exist
    cursor.execute("SELECT COUNT(*) FROM cameras")
    if cursor.fetchone()[0] == 0:
        cams_data = [
            ("CAM-07", "North Gate Perimeter", "BOP-01", "ONLINE", 30, "1080p", "ACTIVE", "PTZ Thermal", "rtsp://edge-bop01.local:554/cam07", "Just now"),
            ("CAM-08", "Sector 3 Fence Line", "BOP-01", "ONLINE", 30, "4K", "ACTIVE", "Fixed Wide", "rtsp://edge-bop01.local:554/cam08", "Just now"),
            ("CAM-04", "Valley Access Road", "BOP-02", "ONLINE", 28, "1080p", "ACTIVE", "ANPR Bullet", "rtsp://edge-bop02.local:554/cam04", "1s ago"),
            ("CAM-03", "Checkpoint Alpha-1", "BOP-01", "ONLINE", 25, "1080p", "ACTIVE", "PTZ Optical", "rtsp://edge-bop01.local:554/cam03", "Just now"),
            ("CAM-11", "River Gorge Trail", "BOP-03", "ONLINE", 22, "720p", "EDGE LOCAL", "Infrared Fixed", "rtsp://edge-bop03.local:554/cam11", "2s ago"),
            ("CAM-02", "Watchtower East", "BOP-02", "DEGRADED", 16, "1080p", "ACTIVE", "PTZ Thermal", "rtsp://edge-bop02.local:554/cam02", "10s ago"),
            ("CAM-01", "HQ Perimeter South", "BOP-04", "ONLINE", 30, "4K", "ACTIVE", "Optical", "rtsp://edge-bop04.local:554/cam01", "Just now"),
            ("CAM-05", "Outpost Perimeter West", "BOP-03", "OFFLINE", 0, "1080p", "STANDBY", "Fixed Thermal", "rtsp://edge-bop03.local:554/cam05", "18m ago")
        ]
        cursor.executemany("""
            INSERT INTO cameras (id, name, bop_id, status, fps, res, ai, type, rtsp_url, last_seen)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, cams_data)

    # Check if events and alerts exist, build initial tamper-evident hash chain
    cursor.execute("SELECT COUNT(*) FROM events")
    if cursor.fetchone()[0] == 0:
        initial_events_raw = [
            ("EVT-20260913-00417", "ANPR Detection", "CAM-04", "BOP-02", "23:10:44", "LOW", json.dumps({"plate": "JK02AA9912", "match": "Authorized Patrol"})),
            ("EVT-20260913-00418", "Night Movement", "CAM-02", "BOP-02", "23:22:15", "HIGH", json.dumps({"thermal_temp_c": 37.2, "velocity_ms": 1.4})),
            ("EVT-20260913-00419", "Suspicious Loitering", "CAM-11", "BOP-03", "23:34:50", "MEDIUM", json.dumps({"duration_seconds": 210, "zone": "Buffer"})),
            ("EVT-20260913-00420", "ANPR Flagged Vehicle", "CAM-04", "BOP-02", "23:38:12", "HIGH", json.dumps({"plate": "DL01AB1234", "flag": "Watchlist Stolen"})),
            ("EVT-20260913-00421", "Virtual Fence Intrusion", "CAM-07", "BOP-01", "23:41:08", "CRITICAL", json.dumps({"target": "PERSON", "confidence": 0.964, "zone": "North Perimeter Sector 3"}))
        ]
        
        current_prev_hash = settings.GENESIS_HASH
        for ev_id, ev_type, cam_id, bop_id, ts, sev, payload in initial_events_raw:
            ev_hash = hash_event_block(ev_id, ts, ev_type, cam_id, bop_id, sev, payload, current_prev_hash)
            cursor.execute("""
                INSERT INTO events (id, type, camera_id, bop_id, timestamp, severity, payload, hash, prev_hash, integrity_status, synced_upstream, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'VERIFIED', 1, ?)
            """, (ev_id, ev_type, cam_id, bop_id, ts, sev, payload, ev_hash, current_prev_hash, datetime.now(timezone.utc).isoformat()))
            current_prev_hash = ev_hash

    # Check if alerts exist
    cursor.execute("SELECT COUNT(*) FROM alerts")
    if cursor.fetchone()[0] == 0:
        cursor.execute("SELECT hash, prev_hash FROM events WHERE id = 'EVT-20260913-00421'")
        row = cursor.fetchone()
        h421, ph421 = (row[0], row[1]) if row else ("8f72a19c", "3a917c2e")
        
        cursor.execute("SELECT hash, prev_hash FROM events WHERE id = 'EVT-20260913-00420'")
        row = cursor.fetchone()
        h420, ph420 = (row[0], row[1]) if row else ("4b21901f", "8f72a19c")
        
        initial_alerts_data = [
            ("ALT-9042", "CRITICAL", "Virtual Fence Intrusion", "CAM-07", "BOP Alpha", "23:41:08", "Active", "96.4%", "TRK-0921", "North Perimeter (Sector 3)", "Unauthorized human movement detected crossing virtual boundary Sector 3 fence line.", h421, ph421, None, None, datetime.now(timezone.utc).isoformat()),
            ("ALT-9041", "HIGH", "Unrecognized Vehicle", "CAM-04", "BOP Bravo", "23:38:12", "Acknowledged", "91.2%", "TRK-0884", "Access Gate 2", "Unregistered heavy utility vehicle idling near restricted barrier.", h420, ph420, "CMD-OP-8042-IN", None, datetime.now(timezone.utc).isoformat()),
            ("ALT-9040", "MEDIUM", "Suspicious Loitering", "CAM-11", "BOP Charlie", "23:34:50", "Active", "84.8%", "TRK-0712", "Gorge Trail Marker 4", "Individual stationary in restricted buffer zone for > 3 minutes.", "7c99210a", "4b21901f", None, None, datetime.now(timezone.utc).isoformat()),
            ("ALT-9039", "HIGH", "Night Movement Detected", "CAM-02", "BOP Bravo", "23:22:15", "Resolved", "92.7%", "TRK-0654", "Ridge Sector B", "Thermal signature crossing international border sector.", "2e11893c", "7c99210a", "CMD-OP-8042-IN", "23:25:00", datetime.now(timezone.utc).isoformat()),
            ("ALT-9038", "LOW", "Face Detection Match", "CAM-03", "BOP Alpha", "23:15:02", "Resolved", "88.1%", "TRK-0540", "Main Gate Terminal", "Authorized personnel biometric scanned at entry terminal.", "90aa321b", "2e11893c", "CMD-OP-8042-IN", "23:16:00", datetime.now(timezone.utc).isoformat())
        ]
        
        cursor.executemany("""
            INSERT INTO alerts (id, severity, type, camera_id, bop_id, timestamp, status, confidence, tracking_id, zone, description, hash, prev_hash, acknowledged_by, resolved_at, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, initial_alerts_data)
