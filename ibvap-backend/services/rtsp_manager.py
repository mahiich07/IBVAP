import asyncio
import io
import time
from datetime import datetime, timezone
from typing import Dict, Any, Optional, Generator
from PIL import Image, ImageDraw, ImageFont
from database import get_db

class RTSPStreamManager:
    def __init__(self):
        self._camera_health: Dict[str, Dict[str, Any]] = {}

    def get_camera_info(self, camera_id: str) -> Optional[Dict[str, Any]]:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.*, b.name as bop_name 
                FROM cameras c 
                JOIN bops b ON c.bop_id = b.id 
                WHERE c.id = ?
            """, (camera_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_all_cameras(self) -> list[Dict[str, Any]]:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT c.id, c.name, b.name as bop, c.bop_id, c.status, c.fps, c.res, c.ai, c.type, c.last_seen as lastSeen
                FROM cameras c
                JOIN bops b ON c.bop_id = b.id
                ORDER BY c.id ASC
            """)
            rows = cursor.fetchall()
            
            # Map detections summary
            results = []
            for r in rows:
                c_dict = dict(r)
                if c_dict["id"] == "CAM-07":
                    c_dict["detections"] = ["PERSON 96.4%", "VEHICLE 89.1%"]
                elif c_dict["id"] == "CAM-08":
                    c_dict["detections"] = ["VIRTUAL FENCE BREACH"]
                elif c_dict["id"] == "CAM-04":
                    c_dict["detections"] = ["VEHICLE: DL01AB1234"]
                elif c_dict["id"] == "CAM-03":
                    c_dict["detections"] = ["FACE RECOGNISED"]
                elif c_dict["id"] == "CAM-11":
                    c_dict["detections"] = ["NIGHT MOVEMENT"]
                elif c_dict["id"] == "CAM-02":
                    c_dict["detections"] = ["SCANNING"]
                elif c_dict["id"] == "CAM-01":
                    c_dict["detections"] = ["CLEAR"]
                else:
                    c_dict["detections"] = ["STANDBY"]
                results.append(c_dict)
            return results

    def generate_synthetic_frame(self, camera_id: str, width: int = 640, height: int = 360) -> bytes:
        """
        Generates an encrypted tactical HUD frame overlay for the camera feed.
        """
        cam = self.get_camera_info(camera_id)
        cam_name = cam["name"] if cam else camera_id
        bop_name = cam["bop_name"] if cam else "BOP Alpha"
        fps = cam["fps"] if cam else 30
        status = cam["status"] if cam else "ONLINE"
        
        # Create dark tactical canvas
        img = Image.new("RGB", (width, height), color=(10, 15, 24))
        draw = ImageDraw.Draw(img)
        
        # Draw grid lines
        for x in range(0, width, 40):
            draw.line([(x, 0), (x, height)], fill=(20, 30, 45), width=1)
        for y in range(0, height, 40):
            draw.line([(0, y), (width, y)], fill=(20, 30, 45), width=1)

        # Tactical Reticle / Crosshair in center
        cx, cy = width // 2, height // 2
        draw.line([(cx - 20, cy), (cx - 5, cy)], fill=(6, 182, 212), width=1)
        draw.line([(cx + 5, cy), (cx + 20, cy)], fill=(6, 182, 212), width=1)
        draw.line([(cx, cy - 20), (cx, cy - 5)], fill=(6, 182, 212), width=1)
        draw.line([(cx, cy + 5), (cx, cy + 20)], fill=(6, 182, 212), width=1)
        draw.ellipse([(cx - 3, cy - 3), (cx + 3, cy + 3)], outline=(6, 182, 212), width=1)

        # Simulated AI Target Bounding Box
        if camera_id in ("CAM-07", "CAM-08"):
            bx1, by1, bx2, by2 = int(width * 0.28), int(height * 0.25), int(width * 0.48), int(height * 0.75)
            draw.rectangle([(bx1, by1), (bx2, by2)], outline=(239, 68, 68), width=2)
            draw.rectangle([(bx1, by1 - 16), (bx1 + 105, by1)], fill=(239, 68, 68))
            draw.text((bx1 + 3, by1 - 14), "PERSON 96.4%", fill=(10, 15, 24))
            draw.text((bx1 + 3, by2 + 3), "TRK-0921 • VEL: 1.2m/s", fill=(248, 113, 113))

        # Top HUD Banner
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        draw.rectangle([(10, 10), (width - 10, 32)], fill=(5, 7, 10))
        draw.text((15, 14), f"{camera_id} • {cam_name.upper()} ({bop_name.upper()})", fill=(226, 232, 240))
        status_color = (52, 211, 153) if status == "ONLINE" else (251, 191, 36)
        draw.text((width - 140, 14), f"LIVE • {fps} FPS", fill=status_color)

        # Bottom HUD Telemetry
        draw.rectangle([(10, height - 30), (width - 10, height - 10)], fill=(5, 7, 10))
        draw.text((15, height - 26), f"TIME: {now_str}", fill=(148, 163, 184))
        draw.text((width - 165, height - 26), "ENCRYPTED RTSP STREAM", fill=(6, 182, 212))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    async def mjpeg_stream_generator(self, camera_id: str):
        """
        Asynchronous generator that yields multipart MJPEG frames.
        """
        while True:
            frame_bytes = self.generate_synthetic_frame(camera_id)
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
            await asyncio.sleep(0.1)  # ~10 FPS preview stream

rtsp_manager = RTSPStreamManager()
