import random
from typing import Dict, Any, List
from datetime import datetime, timezone

WATCHLIST_PLATES = {
    "DL01AB1234": "FLAGGED: Unregistered Heavy Utility Vehicle (Sector 4)",
    "JK02XX9999": "FLAGGED: Known Smuggling Target Vehicle",
    "HR26BK4321": "SUSPICIOUS: Multiple Night Perimeter Passes"
}

VIRTUAL_FENCE_ZONES = {
    "CAM-07": {"zone_name": "North Perimeter (Sector 3)", "boundary_x": 0.40},
    "CAM-08": {"zone_name": "Sector 3 Fence Line", "boundary_x": 0.35},
    "CAM-11": {"zone_name": "Gorge Trail Marker 4", "boundary_x": 0.50}
}

class AIInferencePipeline:
    """
    Simulates Edge AI pipelines operating directly on edge nodes.
    Integrates Human, Vehicle, ANPR, Virtual Fence, and Night-Movement detectors.
    """

    def detect_human(self, camera_id: str) -> List[Dict[str, Any]]:
        conf = round(random.uniform(0.91, 0.98), 3)
        track_id = f"TRK-{random.randint(1000, 9999)}"
        # Normalized bounding box [x, y, w, h]
        x = round(random.uniform(0.35, 0.65), 2)
        y = round(random.uniform(0.30, 0.55), 2)
        w, h = 0.12, 0.32
        
        return [{
            "label": "PERSON",
            "confidence": conf,
            "track_id": track_id,
            "bbox": [x, y, w, h],
            "velocity_ms": round(random.uniform(1.0, 2.2), 1)
        }]

    def detect_vehicle(self, camera_id: str) -> List[Dict[str, Any]]:
        types = ["Heavy Utility Vehicle", "Civilian Pickup", "Armored Patrol", "All-Terrain Vehicle"]
        conf = round(random.uniform(0.88, 0.96), 3)
        track_id = f"TRK-{random.randint(1000, 9999)}"
        return [{
            "label": "VEHICLE",
            "vehicle_type": random.choice(types),
            "confidence": conf,
            "track_id": track_id,
            "bbox": [0.25, 0.40, 0.30, 0.25]
        }]

    def detect_anpr(self, camera_id: str) -> Dict[str, Any]:
        plates = ["DL01AB1234", "JK02AA9912", "HR26BK4321", "PB08CA5541", "DL04CD8821"]
        plate = random.choice(plates)
        conf = round(random.uniform(0.92, 0.99), 3)
        is_flagged = plate in WATCHLIST_PLATES
        
        return {
            "plate": plate,
            "confidence": conf,
            "is_flagged": is_flagged,
            "flag_reason": WATCHLIST_PLATES.get(plate)
        }

    def check_virtual_fence(self, camera_id: str, bbox: List[float]) -> Dict[str, Any]:
        zone = VIRTUAL_FENCE_ZONES.get(camera_id, {"zone_name": "General Perimeter Zone", "boundary_x": 0.45})
        bx = bbox[0]
        # If detection center crosses boundary_x
        breached = bx >= zone["boundary_x"]
        return {
            "breached": breached,
            "zone": zone["zone_name"],
            "boundary_marker": zone["boundary_x"],
            "detection_x": bx
        }

    def detect_night_movement(self, camera_id: str) -> Dict[str, Any]:
        thermal_temp = round(random.uniform(36.5, 38.5), 1)  # Human body heat signature in Celsius
        conf = round(random.uniform(0.89, 0.97), 3)
        return {
            "heat_signature_detected": True,
            "body_temp_c": thermal_temp,
            "confidence": conf,
            "mode": "THERMAL_IR_EDGE",
            "motion_detected": True
        }

    def run_full_inference(self, camera_id: str) -> Dict[str, Any]:
        """
        Executes edge pipeline for given camera feed.
        """
        humans = self.detect_human(camera_id)
        vehicles = self.detect_vehicle(camera_id) if camera_id in ("CAM-04", "CAM-01") else []
        anpr = self.detect_anpr(camera_id) if camera_id == "CAM-04" else None
        
        # Virtual fence check on human detection
        fence_eval = self.check_virtual_fence(camera_id, humans[0]["bbox"])
        
        # Night movement check for thermal cams
        night_eval = self.detect_night_movement(camera_id) if camera_id in ("CAM-02", "CAM-11", "CAM-07") else None

        return {
            "camera_id": camera_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "humans": humans,
            "vehicles": vehicles,
            "anpr": anpr,
            "fence_eval": fence_eval,
            "night_eval": night_eval
        }

    def process_real_frame(self, frame, camera_id: str, bop_id: str = "BOP-01", render_overlays: bool = False):
        """
        Executes production computer vision pipeline (Low-Light -> YOLO -> ByteTrack -> Face -> ANPR) on raw frame.
        """
        from edge_vision.pipeline import EdgeVisionPipeline
        if not hasattr(self, "_edge_pipelines"):
            self._edge_pipelines = {}
        if camera_id not in self._edge_pipelines:
            self._edge_pipelines[camera_id] = EdgeVisionPipeline(camera_id=camera_id, bop_id=bop_id)

        return self._edge_pipelines[camera_id].process_frame(frame, render_overlays=render_overlays)

ai_pipeline = AIInferencePipeline()

