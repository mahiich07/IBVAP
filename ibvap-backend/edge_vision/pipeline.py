"""
Master Edge Computer Vision Pipeline
Orchestrates Low-Light Enhancement, YOLO Detection, ByteTrack Tracking, Face Detection, and ANPR.
Produces structured telemetry payloads for the IBVAP real-time WebSocket Gateway.
"""

import cv2
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

from .low_light_enhancement import LowLightEnhancer
from .object_tracker import YOLOEdgeDetector, ByteTracker, TrackedObject
from .face_anpr import EdgeFaceDetector, ANPRPipeline, DetectedFace, DetectedPlate


class EdgeVisionPipeline:
    """
    Unified edge computer vision inference engine for border outposts.
    """

    def __init__(
        self,
        camera_id: str = "CAM-01",
        bop_id: str = "BOP-01",
        enable_low_light: bool = True,
        low_light_method: str = "auto",
        yolo_weights: Optional[str] = None,
        face_weights: Optional[str] = None,
        ocr_engine: str = "auto",
        conf_thresh: float = 0.40,
        enable_tracking: bool = True,
        enable_faces: bool = True,
        enable_anpr: bool = True
    ):
        self.camera_id = camera_id
        self.bop_id = bop_id
        self.enable_low_light = enable_low_light
        self.enable_tracking = enable_tracking
        self.enable_faces = enable_faces
        self.enable_anpr = enable_anpr

        # Initialize sub-modules
        self.enhancer = LowLightEnhancer(method=low_light_method)
        self.detector = YOLOEdgeDetector(model_path=yolo_weights, conf_threshold=conf_thresh)
        self.tracker = ByteTracker(track_thresh=conf_thresh, track_low_thresh=0.1)
        self.face_detector = EdgeFaceDetector(model_path=face_weights)
        self.anpr_engine = ANPRPipeline(ocr_engine=ocr_engine)

        # Performance monitoring
        self.last_latency_ms: float = 0.0
        self.total_frames_processed: int = 0

    def process_frame(
        self,
        frame: np.ndarray,
        render_overlays: bool = False
    ) -> Tuple[Dict[str, Any], np.ndarray]:
        """
        Executes full multi-stage inference on a raw frame:
        1. Low-light enhancement (Zero-DCE / MSRCR)
        2. YOLO human and vehicle detection
        3. ByteTrack persistent ID tracking with Kalman filtering
        4. Face detection and landmark extraction
        5. ANPR plate localization, character segmentation, and watchlist matching
        6. Optional tactical HUD drawing

        Returns: (telemetry_dict, processed_or_annotated_frame)
        """
        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        # Stage 1: Low-light enhancement
        enhanced_frame = self.enhancer.process(frame) if self.enable_low_light else frame

        # Stage 2: YOLO Object Detection
        detections = self.detector.detect(enhanced_frame)

        # Stage 3: ByteTrack Multi-Target Tracking
        tracked_objects: List[TrackedObject] = []
        if self.enable_tracking:
            tracked_objects = self.tracker.update(detections)

        # Separate vehicle boxes for ANPR targeting
        vehicle_boxes = [t.bbox for t in tracked_objects if t.class_name in ("car", "truck", "bus")]

        # Stage 4: Face Detection
        faces: List[DetectedFace] = []
        if self.enable_faces:
            faces = self.face_detector.detect(enhanced_frame)

        # Stage 5: ANPR License Plate Recognition
        plates: List[DetectedPlate] = []
        if self.enable_anpr:
            plates = self.anpr_engine.process(enhanced_frame, vehicle_bboxes=vehicle_boxes)

        t1 = time.perf_counter()
        self.last_latency_ms = round((t1 - t0) * 1000.0, 2)
        self.total_frames_processed += 1

        # Format Telemetry for IBVAP API and WebSocket
        telemetry = {
            "camera_id": self.camera_id,
            "bop_id": self.bop_id,
            "timestamp": time.time(),
            "latency_ms": self.last_latency_ms,
            "fps": round(1000.0 / max(1e-4, self.last_latency_ms), 1),
            "tracks": [
                {
                    "track_id": f"TRK-{t.track_id:04d}",
                    "class": t.class_name,
                    "confidence": round(t.confidence, 3),
                    "bbox": [round(float(coord), 1) for coord in t.bbox],
                    "velocity": t.velocity,
                    "state": t.state
                }
                for t in tracked_objects
            ],
            "faces": [
                {
                    "bbox": [int(coord) for coord in f.bbox],
                    "confidence": round(f.confidence, 3),
                    "has_landmarks": f.landmarks is not None
                }
                for f in faces
            ],
            "anpr": [
                {
                    "plate": p.plate_text,
                    "confidence": round(p.confidence, 3),
                    "bbox": [int(coord) for coord in p.bbox],
                    "characters_count": len(p.characters),
                    "is_watchlist_match": p.is_watchlist_match,
                    "watchlist_info": p.watchlist_info
                }
                for p in plates
            ]
        }

        # Stage 6: Annotate visual overlays if requested
        output_frame = enhanced_frame.copy() if render_overlays else enhanced_frame
        if render_overlays:
            self._render_tactical_overlays(output_frame, tracked_objects, faces, plates)

        return telemetry, output_frame

    def _render_tactical_overlays(
        self,
        frame: np.ndarray,
        tracks: List[TrackedObject],
        faces: List[DetectedFace],
        plates: List[DetectedPlate]
    ):
        """Draws bounding boxes, trajectory trails, and OCR tags onto the frame."""
        # 1. Draw Tracked Objects
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.bbox)
            color = (0, 255, 127) if t.class_name == "person" else (255, 165, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = f"TRK-{t.track_id} {t.class_name.upper()} {round(t.confidence*100)}%"
            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Draw trajectory path
            if len(t.history) > 1:
                pts = np.array(t.history, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=2)

        # 2. Draw Face Landmarks
        for f in faces:
            x, y, w, h = f.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(frame, "FACE", (x, max(20, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)
            if f.landmarks is not None:
                for lx, ly in f.landmarks:
                    cv2.circle(frame, (int(lx), int(ly)), 2, (0, 255, 255), -1)

        # 3. Draw ANPR Plates
        for p in plates:
            px, py, pw, ph = p.bbox
            tag_color = (0, 0, 255) if p.is_watchlist_match else (0, 255, 0)
            cv2.rectangle(frame, (px, py), (px + pw, py + ph), tag_color, 2)
            tag_str = f"PLATE: {p.plate_text}"
            if p.is_watchlist_match:
                tag_str += " [FLAGGED]"
            cv2.putText(frame, tag_str, (px, max(20, py - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, tag_color, 2)

        # 4. HUD Telemetry Banner
        hud_text = f"IBVAP EDGE AI • LATENCY: {self.last_latency_ms}ms • CAM: {self.camera_id}"
        cv2.putText(frame, hud_text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
