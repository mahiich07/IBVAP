"""
Automated Test Suite for IBVAP Edge Computer Vision Pipeline
Tests Low-Light Enhancement, YOLO Detection, ByteTrack, Face Detection, ANPR, and Optimization Tools.
"""

import unittest
import numpy as np
import cv2
import os
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(backend_dir))

from edge_vision.low_light_enhancement import MultiScaleRetinex, ZeroDCEEnhancer, LowLightEnhancer
from edge_vision.object_tracker import YOLOEdgeDetector, ByteTracker, Detection
from edge_vision.face_anpr import EdgeFaceDetector, ANPRPipeline
from edge_vision.optimization.export_onnx import export_yolo_to_onnx
from edge_vision.optimization.trt_compiler import TensorRTCompiler
from edge_vision.optimization.benchmark import EdgeModelBenchmark
from edge_vision.pipeline import EdgeVisionPipeline


class TestEdgeVisionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a synthetic dark night-time border surveillance frame (640x360)
        cls.dark_frame = np.full((360, 640, 3), 20, dtype=np.uint8)
        # Add some faint targets
        cv2.rectangle(cls.dark_frame, (200, 100), (280, 260), (45, 45, 45), -1)  # Person shape
        cv2.rectangle(cls.dark_frame, (400, 150), (550, 280), (50, 50, 50), -1)  # Vehicle shape
        # Add a simulated license plate region
        cv2.rectangle(cls.dark_frame, (430, 220), (520, 250), (200, 200, 200), -1)
        cv2.putText(cls.dark_frame, "DL01AB1234", (435, 242), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10, 10, 10), 1)

    # =========================================================================
    # 1. LOW-LIGHT ENHANCEMENT TESTS
    # =========================================================================
    def test_01_multiscale_retinex(self):
        msr = MultiScaleRetinex(sigmas=[15.0, 80.0])
        enhanced = msr.enhance(self.dark_frame)
        self.assertEqual(enhanced.shape, self.dark_frame.shape)
        self.assertEqual(enhanced.dtype, np.uint8)
        # Enhanced mean brightness must be higher than original dark frame
        self.assertGreater(float(np.mean(enhanced)), float(np.mean(self.dark_frame)))

    def test_02_zero_dce_enhancement(self):
        zero_dce = ZeroDCEEnhancer(iterations=4)
        enhanced = zero_dce.enhance(self.dark_frame)
        self.assertEqual(enhanced.shape, self.dark_frame.shape)
        self.assertGreater(float(np.mean(enhanced)), float(np.mean(self.dark_frame)))

    def test_03_unified_low_light_auto_switch(self):
        enhancer = LowLightEnhancer(method="auto", dark_threshold_lum=60.0)
        # Dark frame should trigger enhancement
        proc_dark = enhancer.process(self.dark_frame)
        self.assertGreater(float(np.mean(proc_dark)), float(np.mean(self.dark_frame)))

        # Bright daylight frame should pass through unchanged
        bright_frame = np.full((360, 640, 3), 180, dtype=np.uint8)
        proc_bright = enhancer.process(bright_frame)
        self.assertEqual(float(np.mean(proc_bright)), float(np.mean(bright_frame)))

    # =========================================================================
    # 2. OBJECT DETECTION & BYTETRACK TESTS
    # =========================================================================
    def test_04_yolo_detector(self):
        detector = YOLOEdgeDetector(conf_threshold=0.3)
        detections = detector.detect(self.dark_frame)
        self.assertIsInstance(detections, list)
        self.assertGreaterEqual(len(detections), 1)
        det = detections[0]
        self.assertIn(det.class_name, ["person", "car"])
        self.assertEqual(len(det.bbox), 4)

    def test_05_bytetrack_id_persistence(self):
        """
        Feeds consecutive frames of a moving target across 5 frames.
        Verifies that ByteTrack retains the exact same Track ID.
        """
        tracker = ByteTracker(track_thresh=0.4, track_low_thresh=0.1)

        initial_id = None
        # Simulate moving person: [x1, y1, x2, y2]
        for step in range(5):
            x1 = 100.0 + step * 4.0
            y1 = 80.0
            x2 = 150.0 + step * 4.0
            y2 = 200.0
            dets = [Detection(bbox=np.array([x1, y1, x2, y2], dtype=np.float32), confidence=0.92, class_id=0, class_name="person")]

            tracks = tracker.update(dets)
            self.assertGreaterEqual(len(tracks), 1)
            active_track = tracks[0]

            if step == 0:
                initial_id = active_track.track_id
            else:
                # Persistent ID must match across frames!
                self.assertEqual(active_track.track_id, initial_id)
                self.assertEqual(active_track.class_name, "person")

    # =========================================================================
    # 3. FACE DETECTION & ANPR TESTS
    # =========================================================================
    def test_06_edge_face_detector(self):
        face_detector = EdgeFaceDetector()
        # Create a frame with a face-sized skin-tone patch
        face_frame = np.full((360, 640, 3), 100, dtype=np.uint8)
        # Draw face oval in skin tones
        cv2.ellipse(face_frame, (320, 180), (45, 60), 0, 0, 360, (130, 150, 210), -1)

        faces = face_detector.detect(face_frame)
        self.assertIsInstance(faces, list)
        self.assertGreaterEqual(len(faces), 1)
        face = faces[0]
        self.assertGreater(face.confidence, 0.5)
        self.assertEqual(len(face.bbox), 4)

    def test_07_anpr_pipeline(self):
        anpr = ANPRPipeline()
        plates = anpr.process(self.dark_frame)
        self.assertIsInstance(plates, list)
        self.assertGreaterEqual(len(plates), 1)
        plate = plates[0]
        self.assertTrue(len(plate.plate_text) >= 6)
        self.assertGreater(plate.confidence, 0.8)
        # Check watchlist match for DL01AB1234
        if plate.plate_text == "DL01AB1234":
            self.assertTrue(plate.is_watchlist_match)
            self.assertIsNotNone(plate.watchlist_info)

    # =========================================================================
    # 4. EDGE MODEL OPTIMIZATION UTILITIES
    # =========================================================================
    def test_08_onnx_export_utility(self):
        target = export_yolo_to_onnx("models/custom_border_yolo.pt", "models/custom_border_yolo.onnx")
        self.assertEqual(target, "models/custom_border_yolo.onnx")

    def test_09_tensorrt_compiler_command(self):
        compiler = TensorRTCompiler(
            onnx_path="models/yolo11n.onnx",
            precision="fp16",
            use_dla=True,
            dla_core=0
        )
        cmd = compiler.build_trtexec_command()
        self.assertIn("trtexec", cmd)
        self.assertIn("--fp16", cmd)
        self.assertIn("--useDLACore=0", cmd)
        self.assertIn("--allowGPUFallback", cmd)
        self.assertIn("--memPoolSize=workspace:4096MiB", cmd)

        # Test bash script generation
        sh_path = compiler.generate_shell_script("test_compile.sh")
        self.assertTrue(os.path.exists(sh_path))
        os.remove(sh_path)

    def test_10_benchmark_profiler(self):
        benchmark = EdgeModelBenchmark(warmup_runs=2, benchmark_runs=5)
        # Dummy callable
        def dummy_inference(img):
            return cv2.GaussianBlur(img, (5, 5), 0)

        results = benchmark.profile_pipeline(dummy_inference, self.dark_frame)
        self.assertIn("mean_latency_ms", results)
        self.assertIn("throughput_fps", results)
        self.assertGreater(results["throughput_fps"], 0)

    # =========================================================================
    # 5. END-TO-END UNIFIED PIPELINE TEST
    # =========================================================================
    def test_11_end_to_end_pipeline(self):
        pipeline = EdgeVisionPipeline(camera_id="CAM-07", bop_id="BOP-01", enable_low_light=True)
        telemetry, annotated_frame = pipeline.process_frame(self.dark_frame, render_overlays=True)

        self.assertEqual(telemetry["camera_id"], "CAM-07")
        self.assertIn("latency_ms", telemetry)
        self.assertIn("fps", telemetry)
        self.assertIn("tracks", telemetry)
        self.assertIn("faces", telemetry)
        self.assertIn("anpr", telemetry)
        self.assertEqual(annotated_frame.shape, self.dark_frame.shape)


if __name__ == "__main__":
    unittest.main()
