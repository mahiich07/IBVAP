"""
RTSP Stream Handler for Edge Devices (NVIDIA Jetson / Linux)
Provides threaded frame capture, zero-latency frame dropping, and Jetson GStreamer hardware acceleration.
"""

import cv2
import time
import threading
import logging
from typing import Optional, Tuple
import numpy as np

logger = logging.getLogger("ibvap.stream_handler")


class RTSPStreamHandler:
    """
    Thread-safe RTSP stream ingest engine.
    Runs a persistent background thread continuously flushing older frames to prevent RTSP buffer lag.
    Supports NVIDIA Jetson hardware-accelerated GStreamer pipelines (nvv4l2decoder).
    """

    def __init__(
        self,
        stream_url: str,
        camera_id: str = "CAM-01",
        use_jetson_gstreamer: bool = False,
        target_fps: int = 30,
        reconnect_delay_sec: float = 3.0,
        buffer_size: int = 1
    ):
        self.stream_url = stream_url
        self.camera_id = camera_id
        self.use_jetson_gstreamer = use_jetson_gstreamer
        self.target_fps = target_fps
        self.reconnect_delay_sec = reconnect_delay_sec
        self.buffer_size = buffer_size

        self._cap: Optional[cv2.VideoCapture] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None

        # Telemetry metrics
        self.fps_measured: float = 0.0
        self.frame_count: int = 0
        self.dropped_frames: int = 0
        self.is_connected: bool = False
        self.last_frame_timestamp: float = 0.0

    def _build_jetson_gstreamer_pipeline(self) -> str:
        """
        Builds high-performance NVIDIA hardware accelerated GStreamer pipeline for Jetson Orin/Xavier.
        Utilizes nvv4l2decoder (Hardware H.264/H.265 decoder) and nvvidconv.
        """
        pipeline = (
            f"rtspsrc location={self.stream_url} latency=100 ! "
            f"rtph264depay ! h264parse ! "
            f"nvv4l2decoder ! nvvidconv ! "
            f"video/x-raw, format=BGRx ! videoconvert ! "
            f"video/x-raw, format=BGR ! appsink drop=1 sync=false max-buffers={self.buffer_size}"
        )
        return pipeline

    def _open_capture(self) -> bool:
        """Attempts to open the video capture device."""
        if self._cap is not None:
            self._cap.release()

        if self.use_jetson_gstreamer:
            gst_pipeline = self._build_jetson_gstreamer_pipeline()
            logger.info(f"[{self.camera_id}] Opening Jetson GStreamer pipeline...")
            self._cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
        else:
            logger.info(f"[{self.camera_id}] Opening standard OpenCV stream capture...")
            self._cap = cv2.VideoCapture(self.stream_url)

        if self._cap.isOpened():
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
            self.is_connected = True
            logger.info(f"[{self.camera_id}] RTSP stream successfully connected: {self.stream_url}")
            return True
        else:
            self.is_connected = False
            logger.warning(f"[{self.camera_id}] Failed to open RTSP stream: {self.stream_url}")
            return False

    def _capture_loop(self):
        """Background thread loop fetching newest frames continuously."""
        fps_timer = time.time()
        frames_in_interval = 0

        while self._running:
            if not self.is_connected or self._cap is None or not self._cap.isOpened():
                success = self._open_capture()
                if not success:
                    time.sleep(self.reconnect_delay_sec)
                    continue

            # Read frame
            ret, frame = self._cap.read()
            if not ret or frame is None:
                logger.warning(f"[{self.camera_id}] Read error or dropped stream link. Reconnecting...")
                self.is_connected = False
                time.sleep(self.reconnect_delay_sec)
                continue

            now = time.time()
            with self._frame_lock:
                self._latest_frame = frame
                self.last_frame_timestamp = now
                self.frame_count += 1
                frames_in_interval += 1

            # Update FPS metric every 1 second
            if now - fps_timer >= 1.0:
                self.fps_measured = round(frames_in_interval / (now - fps_timer), 1)
                frames_in_interval = 0
                fps_timer = now

            # Throttle slightly to match target FPS
            sleep_budget = (1.0 / self.target_fps) - (time.time() - now)
            if sleep_budget > 0.001:
                time.sleep(sleep_budget)

    def start(self):
        """Starts the capture background thread."""
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(target=self._capture_loop, daemon=True, name=f"rtsp-{self.camera_id}")
        self._worker_thread.start()

    def stop(self):
        """Stops the worker thread and releases OpenCV capture."""
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        self.is_connected = False
        logger.info(f"[{self.camera_id}] Stream handler stopped.")

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retrieves the latest available video frame without blocking.
        Returns: (success_bool, frame_ndarray)
        """
        with self._frame_lock:
            if self._latest_frame is not None:
                # Return a copy to prevent race conditions during inference mutations
                return True, self._latest_frame.copy()
            return False, None
