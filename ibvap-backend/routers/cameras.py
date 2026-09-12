from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import StreamingResponse
from typing import List
from models.schemas import CameraOut
from services.rtsp_manager import rtsp_manager
from database import get_db

router = APIRouter(prefix="/cameras", tags=["Cameras"])

@router.get("", response_model=List[CameraOut])
def list_cameras():
    """
    List all active cameras, BOP locations, and real-time health metrics.
    """
    return rtsp_manager.list_all_cameras()

@router.get("/{camera_id}", response_model=CameraOut)
def get_camera(camera_id: str):
    """
    Retrieve single camera details and telemetry.
    """
    cameras = rtsp_manager.list_all_cameras()
    for cam in cameras:
        if cam["id"].lower() == camera_id.lower():
            return cam
    raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found in fleet")

@router.get("/{camera_id}/snapshot")
def get_camera_snapshot(camera_id: str):
    """
    Returns a real-time tactical JPEG snapshot with military HUD overlay.
    """
    frame_bytes = rtsp_manager.generate_synthetic_frame(camera_id)
    return Response(content=frame_bytes, media_type="image/jpeg")

@router.get("/{camera_id}/stream")
def get_camera_mjpeg_stream(camera_id: str):
    """
    Live multipart MJPEG stream from the edge camera proxy.
    """
    return StreamingResponse(
        rtsp_manager.mjpeg_stream_generator(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )
