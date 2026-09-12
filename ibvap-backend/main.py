import asyncio
import logging
import random
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from database import init_db
from routers.auth import router as auth_router
from routers.cameras import router as cameras_router
from routers.alerts import router as alerts_router
from routers.events import router as events_router
from routers.system import router as system_router
from routers.websocket import router as ws_router

from services.websocket_gateway import ws_gateway
from services.ai_pipeline import ai_pipeline
from services.rules_engine import rules_engine
from services.rtsp_manager import rtsp_manager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ibvap.main")

simulator_task = None

async def background_edge_telemetry_simulator():
    """
    Background worker simulating live edge detection telemetry and periodic threats.
    Broadcasts bounding box detections, camera FPS changes, and triggered alerts.
    """
    logger.info("Starting IBVAP edge telemetry background simulator...")
    cams = [
        ("CAM-07", "BOP-01", "BOP Alpha"),
        ("CAM-08", "BOP-01", "BOP Alpha"),
        ("CAM-04", "BOP-02", "BOP Bravo"),
        ("CAM-02", "BOP-02", "BOP Bravo"),
        ("CAM-11", "BOP-03", "BOP Charlie")
    ]
    
    tick = 0
    while True:
        try:
            await asyncio.sleep(2.5)
            tick += 1
            
            # Select random active camera
            cam_id, bop_id, bop_name = random.choice(cams)
            
            # Run simulated inference
            inference = ai_pipeline.run_full_inference(cam_id)
            
            # Format live detection frame for WebSocket clients
            boxes = []
            detections_summary = []
            for h in inference.get("humans", []):
                bx = h["bbox"]
                boxes.append({
                    "x": bx[0], "y": bx[1], "w": bx[2], "h": bx[3],
                    "label": "PERSON", "confidence": h["confidence"],
                    "track_id": h["track_id"]
                })
                detections_summary.append(f"PERSON {round(h['confidence']*100, 1)}%")

            for v in inference.get("vehicles", []):
                bx = v["bbox"]
                boxes.append({
                    "x": bx[0], "y": bx[1], "w": bx[2], "h": bx[3],
                    "label": "VEHICLE", "confidence": v["confidence"],
                    "track_id": v["track_id"]
                })
                detections_summary.append(f"VEHICLE {round(v['confidence']*100, 1)}%")

            if inference.get("anpr"):
                anpr = inference["anpr"]
                detections_summary.append(f"PLATE: {anpr['plate']}")

            # Broadcast detection frame to connected WebSockets
            if ws_gateway.connection_count > 0:
                await ws_gateway.broadcast("detection_frame", {
                    "camera_id": cam_id,
                    "bop_id": bop_id,
                    "bop_name": bop_name,
                    "fps": random.randint(28, 30),
                    "boxes": boxes,
                    "detections": detections_summary
                })

            # Every ~8-12 ticks, evaluate and trigger an active alert
            if tick % 8 == 0:
                # Force a breach condition for testing
                inference["fence_eval"]["breached"] = True
                await rules_engine.evaluate_and_dispatch(cam_id, bop_id, bop_name, inference)

        except asyncio.CancelledError:
            logger.info("Background simulator stopped.")
            break
        except Exception as e:
            logger.error(f"Simulator error: {e}")
            await asyncio.sleep(5)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Initialize DB and cryptographic ledger
    init_db()
    logger.info("Database and cryptographic hash chain initialized successfully.")
    
    # 2. Launch background telemetry simulator
    global simulator_task
    simulator_task = asyncio.create_task(background_edge_telemetry_simulator())
    
    yield
    
    # Cleanup on shutdown
    if simulator_task:
        simulator_task.cancel()
        try:
            await simulator_task
        except asyncio.CancelledError:
            pass

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="Military-grade Intelligent Border Video Analytics Platform (IBVAP) Edge & Streaming Backend",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(cameras_router, prefix=settings.API_V1_STR)
app.include_router(alerts_router, prefix=settings.API_V1_STR)
app.include_router(events_router, prefix=settings.API_V1_STR)
app.include_router(system_router, prefix=settings.API_V1_STR)
app.include_router(ws_router)

@app.get("/", tags=["Health Check"])
def root():
    return {
        "platform": settings.PROJECT_NAME,
        "status": "OPERATIONAL",
        "node_id": settings.NODE_ID,
        "version": settings.PROJECT_VERSION,
        "docs_url": "/docs"
    }

@app.get("/health", tags=["Health Check"])
def health_check():
    return {
        "status": "healthy",
        "edge_node": settings.NODE_ID,
        "websocket_active_clients": ws_gateway.connection_count
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)
