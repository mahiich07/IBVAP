import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Settings:
    PROJECT_NAME: str = "IBVAP — Intelligent Border Video Analytics Platform"
    PROJECT_VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Server Binding
    HOST: str = os.getenv("IBVAP_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("IBVAP_PORT", 8000))
    
    # Security & Auth
    SECRET_KEY: str = os.getenv("IBVAP_SECRET_KEY", "b0rd3r-d3f3ns3-s1h-2026-s3cur3-k3y-m1l1t4ry-gr4d3")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours
    
    # Database
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", str(BASE_DIR / "ibvap.db"))
    
    # Edge Outpost Node Identity
    NODE_ID: str = os.getenv("IBVAP_NODE_ID", "BOP-ALPHA-EDGE-01")
    DEFAULT_BOP: str = "BOP Alpha"
    IS_EDGE_NODE: bool = True
    
    # CORS
    CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*"
    ]
    
    # Tamper-Evident Genesis Hash
    GENESIS_HASH: str = "0000000000000000000000000000000000000000000000000000000000000000"

settings = Settings()
