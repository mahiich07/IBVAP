import json
from fastapi import APIRouter, Query
from typing import List, Optional
from models.schemas import EventOut, ChainVerificationResponse
from database import get_db
from services.hash_chain import hash_chain

router = APIRouter(prefix="/events", tags=["Events & Hash Chain"])

@router.get("", response_model=List[EventOut])
def get_events(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    camera_id: Optional[str] = None
):
    """
    Retrieve historical events with cryptographic hash-chain verification status.
    """
    with get_db() as conn:
        cursor = conn.cursor()
        query = """
            SELECT id, type, camera_id as camera, bop_id as bop, timestamp, severity, integrity_status as integrity, hash, prev_hash as prevHash, payload
            FROM events
        """
        params = []
        if camera_id:
            query += " WHERE camera_id = ?"
            params.append(camera_id)
            
        query += " ORDER BY rowid DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        results = []
        for r in rows:
            d = dict(r)
            try:
                d["payload"] = json.loads(d["payload"]) if d["payload"] else None
            except Exception:
                d["payload"] = None
            results.append(d)
        return results

@router.post("/verify", response_model=ChainVerificationResponse)
def verify_hash_chain():
    """
    Perform a complete cryptographic audit of the SHA-256 event hash chain.
    Recalculates every block hash from genesis to the latest record.
    """
    result = hash_chain.verify_chain()
    return ChainVerificationResponse(**result)
