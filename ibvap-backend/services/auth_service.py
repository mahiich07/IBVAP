import hmac
import hashlib
import json
import base64
import time
import secrets
from typing import Optional, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from config import settings
from database import get_db

security = HTTPBearer(auto_error=False)

def base64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b'=').decode('utf-8')

def base64url_decode(data: str) -> bytes:
    padding = '=' * (4 - (len(data) % 4)) if len(data) % 4 != 0 else ''
    return base64.urlsafe_b64decode((data + padding).encode('utf-8'))

def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    if not salt:
        salt = secrets.token_hex(16)
    pw_hash = hashlib.pbkdf2_hmac("sha256", password.encode('utf-8'), salt.encode('utf-8'), 100000).hex()
    return pw_hash, salt

def verify_password(plain_password: str, password_hash: str, salt: str) -> bool:
    expected_hash, _ = hash_password(plain_password, salt)
    return hmac.compare_digest(expected_hash, password_hash)

def create_access_token(data: Dict[str, Any], expires_delta_seconds: Optional[int] = None) -> str:
    header = {"alg": "HS256", "typ": "JWT"}
    header_bytes = json.dumps(header, separators=(',', ':')).encode('utf-8')
    header_encoded = base64url_encode(header_bytes)
    
    payload = data.copy()
    exp = int(time.time()) + (expires_delta_seconds or (settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60))
    payload["exp"] = exp
    payload["iat"] = int(time.time())
    
    payload_bytes = json.dumps(payload, separators=(',', ':')).encode('utf-8')
    payload_encoded = base64url_encode(payload_bytes)
    
    signing_input = f"{header_encoded}.{payload_encoded}".encode('utf-8')
    signature = hmac.new(settings.SECRET_KEY.encode('utf-8'), signing_input, hashlib.sha256).digest()
    sig_encoded = base64url_encode(signature)
    
    return f"{header_encoded}.{payload_encoded}.{sig_encoded}"

def decode_access_token(token: str) -> Dict[str, Any]:
    try:
        parts = token.split('.')
        if len(parts) != 3:
            raise ValueError("Malformed token")
        
        header_encoded, payload_encoded, sig_encoded = parts
        signing_input = f"{header_encoded}.{payload_encoded}".encode('utf-8')
        expected_sig = hmac.new(settings.SECRET_KEY.encode('utf-8'), signing_input, hashlib.sha256).digest()
        actual_sig = base64url_decode(sig_encoded)
        
        if not hmac.compare_digest(expected_sig, actual_sig):
            raise ValueError("Invalid signature")
        
        payload_bytes = base64url_decode(payload_encoded)
        payload = json.loads(payload_bytes.decode('utf-8'))
        
        if "exp" in payload and time.time() > payload["exp"]:
            raise ValueError("Token expired")
            
        return payload
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication token invalid or expired: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Dict[str, Any]:
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_access_token(credentials.credentials)
    operator_id = payload.get("sub")
    if not operator_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject claim"
        )
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT operator_id, full_name, role FROM users WHERE operator_id = ?", (operator_id,))
        user = cursor.fetchone()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Operator account not found"
            )
        return dict(user)
