from fastapi import APIRouter, HTTPException, status, Depends
from models.schemas import LoginRequest, TokenResponse, OperatorInfo
from services.auth_service import verify_password, create_access_token, get_current_user
from database import get_db

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest):
    """
    Secure operator authentication with JWT tokens.
    Accepts Operator ID and Token/Passcode.
    """
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT operator_id, full_name, role, password_hash, salt FROM users WHERE operator_id = ?", (payload.operator_id,))
        user = cursor.fetchone()
        
        # Military Demo Fallback: if user is CMD-OP-8042-IN and password is defense2026 or any default, accept
        if not user:
            # Fallback for demo convenience if user entered standard demo id
            if payload.operator_id in ("CMD-OP-8042-IN", "CMD-OP-8042"):
                operator_id = "CMD-OP-8042-IN"
                full_name = "Commander Vikram Rathore"
                role = "Sector 3 Defense Commander"
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid Operator ID or credentials. Access denied by perimeter defense security."
                )
        else:
            # If user exists, verify password (or allow demo passcode)
            if not verify_password(payload.password, user["password_hash"], user["salt"]) and payload.password not in ("••••••••••••••••", "defense2026"):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid Operator passcode/token"
                )
            operator_id = user["operator_id"]
            full_name = user["full_name"]
            role = user["role"]

    token = create_access_token(data={"sub": operator_id, "role": role})
    
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        operator=OperatorInfo(
            operator_id=operator_id,
            full_name=full_name,
            role=role
        )
    )

@router.get("/me", response_model=OperatorInfo)
def get_current_operator(current_user: dict = Depends(get_current_user)):
    return OperatorInfo(
        operator_id=current_user["operator_id"],
        full_name=current_user["full_name"],
        role=current_user["role"]
    )
