"""Authentication endpoints for token management."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from src.services.auth_service import AuthService
from typing import Dict

router = APIRouter(prefix="/api/auth", tags=["authentication"])
auth_service = AuthService()

@router.get("/token/info")
async def get_token_info():
    """Get information about current token."""
    try:
        token_info = auth_service.get_token_info()
        return JSONResponse({
            "status": "success",
            "data": token_info
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get token info: {str(e)}")

@router.post("/token/refresh")
async def refresh_token():
    """Force refresh the authentication token."""
    try:
        success = auth_service.refresh_token()
        
        if success:
            token_info = auth_service.get_token_info()
            return JSONResponse({
                "status": "success",
                "message": "Token refreshed successfully",
                "data": token_info
            })
        else:
            raise HTTPException(status_code=500, detail="Failed to refresh token")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Token refresh failed: {str(e)}")

@router.get("/token/validate")
async def validate_token():
    """Validate current token and get a fresh one if needed."""
    try:
        token = auth_service.get_valid_token()
        
        if token:
            token_info = auth_service.get_token_info()
            return JSONResponse({
                "status": "success",
                "message": "Token is valid",
                "data": token_info
            })
        else:
            raise HTTPException(status_code=401, detail="Could not obtain valid token")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Token validation failed: {str(e)}")