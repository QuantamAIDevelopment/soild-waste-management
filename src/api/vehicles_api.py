"""Vehicle API endpoints for live vehicle management."""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from src.services.vehicle_service import VehicleService
from typing import List, Dict, Optional
import numpy as np

router = APIRouter(prefix="/api/vehicles", tags=["vehicles"])
vehicle_service = VehicleService()

@router.get("/live")
async def get_live_vehicles():
    """Get all live vehicles from SWM API."""
    try:
        vehicles_df = vehicle_service.get_live_vehicles()
        
        if vehicles_df is None or len(vehicles_df) == 0:
            return JSONResponse({
                "status": "warning",
                "message": "No vehicles found",
                "data": [],
                "count": 0
            })
        
        vehicles_df = vehicles_df.replace([np.inf, -np.inf, np.nan], None)
        vehicles_data = vehicles_df.to_dict('records')
        
        return JSONResponse({
            "status": "success",
            "message": f"Retrieved {len(vehicles_data)} vehicles",
            "data": vehicles_data,
            "count": len(vehicles_data),
            "active_count": len(vehicles_df[vehicles_df['status'].str.upper().isin(['ACTIVE', 'AVAILABLE', 'ONLINE'])])
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch live vehicles: {str(e)}")

@router.get("/{vehicle_id}")
async def get_vehicle(vehicle_id: str):
    """Get specific vehicle by ID."""
    try:
        vehicle_data = vehicle_service.get_vehicle_by_id(vehicle_id)
        
        if vehicle_data is None:
            raise HTTPException(status_code=404, detail=f"Vehicle {vehicle_id} not found")
        
        return JSONResponse({
            "status": "success",
            "data": vehicle_data
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch vehicle: {str(e)}")

@router.put("/{vehicle_id}/status")
async def update_vehicle_status(vehicle_id: str, status_data: Dict[str, str]):
    """Update vehicle status."""
    try:
        status = status_data.get('status')
        if not status:
            raise HTTPException(status_code=400, detail="Status is required")
        
        success = vehicle_service.update_vehicle_status(vehicle_id, status)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update vehicle status")
        
        return JSONResponse({
            "status": "success",
            "message": f"Vehicle {vehicle_id} status updated to {status}"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update vehicle status: {str(e)}")

@router.get("/")
async def list_vehicles():
    """List all vehicles with summary information."""
    try:
        vehicles_df = vehicle_service.get_live_vehicles()
        
        if vehicles_df is None or len(vehicles_df) == 0:
            return JSONResponse({
                "status": "warning",
                "message": "No vehicles found",
                "summary": {
                    "total": 0,
                    "active": 0,
                    "inactive": 0
                },
                "vehicles": []
            })
        
        # Calculate summary
        total_vehicles = len(vehicles_df)
        active_vehicles = len(vehicles_df[vehicles_df['status'].str.upper().isin(['ACTIVE', 'AVAILABLE', 'ONLINE'])])
        inactive_vehicles = total_vehicles - active_vehicles
        
        # Clean data before serialization
        vehicles_df = vehicles_df.replace([np.inf, -np.inf, np.nan], None)
        
        # Get vehicle list
        vehicles_list = []
        for _, vehicle in vehicles_df.iterrows():
            vehicles_list.append({
                "vehicle_id": vehicle.get('vehicle_id', 'Unknown'),
                "status": vehicle.get('status', 'Unknown'),
                "vehicle_type": vehicle.get('vehicle_type', 'Unknown'),
                "capacity": vehicle.get('capacity', 0) if vehicle.get('capacity') not in [None, np.nan] else 0
            })
        
        return JSONResponse({
            "status": "success",
            "summary": {
                "total": total_vehicles,
                "active": active_vehicles,
                "inactive": inactive_vehicles
            },
            "vehicles": vehicles_list
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list vehicles: {str(e)}")