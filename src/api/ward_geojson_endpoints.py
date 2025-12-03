"""API endpoints for ward GeoJSON management."""
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Dict, Any
import json
import geopandas as gpd
import tempfile
import os
from src.services.ward_geojson_service import WardGeoJSONService
from loguru import logger

router = APIRouter(prefix="/api/ward-geojson", tags=["ward-geojson"])

ward_service = WardGeoJSONService()

@router.post("/upload/{ward_no}")
async def upload_ward_boundary(
    ward_no: str,
    geojson_file: UploadFile = File(..., description="Ward boundary GeoJSON file"),
    name: str = Form(default="Wardboundary", description="Boundary name")
):
    """
    Upload ward boundary GeoJSON to external API.
    
    - **ward_no**: Ward number (e.g., "Ward 29")
    - **geojson_file**: GeoJSON file containing ward boundary
    - **name**: Optional name for the boundary (default: "Wardboundary")
    """
    try:
        # Validate file type
        if not geojson_file.filename.lower().endswith('.geojson'):
            raise HTTPException(status_code=400, detail="File must be GeoJSON format")
        
        # Read and parse GeoJSON
        with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson') as tmp_file:
            content = await geojson_file.read()
            tmp_file.write(content)
            tmp_file.flush()
            
            # Load with geopandas to validate
            gdf = gpd.read_file(tmp_file.name)
            
            # Convert to WGS84 if needed
            if gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
            
            # Convert to GeoJSON dict
            geojson_data = json.loads(gdf.to_json())
            
            # Clean up temp file
            os.unlink(tmp_file.name)
        
        # Upload to external API
        result = ward_service.upload_ward_geojson(ward_no, geojson_data, name)
        
        if result["status"] == "success":
            return JSONResponse(content=result, status_code=201)
        elif result["status"] == "exists":
            return JSONResponse(content=result, status_code=200)
        else:
            raise HTTPException(status_code=500, detail=result.get("message", "Upload failed"))
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading ward boundary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{ward_no}")
async def get_ward_boundary(ward_no: str):
    """
    Retrieve ward boundary GeoJSON from external API.
    
    - **ward_no**: Ward number (e.g., "Ward 29")
    
    Returns the GeoJSON data for the specified ward.
    """
    try:
        geojson_data = ward_service.get_ward_geojson(ward_no)
        
        if geojson_data is None:
            raise HTTPException(
                status_code=404, 
                detail=f"Ward boundary not found for {ward_no}"
            )
        
        return JSONResponse(content=geojson_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving ward boundary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-and-optimize/{ward_no}")
async def upload_ward_and_optimize(
    ward_no: str,
    ward_geojson: UploadFile = File(..., description="Ward boundary GeoJSON"),
    roads_file: UploadFile = File(..., description="Roads GeoJSON"),
    buildings_file: UploadFile = File(..., description="Buildings GeoJSON"),
    name: str = Form(default="Wardboundary")
):
    """
    Upload ward boundary to external API and then run optimization.
    
    This combines ward upload with route optimization in one step.
    """
    try:
        # First upload ward boundary
        with tempfile.NamedTemporaryFile(delete=False, suffix='.geojson') as tmp_file:
            content = await ward_geojson.read()
            tmp_file.write(content)
            tmp_file.flush()
            
            gdf = gpd.read_file(tmp_file.name)
            if gdf.crs != 'EPSG:4326':
                gdf = gdf.to_crs('EPSG:4326')
            
            geojson_data = json.loads(gdf.to_json())
            os.unlink(tmp_file.name)
        
        # Upload to external API
        upload_result = ward_service.upload_ward_geojson(ward_no, geojson_data, name)
        
        return JSONResponse(content={
            "ward_upload": upload_result,
            "message": f"Ward boundary uploaded. Now you can call /optimize-routes with wardNo={ward_no}",
            "next_step": f"POST /optimize-routes with wardNo={ward_no}"
        })
        
    except Exception as e:
        logger.error(f"Error in upload and optimize: {e}")
        raise HTTPException(status_code=500, detail=str(e))
