"""FastAPI integration for geospatial route optimization."""
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Header, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Any
import numpy as np
import tempfile
import os
import shutil
import geopandas as gpd
import pandas as pd
import folium
from sklearn.cluster import KMeans
import json
import networkx as nx
import numpy as np
from shapely.geometry import Point
from scipy.spatial.distance import cdist
import math
from src.services.vehicle_service import VehicleService
from src.services.scheduler_service import SchedulerService
from src.services.ward_geojson_service import WardGeoJSONService
from src.api.vehicles_api import router as vehicles_router
from src.api.auth_endpoints import router as auth_router
from src.api.ward_geojson_endpoints import router as ward_geojson_router
from src.routing.capacity_optimizer import CapacityRouteOptimizer

from loguru import logger
import warnings
import requests
from dotenv import load_dotenv
from src.configurations.config import Config

# Load environment variables
load_dotenv()

# Suppress specific geographic CRS warnings for intentional lat/lon usage in maps
warnings.filterwarnings('ignore', message='.*Geometry is in a geographic CRS.*')

# API Key for authentication from config
API_KEY = Config.API_KEY

# Security scheme
security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key from Authorization header."""
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials

app = FastAPI(
    title="Geospatial AI Route Optimizer",
    description="Dynamic garbage collection route optimization using live vehicle data and road network",
    version="2.0.0",
    openapi_tags=[
        {
            "name": "clusters",
            "description": "Cluster management and road coordinate retrieval"
        },
        {
            "name": "optimization",
            "description": "Route optimization operations"
        },
        {
            "name": "maps",
            "description": "Map generation and visualization"
        },
        {
            "name": "scheduler",
            "description": "Automatic daily scheduling operations"
        }
    ]
)

# Initialize services
vehicle_service = VehicleService()
scheduler_service = SchedulerService()
ward_geojson_service = WardGeoJSONService()

# Include vehicle API routes
app.include_router(vehicles_router)

# Include authentication API routes
app.include_router(auth_router)

# Include ward GeoJSON API routes
app.include_router(ward_geojson_router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Start the SWM Agent with automatic daily scheduling."""
    logger.info("🚀 Starting SWM Agent...")
    scheduler_service.start_scheduler()
    logger.success("✅ SWM Agent startup complete (daily scheduler enabled)")

@app.on_event("shutdown")
async def shutdown_event():
    """Stop the SWM Agent."""
    logger.info("🛑 Shutting down SWM Agent...")
    if scheduler_service.is_running:
        scheduler_service.stop_scheduler()
    logger.info("✅ SWM Agent shutdown complete")

def safe_argmin(distances):
    """Safely get argmin, handling empty sequences."""
    if not distances or len(distances) == 0:
        return None
    return np.argmin(distances)

def convert_numpy_types(obj):
    """Convert numpy types to JSON serializable types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        val = float(obj)
        # Handle NaN, inf, -inf values
        if not np.isfinite(val):
            return None
        return val
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, float):
        # Handle regular Python floats that might be NaN or inf
        if not np.isfinite(obj):
            return None
        return obj
    return obj

@app.post("/optimize-routes", tags=["optimization"])
async def optimize_routes(
    wardNo: str = Form(..., description="Ward number to fetch data and filter vehicles"),
    roads_file: UploadFile = File(default=None, description="Roads GeoJSON file (optional - fetched from API if not provided)"),
    buildings_file: UploadFile = File(default=None, description="Buildings GeoJSON file (optional - fetched from API if not provided)"), 
    ward_geojson: UploadFile = File(default=None, description="Ward boundary GeoJSON file (optional - fetched from API if not provided)")
):
    """Create clusters using ward data from API or uploaded files."""
    
    # Validate wardNo
    if not wardNo or not wardNo.strip():
        raise HTTPException(status_code=400, detail="Ward number is required")
    
    # Validate uploaded files if provided
    if roads_file and roads_file.filename and not roads_file.filename.lower().endswith('.geojson'):
        raise HTTPException(status_code=400, detail="Roads file must be GeoJSON")
    if buildings_file and buildings_file.filename and not buildings_file.filename.lower().endswith('.geojson'):
        raise HTTPException(status_code=400, detail="Buildings file must be GeoJSON")
    if ward_geojson and ward_geojson.filename and not ward_geojson.filename.lower().endswith('.geojson'):
        raise HTTPException(status_code=400, detail="Ward boundary file must be GeoJSON")
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            roads_path = os.path.join(temp_dir, "roads.geojson")
            buildings_path = os.path.join(temp_dir, "buildings.geojson")
            ward_path = os.path.join(temp_dir, "ward.geojson")
            
            # Check if all files are uploaded or need to fetch from API
            # Prioritize uploaded files over API
            use_api = not (roads_file and roads_file.filename and buildings_file and buildings_file.filename)
            
            if use_api:
                # Fetch all data from API
                logger.info(f"Fetching ward data from API for {wardNo}")
                ward_data = ward_geojson_service.get_ward_data(wardNo.strip())
                if not ward_data:
                    # Try with "Ward " prefix
                    ward_with_prefix = f"Ward {wardNo.strip()}"
                    logger.info(f"Retrying with ward name: {ward_with_prefix}")
                    ward_data = ward_geojson_service.get_ward_data(ward_with_prefix)
                
                # API only has ward boundary, not buildings/roads - require file upload
                if not ward_data:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Ward {wardNo} API only contains boundary data. Please upload buildings_file and roads_file as GeoJSON."
                    )
                
                # This code path won't be reached since API doesn't have complete data
                # Kept for future when API supports buildings/roads
                with open(ward_path, "w") as f:
                    json.dump(ward_data['ward_boundary'], f)
                with open(buildings_path, "w") as f:
                    json.dump(ward_data['buildings'], f)
                with open(roads_path, "w") as f:
                    json.dump(ward_data['roads'], f)
                logger.success(f"Successfully fetched all ward data from API for {wardNo}")
            else:
                # Use uploaded files
                logger.info(f"Using uploaded files for {wardNo}")
                
                with open(roads_path, "wb") as f:
                    shutil.copyfileobj(roads_file.file, f)
                with open(buildings_path, "wb") as f:
                    shutil.copyfileobj(buildings_file.file, f)
                
                # Ward boundary - upload or API
                if ward_geojson and ward_geojson.filename:
                    with open(ward_path, "wb") as f:
                        shutil.copyfileobj(ward_geojson.file, f)
                else:
                    ward_data = ward_geojson_service.get_ward_geojson(wardNo.strip())
                    if not ward_data:
                        raise HTTPException(status_code=404, detail=f"Ward boundary not found for {wardNo}")
                    with open(ward_path, "w") as f:
                        json.dump(ward_data, f)
            
            # Load geospatial data
            buildings_gdf = gpd.read_file(buildings_path)
            roads_gdf = gpd.read_file(roads_path)
            
            # Validate data
            if len(buildings_gdf) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"No buildings found in ward {wardNo}. Please upload a valid buildings GeoJSON file."
                )
            if len(roads_gdf) == 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"No roads found in ward {wardNo}. Please upload a valid roads GeoJSON file."
                )
            
            # Convert to WGS84 if needed
            if buildings_gdf.crs != 'EPSG:4326':
                buildings_gdf = buildings_gdf.to_crs('EPSG:4326')
            if roads_gdf.crs != 'EPSG:4326':
                roads_gdf = roads_gdf.to_crs('EPSG:4326')
            
            # Get vehicle data from live API
            try:
                if not wardNo or wardNo.strip() in ["string", ""]:
                    raise HTTPException(status_code=400, detail="Valid ward number required")
                
                # Get ALL vehicles in ward (including INACTIVE) - filtering happens later
                vehicles_df = vehicle_service.get_vehicles_by_ward(wardNo.strip(), include_all_status=True)
                vehicles_csv_path = os.path.join(temp_dir, "vehicles.csv")
                vehicles_df.to_csv(vehicles_csv_path, index=False)
                vehicles_path = vehicles_csv_path
                vehicle_source = "Live API (Ward Filtered)"
                logger.info(f"[API] Using {len(vehicles_df)} vehicles from ward {wardNo} (all statuses)")
                
                if len(vehicles_df) == 0:
                    raise HTTPException(status_code=404, detail=f"No vehicles found in ward {wardNo}")
                
                # Filter for Swachh Auto vehicles only
                swachh_auto_vehicles = vehicles_df[
                    vehicles_df['vehicle_type'].notna() & 
                    (vehicles_df['vehicle_type'].str.upper().str.contains('SWACHH AUTO|SWACHAUTO|SWACHH_AUTO', na=False))
                ]
                
                if len(swachh_auto_vehicles) == 0:
                    return JSONResponse(
                        status_code=200,
                        content={
                            "status": "error",
                            "error_type": "no_swachh_auto",
                            "message": f"No Swachh Auto vehicles found in ward {wardNo}. Cannot create clusters.",
                            "wardNo": wardNo,
                            "total_vehicles_in_ward": len(vehicles_df),
                            "swachh_auto_count": 0,
                            "vehicle_types_found": vehicles_df['vehicle_type'].value_counts().to_dict() if 'vehicle_type' in vehicles_df.columns else {},
                            "recommendation": "Please ensure Swachh Auto vehicles are available in the ward."
                        }
                    )
                
                # Filter for ACTIVE Swachh Auto vehicles only
                active_swachh_auto = swachh_auto_vehicles[
                    swachh_auto_vehicles['status'].notna() & 
                    (swachh_auto_vehicles['status'].str.upper().isin(['ACTIVE', 'AVAILABLE', 'ONLINE', 'READY', 'OPERATIONAL', 'IN_SERVICE', 'HALTED', 'IDEL']))
                ]
                
                if len(active_swachh_auto) == 0:
                    status_distribution = swachh_auto_vehicles['status'].value_counts().to_dict()
                    return JSONResponse(
                        status_code=200,
                        content={
                            "status": "error",
                            "error_type": "no_active_swachh_auto",
                            "message": f"No ACTIVE Swachh Auto vehicles found in ward {wardNo}. Cannot generate routes.",
                            "wardNo": wardNo,
                            "total_swachh_auto": len(swachh_auto_vehicles),
                            "active_swachh_auto": 0,
                            "status_distribution": status_distribution,
                            "recommendation": "Please activate Swachh Auto vehicles in the ward to generate routes."
                        }
                    )
                
                # Create clusters based on SWACHH AUTO count (all statuses)
                from src.clustering.assign_buildings import BuildingClusterer
                clusterer = BuildingClusterer()
                
                if 'wardNo' not in buildings_gdf.columns and 'wardNo' not in buildings_gdf.columns:
                    buildings_gdf['wardNo'] = wardNo
                
                swachh_auto_count = len(swachh_auto_vehicles)
                clustered_buildings = clusterer.cluster_buildings_with_vehicles(buildings_gdf, swachh_auto_vehicles)
                
                # Check if clustering was successful
                if clustered_buildings is None or len(clustered_buildings) == 0 or 'cluster' not in clustered_buildings.columns:
                    raise HTTPException(
                        status_code=400,
                        detail=f"No buildings found in ward {wardNo}. The GeoJSON data may be empty or invalid."
                    )
                
                # Extract cluster groups
                cluster_groups = clustered_buildings.groupby('cluster')
                clusters_list = [(cluster_id, group) for cluster_id, group in cluster_groups]
                
                optimization_result = {
                    'swachh_auto_count': swachh_auto_count,
                    'active_swachh_auto': len(active_swachh_auto),
                    'clusters_created': len(clusters_list),
                    'total_houses': len(buildings_gdf)
                }
                
                print(f"Clusters created: {len(clusters_list)} clusters from {swachh_auto_count} Swachh Auto vehicles")
                
            except HTTPException:
                raise
            except ValueError as ve:
                print(f"Clustering failed: {ve}")
                raise HTTPException(status_code=400, detail=str(ve))
            except Exception as vehicle_error:
                print(f"Failed to get vehicle data: {vehicle_error}")
                import traceback
                print(traceback.format_exc())
                raise HTTPException(status_code=500, detail=f"Failed to process data: {str(vehicle_error)}")
            
            # Save data files for later use
            os.makedirs("output", exist_ok=True)
            shutil.copy(ward_path, "output/ward.geojson")
            shutil.copy(buildings_path, "output/buildings.geojson")
            shutil.copy(roads_path, "output/roads.geojson")
            shutil.copy(vehicles_path, "output/vehicles.csv")
            
            # Save clustering results
            with open("output/optimization_result.json", "w") as f:
                json.dump(convert_numpy_types({
                    "wardNo": wardNo,
                    "swachh_auto_count": optimization_result['swachh_auto_count'],
                    "active_swachh_auto": optimization_result['active_swachh_auto'],
                    "clusters_created": optimization_result['clusters_created'],
                    "total_houses": optimization_result['total_houses']
                }), f, indent=2)
            
            print("Clustering data saved successfully")
            
            # Prepare vehicle details
            vehicle_details = []
            for idx, vehicle in swachh_auto_vehicles.iterrows():
                vehicle_details.append({
                    "vehicle_id": str(vehicle.get('vehicle_id', '')),
                    "vehicle_type": str(vehicle.get('vehicle_type', '')),
                    "status": str(vehicle.get('status', '')),
                    "capacity": vehicle.get('capacity', 0),
                    "driver_name": str(vehicle.get('driverName', 'N/A'))
                })
            
            # Prepare response data
            response_data = {
                "status": "success",
                "message": f"Created {optimization_result['clusters_created']} clusters from {optimization_result['swachh_auto_count']} Swachh Auto vehicles. Use /assign-routes-by-vehicle to assign routes.",
                "wardNo": wardNo,
                "swachh_auto_count": optimization_result['swachh_auto_count'],
                "active_swachh_auto": optimization_result['active_swachh_auto'],
                "clusters_created": optimization_result['clusters_created'],
                "total_houses": optimization_result['total_houses'],
                "vehicle_source": vehicle_source,
                "vehicles": vehicle_details,
                "next_step": "Use POST /assign-routes-by-vehicle with vehicle IDs to generate routes and maps"
            }
            
            # Ensure all data is JSON serializable
            response_data = convert_numpy_types(response_data)
            
            return JSONResponse(response_data)
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error details: {error_details}")
            raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")







class Coordinate(BaseModel):
    longitude: float
    latitude: float

class RoadSegment(BaseModel):
    start_coordinate: Coordinate
    end_coordinate: Coordinate
    distance_meters: float
    house_count: int

class VehicleInfo(BaseModel):
    vehicle_id: str
    vehicle_type: str
    status: str
    capacity: int

class ClusterBounds(BaseModel):
    min_longitude: float
    max_longitude: float
    min_latitude: float
    max_latitude: float

class ClusterRoadsResponse(BaseModel):
    wardNo: str
    cluster_id: int
    vehicle_info: VehicleInfo
    buildings_count: int
    roads: List[RoadSegment]
    total_road_segments: int
    cluster_bounds: ClusterBounds

class AllClustersRoadsResponse(BaseModel):
    total_clusters: int
    clusters: List[ClusterRoadsResponse]

class RouteCoordinate(BaseModel):
    lat: float
    lon: float
    time: str

class ClusterRouteResponse(BaseModel):
    vehicleNumber: str
    routeName: str
    coordinates: List[RouteCoordinate]

@app.get("/cluster/{cluster_id}", tags=["clusters"], response_model=ClusterRoadsResponse)
async def get_cluster_roads(cluster_id: int):
    """Get cluster roads with coordinates for a specific cluster.
    
    Returns all road segments within the cluster along with their start/end coordinates.
    Each road segment includes the geographic coordinates, distance in meters, house count, and ward number.
    """
    try:
        # Load processed data from output directory
        roads_path = os.path.join("output", "roads.geojson")
        buildings_path = os.path.join("output", "buildings.geojson")
        vehicles_path = os.path.join("output", "vehicles.csv")
        optimization_path = os.path.join("output", "optimization_result.json")
        
        if not all(os.path.exists(p) for p in [roads_path, buildings_path, vehicles_path]):
            raise HTTPException(status_code=404, detail="Cluster data not found. Run /optimize-routes first")
        
        # Load optimization result to get wardNo
        wardNo = "1"
        if os.path.exists(optimization_path):
            with open(optimization_path, 'r') as f:
                optimization_result = json.load(f)
                wardNo = optimization_result.get('wardNo', '1')
        
        # Load data
        roads_gdf = gpd.read_file(roads_path)
        buildings_gdf = gpd.read_file(buildings_path)
        vehicles_df = pd.read_csv(vehicles_path)
        
        # Convert to WGS84 if needed
        if roads_gdf.crs != 'EPSG:4326':
            roads_gdf = roads_gdf.to_crs('EPSG:4326')
        if buildings_gdf.crs != 'EPSG:4326':
            buildings_gdf = buildings_gdf.to_crs('EPSG:4326')
        
        # Get active vehicles and create clusters
        active_vehicles = vehicles_df[
            vehicles_df['status'].str.upper().isin(['ACTIVE', 'AVAILABLE', 'ONLINE'])
        ]
        
        if cluster_id >= len(active_vehicles):
            raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found")
        
        # Use fixed clustering for consistent assignments
        from src.clustering.assign_buildings import BuildingClusterer
        clusterer = BuildingClusterer()
        
        if 'wardNo' not in buildings_gdf.columns and 'wardNo' not in buildings_gdf.columns:
            buildings_gdf['wardNo'] = '1'
        
        clustered_buildings = clusterer.cluster_buildings_with_vehicles(buildings_gdf, active_vehicles)
        
        building_clusters = []
        for idx, row in clustered_buildings.iterrows():
            cluster_str = row.get('cluster', 'cluster_0')
            if 'vehicle_' in cluster_str:
                cluster_id = int(cluster_str.split('_')[-1])
            else:
                cluster_id = 0
            building_clusters.append(cluster_id)
        
        # Get buildings for this cluster
        cluster_buildings = buildings_gdf[[i == cluster_id for i in building_clusters]]
        
        if len(cluster_buildings) == 0:
            raise HTTPException(status_code=404, detail=f"No buildings found in cluster {cluster_id}")
        
        # Build road network graph
        G = nx.Graph()
        road_coordinates = []
        
        for idx, road in roads_gdf.iterrows():
            geom = road.geometry
            if geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    coords = list(line.coords)
                    road_coordinates.extend(coords)
                    for i in range(len(coords)-1):
                        p1, p2 = coords[i], coords[i+1]
                        dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
                        G.add_edge(p1, p2, weight=dist)
            else:
                coords = list(geom.coords)
                road_coordinates.extend(coords)
                for i in range(len(coords)-1):
                    p1, p2 = coords[i], coords[i+1]
                    dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
                    G.add_edge(p1, p2, weight=dist)
        
        # Find roads used by this cluster
        cluster_road_points = list(set(road_coordinates))
        house_locations = [(pt.x, pt.y) for pt in cluster_buildings.geometry.centroid]
        
        # Find nearest road points to houses and count houses per road segment
        cluster_roads = []
        road_segment_houses = {}  # Track house count per road segment
        
        for house_pt in house_locations:
            if cluster_road_points:
                distances = [((house_pt[0]-rp[0])**2 + (house_pt[1]-rp[1])**2)**0.5 for rp in cluster_road_points]
                nearest_idx = np.argmin(distances)
                nearest_road_point = cluster_road_points[nearest_idx]
                
                # Find all road segments connected to this point
                for edge in G.edges(nearest_road_point, data=True):
                    segment_key = (edge[0], edge[1])
                    road_segment_houses[segment_key] = road_segment_houses.get(segment_key, 0) + 1
        
        # Build road segments with house counts
        for segment_key, house_count in road_segment_houses.items():
            road_segment = {
                "start_coordinate": {"longitude": float(segment_key[0][0]), "latitude": float(segment_key[0][1])},
                "end_coordinate": {"longitude": float(segment_key[1][0]), "latitude": float(segment_key[1][1])},
                "distance_meters": float(G[segment_key[0]][segment_key[1]]['weight'] * 111000),
                "house_count": house_count
            }
            cluster_roads.append(road_segment)
        
        # Get vehicle info
        vehicle_info = active_vehicles.iloc[cluster_id] if cluster_id < len(active_vehicles) else {}
        
        response_data = {
            "wardNo": wardNo,
            "cluster_id": cluster_id,
            "vehicle_info": {
                "vehicle_id": str(vehicle_info.get('vehicle_id', f'vehicle_{cluster_id}')),
                "vehicle_type": str(vehicle_info.get('vehicle_type', 'standard')),
                "status": str(vehicle_info.get('status', 'active')),
                "capacity": vehicle_info.get('capacity', 1000)
            },
            "buildings_count": len(cluster_buildings),
            "roads": cluster_roads,
            "total_road_segments": len(cluster_roads),
            "cluster_bounds": {
                "min_longitude": cluster_buildings.bounds.minx.min(),
                "max_longitude": cluster_buildings.bounds.maxx.max(),
                "min_latitude": cluster_buildings.bounds.miny.min(),
                "max_latitude": cluster_buildings.bounds.maxy.max()
            }
        }
        
        # Convert all numpy types to JSON serializable types
        response_data = convert_numpy_types(response_data)
        
        return JSONResponse(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cluster roads: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cluster roads: {str(e)}")

@app.get("/cluster-routes", tags=["clusters"], response_model=List[ClusterRouteResponse])
async def get_cluster_route_coordinates():
    """Get road coordinates for every cluster and upload to external URL.
    
    Returns route coordinates for all clusters with vehicle information,
    route names, coordinate details including timestamps, house count, and ward number.
    """
    try:
        roads_path = os.path.join("output", "roads.geojson")
        buildings_path = os.path.join("output", "buildings.geojson")
        vehicles_path = os.path.join("output", "vehicles.csv")
        optimization_path = os.path.join("output", "optimization_result.json")
        
        if not all(os.path.exists(p) for p in [roads_path, buildings_path, vehicles_path]):
            raise HTTPException(status_code=404, detail="Cluster data not found. Run /optimize-routes first")
        
        # Load optimization result to get wardNo
        wardNo = "1"
        if os.path.exists(optimization_path):
            with open(optimization_path, 'r') as f:
                optimization_result_data = json.load(f)
                wardNo = optimization_result_data.get('wardNo', '1')
        
        
        roads_gdf = gpd.read_file(roads_path)
        buildings_gdf = gpd.read_file(buildings_path)
        vehicles_df = pd.read_csv(vehicles_path)
        
        if roads_gdf.crs != 'EPSG:4326':
            roads_gdf = roads_gdf.to_crs('EPSG:4326')
        if buildings_gdf.crs != 'EPSG:4326':
            buildings_gdf = buildings_gdf.to_crs('EPSG:4326')
        
        active_vehicles = vehicles_df[
            vehicles_df['status'].str.upper().isin(['ACTIVE', 'AVAILABLE', 'ONLINE'])
        ]
        
        # Load route assignments if available
        route_assignments_path = os.path.join("output", "route_assignments.json")
        optimization_result = None
        if os.path.exists(route_assignments_path):
            with open(route_assignments_path, 'r') as f:
                optimization_result = json.load(f)
        elif os.path.exists(optimization_path):
            with open(optimization_path, 'r') as f:
                optimization_result = json.load(f)
        
        # Build road network
        G = nx.Graph()
        road_coordinates = []
        
        for idx, road in roads_gdf.iterrows():
            geom = road.geometry
            if geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    coords = list(line.coords)
                    road_coordinates.extend(coords)
                    for i in range(len(coords)-1):
                        p1, p2 = coords[i], coords[i+1]
                        dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
                        G.add_edge(p1, p2, weight=dist)
            else:
                coords = list(geom.coords)
                road_coordinates.extend(coords)
                for i in range(len(coords)-1):
                    p1, p2 = coords[i], coords[i+1]
                    dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
                    G.add_edge(p1, p2, weight=dist)
        
        cluster_routes = []
        
        if optimization_result and 'route_assignments' in optimization_result:
            # Use optimization results
            for vehicle_id, assignment in optimization_result['route_assignments'].items():
                vehicle_info = assignment['vehicle_info']
                
                for trip_idx, trip in enumerate(assignment['trips']):
                    # Get house coordinates for this trip
                    house_indices = trip.get('houses', [])
                    house_coords = []
                    
                    for house_idx in house_indices:
                        if house_idx < len(buildings_gdf):
                            pt = buildings_gdf.iloc[house_idx].geometry.centroid
                            house_coords.append((pt.x, pt.y))
                    
                    if house_coords:
                        # Generate road route coordinates with house counts per segment
                        road_points = list(set(road_coordinates))
                        
                        # Track house count per road point
                        road_point_houses = {}
                        for house_coord in house_coords:
                            if road_points:
                                distances = [((house_coord[0]-rp[0])**2 + (house_coord[1]-rp[1])**2)**0.5 for rp in road_points]
                                nearest_road = road_points[np.argmin(distances)]
                                road_point_houses[nearest_road] = road_point_houses.get(nearest_road, 0) + 1
                        
                        # Create route through nearest road points
                        route_path = list(road_point_houses.keys())
                        
                        # Add road segments between points
                        full_route = []
                        for i in range(len(route_path)):
                            full_route.append(route_path[i])
                            if i < len(route_path) - 1:
                                try:
                                    path = nx.shortest_path(G, route_path[i], route_path[i+1], weight='weight')
                                    full_route.extend(path[1:])
                                except:
                                    pass
                        
                        # Build road segments with house counts
                        road_segments = []
                        for i in range(len(full_route) - 1):
                            p1, p2 = full_route[i], full_route[i+1]
                            if G.has_edge(p1, p2):
                                segment_houses = road_point_houses.get(p1, 0) + road_point_houses.get(p2, 0)
                                distance_meters = G[p1][p2]['weight'] * 111000
                                
                                road_segments.append({
                                    "coordinates": [
                                        {"longitude": float(p1[0]), "latitude": float(p1[1])},
                                        {"longitude": float(p2[0]), "latitude": float(p2[1])}
                                    ],
                                    "house_count": segment_houses
                                })
                        
                        cluster_routes.append({
                            "vehicleNumber": str(vehicle_info['vehicle_id']),
                            "routeName": f"Route-{vehicle_id}-Trip-{trip_idx + 1}",
                            "segments": road_segments,
                            "house_count": len(house_coords)
                        })
        else:
            # Use fixed clustering for consistency
            from src.clustering.assign_buildings import BuildingClusterer
            clusterer = BuildingClusterer()
            
            if 'wardNo' not in buildings_gdf.columns and 'wardNo' not in buildings_gdf.columns:
                buildings_gdf['wardNo'] = '1'
            
            clustered_buildings = clusterer.cluster_buildings_with_vehicles(buildings_gdf, active_vehicles)
            
            building_clusters = []
            for idx, row in clustered_buildings.iterrows():
                cluster_str = row.get('cluster', 'cluster_0')
                if 'vehicle_' in cluster_str:
                    cluster_id = int(cluster_str.split('_')[-1])
                else:
                    cluster_id = 0
                building_clusters.append(cluster_id)
            
            for cluster_id in range(min(len(active_vehicles), max(building_clusters) + 1)):
                cluster_buildings = [i for i, c in enumerate(building_clusters) if c == cluster_id]
                
                if cluster_buildings:
                    house_coords = []
                    for i in cluster_buildings:
                        pt = buildings_gdf.iloc[i].geometry.centroid
                        house_coords.append((pt.x, pt.y))
                    
                    road_points = list(set(road_coordinates))
                    
                    # Track house count per road point
                    road_point_houses = {}
                    for house_coord in house_coords:
                        if road_points:
                            distances = [((house_coord[0]-rp[0])**2 + (house_coord[1]-rp[1])**2)**0.5 for rp in road_points]
                            nearest_road = road_points[np.argmin(distances)]
                            road_point_houses[nearest_road] = road_point_houses.get(nearest_road, 0) + 1
                    
                    # Create route through nearest road points
                    route_path = list(road_point_houses.keys())
                    
                    # Add road segments between points
                    full_route = []
                    for i in range(len(route_path)):
                        full_route.append(route_path[i])
                        if i < len(route_path) - 1:
                            try:
                                path = nx.shortest_path(G, route_path[i], route_path[i+1], weight='weight')
                                full_route.extend(path[1:])
                            except:
                                pass
                    
                    # Build road segments with house counts
                    road_segments = []
                    for i in range(len(full_route) - 1):
                        p1, p2 = full_route[i], full_route[i+1]
                        if G.has_edge(p1, p2):
                            segment_houses = road_point_houses.get(p1, 0) + road_point_houses.get(p2, 0)
                            distance_meters = G[p1][p2]['weight'] * 111000
                            
                            road_segments.append({
                                "coordinates": [
                                    {"longitude": float(p1[0]), "latitude": float(p1[1])},
                                    {"longitude": float(p2[0]), "latitude": float(p2[1])}
                                ],
                                "house_count": segment_houses
                            })
                    
                    vehicle_info = active_vehicles.iloc[cluster_id] if cluster_id < len(active_vehicles) else {}
                    cluster_routes.append({
                        "vehicleNumber": str(vehicle_info.get('vehicle_id', f'vehicle_{cluster_id}')),
                        "routeName": f"Route-Cluster-{cluster_id + 1}",
                        "segments": road_segments,
                        "no_of_houses": len(house_coords)
                    })
        
        # Upload data to external URL
        upload_status = {"success": False, "uploaded_count": 0, "failed_count": 0, "details": []}
        try:
            load_dotenv()
            
            external_url = Config.EXTERNAL_UPLOAD_URL
            swm_token = Config.SWM_TOKEN
            
            if external_url and swm_token:
                # Clean the URL and ensure proper endpoint
                external_url = external_url.rstrip("'").rstrip('/')
                
                # Add the correct endpoint path if not present
                if not external_url.endswith('/api/vehicle-routes/upload-data'):
                    external_url = external_url + '/api/vehicle-routes/upload-data'
                
                logger.info(f"Upload URL: {external_url}")
                
                headers = {
                    'Authorization': 'Bearer ' + swm_token.strip("'\""),
                    'Content-Type': 'application/json'
                }
                
                logger.info(f"Uploading {len(cluster_routes)} routes to {external_url}")
                
                # Log sample payload for debugging
                if cluster_routes:
                    sample = cluster_routes[0]
                    logger.debug(f"Sample payload structure: vehicleNumber={sample.get('vehicleNumber')}, routeName={sample.get('routeName')}, segments={len(sample.get('segments', []))}")
                
                # Upload each route individually with retry logic
                from datetime import datetime
                import time
                for route_idx, route in enumerate(cluster_routes):
                    success = False
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{route_idx:03d}"
                    route['routeName'] = f"{route['routeName']}-{timestamp}"
                    
                    logger.debug(f"Uploading route: {route['routeName']}, segments: {len(route.get('segments', []))}")
                    
                    # Create clean payload with wardNo and segments
                    payload = {
                        "wardNo": wardNo,
                        "vehicleNumber": route['vehicleNumber'],
                        "routeName": route['routeName'],
                        "segments": route['segments'],
                        "no_of_houses": route.get('no_of_houses', route.get('house_count', 0))
                    }
                    
                    # Print payload details for debugging
                    print(f"\n=== Sending to External API ===")
                    print(f"Route: {route['routeName']}")
                    print(f"Vehicle: {route['vehicleNumber']}")
                    print(f"Ward: {wardNo}")
                    print(f"Houses in this route: {route.get('no_of_houses', route.get('house_count', 0))}")
                    print(f"Segments: {len(route.get('segments', []))}")
                    print(f"================================\n")
                    
                    for attempt in range(3):
                        try:
                            if attempt > 0:
                                wait_time = 2 ** attempt
                                logger.debug(f"Retry {attempt + 1} after {wait_time}s")
                                time.sleep(wait_time)
                            
                            response = requests.post(
                                external_url,
                                json=payload,
                                headers=headers,
                                timeout=30
                            )
                            
                            if response.status_code in [200, 201]:
                                upload_status["uploaded_count"] += 1
                                upload_status["details"].append({"route": route['routeName'], "status": "success"})
                                success = True
                                logger.info(f"Uploaded {route['routeName']}")
                                break
                            elif response.status_code == 409:
                                upload_status["uploaded_count"] += 1
                                upload_status["details"].append({"route": route['routeName'], "status": "exists"})
                                success = True
                                logger.info(f"{route['routeName']} already exists")
                                break
                            else:
                                error_body = response.text[:500] if response.text else "No response body"
                                logger.warning(f"Attempt {attempt + 1}: {response.status_code}")
                                if attempt == 2:
                                    upload_status["failed_count"] += 1
                                    upload_status["details"].append({
                                        "route": route['routeName'],
                                        "status": "failed",
                                        "code": response.status_code
                                    })
                        except requests.exceptions.Timeout:
                            logger.warning(f"Attempt {attempt + 1} timeout")
                            if attempt == 2:
                                upload_status["failed_count"] += 1
                                upload_status["details"].append({
                                    "route": route['routeName'],
                                    "status": "timeout"
                                })
                        except Exception as route_error:
                            logger.warning(f"Attempt {attempt + 1} error")
                            if attempt == 2:
                                upload_status["failed_count"] += 1
                                upload_status["details"].append({
                                    "route": route['routeName'],
                                    "status": "error"
                                })
                    
                    if not success:
                        logger.error(f"Failed to upload {route['routeName']} after 3 attempts")
                    else:
                        time.sleep(0.2)
                
                upload_status["success"] = upload_status["failed_count"] == 0
                logger.info(f"Upload complete: {upload_status['uploaded_count']} success, {upload_status['failed_count']} failed")
            else:
                upload_status["details"].append({"status": "skipped", "reason": "URL or token not configured"})
                logger.warning("Upload skipped: URL or token not configured")
        except Exception as upload_error:
            upload_status["details"].append({"status": "error", "error": str(upload_error)})
            logger.error(f"Failed to upload to external URL: {upload_error}")
        
        # Calculate total houses count
        total_houses = sum(route.get('house_count', 0) for route in cluster_routes)
        
        return JSONResponse(convert_numpy_types({
            "wardNo": wardNo,
            "total_houses": total_houses,
            "routes": cluster_routes,
            "upload_status": upload_status
        }))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cluster route coordinates: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cluster route coordinates: {str(e)}")

@app.get("/clusters", tags=["clusters"], response_model=AllClustersRoadsResponse)
async def get_all_cluster_roads():
    """Get cluster roads with coordinates for all clusters.
    
    Returns all road segments within each cluster along with their start/end coordinates.
    Each road segment includes the geographic coordinates, distance in meters, house count, and ward number.
    """
    try:
        roads_path = os.path.join("output", "roads.geojson")
        buildings_path = os.path.join("output", "buildings.geojson")
        vehicles_path = os.path.join("output", "vehicles.csv")
        optimization_path = os.path.join("output", "optimization_result.json")
        
        if not all(os.path.exists(p) for p in [roads_path, buildings_path, vehicles_path]):
            raise HTTPException(status_code=404, detail="Cluster data not found. Run /optimize-routes first")
        
        # Load optimization result to get wardNo
        wardNo = "1"
        if os.path.exists(optimization_path):
            with open(optimization_path, 'r') as f:
                optimization_result = json.load(f)
                wardNo = optimization_result.get('wardNo', '1')
        
        roads_gdf = gpd.read_file(roads_path)
        buildings_gdf = gpd.read_file(buildings_path)
        vehicles_df = pd.read_csv(vehicles_path)
        
        if roads_gdf.crs != 'EPSG:4326':
            roads_gdf = roads_gdf.to_crs('EPSG:4326')
        if buildings_gdf.crs != 'EPSG:4326':
            buildings_gdf = buildings_gdf.to_crs('EPSG:4326')
        
        active_vehicles = vehicles_df[
            vehicles_df['status'].str.upper().isin(['ACTIVE', 'AVAILABLE', 'ONLINE'])
        ]
        
        # Use fixed clustering for all cluster operations
        from src.clustering.assign_buildings import BuildingClusterer
        clusterer = BuildingClusterer()
        
        if 'wardNo' not in buildings_gdf.columns and 'wardNo' not in buildings_gdf.columns:
            buildings_gdf['wardNo'] = '1'
        
        clustered_buildings = clusterer.cluster_buildings_with_vehicles(buildings_gdf, active_vehicles)
        
        building_clusters = []
        for idx, row in clustered_buildings.iterrows():
            cluster_str = row.get('cluster', 'cluster_0')
            if 'vehicle_' in cluster_str:
                cluster_id = int(cluster_str.split('_')[-1])
            else:
                cluster_id = 0
            building_clusters.append(cluster_id)
        
        G = nx.Graph()
        road_coordinates = []
        
        for idx, road in roads_gdf.iterrows():
            geom = road.geometry
            if geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    coords = list(line.coords)
                    road_coordinates.extend(coords)
                    for i in range(len(coords)-1):
                        p1, p2 = coords[i], coords[i+1]
                        dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
                        G.add_edge(p1, p2, weight=dist)
            else:
                coords = list(geom.coords)
                road_coordinates.extend(coords)
                for i in range(len(coords)-1):
                    p1, p2 = coords[i], coords[i+1]
                    dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
                    G.add_edge(p1, p2, weight=dist)
        
        cluster_road_points = list(set(road_coordinates))
        all_clusters = []
        
        for cluster_id in range(len(active_vehicles)):
            cluster_buildings = buildings_gdf[[i == cluster_id for i in building_clusters]]
            
            if len(cluster_buildings) == 0:
                continue
            
            house_locations_wgs84 = []
            for i in range(len(buildings_gdf)):
                if building_clusters[i] == cluster_id:
                    pt = buildings_gdf.iloc[i].geometry.centroid
                    house_locations_wgs84.append((pt.x, pt.y))
            
            # Track house count per road segment
            road_segment_houses = {}
            for house_pt in house_locations_wgs84:
                if cluster_road_points:
                    distances = [((house_pt[0]-rp[0])**2 + (house_pt[1]-rp[1])**2)**0.5 for rp in cluster_road_points]
                    nearest_idx = np.argmin(distances)
                    nearest_road_point = cluster_road_points[nearest_idx]
                    
                    for edge in G.edges(nearest_road_point, data=True):
                        segment_key = (edge[0], edge[1])
                        road_segment_houses[segment_key] = road_segment_houses.get(segment_key, 0) + 1
            
            # Build road segments with house counts
            cluster_roads = []
            for segment_key, house_count in road_segment_houses.items():
                road_segment = {
                    "start_coordinate": {"longitude": float(segment_key[0][0]), "latitude": float(segment_key[0][1])},
                    "end_coordinate": {"longitude": float(segment_key[1][0]), "latitude": float(segment_key[1][1])},
                    "distance_meters": float(G[segment_key[0]][segment_key[1]]['weight'] * 111000),
                    "house_count": house_count
                }
                cluster_roads.append(road_segment)
            
            vehicle_info = active_vehicles.iloc[cluster_id] if cluster_id < len(active_vehicles) else {}
            
            cluster_data = {
                "wardNo": wardNo,
                "cluster_id": cluster_id,
                "vehicle_info": {
                    "vehicle_id": str(vehicle_info.get('vehicle_id', f'vehicle_{cluster_id}')),
                    "vehicle_type": str(vehicle_info.get('vehicle_type', 'standard')),
                    "status": str(vehicle_info.get('status', 'active')),
                    "capacity": vehicle_info.get('capacity', 1000)
                },
                "buildings_count": len(cluster_buildings),
                "roads": cluster_roads,
                "total_road_segments": len(cluster_roads),
                "cluster_bounds": {
                    "min_longitude": cluster_buildings.bounds.minx.min(),
                    "max_longitude": cluster_buildings.bounds.maxx.max(),
                    "min_latitude": cluster_buildings.bounds.miny.min(),
                    "max_latitude": cluster_buildings.bounds.maxy.max()
                }
            }
            
            all_clusters.append(cluster_data)
        
        response_data = {
            "total_clusters": len(all_clusters),
            "clusters": all_clusters
        }
        
        response_data = convert_numpy_types(response_data)
        
        return JSONResponse(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting all cluster roads: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get all cluster roads: {str(e)}")

@app.get("/generate-map", tags=["maps"])
async def generate_map():
    """Generate and return route map HTML."""
    file_path = os.path.join("output", "route_map.html")
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Map not found. Please upload files first using /optimize-routes")
    
    with open(file_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    
    # Delete the file after reading
    try:
        os.remove(file_path)
        print("Map file deleted after serving")
    except Exception as e:
        print(f"Warning: Could not delete map file: {e}")
    
    return HTMLResponse(content=html_content)

def generate_map_from_files(ward_file, roads_file, buildings_file, vehicles_file=None):
    """Generate map with capacity-based route optimization."""
    # Load GeoJSON data using geopandas
    ward_gdf = gpd.read_file(ward_file)
    roads_gdf = gpd.read_file(roads_file)
    buildings_gdf = gpd.read_file(buildings_file)
    
    # Convert to WGS84 if needed
    if ward_gdf.crs != 'EPSG:4326':
        ward_gdf = ward_gdf.to_crs('EPSG:4326')
    if roads_gdf.crs != 'EPSG:4326':
        roads_gdf = roads_gdf.to_crs('EPSG:4326')
    if buildings_gdf.crs != 'EPSG:4326':
        buildings_gdf = buildings_gdf.to_crs('EPSG:4326')
    
    # Clean data - keep only geometry and essential columns
    ward_clean = ward_gdf[['geometry']]
    buildings_clean = buildings_gdf[['geometry']]
    
    # Get center coordinates from ward bounds
    bounds = ward_gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=15)
    
    # Add ward boundaries to base layer
    ward_layer = folium.FeatureGroup(name="Ward Boundary", show=True)
    folium.GeoJson(
        json.loads(ward_clean.to_json()),
        style_function=lambda x: {
            'fillColor': 'transparent',
            'color': 'darkblue',
            'weight': 3,
            'fillOpacity': 0
        }
    ).add_to(ward_layer)
    ward_layer.add_to(m)
    
    # Load vehicle data if available
    vehicles_df = None
    if vehicles_file and os.path.exists(vehicles_file):
        import pandas as pd
        vehicles_df = pd.read_csv(vehicles_file)
        print(f"Loaded {len(vehicles_df)} vehicles for map generation")
        
        # Filter for Swachh Auto vehicles only (same as clustering logic)
        swachh_auto_vehicles = vehicles_df[
            vehicles_df['vehicle_type'].notna() & 
            (vehicles_df['vehicle_type'].str.upper().str.contains('SWACHH AUTO|SWACHAUTO|SWACHH_AUTO', na=False))
        ]
        
        if len(swachh_auto_vehicles) == 0:
            print("No Swachh Auto vehicles found, using all vehicles")
            swachh_auto_vehicles = vehicles_df
        
        # Use fixed clustering based on Swachh Auto count
        from src.clustering.assign_buildings import BuildingClusterer
        clusterer = BuildingClusterer()
        
        # Add ward info if not present
        if 'wardNo' not in buildings_gdf.columns and 'wardNo' not in buildings_gdf.columns:
            buildings_gdf['wardNo'] = '1'  # Default ward for map generation
        
        # Apply fixed clustering using Swachh Auto vehicles only
        clustered_buildings = clusterer.cluster_buildings_with_vehicles(buildings_gdf, swachh_auto_vehicles)
        
        # Extract cluster assignments for map generation
        building_clusters = []
        for idx, row in clustered_buildings.iterrows():
            cluster_str = row.get('cluster', 'cluster_0')
            if 'vehicle_' in cluster_str:
                cluster_id = int(cluster_str.split('_')[-1])
            else:
                cluster_id = 0
            building_clusters.append(cluster_id)
    else:
        # Fallback to fixed clustering with default settings
        from src.clustering.assign_buildings import BuildingClusterer
        clusterer = BuildingClusterer()
        
        if 'wardNo' not in buildings_gdf.columns and 'wardNo' not in buildings_gdf.columns:
            buildings_gdf['wardNo'] = '1'
        
        # Create fallback vehicles for map generation
        fallback_vehicles = pd.DataFrame([
            {'vehicle_id': f'vehicle_{i}', 'status': 'active', 'vehicle_type': 'truck'} 
            for i in range(min(5, len(buildings_gdf)))
        ])
        clustered_buildings = clusterer.cluster_buildings_with_vehicles(buildings_gdf, fallback_vehicles)
        
        building_clusters = []
        for idx, row in clustered_buildings.iterrows():
            cluster_str = row.get('cluster', 'cluster_0')
            if 'vehicle_' in cluster_str:
                cluster_id = int(cluster_str.split('_')[-1])
            else:
                cluster_id = 0
            building_clusters.append(cluster_id)
    
    # Create road network graph
    G = nx.Graph()
    
    # Build road network for routing
    for idx, road in roads_gdf.iterrows():
        geom = road.geometry
        if geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                coords = list(line.coords)
                for i in range(len(coords)-1):
                    p1, p2 = coords[i], coords[i+1]
                    dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
                    G.add_edge(p1, p2, weight=dist)
        else:
            coords = list(geom.coords)
            for i in range(len(coords)-1):
                p1, p2 = coords[i], coords[i+1]
                dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5
                G.add_edge(p1, p2, weight=dist)
    
    # Colors and vehicle names from active Swachh Auto vehicles or defaults
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    if vehicles_df is not None and len(swachh_auto_vehicles) > 0:
        active_swachh_auto = swachh_auto_vehicles[
            swachh_auto_vehicles['status'].str.upper().isin(['ACTIVE', 'AVAILABLE', 'ONLINE', 'HALTED', 'IDEL'])
        ]
        vehicle_names = active_swachh_auto['vehicle_id'].tolist()[:len(set(building_clusters))]
    else:
        vehicle_names = ['Vehicle A', 'Vehicle B', 'Vehicle C', 'Vehicle D', 'Vehicle E']
    
    # Process each cluster with separate layers
    n_clusters = len(set(building_clusters))
    for cluster_id in range(n_clusters):
        cluster_buildings = [i for i, c in enumerate(building_clusters) if c == cluster_id]
        
        if not cluster_buildings:
            continue
        
        # Create separate layer for each cluster with active vehicle info
        vehicle_name = vehicle_names[cluster_id] if cluster_id < len(vehicle_names) else f"Vehicle {cluster_id + 1}"
        vehicle_info = ""
        if vehicles_df is not None and len(swachh_auto_vehicles) > 0 and cluster_id < len(active_swachh_auto):
            vehicle = active_swachh_auto.iloc[cluster_id]
            vehicle_info = f" ({vehicle.get('vehicle_type', 'N/A')} - {vehicle.get('status', 'ACTIVE')})"
        
        cluster_layer = folium.FeatureGroup(
            name=f"🚛 Trip-{cluster_id + 1} ({vehicle_name}) - {len(cluster_buildings)} houses",
            show=True
        )
            
        # Find all road points for routing
        cluster_road_points = []
        for idx, road in roads_gdf.iterrows():
            geom = road.geometry
            if geom.geom_type == 'MultiLineString':
                for line in geom.geoms:
                    cluster_road_points.extend(list(line.coords))
            else:
                cluster_road_points.extend(list(geom.coords))
        
        # Create waste collection route covering all houses
        if cluster_buildings:
            cluster_road_points = list(set(cluster_road_points))
            
            # Get house locations for this cluster (already in WGS84)
            house_locations_wgs84 = []
            for i in cluster_buildings:
                pt = buildings_gdf.iloc[i].geometry.centroid
                house_locations_wgs84.append((pt.x, pt.y))
            
            # Find nearest road points to houses
            house_road_points = []
            for house_pt in house_locations_wgs84:
                if cluster_road_points:
                    distances = [((house_pt[0]-rp[0])**2 + (house_pt[1]-rp[1])**2)**0.5 for rp in cluster_road_points]
                    nearest_idx = np.argmin(distances)
                    house_road_points.append(cluster_road_points[nearest_idx])
            
            # Create collection route through all houses
            if house_road_points and len(house_road_points) > 0:
                # Remove duplicates while preserving order
                unique_house_points = []
                for pt in house_road_points:
                    if pt not in unique_house_points:
                        unique_house_points.append(pt)
                
                if len(unique_house_points) >= 1:
                    # Start from first house location
                    start_point = unique_house_points[0]
                    route_points = [start_point]
                    
                    # Visit all other houses
                    current_point = start_point
                    remaining_points = unique_house_points[1:].copy()
                    
                    while remaining_points:
                        # Find nearest unvisited house
                        distances = [((current_point[0]-pt[0])**2 + (current_point[1]-pt[1])**2)**0.5 for pt in remaining_points]
                        nearest_idx = np.argmin(distances)
                        next_point = remaining_points.pop(nearest_idx)
                        
                        # Find shortest path on roads between current and next point
                        try:
                            path_segment = nx.shortest_path(G, current_point, next_point, weight='weight')
                            route_points.extend(path_segment[1:])  # Skip first point to avoid duplication
                        except:
                            # If no path found, add direct connection
                            route_points.append(next_point)
                        
                        current_point = next_point
                    
                    # Convert to lat/lon for folium
                    route_coords = [[pt[1], pt[0]] for pt in route_points]
                    
                    # Add collection route with direction arrows
                    route_popup = f"{vehicle_name} - {len(cluster_buildings)} Houses"
                    if vehicles_df is not None and len(swachh_auto_vehicles) > 0 and cluster_id < len(swachh_auto_vehicles):
                        vehicle = swachh_auto_vehicles.iloc[cluster_id]
                        route_popup += f"\nType: {vehicle.get('vehicle_type', 'N/A')}\nDriver: {vehicle.get('driverName', 'N/A')}"
                    
                    folium.PolyLine(
                        route_coords,
                        color=colors[cluster_id % len(colors)],
                        weight=4,
                        opacity=0.8,
                        popup=route_popup
                    ).add_to(cluster_layer)
                    
                    # Add directional arrows along the route
                    for i in range(0, len(route_coords)-1, max(1, len(route_coords)//10)):
                        if i+1 < len(route_coords):
                            # Calculate arrow direction
                            lat1, lon1 = route_coords[i]
                            lat2, lon2 = route_coords[i+1]
                            
                            # Calculate bearing for arrow rotation
                            dlon = math.radians(lon2 - lon1)
                            dlat = math.radians(lat2 - lat1)
                            bearing = math.degrees(math.atan2(dlon, dlat))
                            
                            # Add arrow marker
                            folium.Marker(
                                [lat1, lon1],
                                icon=folium.DivIcon(
                                    html=f'<div style="transform: rotate({bearing}deg); color: {colors[cluster_id % len(colors)]}; font-size: 16px;">➤</div>',
                                    icon_size=(20, 20),
                                    icon_anchor=(10, 10)
                                )
                            ).add_to(cluster_layer)
                    
                    # Add start marker with active vehicle info
                    start_popup = f"{vehicle_name} Start"
                    if vehicles_df is not None and len(swachh_auto_vehicles) > 0 and cluster_id < len(active_swachh_auto):
                        vehicle = active_swachh_auto.iloc[cluster_id]
                        start_popup += f" ({vehicle.get('status', 'ACTIVE')})\nID: {vehicle.get('vehicle_id', 'N/A')}\nCapacity: {vehicle.get('capacity', 'N/A')}"
                    
                    folium.Marker(
                        [start_point[1], start_point[0]],
                        popup=start_popup,
                        icon=folium.Icon(color='green', icon='play')
                    ).add_to(cluster_layer)
                    
                    # Add end marker
                    folium.Marker(
                        [current_point[1], current_point[0]],
                        popup=f"{vehicle_name} End",
                        icon=folium.Icon(color='red', icon='stop')
                    ).add_to(cluster_layer)
        
        # Add clustered buildings as polygons
        for house_number, building_idx in enumerate(cluster_buildings, 1):
            building = buildings_clean.iloc[building_idx]
            folium.GeoJson(
                json.loads(gpd.GeoSeries([building.geometry]).to_json()),
                style_function=lambda x, c=colors[cluster_id % len(colors)]: {
                    'fillColor': c,
                    'color': c,
                    'weight': 1,
                    'fillOpacity': 0.6
                },
                popup=f"{vehicle_name} - House {house_number}",
                tooltip=f"C{cluster_id + 1}-H{house_number}"
            ).add_to(cluster_layer)
        
        # Add cluster layer to map
        cluster_layer.add_to(m)
    
    # Add layer control
    folium.LayerControl(position='topleft', collapsed=False).add_to(m)
    
    # Add cluster dashboard panel with layer toggle functionality
    cluster_stats = []
    for cluster_id in range(n_clusters):
        cluster_buildings = [i for i, c in enumerate(building_clusters) if c == cluster_id]
        if cluster_buildings:
            vehicle_name = vehicle_names[cluster_id] if cluster_id < len(vehicle_names) else f"Vehicle {cluster_id + 1}"
            vehicle_details = ""
            if vehicles_df is not None and len(swachh_auto_vehicles) > 0 and cluster_id < len(active_swachh_auto):
                vehicle = active_swachh_auto.iloc[cluster_id]
                vehicle_details = f" • {vehicle.get('vehicle_type', 'N/A')} • {vehicle.get('status', 'ACTIVE')} • Cap: {vehicle.get('capacity', 'N/A')}"
            
            cluster_stats.append(f'''
            <div style="margin:5px 0;padding:8px;border:1px solid #ddd;border-radius:4px;background:#f9f9f9;">
                <div style="display:flex;align-items:center;justify-content:space-between;">
                    <div>
                        <span style="color:{colors[cluster_id % len(colors)]};font-size:14px;">●</span> 
                        <strong>Trip-{cluster_id + 1}</strong>
                    </div>
                    <button onclick="toggleCluster({cluster_id})" style="padding:2px 6px;font-size:10px;border:1px solid {colors[cluster_id % len(colors)]};background:white;border-radius:3px;cursor:pointer;">Toggle</button>
                </div>
                <div style="font-size:11px;margin-top:5px;">
                    Houses: {len(cluster_buildings)}<br>
                    <small>{vehicle_name}{vehicle_details} • {len(cluster_buildings) * 0.5:.1f}km • {len(cluster_buildings) * 3:.0f}min</small>
                </div>
            </div>
            ''')
    
    panel_html = f'''
    <div style="position:fixed;top:10px;right:10px;width:280px;max-height:70vh;background:white;border:2px solid #333;z-index:9999;font-size:12px;border-radius:5px;box-shadow:0 2px 10px rgba(0,0,0,0.3);">
        <div style="background:#333;color:white;padding:8px;border-radius:3px 3px 0 0;">
            <strong>📊 Trip Dashboard</strong>
            <div style="font-size:10px;margin-top:3px;">{len([c for c in cluster_stats if c])} trips • {len(buildings_gdf)} houses</div>
            <div style="margin-top:5px;">
                <button onclick="showAllTrips()" style="padding:3px 8px;font-size:10px;border:1px solid white;background:none;color:white;border-radius:3px;cursor:pointer;margin-right:5px;">Show All</button>
                <button onclick="hideAllTrips()" style="padding:3px 8px;font-size:10px;border:1px solid white;background:none;color:white;border-radius:3px;cursor:pointer;">Hide All</button>
            </div>
        </div>
        <div style="padding:8px;max-height:50vh;overflow-y:auto;">
            {''.join(cluster_stats)}
        </div>
    </div>
    
    <script>
    function toggleCluster(clusterId) {{
        var layerControls = document.querySelectorAll('.leaflet-control-layers-selector');
        layerControls.forEach(function(control) {{
            var label = control.nextSibling;
            if (label && label.textContent.includes('Trip-' + (clusterId + 1))) {{
                control.click();
            }}
        }});
    }}
    
    function showAllTrips() {{
        var layerControls = document.querySelectorAll('.leaflet-control-layers-selector');
        layerControls.forEach(function(control) {{
            var label = control.nextSibling;
            if (label && label.textContent.includes('Trip-') && !control.checked) {{
                control.click();
            }}
        }});
    }}
    
    function hideAllTrips() {{
        var layerControls = document.querySelectorAll('.leaflet-control-layers-selector');
        layerControls.forEach(function(control) {{
            var label = control.nextSibling;
            if (label && label.textContent.includes('Trip-') && control.checked) {{
                control.click();
            }}
        }});
    }}
    </script>
    '''
    
    m.get_root().html.add_child(folium.Element(panel_html))
    
    return m._repr_html_()

@app.post("/assign-routes-by-vehicle", tags=["optimization"])
async def assign_routes_by_vehicle(vehicle_ids: str = Form(..., description="Comma-separated vehicle IDs")):
    """Assign existing clusters to specified vehicles and generate routes with map."""
    try:
        vehicle_id_list = [vid.strip() for vid in vehicle_ids.split(',')]
        
        buildings_path = os.path.join("output", "buildings.geojson")
        vehicles_path = os.path.join("output", "vehicles.csv")
        optimization_path = os.path.join("output", "optimization_result.json")
        roads_path = os.path.join("output", "roads.geojson")
        ward_path = os.path.join("output", "ward.geojson")
        
        if not all(os.path.exists(p) for p in [buildings_path, vehicles_path, optimization_path]):
            raise HTTPException(status_code=404, detail="Run /optimize-routes first")
        
        buildings_gdf = gpd.read_file(buildings_path)
        vehicles_df = pd.read_csv(vehicles_path)
        
        selected_vehicles = vehicles_df[vehicles_df['vehicle_id'].isin(vehicle_id_list)]
        if len(selected_vehicles) != len(vehicle_id_list):
            missing = set(vehicle_id_list) - set(selected_vehicles['vehicle_id'].tolist())
            raise HTTPException(status_code=404, detail=f"Vehicles not found: {missing}")
        
        with open(optimization_path, 'r') as f:
            clustering_result = json.load(f)
        
        # Get existing clusters from optimize-routes
        swachh_auto_vehicles = vehicles_df[
            vehicles_df['vehicle_type'].notna() & 
            (vehicles_df['vehicle_type'].str.upper().str.contains('SWACHH AUTO|SWACHAUTO|SWACHH_AUTO', na=False))
        ]
        
        from src.clustering.assign_buildings import BuildingClusterer
        clusterer = BuildingClusterer()
        
        if 'wardNo' not in buildings_gdf.columns:
            buildings_gdf['wardNo'] = clustering_result.get('wardNo', '1')
        
        clustered_buildings = clusterer.cluster_buildings_with_vehicles(buildings_gdf, swachh_auto_vehicles)
        cluster_groups = clustered_buildings.groupby('cluster')
        clusters_list = [(cluster_id, group) for cluster_id, group in cluster_groups]
        
        # Assign existing clusters to selected vehicles
        route_assignments = {}
        for idx, (cluster_id, cluster_buildings_df) in enumerate(clusters_list):
            vehicle_idx = idx % len(selected_vehicles)
            vehicle = selected_vehicles.iloc[vehicle_idx]
            vehicle_id = vehicle['vehicle_id']
            
            if vehicle_id not in route_assignments:
                route_assignments[vehicle_id] = {
                    'vehicle_info': {
                        'vehicle_id': vehicle_id,
                        'vehicle_type': vehicle.get('vehicle_type', 'garbage_truck'),
                        'status': vehicle.get('status', 'ACTIVE'),
                        'trips_assigned': 0,
                        'houses_assigned': 0,
                        'capacity_per_trip': vehicle.get('capacity', 500)
                    },
                    'trips': []
                }
            
            trip_num = len(route_assignments[vehicle_id]['trips']) + 1
            route_assignments[vehicle_id]['trips'].append({
                'trip_id': f"{vehicle_id}_trip_{trip_num}",
                'house_count': len(cluster_buildings_df),
                'cluster_id': cluster_id,
                'houses': cluster_buildings_df.index.tolist()
            })
            route_assignments[vehicle_id]['vehicle_info']['houses_assigned'] += len(cluster_buildings_df)
            route_assignments[vehicle_id]['vehicle_info']['trips_assigned'] = trip_num
        
        # Generate map with route assignments
        try:
            map_html = generate_map_from_files(ward_path, roads_path, buildings_path, vehicles_path)
            
            os.makedirs("output", exist_ok=True)
            with open("output/route_map.html", "w", encoding="utf-8") as f:
                f.write(map_html)
            
            import webbrowser, time
            map_path = os.path.abspath("output/route_map.html")
            webbrowser.open(f"file://{map_path}")
            time.sleep(5)
            os.remove("output/route_map.html")
        except Exception as e:
            logger.error(f"Map generation error: {e}")
        
        # Save route assignments
        with open("output/route_assignments.json", "w") as f:
            json.dump(convert_numpy_types({
                "wardNo": clustering_result.get('wardNo', '1'),
                "assigned_vehicles": len(selected_vehicles),
                "total_houses": len(buildings_gdf),
                "route_assignments": {k: {"vehicle_info": v["vehicle_info"], "trips": v["trips"]} for k, v in route_assignments.items()}
            }), f, indent=2)
        
        vehicle_data = []
        route_summary = []
        for vehicle_id, assignment in route_assignments.items():
            vi = assignment['vehicle_info']
            vehicle_data.append({
                "vehicle_id": str(vi['vehicle_id']),
                "vehicle_type": str(vi['vehicle_type']),
                "status": str(vi['status']),
                "trips_assigned": vi['trips_assigned'],
                "houses_assigned": vi['houses_assigned'],
                "capacity_per_trip": vi['capacity_per_trip']
            })
            for trip in assignment['trips']:
                route_summary.append({
                    "trip_id": trip['trip_id'],
                    "vehicle_id": vehicle_id,
                    "house_count": trip['house_count'],
                    "cluster_id": trip['cluster_id']
                })
        
        return JSONResponse(convert_numpy_types({
            "status": "success",
            "message": f"Assigned {len(clusters_list)} clusters to {len(selected_vehicles)} vehicles and generated routes with map",
            "maps": {"route_map": "/generate-map"},
            "dashboard": "/cluster-dashboard",
            "wardNo": clustering_result.get('wardNo', '1'),
            "swachh_auto_count": clustering_result.get('swachh_auto_count', 0),
            "active_swachh_auto": clustering_result.get('active_swachh_auto', 0),
            "clusters_created": len(clusters_list),
            "total_houses": len(buildings_gdf),
            "total_trips": len(route_summary),
            "trips_per_vehicle": {vid: len(assignment['trips']) for vid, assignment in route_assignments.items()},
            "vehicle_source": "Selected Vehicles",
            "vehicles_with_routes": vehicle_data,
            "route_summary": route_summary
        }))

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        logger.error(f"Error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test-ward-api/{ward_no}", tags=["optimization"])
async def test_ward_api(ward_no: str):
    """Test ward API endpoint to see what data is returned."""
    try:
        base_url = Config.SWM_API_BASE_URL
        token = Config.SWM_TOKEN.strip("'")
        
        url = f"{base_url}/api/ward-geojson/{ward_no}"
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        
        logger.info(f"Testing API: {url}")
        response = requests.get(url, headers=headers, timeout=30)
        
        return JSONResponse({
            "status_code": response.status_code,
            "url": url,
            "response_type": str(type(response.text)),
            "response_length": len(response.text),
            "response_preview": response.text[:500] if response.text else "Empty",
            "headers": dict(response.headers)
        })
    except Exception as e:
        import traceback as tb
        return JSONResponse({
            "error": str(e),
            "traceback": tb.format_exc()
        })

@app.delete("/cleanup", tags=["optimization"])
async def cleanup_data():
    """Clean up stored cluster data files."""
    try:
        files_to_remove = [
            "output/ward.geojson",
            "output/buildings.geojson", 
            "output/roads.geojson",
            "output/vehicles.csv",
            "output/optimization_result.json"
        ]
        
        removed_files = []
        for file_path in files_to_remove:
            if os.path.exists(file_path):
                os.remove(file_path)
                removed_files.append(file_path)
        
        return JSONResponse({
            "status": "success",
            "message": f"Cleaned up {len(removed_files)} files",
            "removed_files": removed_files
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")



@app.get("/")
async def root():
    return HTMLResponse(content="""
    <html>
        <body>
            <h2>🗺️ Geospatial AI Route Optimizer</h2>
            <div style="background:#fff3cd;border:1px solid #ffeaa7;padding:10px;margin:10px 0;border-radius:5px;">
                <strong>🔐 API Key Required:</strong> <code>swm-2024-secure-key</code><br>
                <small>Add to Authorization header: <code>Bearer swm-2024-secure-key</code></small>
            </div>
            <h3>Available Endpoints:</h3>
            <ul>
                <li><strong>POST /optimize-routes</strong> - Upload files with wardNo and generate optimized routes using live vehicles</li>
                <li><strong>POST /assign-routes-by-vehicle</strong> - Assign routes to specific vehicles (comma-separated IDs) from existing clusters</li>
                <li><strong>GET /cluster-routes</strong> - Get road coordinates for every cluster with timestamps</li>
                <li><strong>GET /cluster/{cluster_id}</strong> - Get cluster roads with coordinates for specific cluster</li>
                <li><strong>GET /clusters</strong> - Get cluster roads with coordinates for all clusters</li>
                <li><strong>GET /generate-map/route_map</strong> - View interactive map with layer controls</li>
                <li><strong>GET /api/vehicles/live</strong> - Get live vehicle data from SWM API</li>
                <li><strong>GET /api/vehicles/{vehicle_id}</strong> - Get specific vehicle details</li>
                <li><strong>PUT /api/vehicles/{vehicle_id}/status</strong> - Update vehicle status</li>
                <li><strong>GET /api/auth/token/info</strong> - Check current token status</li>
                <li><strong>POST /api/auth/token/refresh</strong> - Force refresh token</li>
            </ul>
            <h3>Features:</h3>
            <ul>
                <li>🌐 <strong>Ward-based Vehicle Filtering</strong> - Real-time vehicle data filtered by ward number</li>
                <li>⏰ <strong>Automatic Daily Scheduling</strong> - Daily vehicle data fetch at 5:30 AM</li>
                <li>✅ Interactive cluster dashboard panel</li>
                <li>✅ Layer controls to show/hide individual clusters</li>
                <li>✅ Toggle buttons for each cluster</li>
                <li>✅ Show All / Hide All cluster controls</li>
                <li>✅ Color-coded routes and buildings</li>
                <li>🔐 API Key authentication</li>
                <li>📱 RESTful vehicle management endpoints</li>
                <li>🏘️ Ward-based vehicle clustering and optimization</li>
                <li>📊 Automatic daily data logging with timestamps</li>
            </ul>
            <p><a href="/docs" style="background:#007bff;color:white;padding:10px 20px;text-decoration:none;border-radius:5px;">📚 API Documentation</a></p>
        </body>
    </html>
    """)

if __name__ == "__main__":
    import uvicorn
    load_dotenv()
    port = int(os.getenv('PORT', 8000))
    uvicorn.run(app, host="127.0.0.1", port=port)
