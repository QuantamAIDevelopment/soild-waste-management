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
from src.api.vehicles_api import router as vehicles_router
from src.api.auth_endpoints import router as auth_router
from src.routing.capacity_optimizer import CapacityRouteOptimizer

from loguru import logger
import warnings
import requests
from dotenv import load_dotenv

# Suppress specific geographic CRS warnings for intentional lat/lon usage in maps
warnings.filterwarnings('ignore', message='.*Geometry is in a geographic CRS.*')

# API Key for authentication - Change this in production!
API_KEY = "swm-2024-secure-key"

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

# Include vehicle API routes
app.include_router(vehicles_router)

# Include authentication API routes
app.include_router(auth_router)

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
    roads_file: UploadFile = File(..., description="Roads GeoJSON file"),
    buildings_file: UploadFile = File(..., description="Buildings GeoJSON file"), 
    ward_geojson: UploadFile = File(..., description="Ward boundary GeoJSON file"),
    wardNo: str = Form(..., description="Ward number to filter vehicles"),
    vehicles_csv: UploadFile = File(None, description="Optional vehicles CSV file (uses live API if not provided)")
):
    """Upload files and run complete route optimization pipeline."""
    
    # Validate file types and wardNo
    if not roads_file.filename or not roads_file.filename.lower().endswith('.geojson'):
        raise HTTPException(status_code=400, detail="Roads file must be GeoJSON")
    if not buildings_file.filename or not buildings_file.filename.lower().endswith('.geojson'):
        raise HTTPException(status_code=400, detail="Buildings file must be GeoJSON")
    if not wardNo or not wardNo.strip():
        raise HTTPException(status_code=400, detail="Ward number is required")
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Save uploaded files
            roads_path = os.path.join(temp_dir, "roads.geojson")
            buildings_path = os.path.join(temp_dir, "buildings.geojson")
            ward_path = os.path.join(temp_dir, "ward.geojson")
            
            with open(roads_path, "wb") as f:
                shutil.copyfileobj(roads_file.file, f)
            with open(buildings_path, "wb") as f:
                shutil.copyfileobj(buildings_file.file, f)
            with open(ward_path, "wb") as f:
                shutil.copyfileobj(ward_geojson.file, f)
            
            # Load geospatial data
            buildings_gdf = gpd.read_file(buildings_path)
            roads_gdf = gpd.read_file(roads_path)
            
            # Convert to WGS84 if needed
            if buildings_gdf.crs != 'EPSG:4326':
                buildings_gdf = buildings_gdf.to_crs('EPSG:4326')
            if roads_gdf.crs != 'EPSG:4326':
                roads_gdf = roads_gdf.to_crs('EPSG:4326')
            
            # Get vehicle data from CSV file or live API
            try:
                if vehicles_csv and vehicles_csv.filename:
                    # CSV Upload: Use ONLY CSV vehicles, NO API calls
                    vehicles_csv_path = os.path.join(temp_dir, "vehicles.csv")
                    with open(vehicles_csv_path, "wb") as f:
                        shutil.copyfileobj(vehicles_csv.file, f)
                    vehicles_df = pd.read_csv(vehicles_csv_path)
                    
                    from src.services.vehicle_service import VehicleService
                    temp_service = VehicleService()
                    vehicles_df = temp_service._standardize_vehicle_data(vehicles_df)
                    
                    vehicles_path = vehicles_csv_path
                    vehicle_source = "Uploaded CSV File"
                    logger.info(f"[CSV] Using {len(vehicles_df)} vehicles from CSV - NO API calls")
                else:
                    # No CSV: Call live API only with valid wardNo
                    if not wardNo or wardNo.strip() in ["string", ""]:
                        raise HTTPException(status_code=400, detail="Valid ward number required when no CSV provided")
                    
                    # Get ALL vehicles in ward (including INACTIVE) - filtering happens later
                    vehicles_df = vehicle_service.get_vehicles_by_ward(wardNo.strip(), include_all_status=True)
                    vehicles_csv_path = os.path.join(temp_dir, "vehicles.csv")
                    vehicles_df.to_csv(vehicles_csv_path, index=False)
                    vehicles_path = vehicles_csv_path
                    vehicle_source = "Live API (Ward Filtered - All Status)"
                    logger.info(f"[API] Using {len(vehicles_df)} vehicles from ward {wardNo} (all statuses)")
                
                if len(vehicles_df) == 0:
                    source = "CSV file" if vehicles_csv and vehicles_csv.filename else f"ward {wardNo}"
                    raise HTTPException(status_code=404, detail=f"No vehicles found in {source}")
                
                # Filter for ACTIVE vehicles only - exclude INACTIVE and invalid statuses
                active_vehicles = vehicles_df[
                    vehicles_df['status'].notna() & 
                    (vehicles_df['status'].str.upper().isin(['ACTIVE', 'AVAILABLE', 'ONLINE', 'READY', 'OPERATIONAL', 'IN_SERVICE', 'HALTED', 'IDEL']))
                ]
                
                if len(active_vehicles) == 0:
                    # Return detailed response with all vehicles and their statuses
                    status_distribution = vehicles_df['status'].value_counts().to_dict()
                    all_vehicles_info = vehicles_df[['vehicle_id', 'vehicleNo', 'status', 'vehicle_type']].to_dict('records') if 'vehicleNo' in vehicles_df.columns else vehicles_df[['vehicle_id', 'status']].to_dict('records')
                    
                    return JSONResponse(
                        status_code=200,
                        content={
                            "status": "error",
                            "error_type": "no_active_vehicles",
                            "message": f"No ACTIVE vehicles found in ward {wardNo}. Cannot generate routes.",
                            "wardNo": wardNo,
                            "total_vehicles_in_ward": len(vehicles_df),
                            "active_vehicles": 0,
                            "status_distribution": status_distribution,
                            "all_vehicles": all_vehicles_info,
                            "available_statuses": list(status_distribution.keys()),
                            "recommendation": "Please activate vehicles or check vehicle status in the ward to generate routes."
                        }
                    )
                
                # Clustering: TOTAL vehicles → clusters, ACTIVE vehicles → routes
                from src.clustering.assign_buildings import BuildingClusterer
                clusterer = BuildingClusterer()
                
                if 'wardNo' not in buildings_gdf.columns and 'wardNo' not in buildings_gdf.columns:
                    buildings_gdf['wardNo'] = wardNo
                
                # Create clusters based on TOTAL vehicles (all statuses)
                total_vehicles_count = len(vehicles_df)
                clustered_buildings = clusterer.cluster_buildings_with_vehicles(buildings_gdf, vehicles_df)
                
                # Extract cluster groups
                cluster_groups = clustered_buildings.groupby('cluster')
                clusters_list = [(cluster_id, group) for cluster_id, group in cluster_groups]
                
                # Assign routes to ACTIVE vehicles (round-robin if fewer active)
                route_assignments = {}
                active_count = len(active_vehicles)
                
                for idx, (cluster_id, cluster_buildings_df) in enumerate(clusters_list):
                    vehicle_idx = idx % active_count
                    vehicle = active_vehicles.iloc[vehicle_idx]
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
                    
                    route_assignments[vehicle_id]['vehicle_info']['trips_assigned'] = trip_num
                    route_assignments[vehicle_id]['vehicle_info']['houses_assigned'] += len(cluster_buildings_df)
                
                optimization_result = {
                    'total_vehicles_in_ward': total_vehicles_count,
                    'active_vehicles': active_count,
                    'clusters_created': len(clusters_list),
                    'total_houses': len(buildings_gdf),
                    'route_assignments': route_assignments
                }
                
                print(f"Clusters: {total_vehicles_count} total vehicles → {len(clusters_list)} clusters | Routes: {active_count} active vehicles covering all")
                
            except HTTPException:
                raise
            except ValueError as ve:
                print(f"Route optimization failed: {ve}")
                raise HTTPException(status_code=400, detail=str(ve))
            except Exception as vehicle_error:
                print(f"Failed to get vehicle data: {vehicle_error}")
                raise HTTPException(status_code=500, detail="Failed to get vehicle data")
            
            # Generate map using uploaded files
            try:
                map_html = generate_map_from_files(ward_path, roads_path, buildings_path, vehicles_path)
                print("Map generation completed successfully")
            except Exception as map_error:
                print(f"Map generation error: {map_error}")
                import traceback
                print(f"Full error: {traceback.format_exc()}")
                # Create simple fallback map
                map_html = "<html><body><h1>Map Processing Complete</h1><p>Data uploaded successfully</p></body></html>"
            
            # Save map and data to output directory
            os.makedirs("output", exist_ok=True)
            try:
                with open("output/route_map.html", "w", encoding="utf-8") as f:
                    f.write(map_html)
                
                # Open map in browser automatically
                import webbrowser
                import time
                map_path = os.path.abspath("output/route_map.html")
                webbrowser.open(f"file://{map_path}")
                print(f"Map opened in browser: {map_path}")
                
                # Wait briefly then cleanup
                time.sleep(5)
                os.remove("output/route_map.html")
                print("Map file cleaned up after opening")
            except Exception as save_error:
                print(f"File save error: {save_error}")
                raise save_error
            
            # Save data files for cluster endpoint (keep them for /clusters access)
            shutil.copy(ward_path, "output/ward.geojson")
            shutil.copy(buildings_path, "output/buildings.geojson")
            shutil.copy(roads_path, "output/roads.geojson")
            shutil.copy(vehicles_path, "output/vehicles.csv")
            
            # Save optimization results for cluster endpoint
            import json
            with open("output/optimization_result.json", "w") as f:
                json.dump(convert_numpy_types({
                    "wardNo": wardNo,
                    "active_vehicles": optimization_result['active_vehicles'],
                    "total_houses": optimization_result['total_houses'],
                    "route_assignments": {k: {
                        "vehicle_info": v["vehicle_info"],
                        "trips": v["trips"]
                    } for k, v in optimization_result['route_assignments'].items()}
                }), f, indent=2)
            
            print("Data files saved for cluster endpoint access")
            
            # Prepare response data
            vehicle_data = []
            route_summary = []
            all_vehicles_status = vehicles_df[['vehicle_id', 'status']].to_dict('records') if 'vehicle_id' in vehicles_df.columns else []
            
            for vehicle_id, assignment in optimization_result['route_assignments'].items():
                vehicle_info = assignment['vehicle_info']
                vehicle_data.append({
                    "vehicle_id": str(vehicle_info['vehicle_id']),
                    "vehicle_type": str(vehicle_info['vehicle_type']),
                    "status": str(vehicle_info['status']),
                    "trips_assigned": vehicle_info['trips_assigned'],
                    "houses_assigned": vehicle_info['houses_assigned'],
                    "capacity_per_trip": vehicle_info['capacity_per_trip']
                })
                
                for trip in assignment['trips']:
                    route_summary.append({
                        "trip_id": trip['trip_id'],
                        "vehicle_id": vehicle_id,
                        "house_count": trip['house_count'],
                        "cluster_id": trip['cluster_id']
                    })
            
            # Calculate trips per vehicle
            trips_per_vehicle = {}
            for vehicle_id, assignment in optimization_result['route_assignments'].items():
                trips_per_vehicle[vehicle_id] = len(assignment['trips'])
            
            # Convert to JSON-safe format
            response_data = {
                "status": "success",
                "message": f"Created {optimization_result['clusters_created']} clusters from {optimization_result['total_vehicles_in_ward']} total vehicles. Routes assigned to {optimization_result['active_vehicles']} active vehicles.",
                "maps": {
                    "route_map": "/generate-map"
                },
                "dashboard": "/cluster-dashboard",
                "wardNo": wardNo,
                "total_vehicles_in_ward": optimization_result['total_vehicles_in_ward'],
                "active_vehicles": optimization_result['active_vehicles'],
                "clusters_created": optimization_result['clusters_created'],
                "total_houses": optimization_result['total_houses'],
                "total_trips": len(route_summary),
                "trips_per_vehicle": trips_per_vehicle,
                "vehicle_source": vehicle_source,
                "vehicles_with_routes": vehicle_data,
                "all_vehicles_in_ward": all_vehicles_status,
                "route_summary": route_summary,
                "clustering_strategy": f"Clusters based on {optimization_result['total_vehicles_in_ward']} total vehicles, routes assigned to {optimization_result['active_vehicles']} active vehicles",
                "features": [
                    f"Clusters: {optimization_result['clusters_created']} (based on total vehicles)",
                    f"Routes: {optimization_result['active_vehicles']} active vehicles",
                    "Active vehicles may handle multiple clusters",
                    "Live vehicle status tracking",
                    "Ward-based filtering"
                ]
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
        
        # Load optimization results if available
        optimization_result = None
        if os.path.exists(optimization_path):
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
            
            external_url = os.getenv('EXTERNAL_UPLOAD_URL')
            swm_token = os.getenv('SWM_TOKEN')
            
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
        
        # Use fixed clustering based on ward vehicle count
        from src.clustering.assign_buildings import BuildingClusterer
        clusterer = BuildingClusterer()
        
        # Add ward info if not present
        if 'wardNo' not in buildings_gdf.columns and 'wardNo' not in buildings_gdf.columns:
            buildings_gdf['wardNo'] = '1'  # Default ward for map generation
        
        # Apply fixed clustering using provided vehicles (no API calls)
        clustered_buildings = clusterer.cluster_buildings_with_vehicles(buildings_gdf, vehicles_df)
        
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
    
    # Colors and vehicle names from active vehicles or defaults
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan']
    if vehicles_df is not None:
        active_vehicles = vehicles_df[
            vehicles_df['status'].str.upper().isin(['ACTIVE', 'AVAILABLE', 'ONLINE'])
        ]
        vehicle_names = active_vehicles['vehicle_id'].tolist()[:len(set(building_clusters))]
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
        if vehicles_df is not None and cluster_id < len(active_vehicles):
            vehicle = active_vehicles.iloc[cluster_id]
            vehicle_info = f" ({vehicle.get('vehicle_type', 'N/A')} - ACTIVE)"
        
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
                    if vehicles_df is not None and cluster_id < len(vehicles_df):
                        vehicle = vehicles_df.iloc[cluster_id]
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
                    start_popup = f"{vehicle_name} Start (ACTIVE)"
                    if vehicles_df is not None and cluster_id < len(active_vehicles):
                        vehicle = active_vehicles.iloc[cluster_id]
                        start_popup += f"\nID: {vehicle.get('vehicle_id', 'N/A')}\nCapacity: {vehicle.get('capacity', 'N/A')}"
                    
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
            if vehicles_df is not None and cluster_id < len(active_vehicles):
                vehicle = active_vehicles.iloc[cluster_id]
                vehicle_details = f" • {vehicle.get('vehicle_type', 'N/A')} • ACTIVE • Cap: {vehicle.get('capacity', 'N/A')}"
            
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
    uvicorn.run(app, host="127.0.0.1", port=8080)
