# Intelligent Garbage Collection Route Assignment System

A production-ready Python system for optimizing garbage collection routes using geospatial AI, clustering algorithms, and VRP solving with live vehicle data integration.

## Features

- **Live Vehicle Integration**: Real-time vehicle data from SWM API with automatic authentication
- **Ward-based Filtering**: Filter vehicles by ward number for targeted route optimization
- **Capacity-based Optimization**: Multi-trip assignment based on vehicle capacity constraints
- **Interactive Route Maps**: Folium-based maps with cluster controls and trip visualization
- **OSRM Integration**: Real-world driving directions and turn-by-turn navigation
- **Automatic Scheduling**: Daily vehicle data fetch at 5:30 AM with APScheduler
- **RESTful API**: FastAPI with OpenAPI/Swagger documentation
- **Geospatial Processing**: NetworkX graphs, GeoPandas, and spatial clustering

## Architecture

```
├── src/
│   ├── api/               # FastAPI endpoints and route handlers
│   │   ├── geospatial_routes.py  # Main optimization API
│   │   ├── vehicles_api.py       # Vehicle management endpoints
│   │   └── auth_endpoints.py     # Authentication endpoints
│   ├── clustering/        # Building clustering and trip assignment
│   │   ├── assign_buildings.py   # KMeans/DBSCAN clustering
│   │   └── trip_assignment.py    # Capacity-based trip planning
│   ├── configurations/    # System configuration
│   │   └── config.py             # Environment and app settings
│   ├── data_processing/   # Geospatial data processing
│   │   ├── load_road_network.py  # Road network graph building
│   │   └── snap_buildings.py     # Building-to-road snapping
│   ├── routing/           # Route computation and optimization
│   │   ├── capacity_optimizer.py # Capacity-based optimization
│   │   ├── compute_routes.py     # OR-Tools VRP solver
│   │   └── get_osrm_directions.py # OSRM turn-by-turn directions
│   ├── services/          # External service integration
│   │   ├── vehicle_service.py    # Live vehicle API integration
│   │   ├── auth_service.py       # JWT token management
│   │   └── scheduler_service.py  # Daily scheduling service
│   ├── tools/             # Utility tools and algorithms
│   │   ├── road_snapper.py       # Road network snapping
│   │   ├── vrp_solver.py         # VRP optimization
│   │   └── osrm_routing.py       # OSRM routing utilities
│   └── visualization/     # Map generation and export
│       ├── folium_map.py         # Interactive map generation
│       └── export_to_geojson.py  # GeoJSON export utilities
├── tests/                 # Unit and integration tests
├── output/                # Generated maps and results (auto-created)
└── main.py               # CLI and API entry point
```

## Quick Start

### Prerequisites

- Python 3.11+
- Git
- Internet connection for OSRM routing and live vehicle data

### Installation

1. Clone and setup:
```bash
git clone <repository>
cd Solid_waste_management
pip install -r requirements.txt
```

2. Configure environment variables in `.env`:
```bash
# SWM API Configuration
SWM_API_BASE_URL=https://uat-swm-main-service-hdaqcdcscbfedhhn.centralindia-01.azurewebsites.net
SWM_USERNAME=your_username
SWM_PASSWORD=your_password

# Optional: External upload URL for route data
EXTERNAL_UPLOAD_URL=your_external_api_url
SWM_TOKEN=your_jwt_token

# API Security
API_KEY=swm-2024-secure-key
```

3. Run the API server:
```bash
python main.py --api --port 8081
```

**Automatic Features**:
- Vehicle data fetched daily at 5:30 AM
- JWT token auto-refresh
- Interactive maps with cluster controls

Or run CLI mode:
```bash
python main.py --roads roads.geojson --buildings buildings.geojson --output results/
```

The API will be available at `http://localhost:8081` with Swagger UI at `http://localhost:8081/docs`.

## API Usage

### Authentication
All endpoints require Bearer token authentication:
```bash
# Add to all requests
-H "Authorization: Bearer swm-2024-secure-key"
```

### 1. Upload Files and Optimize Routes
```bash
curl -X POST "http://localhost:8081/optimize-routes" \
  -H "Authorization: Bearer swm-2024-secure-key" \
  -F "roads_file=@roads.geojson" \
  -F "buildings_file=@buildings.geojson" \
  -F "ward_geojson=@ward.geojson" \
  -F "ward_no=1" \
  -F "vehicles_csv=@vehicles.csv"
```

### 2. Get Cluster Roads with Coordinates
```bash
curl -H "Authorization: Bearer swm-2024-secure-key" \
  "http://localhost:8081/cluster/0"
```

**Response:**
```json
{
  "cluster_id": 0,
  "vehicle_info": {
    "vehicle_id": "SWM001",
    "vehicle_type": "garbage_truck",
    "status": "active",
    "capacity": 500
  },
  "buildings_count": 15,
  "roads": [
    {
      "start_coordinate": {"longitude": 77.123, "latitude": 12.456},
      "end_coordinate": {"longitude": 77.124, "latitude": 12.457},
      "distance_meters": 125.5
    }
  ],
  "total_road_segments": 8,
  "cluster_bounds": {
    "min_longitude": 77.120,
    "max_longitude": 77.130,
    "min_latitude": 12.450,
    "max_latitude": 12.460
  }
}
```

### 3. Get Route Coordinates with Timestamps
```bash
curl -H "Authorization: Bearer swm-2024-secure-key" \
  "http://localhost:8081/cluster-routes"
```

### 4. Live Vehicle Data
```bash
# All vehicles
curl -H "Authorization: Bearer swm-2024-secure-key" \
  "http://localhost:8081/api/vehicles/live"

# Vehicles by ward
curl -H "Authorization: Bearer swm-2024-secure-key" \
  "http://localhost:8081/api/vehicles/ward/1"
```

### 5. Scheduler Management
```bash
# Check scheduler status
curl -H "Authorization: Bearer swm-2024-secure-key" \
  "http://localhost:8081/scheduler/status"

# Manually trigger vehicle fetch
curl -X POST -H "Authorization: Bearer swm-2024-secure-key" \
  "http://localhost:8081/scheduler/trigger"
```

### 6. Interactive Maps
- Route Map: `http://localhost:8081/generate-map`
- API Documentation: `http://localhost:8081/docs`

## Automatic Scheduling

### Daily Vehicle Data Fetch
The system automatically fetches vehicle data at **5:30 AM daily** using APScheduler:

- **Automatic Start**: Scheduler starts with FastAPI server
- **JWT Token Management**: Automatic token refresh before data fetch
- **Data Storage**: Saves to `output/vehicles_daily_YYYYMMDD_HHMMSS.csv`
- **Logging**: Timestamped logs for every scheduled run
- **Error Handling**: Fallback to cached data if API fails

### Scheduler Endpoints
| Endpoint | Method | Description |
|----------|--------|--------------|
| `/scheduler/status` | GET | Check scheduler status and next run time |
| `/scheduler/trigger` | POST | Manually trigger vehicle data fetch |
| `/scheduler/start` | POST | Start scheduler if stopped |
| `/scheduler/stop` | POST | Stop scheduler |

### Authentication Management
```bash
# Check token status
curl -H "Authorization: Bearer swm-2024-secure-key" \
  "http://localhost:8081/api/auth/token/info"

# Force token refresh
curl -X POST -H "Authorization: Bearer swm-2024-secure-key" \
  "http://localhost:8081/api/auth/token/refresh"
```

### Testing Scheduler
```bash
# Run test script
python test_scheduler.py

# Check output files
dir output\vehicles_daily_*.csv
```

## Input File Formats

### Ward Boundaries (GeoJSON)
```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature",
    "properties": {"ward_id": 1, "ward_name": "Ward 1"},
    "geometry": {"type": "Polygon", "coordinates": [...]}
  }]
}
```

### Road Network (GeoJSON)
```json
{
  "type": "FeatureCollection", 
  "features": [{
    "type": "Feature",
    "properties": {"road_id": "R001", "road_name": "Main St"},
    "geometry": {"type": "LineString", "coordinates": [...]}
  }]
}
```

### Buildings/Houses (GeoJSON)
```json
{
  "type": "FeatureCollection",
  "features": [{
    "type": "Feature", 
    "properties": {"house_id": "H001", "ward_no": 1},
    "geometry": {"type": "Point", "coordinates": [...]}
  }]
}
```

### Vehicles (CSV - Optional)
```csv
vehicle_id,vehicle_type,ward_no,status,capacity,driverName
SWM001,garbage_truck,1,active,500,Driver1
SWM002,garbage_truck,1,active,500,Driver2
```

**Note**: Vehicle data is automatically fetched from live SWM API. CSV upload is optional for testing.

## Algorithm Details

### 1. Data Processing Pipeline
- **CRS Conversion**: EPSG:4326 (WGS84) for display, EPSG:3857 (Web Mercator) for calculations
- **Building Snapping**: Snap buildings to nearest road network nodes
- **Graph Construction**: NetworkX graph from road network for routing
- **Live Data Integration**: Real-time vehicle data with authentication

### 2. Capacity-based Clustering
- **Vehicle Filtering**: Active vehicles only (status: ACTIVE/AVAILABLE/ONLINE)
- **KMeans Clustering**: k = number of active vehicles
- **Trip Assignment**: Multiple trips per vehicle based on capacity (500 houses/trip)
- **Ward Filtering**: Vehicles filtered by ward number for targeted optimization

### 3. Route Optimization
- **VRP Solver**: OR-Tools with capacity constraints and time limits
- **Distance Matrix**: NetworkX shortest paths on road network
- **Multi-trip Support**: Up to 3 trips per vehicle per day
- **OSRM Integration**: Real-world driving directions and turn-by-turn navigation

### 4. Interactive Visualization
- **Folium Maps**: Color-coded routes with cluster controls
- **Layer Management**: Show/hide individual trips and clusters
- **Route Details**: Start/end markers, directional arrows, trip statistics
- **Dashboard Panel**: Trip summary with toggle controls

## Testing

Run unit tests:
```bash
pytest tests/ -v
```

Run integration test:
```bash
pytest tests/test_snapper_vrp.py::TestIntegration::test_end_to_end_small_dataset -v
```

## Configuration

Edit `src/configurations/config.py`:

```python
class Config:
    # Spatial reference systems
    TARGET_CRS = "EPSG:3857"  # Web Mercator for calculations
    
    # Vehicle capacity settings
    HOUSES_PER_VEHICLE_PER_TRIP = 500
    MAX_TRIPS_PER_DAY = 3
    
    # VRP solver settings
    VRP_TIME_LIMIT_SECONDS = 30
    VRP_VEHICLE_CAPACITY = 999999  # Effectively infinite
    
    # API settings
    API_HOST = "127.0.0.1"
    API_PORT = 8080
    
    # Deterministic results
    RANDOM_SEED = 42
```

## Performance

- **Scalability**: Optimized for ward sizes up to several thousand houses
- **Clustering Efficiency**: KMeans clustering limits VRP problem size
- **Spatial Indexing**: GeoPandas spatial operations for fast nearest neighbor queries
- **Caching**: JWT token caching and session reuse for API calls
- **Deterministic Results**: Configurable random seed for reproducible optimization
- **Memory Management**: Automatic cleanup of temporary files and maps

## Logging

The system provides comprehensive logging:
- Clustering assignments and decisions
- Road segment assignment rationale  
- OR-Tools objective values
- Conflict resolution steps
- Performance metrics

## Docker Support

```dockerfile
FROM python:3.11-slim

# Install system dependencies for geospatial libraries
RUN apt-get update && apt-get install -y \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8081

# Run API server by default
CMD ["python", "main.py", "--api", "--port", "8081"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  swm-optimizer:
    build: .
    ports:
      - "8081:8081"
    environment:
      - SWM_API_BASE_URL=${SWM_API_BASE_URL}
      - SWM_USERNAME=${SWM_USERNAME}
      - SWM_PASSWORD=${SWM_PASSWORD}
      - API_KEY=${API_KEY}
    volumes:
      - ./output:/app/output
```

## Key Features Summary

- 🌐 **Live Vehicle Integration**: Real-time SWM API with JWT authentication
- 🗺️ **Interactive Maps**: Folium-based visualization with cluster controls
- 🚛 **Capacity Optimization**: Multi-trip assignment based on vehicle capacity
- 📍 **OSRM Routing**: Real-world driving directions and navigation
- ⏰ **Automatic Scheduling**: Daily data fetch at 5:30 AM
- 🔐 **Secure API**: Bearer token authentication for all endpoints
- 📊 **Ward-based Filtering**: Target specific wards for optimization
- 🎯 **Geospatial AI**: NetworkX graphs, spatial clustering, and VRP solving

## License

MIT License - see LICENSE file for details.