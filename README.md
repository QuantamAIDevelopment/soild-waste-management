# 🗺️ Geospatial AI Route Optimizer for Solid Waste Management

AI-powered garbage collection route optimization using real-time vehicle data, geospatial clustering, and road network analysis.

## 🚀 Key Features

- **Live API Integration**: Real-time vehicle data from SWM API
- **Swachh Auto Focused**: Clusters based on Swachh Auto vehicle count
- **Ward-Based Filtering**: Target specific wards for optimization
- **Geographic Clustering**: KMeans spatial clustering
- **Interactive Maps**: Folium visualization with layer controls
- **Automatic Scheduling**: Daily data fetch at 5:30 AM
- **Auto-Upload**: Routes uploaded to external API automatically
- **Secure API**: Bearer token authentication

## 📋 Prerequisites

- Python 3.11+
- GDAL (for geospatial operations)

## 🔧 Installation

```bash
# Clone repository
git clone <repository-url>
cd swm2

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your credentials
```

### Environment Variables (.env)
```env
SWM_API_BASE_URL=https://uat-swm-main-service-hdaqcdcscbfedhhn.centralindia-01.azurewebsites.net
SWM_USERNAME=your_username
SWM_PASSWORD=your_password
SWM_TOKEN='your_jwt_token'
PORT=8001
API_KEY=swm-2024-secure-key
EXTERNAL_UPLOAD_URL=http://192.168.100.154:8081/api/vehicle-routes/upload-data
```

## 🏃 Quick Start

### Start API Server
```bash
python main.py --api
# Server starts on http://127.0.0.1:8001
```

### CLI Mode
```bash
python main.py --roads roads.geojson --buildings buildings.geojson --output output/
```

## 📡 API Endpoints

All endpoints require: `Authorization: Bearer swm-2024-secure-key`

### Core Endpoints

**1. Optimize Routes**
```bash
POST /optimize-routes
Form Data:
- wardNo: string (required)
- roads_file: file (optional)
- buildings_file: file (optional)
- ward_geojson: file (optional)
```

**2. Assign Routes to Vehicles**
```bash
POST /assign-routes-by-vehicle
Form Data:
- vehicle_ids: string (comma-separated)
```

**3. Get Cluster Routes**
```bash
GET /cluster-routes
# Returns routes with auto-upload to external API
```

**4. Get Specific Cluster**
```bash
GET /cluster/{cluster_id}
```

**5. Generate Map**
```bash
GET /generate-map
```

### Vehicle Management

```bash
GET /api/vehicles/live?wardNo=29
GET /api/vehicles/{vehicle_id}
PUT /api/vehicles/{vehicle_id}/status
```

### Authentication

```bash
GET /api/auth/token/info
POST /api/auth/token/refresh
```

### Ward GeoJSON

```bash
POST /api/ward-geojson/upload
GET /api/ward-geojson/{ward_no}
```

## 🔄 Complete Workflow

```bash
# Step 1: Upload ward data and create clusters
curl -X POST "http://localhost:8001/optimize-routes" \
  -H "Authorization: Bearer swm-2024-secure-key" \
  -F "wardNo=Ward 29" \
  -F "buildings_file=@buildings.geojson" \
  -F "roads_file=@roads.geojson"

# Step 2: Assign routes to specific vehicles
curl -X POST "http://localhost:8001/assign-routes-by-vehicle" \
  -H "Authorization: Bearer swm-2024-secure-key" \
  -F "vehicle_ids=SA001,SA002,SA003"

# Step 3: Get routes and auto-upload
curl -X GET "http://localhost:8001/cluster-routes" \
  -H "Authorization: Bearer swm-2024-secure-key"
```

## 🎯 How It Works

### Clustering Logic

1. **Filter Swachh Auto**: Only Swachh Auto vehicles used for clustering
2. **Create Clusters**: Number of clusters = Total Swachh Auto count (all statuses)
3. **Filter Active**: Only ACTIVE/AVAILABLE/ONLINE vehicles get routes
4. **Assign Routes**: Each active vehicle gets one cluster

### Example

**Ward 29:** 5 Swachh Auto (3 active, 2 inactive), 1000 buildings

**Result:**
- 5 clusters created (based on total Swachh Auto)
- 3 routes assigned (to active vehicles only)
- 2 clusters unassigned (for inactive vehicles)

| Vehicle | Status | Cluster | Buildings | Route |
|---------|--------|---------|-----------|-------|
| SA001 | ACTIVE | 0 | 200 | ✅ |
| SA002 | ACTIVE | 1 | 200 | ✅ |
| SA003 | ACTIVE | 2 | 200 | ✅ |
| SA004 | INACTIVE | - | - | ❌ |
| SA005 | INACTIVE | - | - | ❌ |

## 📊 Configuration

Edit `src/configurations/config.py`:

```python
class Config:
    TARGET_CRS = "EPSG:3857"
    HOUSES_PER_VEHICLE_PER_TRIP = 500
    MAX_TRIPS_PER_DAY = 3
    VRP_TIME_LIMIT_SECONDS = 30
    API_HOST = "127.0.0.1"
    API_PORT = 8080
    RANDOM_SEED = 42
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test
pytest tests/test_snapper_vrp.py::TestIntegration::test_end_to_end_small_dataset -v
```

## 📦 Project Structure

```
swm2/
├── src/
│   ├── api/                 # FastAPI endpoints
│   ├── clustering/          # Clustering algorithms
│   ├── configurations/      # Config settings
│   ├── data_processing/     # Data loading
│   ├── routing/             # Route optimization
│   ├── services/            # Business logic
│   ├── tools/               # Utility tools
│   └── visualization/       # Map generation
├── tests/                   # Tests
├── manifests/               # Kubernetes manifests
│   ├── deployment.yml
│   ├── service.yml
│   └── secrets.yaml
├── main.py                  # Entry point
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🐳 Docker Deployment

### Build and Run
```bash
docker build -t swm-optimizer .
docker run -p 8001:8001 --env-file .env swm-optimizer
```

### Kubernetes Deployment
```bash
# Apply secrets
kubectl apply -f manifests/secrets.yaml

# Deploy application
kubectl apply -f manifests/deployment.yml
kubectl apply -f manifests/service.yml
```

## ⚠️ Troubleshooting

### No Swachh Auto Vehicles
**Error:** "No Swachh Auto vehicles found in ward X"

**Solution:** Verify ward has Swachh Auto vehicles with correct naming

### No Active Swachh Auto
**Error:** "No ACTIVE Swachh Auto vehicles found"

**Solution:** Activate at least one Swachh Auto vehicle (valid statuses: ACTIVE, AVAILABLE, ONLINE, HALTED, IDEL)

### Port Already in Use
```bash
python main.py --api --port 8002
# Or update PORT in .env
```

### Token Expiration
```bash
curl -X POST "http://localhost:8001/api/auth/token/refresh" \
  -H "Authorization: Bearer swm-2024-secure-key"
```

## 📈 Performance

- Handles wards with thousands of buildings
- KMeans for efficient spatial grouping
- JWT token caching and session reuse
- Deterministic results with configurable seed
- Automatic cleanup of temporary files

## 🔄 Automatic Scheduling

System automatically fetches vehicle data daily at 5:30 AM.

Check status:
```bash
GET /api/scheduler/status
```

Manual trigger:
```bash
POST /api/scheduler/trigger
```

## 🌐 Interactive Maps

Maps include:
- Ward boundary (blue outline)
- Color-coded clusters by vehicle
- Directional route arrows
- Building polygons
- Trip dashboard with statistics
- Layer controls (show/hide clusters)
- Toggle buttons for individual clusters

## 📝 Logging

Comprehensive logging with Loguru:
- Clustering decisions
- Vehicle assignments
- API calls and responses
- Route optimization metrics
- Upload status

## 🔐 Security

- Never commit `.env` file
- Use `.env.example` as template
- JWT tokens auto-refresh
- API keys required for all endpoints
- Kubernetes secrets for sensitive data

## 📄 License

MIT License

## 📞 Support

- API Documentation: `http://localhost:8001/docs`
- Check logs for detailed error information
- Review troubleshooting section above
