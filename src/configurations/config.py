"""Configuration settings for the garbage collection system."""
import os

class Config:
    # SWM API Configuration
    SWM_API_BASE_URL: str = os.getenv("SWM_API_BASE_URL", "https://swm-main-service-wap-htangag4ghgsfeg4.centralindia-01.azurewebsites.net/")
    SWM_USERNAME: str = os.getenv("SWM_USERNAME", "swmsuperadmin")
    SWM_PASSWORD: str = os.getenv("SWM_PASSWORD", "Admin@123")
    SWM_TOKEN: str = os.getenv("SWM_TOKEN", "")
    
    # Server Configuration
    PORT: int = int(os.getenv("PORT", "8000"))
    API_KEY: str = os.getenv("API_KEY", "swm-2024-secure-key")
    
    # External Upload URL
    EXTERNAL_UPLOAD_URL: str = os.getenv("EXTERNAL_UPLOAD_URL", "https://swm-main-service-wap-htangag4ghgsfeg4.centralindia-01.azurewebsites.net/api/vehicle-routes/upload-data")
    
    # Spatial reference system
    TARGET_CRS: str = "EPSG:3857"  # Web Mercator for distance calculations
    
    # Random seed for deterministic results
    RANDOM_SEED: int = 42
    
    # VRP solver settings
    VRP_TIME_LIMIT_SECONDS: int = 30
    VRP_VEHICLE_CAPACITY: int = 999999  # Effectively infinite as per spec
    
    # Vehicle capacity settings
    HOUSES_PER_VEHICLE_PER_TRIP: int = 500
    MAX_TRIPS_PER_DAY: int = 3
    
    # File upload settings
    MAX_FILE_SIZE_MB: int = 100
    UPLOAD_DIR: str = "uploads"
    
    # API settings (legacy - use PORT instead)
    API_HOST: str = "127.0.0.1"
    API_PORT: int = int(os.getenv("PORT", "8000"))
    
    # Logging
    LOG_LEVEL: str = "INFO"