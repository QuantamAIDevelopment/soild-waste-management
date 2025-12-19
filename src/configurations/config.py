"""Configuration settings for the garbage collection system."""
import os

class Config:
    # SWM API Configuration
    SWM_API_BASE_URL: str = os.getenv("SWM_API_BASE_URL") or ""
    
    # Server Configuration
    PORT: int = int(os.getenv("PORT", "8000"))
    
    # External Upload URL
    EXTERNAL_UPLOAD_URL: str = os.getenv("EXTERNAL_UPLOAD_URL") or ""
    
    @classmethod
    def validate(cls):
        try:
            required = ["SWM_API_BASE_URL", "EXTERNAL_UPLOAD_URL"]
            missing = [var for var in required if not getattr(cls, var)]
            if missing:
                raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        except AttributeError as e:
            raise ValueError(f"Configuration error: {e}")
        except Exception as e:
            raise ValueError(f"Validation failed: {e}")
    
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