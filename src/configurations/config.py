"""Configuration settings for the garbage collection system."""
import os
from typing import Optional

class Config:
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
    
    # API settings
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8080
    
    # Logging
    LOG_LEVEL: str = "INFO"