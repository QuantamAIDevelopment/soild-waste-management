"""Data models for blackboard entries."""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime
import geopandas as gpd
import pandas as pd

@dataclass
class UploadData:
    upload_id: str
    ward_boundaries: gpd.GeoDataFrame
    road_network: gpd.GeoDataFrame
    houses: gpd.GeoDataFrame
    vehicles: pd.DataFrame
    timestamp: datetime

@dataclass
class RouteResult:
    vehicle_id: str
    route_id: str
    ordered_house_ids: List[str]
    road_segment_ids: List[str]
    start_node: str
    end_node: str
    total_distance_meters: float
    status: str
    geometry: Any  # LineString geometry

@dataclass
class BlackboardEntry:
    entry_id: str
    entry_type: str
    data: Dict[str, Any]
    timestamp: datetime
    status: str = "pending"
    
    def __post_init__(self):
        if not self.entry_id or not isinstance(self.entry_id, str):
            raise ValueError("entry_id must be a non-empty string")
        if not self.entry_type or not isinstance(self.entry_type, str):
            raise ValueError("entry_type must be a non-empty string")
        if not isinstance(self.data, dict):
            raise ValueError("data must be a dictionary")
        if not isinstance(self.timestamp, datetime):
            raise ValueError("timestamp must be a datetime object")
        if self.status not in ["pending", "processing", "completed", "failed"]:
            raise ValueError("status must be one of: pending, processing, completed, failed")