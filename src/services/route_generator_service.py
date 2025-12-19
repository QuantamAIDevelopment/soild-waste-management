"""Route generation service based on vehicle status and clustering."""
import pandas as pd
import geopandas as gpd
from typing import Dict, List, Tuple, Optional
from loguru import logger
from .vehicle_service import VehicleService
from ..routing.hierarchical_clustering import HierarchicalSpatialClustering
from ..clustering.trip_assignment import TripAssignmentManager
from ..configurations.config import Config

class RouteGeneratorService:
    """Generate routes based on vehicle status and create clusters based on total vehicles."""
    
    def __init__(self):
        self.vehicle_service = VehicleService()
        self.hierarchical_clustering = HierarchicalSpatialClustering()
        self.trip_manager = TripAssignmentManager()
        
    def generate_routes_by_status(self, ward_no: str = None, 
                                status_filter: List[str] = None) -> Dict:
        """Generate routes: clusters based on TOTAL vehicles, routes assigned by ACTIVE vehicles."""
        
        # Get ALL vehicles in ward (all statuses)
        all_vehicles_df = self.vehicle_service.get_vehicles_by_ward(ward_no, include_all_status=True)
        
        if all_vehicles_df.empty:
            logger.warning(f"No vehicles found for ward {ward_no}")
            return self._create_empty_result()
        
        # Analyze vehicle status distribution
        status_analysis = self._analyze_vehicle_status(all_vehicles_df)
        
        # Create clusters based on TOTAL vehicle count (all statuses)
        total_vehicles = len(all_vehicles_df)
        clusters = self._create_vehicle_clusters(all_vehicles_df, total_vehicles)
        
        # Filter ACTIVE vehicles for route assignment
        active_vehicles_df = all_vehicles_df[
            all_vehicles_df['status'].str.upper().isin(['ACTIVE', 'AVAILABLE', 'ONLINE', 'READY', 'OPERATIONAL'])
        ]
        
        # Assign routes using active vehicles to cover all clusters
        routes_by_status = self._assign_routes_to_active_vehicles(
            all_vehicles_df, active_vehicles_df, clusters
        )
        
        return {
            'ward_no': ward_no,
            'total_vehicles': total_vehicles,
            'active_vehicles': len(active_vehicles_df),
            'status_analysis': status_analysis,
            'clusters': clusters,
            'routes': routes_by_status,
            'generation_timestamp': pd.Timestamp.now().isoformat()
        }
    
    def _get_vehicles_by_status(self, ward_no: str, status_filter: List[str]) -> pd.DataFrame:
        """Get vehicles filtered by status."""
        all_vehicles = self.vehicle_service.get_vehicles_by_ward(ward_no, include_all_status=True)
        
        if all_vehicles.empty:
            return all_vehicles
        
        # Check for status column variations
        status_cols = ['status', 'vehicleStatus', 'vehicle_status', 'state']
        status_col = None
        
        for col in status_cols:
            if col in all_vehicles.columns:
                status_col = col
                break
        
        if status_col is None:
            logger.warning("No status column found, returning all vehicles")
            return all_vehicles
        
        # Filter by status
        filtered_vehicles = all_vehicles[
            all_vehicles[status_col].isin(status_filter)
        ]
        
        logger.info(f"Filtered {len(filtered_vehicles)} vehicles by status: {status_filter}")
        return filtered_vehicles
    
    def _analyze_vehicle_status(self, vehicles_df: pd.DataFrame) -> Dict:
        """Analyze vehicle status distribution."""
        status_cols = ['status', 'vehicleStatus', 'vehicle_status', 'state']
        status_col = None
        
        for col in status_cols:
            if col in vehicles_df.columns:
                status_col = col
                break
        
        if status_col is None:
            return {'total_vehicles': len(vehicles_df), 'status_distribution': {}}
        
        status_counts = vehicles_df[status_col].value_counts().to_dict()
        
        return {
            'total_vehicles': len(vehicles_df),
            'status_distribution': status_counts,
            'active_vehicles': status_counts.get('ACTIVE', 0),
            'inactive_vehicles': status_counts.get('INACTIVE', 0),
            'maintenance_vehicles': status_counts.get('MAINTENANCE', 0)
        }
    
    def _create_vehicle_clusters(self, vehicles_df: pd.DataFrame, total_vehicles: int) -> Dict:
        """Create clusters based on total number of vehicles."""
        
        # Extract coordinates if available
        coordinates = self._extract_vehicle_coordinates(vehicles_df)
        
        if not coordinates:
            logger.warning("No coordinates found for vehicles, creating simple clusters")
            return self._create_simple_clusters(total_vehicles)
        
        # Determine optimal cluster count based on vehicle count
        optimal_clusters = self._calculate_optimal_clusters(total_vehicles)
        
        # Create hierarchical clusters
        ward_value = None
        if 'ward' in vehicles_df.columns and not vehicles_df.empty:
            try:
                ward_value = vehicles_df['ward'].iloc[0]
            except (IndexError, KeyError):
                ward_value = None
                
        hierarchical_clusters = self.hierarchical_clustering.create_fixed_clusters_by_ward(
            coordinates, optimal_clusters, ward_value
        )
        
        # Add vehicle assignments to clusters
        cluster_assignments = self._assign_vehicles_to_clusters(vehicles_df, hierarchical_clusters)
        
        return {
            'total_vehicles': total_vehicles,
            'cluster_count': optimal_clusters,
            'clusters': hierarchical_clusters,
            'vehicle_assignments': cluster_assignments
        }
    
    def _extract_vehicle_coordinates(self, vehicles_df: pd.DataFrame) -> List[Tuple]:
        """Extract coordinates from vehicle data."""
        coordinates = []
        
        # Check for coordinate columns
        lat_cols = ['latitude', 'lat', 'y', 'currentLatitude']
        lon_cols = ['longitude', 'lon', 'lng', 'x', 'currentLongitude']
        
        lat_col = lon_col = None
        
        for col in lat_cols:
            if col in vehicles_df.columns:
                lat_col = col
                break
        
        for col in lon_cols:
            if col in vehicles_df.columns:
                lon_col = col
                break
        
        if lat_col and lon_col:
            for _, vehicle in vehicles_df.iterrows():
                try:
                    lat = float(vehicle[lat_col])
                    lon = float(vehicle[lon_col])
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        coordinates.append((lon, lat))
                except (ValueError, TypeError):
                    continue
        
        logger.info(f"Extracted {len(coordinates)} valid coordinates from {len(vehicles_df)} vehicles")
        return coordinates
    
    def _calculate_optimal_clusters(self, total_vehicles: int) -> int:
        """Calculate optimal number of clusters based on vehicle count."""
        if total_vehicles <= 5:
            return total_vehicles
        elif total_vehicles <= 20:
            return max(3, total_vehicles // 3)
        elif total_vehicles <= 50:
            return max(5, total_vehicles // 5)
        else:
            return max(8, min(15, total_vehicles // 8))
    
    def _create_simple_clusters(self, total_vehicles: int) -> Dict:
        """Create simple clusters when coordinates are not available."""
        optimal_clusters = self._calculate_optimal_clusters(total_vehicles)
        vehicles_per_cluster = total_vehicles // optimal_clusters
        remaining_vehicles = total_vehicles % optimal_clusters
        
        clusters = {}
        vehicle_idx = 0
        
        for cluster_id in range(optimal_clusters):
            cluster_size = vehicles_per_cluster
            if cluster_id < remaining_vehicles:
                cluster_size += 1
            
            clusters[cluster_id] = {
                'vehicle_idx': cluster_id,
                'houses': list(range(vehicle_idx, vehicle_idx + cluster_size)),
                'coordinates': [],
                'fixed_assignment': True,
                'vehicle_count': cluster_size
            }
            
            vehicle_idx += cluster_size
        
        return clusters
    
    def _assign_vehicles_to_clusters(self, vehicles_df: pd.DataFrame, 
                                   hierarchical_clusters: Dict) -> Dict:
        """Assign specific vehicles to clusters."""
        assignments = {}
        vehicle_list = vehicles_df.to_dict('records')
        
        for cluster_id, cluster_data in hierarchical_clusters.items():
            cluster_vehicles = []
            house_indices = cluster_data.get('houses', [])
            
            for house_idx in house_indices:
                if house_idx < len(vehicle_list):
                    cluster_vehicles.append(vehicle_list[house_idx])
            
            assignments[cluster_id] = {
                'cluster_id': cluster_id,
                'vehicles': cluster_vehicles,
                'vehicle_count': len(cluster_vehicles),
                'coordinates': cluster_data.get('coordinates', [])
            }
        
        return assignments
    
    def _assign_routes_to_active_vehicles(self, all_vehicles_df: pd.DataFrame,
                                          active_vehicles_df: pd.DataFrame, 
                                          clusters: Dict) -> Dict:
        """Assign routes to ACTIVE vehicles to cover all clusters created from total vehicles."""
        routes = {}
        
        if len(active_vehicles_df) == 0:
            logger.warning("No active vehicles available for route assignment")
            return routes
        
        total_clusters = len(clusters.get('clusters', {}))
        active_count = len(active_vehicles_df)
        
        logger.info(f"Assigning {total_clusters} clusters to {active_count} active vehicles")
        
        # Assign clusters to active vehicles (active vehicles may handle multiple clusters)
        cluster_list = list(clusters.get('clusters', {}).items())
        
        for idx, (cluster_id, cluster_data) in enumerate(cluster_list):
            # Round-robin assignment to active vehicles
            vehicle_idx = idx % active_count
            vehicle = active_vehicles_df.iloc[vehicle_idx]
            
            route = {
                'cluster_id': cluster_id,
                'vehicle_id': vehicle['vehicle_id'],
                'vehicle_status': vehicle.get('status', 'ACTIVE'),
                'vehicle_type': vehicle.get('vehicle_type', 'truck'),
                'coordinates': cluster_data.get('coordinates', []),
                'house_count': len(cluster_data.get('houses', [])),
                'route_sequence': self._create_route_sequence(cluster_data.get('coordinates', [])),
                'estimated_distance': self._estimate_route_distance(cluster_data.get('coordinates', [])),
                'assigned_from_total': total_clusters,
                'active_vehicles_used': active_count
            }
            
            routes[cluster_id] = route
        
        logger.info(f"Generated {len(routes)} routes using {active_count} active vehicles for {total_clusters} clusters")
        return routes
    
    def _generate_routes_for_group(self, vehicles_df: pd.DataFrame, 
                                  clusters: Dict, group_name: str) -> Dict:
        """Generate routes for a specific vehicle group."""
        routes = {}
        
        # Assign vehicles to clusters
        vehicle_assignments = self._assign_vehicles_to_clusters(vehicles_df, clusters)
        
        for cluster_id, assignment in vehicle_assignments.items():
            cluster_vehicles = assignment['vehicles']
            cluster_coords = assignment['coordinates']
            
            if not cluster_vehicles:
                continue
            
            # Create route for this cluster
            route = {
                'cluster_id': cluster_id,
                'group_name': group_name,
                'vehicle_count': len(cluster_vehicles),
                'vehicles': cluster_vehicles,
                'coordinates': cluster_coords,
                'route_sequence': self._create_route_sequence(cluster_coords),
                'estimated_distance': self._estimate_route_distance(cluster_coords),
                'route_metadata': {
                    'start_point': cluster_coords[0] if cluster_coords else None,
                    'end_point': cluster_coords[-1] if cluster_coords else None,
                    'waypoints': len(cluster_coords)
                }
            }
            
            routes[cluster_id] = route
        
        logger.info(f"Generated {len(routes)} routes for group '{group_name}'")
        return routes
    
    def _create_route_sequence(self, coordinates: List[Tuple]) -> List[int]:
        """Create optimal route sequence for coordinates."""
        if not coordinates:
            return []
        
        # Simple nearest neighbor sequence
        sequence = [0]
        remaining = set(range(1, len(coordinates)))
        current = 0
        
        while remaining:
            nearest = min(remaining, key=lambda i: self._distance(
                coordinates[current], coordinates[i]
            ))
            sequence.append(nearest)
            remaining.remove(nearest)
            current = nearest
        
        return sequence
    
    def _estimate_route_distance(self, coordinates: List[Tuple]) -> float:
        """Estimate total route distance."""
        if len(coordinates) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(coordinates) - 1):
            total_distance += self._distance(coordinates[i], coordinates[i + 1])
        
        return total_distance
    
    def _distance(self, coord1: Tuple, coord2: Tuple) -> float:
        """Calculate Euclidean distance between two coordinates."""
        return ((coord1[0] - coord2[0]) ** 2 + (coord1[1] - coord2[1]) ** 2) ** 0.5
    
    def _create_empty_result(self) -> Dict:
        """Create empty result structure."""
        return {
            'ward_no': None,
            'total_vehicles': 0,
            'status_analysis': {'total_vehicles': 0, 'status_distribution': {}},
            'clusters': {},
            'routes': {},
            'generation_timestamp': pd.Timestamp.now().isoformat()
        }
