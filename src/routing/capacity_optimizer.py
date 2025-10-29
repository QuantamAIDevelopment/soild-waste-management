"""Capacity-based route optimizer for active vehicles with trip assignments."""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any
from sklearn.cluster import KMeans
import networkx as nx
from src.routing.hierarchical_clustering import HierarchicalSpatialClustering
import warnings

# Suppress specific geographic CRS warnings for intentional lat/lon usage
warnings.filterwarnings('ignore', message='.*Geometry is in a geographic CRS.*')

class CapacityRouteOptimizer:
    def __init__(self, max_houses_per_trip: int = 500):
        self.max_houses_per_trip = max_houses_per_trip
    
    def optimize_routes_with_capacity(self, buildings_gdf, vehicles_df, roads_gdf=None) -> Dict[str, Any]:
        """Optimize routes considering only active vehicles and capacity constraints."""
        
        # Filter only active vehicles
        active_vehicles = vehicles_df[
            vehicles_df['status'].str.upper().isin(['ACTIVE', 'AVAILABLE', 'ONLINE'])
        ].copy()
        
        if len(active_vehicles) == 0:
            raise ValueError("No active vehicles available for route optimization. Please ensure vehicles are active in the specified ward.")
        
        # Get building centroids with proper CRS handling
        buildings_projected = buildings_gdf.to_crs('EPSG:3857')  # Web Mercator for accurate distance
        building_centroids = [(pt.x, pt.y) for pt in buildings_projected.geometry.centroid]
        # Keep original for final output
        building_centroids_wgs84 = [(pt.x, pt.y) for pt in buildings_gdf.geometry.centroid]
        total_houses = len(building_centroids)
        
        # Calculate trips needed per vehicle based on capacity
        vehicle_assignments = self._assign_trips_to_vehicles(active_vehicles, total_houses)
        
        # Create clusters based on active vehicles and their trip capacity
        route_assignments = self._create_capacity_based_clusters(
            building_centroids, vehicle_assignments, roads_gdf
        )
        
        return {
            'active_vehicles': len(active_vehicles),
            'total_houses': total_houses,
            'vehicle_assignments': vehicle_assignments,
            'route_assignments': route_assignments
        }
    
    def _assign_trips_to_vehicles(self, active_vehicles: pd.DataFrame, total_houses: int) -> List[Dict]:
        """Distribute houses evenly among active vehicles, assign multiple trips if needed."""
        num_active_vehicles = len(active_vehicles)
        houses_per_vehicle = total_houses // num_active_vehicles
        remaining_houses = total_houses % num_active_vehicles
        
        vehicle_assignments = []
        
        for i, (_, vehicle) in enumerate(active_vehicles.iterrows()):
            # Distribute houses evenly, give extra house to first few vehicles
            vehicle_houses = houses_per_vehicle + (1 if i < remaining_houses else 0)
            
            # Calculate trips needed (500 houses per trip max)
            trips_needed = max(1, (vehicle_houses + self.max_houses_per_trip - 1) // self.max_houses_per_trip)
            
            vehicle_assignments.append({
                'vehicle_id': vehicle.get('vehicle_id'),
                'vehicle_type': vehicle.get('vehicle_type', 'garbage_truck'),
                'status': vehicle.get('status'),
                'capacity_per_trip': self.max_houses_per_trip,
                'trips_assigned': trips_needed,
                'houses_assigned': vehicle_houses,
                'total_capacity': trips_needed * self.max_houses_per_trip
            })
        
        return vehicle_assignments
    
    def _create_capacity_based_clusters(self, building_centroids: List[Tuple], 
                                      vehicle_assignments: List[Dict], 
                                      roads_gdf=None) -> Dict[str, Any]:
        """Create hierarchical non-overlapping clusters like states in a country."""
        
        # Create hierarchical clustering
        hierarchical_clusterer = HierarchicalSpatialClustering()
        
        # Get trips per vehicle
        trips_per_vehicle = [v['trips_assigned'] for v in vehicle_assignments]
        num_vehicles = len(vehicle_assignments)
        
        # Create non-overlapping hierarchical clusters
        hierarchical_clusters = hierarchical_clusterer.create_non_overlapping_clusters(
            building_centroids, num_vehicles, trips_per_vehicle
        )
        
        # Balance cluster sizes to respect capacity
        balanced_clusters = hierarchical_clusterer.balance_cluster_sizes(
            hierarchical_clusters, self.max_houses_per_trip
        )
        
        # Assign balanced clusters to vehicles and trips
        route_assignments = {}
        
        for vehicle_idx, vehicle in enumerate(vehicle_assignments):
            vehicle_id = vehicle['vehicle_id']
            vehicle_routes = []
            
            # Find clusters assigned to this vehicle
            vehicle_clusters = [
                cluster for cluster in balanced_clusters.values() 
                if cluster['vehicle_idx'] == vehicle_idx
            ]
            
            # Sort clusters by trip index
            vehicle_clusters.sort(key=lambda x: x.get('trip_idx', 0))
            
            for trip_idx, cluster in enumerate(vehicle_clusters):
                if cluster['houses']:
                    vehicle_routes.append({
                        'trip_id': f"{vehicle_id}_trip_{trip_idx + 1}",
                        'cluster_id': cluster['cluster_id'],
                        'houses': cluster['houses'],
                        'house_count': len(cluster['houses']),
                        'coordinates': cluster['coordinates'],
                        'route_points': self._optimize_trip_route(cluster['coordinates'], roads_gdf)
                    })
            
            route_assignments[vehicle_id] = {
                'vehicle_info': vehicle,
                'trips': vehicle_routes,
                'total_houses': sum(trip['house_count'] for trip in vehicle_routes)
            }
        
        return route_assignments
    
    def _optimize_trip_route(self, cluster_coords: List[Tuple], roads_gdf=None) -> List[Tuple]:
        """Optimize route for a single trip using nearest neighbor."""
        if not cluster_coords:
            return []
        
        if len(cluster_coords) == 1:
            return cluster_coords
        
        # Simple nearest neighbor optimization
        route = [cluster_coords[0]]
        remaining = cluster_coords[1:].copy()
        current = cluster_coords[0]
        
        while remaining:
            # Find nearest unvisited point
            distances = [
                ((current[0] - pt[0])**2 + (current[1] - pt[1])**2)**0.5 
                for pt in remaining
            ]
            nearest_idx = np.argmin(distances)
            next_point = remaining.pop(nearest_idx)
            route.append(next_point)
            current = next_point
        
        return route