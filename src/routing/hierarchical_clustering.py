"""Hierarchical clustering for non-overlapping spatial regions."""
import numpy as np
from typing import List, Tuple, Dict
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

class HierarchicalSpatialClustering:
    def __init__(self):
        self.clusters = {}
        self.fixed_assignments = {}  # Store fixed vehicle-cluster assignments
        
    def create_fixed_clusters_by_ward(self, coordinates: List[Tuple], num_vehicles: int, ward_no: str = None) -> Dict:
        """Create fixed clusters based on total vehicles in ward."""
        if num_vehicles == 0:
            return {}
            
        # Create fixed spatial clusters based on vehicle count
        vehicle_clusters = self._create_fixed_vehicle_clusters(coordinates, num_vehicles, ward_no)
        
        # Store fixed assignments for consistency
        if ward_no:
            self.fixed_assignments[ward_no] = vehicle_clusters
        
        hierarchical_clusters = {}
        for vehicle_idx, vehicle_houses in vehicle_clusters.items():
            hierarchical_clusters[vehicle_idx] = {
                'vehicle_idx': vehicle_idx,
                'ward_no': ward_no,
                'houses': vehicle_houses,
                'coordinates': [coordinates[i] for i in vehicle_houses],
                'fixed_assignment': True
            }
        
        return hierarchical_clusters
    
    def create_non_overlapping_clusters(self, coordinates: List[Tuple], num_vehicles: int, trips_per_vehicle: List[int]) -> Dict:
        """Create hierarchical clusters like states divided in a country."""
        
        # Step 1: Create vehicle-level clusters (like states)
        vehicle_clusters = self._create_vehicle_clusters(coordinates, num_vehicles)
        
        # Step 2: Subdivide each vehicle cluster into trip clusters (like districts in states)
        hierarchical_clusters = {}
        cluster_id = 0
        
        for vehicle_idx, (vehicle_id, trips_needed) in enumerate(zip(range(num_vehicles), trips_per_vehicle)):
            vehicle_houses = vehicle_clusters[vehicle_idx]
            vehicle_coords = [coordinates[i] for i in vehicle_houses]
            
            if trips_needed > 1 and len(vehicle_coords) > 1:
                # Subdivide vehicle cluster into trip clusters
                trip_clusters = self._subdivide_cluster(vehicle_coords, vehicle_houses, trips_needed)
            else:
                # Single trip for this vehicle
                trip_clusters = [vehicle_houses]
            
            # Assign trip clusters to hierarchical structure
            for trip_idx, trip_houses in enumerate(trip_clusters):
                hierarchical_clusters[cluster_id] = {
                    'vehicle_idx': vehicle_idx,
                    'trip_idx': trip_idx,
                    'houses': trip_houses,
                    'coordinates': [coordinates[i] for i in trip_houses]
                }
                cluster_id += 1
        
        return hierarchical_clusters
    
    def _create_fixed_vehicle_clusters(self, coordinates: List[Tuple], num_vehicles: int, ward_no: str = None) -> Dict:
        """Create fixed vehicle clusters for consistent assignment."""
        if num_vehicles == 1:
            return {0: list(range(len(coordinates)))}
        
        # Use deterministic clustering with fixed seed for consistency
        kmeans = KMeans(n_clusters=num_vehicles, random_state=hash(ward_no or 'default') % 2147483647, n_init=10)
        labels = kmeans.fit_predict(coordinates)
        
        # Group houses by vehicle cluster
        vehicle_clusters = {}
        for house_idx, cluster_label in enumerate(labels):
            if cluster_label not in vehicle_clusters:
                vehicle_clusters[cluster_label] = []
            vehicle_clusters[cluster_label].append(house_idx)
        
        return vehicle_clusters
    
    def _create_vehicle_clusters(self, coordinates: List[Tuple], num_vehicles: int) -> Dict:
        """Create main vehicle clusters using spatial boundaries."""
        if num_vehicles == 1:
            return {0: list(range(len(coordinates)))}
        
        # Use KMeans to create vehicle-level clusters
        kmeans = KMeans(n_clusters=num_vehicles, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coordinates)
        
        # Group houses by vehicle cluster
        vehicle_clusters = {}
        for house_idx, cluster_label in enumerate(labels):
            if cluster_label not in vehicle_clusters:
                vehicle_clusters[cluster_label] = []
            vehicle_clusters[cluster_label].append(house_idx)
        
        return vehicle_clusters
    
    def _subdivide_cluster(self, cluster_coords: List[Tuple], cluster_houses: List[int], num_subdivisions: int) -> List[List[int]]:
        """Subdivide a cluster into non-overlapping sub-clusters."""
        if num_subdivisions == 1 or len(cluster_coords) <= num_subdivisions:
            return [cluster_houses]
        
        # Use KMeans to subdivide the cluster
        kmeans = KMeans(n_clusters=num_subdivisions, random_state=42, n_init=10)
        sub_labels = kmeans.fit_predict(cluster_coords)
        
        # Group houses by sub-cluster
        sub_clusters = {}
        for local_idx, sub_label in enumerate(sub_labels):
            if sub_label not in sub_clusters:
                sub_clusters[sub_label] = []
            sub_clusters[sub_label].append(cluster_houses[local_idx])
        
        return list(sub_clusters.values())
    
    def balance_cluster_sizes(self, clusters: Dict, max_houses_per_cluster: int = 500) -> Dict:
        """Balance cluster sizes to respect capacity constraints."""
        balanced_clusters = {}
        cluster_id = 0
        
        for original_id, cluster_data in clusters.items():
            houses = cluster_data['houses']
            coordinates = cluster_data['coordinates']
            
            if len(houses) <= max_houses_per_cluster:
                # Cluster is within capacity
                balanced_clusters[cluster_id] = cluster_data.copy()
                balanced_clusters[cluster_id]['cluster_id'] = cluster_id
                cluster_id += 1
            else:
                # Split oversized cluster
                num_splits = (len(houses) + max_houses_per_cluster - 1) // max_houses_per_cluster
                split_clusters = self._subdivide_cluster(coordinates, houses, num_splits)
                
                for split_houses in split_clusters:
                    balanced_clusters[cluster_id] = {
                        'vehicle_idx': cluster_data['vehicle_idx'],
                        'trip_idx': cluster_data['trip_idx'],
                        'houses': split_houses,
                        'coordinates': [coordinates[houses.index(h)] for h in split_houses if h in houses],
                        'cluster_id': cluster_id
                    }
                    cluster_id += 1
        
        return balanced_clusters