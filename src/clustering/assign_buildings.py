"""Cluster buildings based on number of available vehicles."""
import pandas as pd
import geopandas as gpd
from sklearn.cluster import KMeans, DBSCAN
from loguru import logger
import numpy as np
from src.services.vehicle_service import VehicleService
from src.clustering.trip_assignment import TripAssignmentManager
from src.routing.hierarchical_clustering import HierarchicalSpatialClustering

class BuildingClusterer:
    def __init__(self):
        self.clusters = None
        self.vehicle_service = VehicleService()
        self.trip_manager = TripAssignmentManager()
        self.hierarchical_clusterer = HierarchicalSpatialClustering()
        self.ward_vehicle_assignments = {}  # Store fixed assignments per ward
        
    def load_vehicles(self, csv_path: str = None) -> pd.DataFrame:
        """Load vehicle data from live API or fallback to CSV."""
        try:
            # Try to get live vehicle data first
            vehicles_df = self.vehicle_service.get_live_vehicles()
            
            if vehicles_df is not None and len(vehicles_df) > 0:
                logger.info(f"Loaded {len(vehicles_df)} vehicles from live API")
                return vehicles_df
            
            # Fallback to CSV if provided
            if csv_path:
                logger.warning("Live API failed, falling back to CSV")
                vehicles_df = pd.read_csv(csv_path)
                active_vehicles = vehicles_df[vehicles_df.get('status', 'active') == 'active']
                logger.info(f"Loaded {len(active_vehicles)} active vehicles from {csv_path}")
                return active_vehicles
            
            # Create fallback data if no CSV provided
            logger.warning("No CSV provided, using fallback vehicle data")
            return self.vehicle_service._create_fallback_vehicles()
            
        except Exception as e:
            logger.error(f"Failed to load vehicles: {e}")
            # Return fallback data instead of raising
            return self.vehicle_service._create_fallback_vehicles()
    
    def cluster_buildings_by_ward(self, buildings_gdf: gpd.GeoDataFrame, ward_no: str) -> gpd.GeoDataFrame:
        """Cluster buildings based on TOTAL vehicles in ward (all statuses)."""
        if len(buildings_gdf) == 0:
            logger.warning(f"No buildings to cluster in ward {ward_no}")
            return buildings_gdf
        
        # Get ALL vehicles for this ward (all statuses)
        ward_vehicles = self.vehicle_service.get_vehicles_by_ward(ward_no, include_all_status=True)
        total_vehicles = len(ward_vehicles)
        
        if total_vehicles == 0:
            logger.warning(f"No vehicles found in ward {ward_no}")
            buildings_gdf['cluster'] = 'no_vehicle'
            buildings_gdf['vehicle_id'] = None
            return buildings_gdf
        
        # Extract coordinates for clustering
        coordinates = [(geom.x, geom.y) for geom in buildings_gdf.geometry.centroid]
        
        # Create clusters based on TOTAL vehicle count (not just active)
        fixed_clusters = self.hierarchical_clusterer.create_fixed_clusters_by_ward(
            coordinates, total_vehicles, ward_no
        )
        
        # Assign buildings to clusters
        buildings_gdf = buildings_gdf.copy()
        buildings_gdf['cluster'] = 'unassigned'
        buildings_gdf['vehicle_id'] = None
        buildings_gdf['ward_no'] = ward_no
        
        for cluster_id, cluster_data in fixed_clusters.items():
            house_indices = cluster_data['houses']
            vehicle_id = ward_vehicles.iloc[cluster_id % total_vehicles]['vehicle_id']
            
            for house_idx in house_indices:
                if house_idx < len(buildings_gdf):
                    buildings_gdf.iloc[house_idx, buildings_gdf.columns.get_loc('cluster')] = f"ward_{ward_no}_vehicle_{cluster_id}"
                    buildings_gdf.iloc[house_idx, buildings_gdf.columns.get_loc('vehicle_id')] = vehicle_id
        
        # Store ward assignments
        self.ward_vehicle_assignments[ward_no] = {
            'total_vehicles': total_vehicles,
            'clusters': fixed_clusters,
            'all_vehicles': ward_vehicles
        }
        
        logger.info(f"Ward {ward_no}: Created {total_vehicles} clusters based on total vehicles (all statuses)")
        return buildings_gdf
    
    def cluster_buildings_with_vehicles(self, buildings_gdf: gpd.GeoDataFrame, vehicles_df: pd.DataFrame) -> gpd.GeoDataFrame:
        """Cluster buildings using provided vehicles with geographic clustering (no API calls)."""
        if len(buildings_gdf) == 0:
            logger.warning("No buildings to cluster")
            return buildings_gdf
        
        if len(vehicles_df) == 0:
            logger.error("No vehicles provided for clustering")
            return buildings_gdf
        
        buildings_gdf = buildings_gdf.copy()
        num_vehicles = len(vehicles_df)
        
        # Extract coordinates for clustering
        coordinates = np.array([(geom.x, geom.y) for geom in buildings_gdf.geometry.centroid])
        
        # Use KMeans for geographic clustering
        kmeans = KMeans(n_clusters=num_vehicles, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(coordinates)
        
        # Assign clusters to buildings
        buildings_gdf['cluster'] = 'unassigned'
        buildings_gdf['vehicle_id'] = None
        
        for i, (_, vehicle) in enumerate(vehicles_df.iterrows()):
            vehicle_id = vehicle['vehicle_id']
            
            # Find buildings assigned to this cluster
            cluster_mask = cluster_labels == i
            cluster_indices = np.where(cluster_mask)[0]
            
            # Assign buildings to this vehicle
            for idx in cluster_indices:
                buildings_gdf.iloc[idx, buildings_gdf.columns.get_loc('cluster')] = f"vehicle_{i}"
                buildings_gdf.iloc[idx, buildings_gdf.columns.get_loc('vehicle_id')] = vehicle_id
        
        # Verify no overlaps
        cluster_counts = buildings_gdf['cluster'].value_counts()
        total_assigned = cluster_counts.sum()
        
        logger.info(f"Geographic clustering: {len(buildings_gdf)} buildings to {num_vehicles} vehicles (no overlaps)")
        logger.info(f"Cluster distribution: {dict(cluster_counts)}")
        
        return buildings_gdf
    
    def cluster_buildings(self, buildings_gdf: gpd.GeoDataFrame, num_vehicles: int, method='kmeans') -> gpd.GeoDataFrame:
        """Cluster buildings based on vehicle capacity and trip constraints."""
        if len(buildings_gdf) == 0:
            logger.warning("No buildings to cluster")
            return buildings_gdf
        
        # REMOVED: Ward-specific clustering that makes API calls
        # This was causing API calls even when CSV was uploaded
        
        # Fallback to original trip-based clustering
        trip_assignments = self.trip_manager.assign_trips(buildings_gdf, num_vehicles)
        
        # Validate no overlap
        if not self.trip_manager.validate_no_overlap(trip_assignments):
            logger.error("Trip assignment validation failed")
        
        # Flatten trip assignments into a single GeoDataFrame
        all_buildings = []
        for trip_num, trip_data in trip_assignments['assignments'].items():
            for vehicle_id, vehicle_data in trip_data.items():
                all_buildings.append(vehicle_data['buildings'])
        
        if all_buildings:
            result_gdf = pd.concat(all_buildings, ignore_index=True)
            logger.info(f"Assigned {len(result_gdf)} buildings across {trip_assignments['num_trips']} trips")
            
            # Store trip assignments for later use
            self.trip_assignments = trip_assignments
            return result_gdf
        else:
            logger.warning("No buildings assigned")
            return buildings_gdf
    
    def _calculate_optimal_eps(self, coords: np.ndarray) -> float:
        """Calculate optimal eps for DBSCAN based on data spread."""
        from sklearn.neighbors import NearestNeighbors
        
        neighbors = NearestNeighbors(n_neighbors=4)
        neighbors_fit = neighbors.fit(coords)
        distances, indices = neighbors_fit.kneighbors(coords)
        distances = np.sort(distances[:, 3], axis=0)
        
        # Use knee point detection or simple heuristic
        return np.percentile(distances, 75)
    
    def _reassign_noise_points(self, labels: np.ndarray, coords: np.ndarray, target_clusters: int) -> np.ndarray:
        """Reassign noise points (-1) to nearest valid clusters."""
        noise_mask = labels == -1
        if not np.any(noise_mask):
            return labels
        
        # If we have fewer clusters than vehicles, use KMeans fallback
        unique_labels = set(labels[~noise_mask])
        if len(unique_labels) < target_clusters:
            kmeans = KMeans(n_clusters=target_clusters, random_state=42, n_init=10)
            return kmeans.fit_predict(coords)
        
        # Assign noise points to nearest cluster centroid
        for i, label in enumerate(labels):
            if label == -1:
                min_dist = float('inf')
                best_cluster = 0
                point = coords[i]
                
                for cluster_id in unique_labels:
                    cluster_points = coords[labels == cluster_id]
                    centroid = np.mean(cluster_points, axis=0)
                    dist = np.linalg.norm(point - centroid)
                    
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster = cluster_id
                
                labels[i] = best_cluster
        
        return labels
    
    def get_cluster_summary(self, buildings_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Generate cluster summary statistics including trip information."""
        if hasattr(self, 'trip_assignments'):
            # Use trip-based summary
            return self.trip_manager.get_trip_summary(self.trip_assignments)
        elif self.ward_vehicle_assignments:
            # Use ward-based fixed assignment summary
            return self._get_ward_assignment_summary(buildings_gdf)
        else:
            # Fallback to original cluster summary
            summary = buildings_gdf.groupby('cluster').agg({
                'geometry': 'count',
                'snap_distance': ['mean', 'max'] if 'snap_distance' in buildings_gdf.columns else ['count', 'count']
            }).round(4)
            
            if 'snap_distance' in buildings_gdf.columns:
                summary.columns = ['building_count', 'avg_snap_distance', 'max_snap_distance']
            else:
                summary.columns = ['building_count', 'total_buildings', 'cluster_size']
            summary = summary.reset_index()
            
            logger.info(f"Generated summary for {len(summary)} clusters")
            return summary
    
    def get_ward_vehicle_count(self, ward_no: str) -> int:
        """Get the TOTAL number of vehicles in ward (all statuses)."""
        ward_vehicles = self.vehicle_service.get_vehicles_by_ward(ward_no, include_all_status=True)
        return len(ward_vehicles)
    
    def get_fixed_assignment_for_ward(self, ward_no: str) -> dict:
        """Get the fixed cluster assignment for a specific ward."""
        return self.ward_vehicle_assignments.get(ward_no, {})
    
    def _get_ward_assignment_summary(self, buildings_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Generate summary for ward-based fixed assignments."""
        summary_data = []
        
        for ward_no, assignment_data in self.ward_vehicle_assignments.items():
            num_vehicles = assignment_data['num_vehicles']
            vehicles = assignment_data['vehicles']
            
            # Count buildings per vehicle in this ward
            ward_buildings = buildings_gdf[buildings_gdf.get('ward_no', '') == ward_no]
            
            for i, vehicle in vehicles.iterrows():
                vehicle_id = vehicle['vehicle_id']
                vehicle_buildings = ward_buildings[ward_buildings.get('vehicle_id', '') == vehicle_id]
                
                summary_data.append({
                    'ward_no': ward_no,
                    'vehicle_id': vehicle_id,
                    'building_count': len(vehicle_buildings),
                    'cluster': f"ward_{ward_no}_vehicle_{i}",
                    'fixed_assignment': True
                })
        
        summary_df = pd.DataFrame(summary_data)
        logger.info(f"Generated ward-based summary for {len(summary_data)} vehicle assignments")
        return summary_df