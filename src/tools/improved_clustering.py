"""Improved clustering for garbage collection routes."""
import logging
import numpy as np
import geopandas as gpd
from sklearn.cluster import KMeans
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union
from scipy.spatial.distance import cdist
from typing import List, Dict, Tuple
from src.configurations.config import Config

logger = logging.getLogger(__name__)

class ImprovedClustering:
    def __init__(self, random_seed: int = None):
        self.random_seed = random_seed or Config.RANDOM_SEED
        
    def create_geographic_clusters(self, roads: gpd.GeoDataFrame, 
                                 snapped_houses: gpd.GeoDataFrame, 
                                 n_clusters: int) -> List[List]:
        """Create optimized K-means clusters for garbage collection routes."""
        
        house_counts = snapped_houses.groupby('road_id').size().to_dict()
        all_road_ids = list(roads['road_id'].unique())
        
        if len(all_road_ids) <= n_clusters:
            return [[road_id] for road_id in all_road_ids[:n_clusters]]
        
        # Step 1: K-means clustering on house locations
        kmeans_clusters = self._kmeans_clustering(snapped_houses, n_clusters)
        
        # Step 2: Assign roads to clusters based on house assignments
        road_clusters = self._assign_roads_to_clusters(kmeans_clusters, snapped_houses, roads)
        
        # Step 3: Balance clusters for optimal route distribution
        balanced_clusters = self._balance_clusters(road_clusters, roads, snapped_houses)
        
        # Log results
        for i, cluster in enumerate(balanced_clusters):
            cluster_houses = sum(house_counts.get(road_id, 0) for road_id in cluster)
            logger.info(f"K-means cluster {i}: {len(cluster)} roads, {cluster_houses} houses")
        
        return balanced_clusters
    
    def _kmeans_clustering(self, snapped_houses: gpd.GeoDataFrame, 
                          n_clusters: int) -> np.ndarray:
        """Perform K-means clustering on house locations."""
        
        # Extract coordinates
        house_coords = []
        for _, house in snapped_houses.iterrows():
            centroid = house.geometry.centroid
            house_coords.append([centroid.x, centroid.y])
        
        house_coords = np.array(house_coords)
        
        # Apply K-means with multiple initializations for better results
        kmeans = KMeans(
            n_clusters=n_clusters, 
            random_state=self.random_seed, 
            n_init=20,
            max_iter=300,
            algorithm='lloyd'
        )
        
        cluster_labels = kmeans.fit_predict(house_coords)
        
        logger.info(f"K-means clustering completed with {n_clusters} clusters")
        logger.info(f"Cluster sizes: {np.bincount(cluster_labels)}")
        
        return cluster_labels
    
    def _assign_roads_to_clusters(self, house_labels: np.ndarray, 
                                 snapped_houses: gpd.GeoDataFrame,
                                 roads: gpd.GeoDataFrame) -> List[List]:
        """Assign roads to clusters based on house cluster assignments."""
        
        n_clusters = len(np.unique(house_labels))
        road_cluster_votes = {road_id: [0] * n_clusters for road_id in roads['road_id']}
        
        # Vote for road assignments based on house clusters
        for idx, (df_idx, house) in enumerate(snapped_houses.iterrows()):
            road_id = house['road_id']
            cluster_label = house_labels[idx]
            if road_id in road_cluster_votes:
                road_cluster_votes[road_id][cluster_label] += 1
        
        # Assign roads to clusters with highest votes
        clusters = [[] for _ in range(n_clusters)]
        for road_id, votes in road_cluster_votes.items():
            if sum(votes) > 0:
                best_cluster = votes.index(max(votes))
                clusters[best_cluster].append(road_id)
        
        return clusters
    
    def _create_cluster_zones(self, clusters: List[List], roads: gpd.GeoDataFrame) -> List[Polygon]:
        """Create geographic zones (convex hulls) for each cluster."""
        
        cluster_zones = []
        
        for cluster_roads in clusters:
            if not cluster_roads:
                cluster_zones.append(None)
                continue
                
            cluster_roads_gdf = roads[roads['road_id'].isin(cluster_roads)]
            
            if cluster_roads_gdf.empty or cluster_roads_gdf.geometry.empty:
                cluster_zones.append(None)
                continue
            
            try:
                # Create convex hull of all roads in cluster
                cluster_geom = unary_union(cluster_roads_gdf.geometry)
                convex_hull = cluster_geom.convex_hull
                
                # Expand hull slightly to include nearby houses
                expanded_hull = convex_hull.buffer(50)  # 50m buffer
                
                cluster_zones.append(expanded_hull)
            except Exception as e:
                logger.warning(f"Failed to create cluster zone: {e}")
                cluster_zones.append(None)
        
        return cluster_zones
    
    def _enforce_geographic_constraints(self, clusters: List[List], 
                                      cluster_zones: List[Polygon],
                                      roads: gpd.GeoDataFrame,
                                      snapped_houses: gpd.GeoDataFrame) -> List[List]:
        """Reassign houses to ensure they're within their cluster's geographic zone."""
        
        # Create mapping of road to cluster
        road_to_cluster = {}
        for cluster_idx, cluster_roads in enumerate(clusters):
            for road_id in cluster_roads:
                road_to_cluster[road_id] = cluster_idx
        
        violations = []
        
        # Check each house
        for _, house in snapped_houses.iterrows():
            house_road_id = house['road_id']
            if house_road_id not in road_to_cluster:
                continue
                
            assigned_cluster = road_to_cluster[house_road_id]
            house_point = house.geometry.centroid
            
            # Check if house is within its assigned cluster's zone
            if cluster_zones[assigned_cluster] and not cluster_zones[assigned_cluster].contains(house_point):
                violations.append((house_road_id, assigned_cluster, house_point))
        
        # Reassign violating roads
        for road_id, current_cluster, house_point in violations:
            best_cluster = self._find_best_cluster_for_point(house_point, cluster_zones)
            
            if best_cluster != current_cluster and best_cluster is not None:
                # Move road to better cluster
                if road_id in clusters[current_cluster]:
                    clusters[current_cluster].remove(road_id)
                    clusters[best_cluster].append(road_id)
                    road_to_cluster[road_id] = best_cluster
        
        logger.info(f"Fixed {len(violations)} geographic constraint violations")
        return clusters
    
    def _find_best_cluster_for_point(self, point: Point, cluster_zones: List[Polygon]) -> int:
        """Find the best cluster zone for a given point."""
        
        min_distance = float('inf')
        best_cluster = None
        
        for cluster_idx, zone in enumerate(cluster_zones):
            if zone is None:
                continue
                
            if zone.contains(point):
                return cluster_idx
            
            distance = zone.distance(point)
            if distance < min_distance:
                min_distance = distance
                best_cluster = cluster_idx
        
        return best_cluster
    
    def _balance_clusters(self, clusters: List[List], 
                         roads: gpd.GeoDataFrame,
                         snapped_houses: gpd.GeoDataFrame) -> List[List]:
        """Balance cluster sizes while maintaining geographic constraints."""
        
        house_counts = snapped_houses.groupby('road_id').size().to_dict()
        
        # Calculate current cluster sizes
        cluster_sizes = []
        for cluster_roads in clusters:
            size = sum(house_counts.get(road_id, 0) for road_id in cluster_roads)
            cluster_sizes.append(size)
        
        if not cluster_sizes:
            return clusters
            
        target_size = sum(cluster_sizes) / len(clusters)
        
        # Find imbalanced clusters
        oversized = [(i, size) for i, size in enumerate(cluster_sizes) if size > target_size * 1.3]
        undersized = [(i, size) for i, size in enumerate(cluster_sizes) if size < target_size * 0.7]
        
        # Rebalance by moving roads from oversized to undersized clusters
        for oversized_idx, _ in oversized:
            for undersized_idx, _ in undersized:
                if cluster_sizes[oversized_idx] <= target_size * 1.1:
                    break
                    
                # Find roads on the boundary between clusters
                boundary_roads = self._find_boundary_roads(
                    clusters[oversized_idx], clusters[undersized_idx], roads
                )
                
                # Move smallest boundary roads
                for road_id in boundary_roads:
                    road_houses = house_counts.get(road_id, 0)
                    if cluster_sizes[oversized_idx] - road_houses >= target_size * 0.9:
                        clusters[oversized_idx].remove(road_id)
                        clusters[undersized_idx].append(road_id)
                        cluster_sizes[oversized_idx] -= road_houses
                        cluster_sizes[undersized_idx] += road_houses
                        break
        
        return clusters
    
    def _find_boundary_roads(self, cluster1_roads: List, cluster2_roads: List, 
                           roads: gpd.GeoDataFrame) -> List:
        """Find roads on the boundary between two clusters."""
        
        if not cluster1_roads or not cluster2_roads:
            return []
            
        cluster1_gdf = roads[roads['road_id'].isin(cluster1_roads)]
        cluster2_gdf = roads[roads['road_id'].isin(cluster2_roads)]
        
        cluster2_union = unary_union(cluster2_gdf.geometry)
        
        boundary_roads = []
        for _, road in cluster1_gdf.iterrows():
            if road.geometry.distance(cluster2_union) < 100:  # 100m threshold
                boundary_roads.append(road['road_id'])
        
        return boundary_roads[:3]  # Return max 3 boundary roads