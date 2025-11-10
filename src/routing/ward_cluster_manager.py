"""Ward cluster manager for fixed N clusters with vehicle assignment rules."""
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class Config:
    max_trips_per_vehicle: int = 3

class WardClusterManager:
    def __init__(self, config: Config = None):
        self.config = config or Config()
    
    def create_ward_clusters(self, ward_no: str, total_vehicles: int, 
                           active_vehicles: List[str], pending_trips: List[str] = None) -> Dict:
        """Create exactly N fixed clusters and assign vehicles according to rules."""
        N = total_vehicles
        clusters = {}
        
        # Create N fixed clusters
        for i in range(N):
            clusters[i] = {
                'cluster_id': i,
                'ward_no': ward_no,
                'assigned_vehicle': None,
                'trip_name': None,
                'is_secondary_trip': False
            }
        
        # Step 1: Assign active vehicles to clusters (one-to-one)
        for i, vehicle in enumerate(active_vehicles[:N]):
            clusters[i]['assigned_vehicle'] = vehicle
        
        # Step 2: Fill remaining clusters with secondary trips from active vehicles
        remaining_clusters = N - len(active_vehicles)
        if remaining_clusters > 0 and active_vehicles:
            vehicle_trip_count = {v: 1 for v in active_vehicles[:N]}
            
            for i in range(len(active_vehicles), N):
                # Find vehicle with least trips that hasn't exceeded max_trips
                available_vehicles = [v for v, count in vehicle_trip_count.items() 
                                    if count < self.config.max_trips_per_vehicle]
                
                if available_vehicles:
                    # Select vehicle with minimum trips
                    selected_vehicle = min(available_vehicles, key=lambda v: vehicle_trip_count[v])
                    clusters[i]['assigned_vehicle'] = selected_vehicle
                    clusters[i]['is_secondary_trip'] = True
                    vehicle_trip_count[selected_vehicle] += 1
        
        # Step 3: Assign trip names
        if pending_trips:
            # Map pending trips to clusters in order
            for i, trip in enumerate(pending_trips[:N]):
                clusters[i]['trip_name'] = trip
        else:
            # Create placeholder trip names
            vehicle_trip_counter = {}
            for i in range(N):
                vehicle = clusters[i]['assigned_vehicle']
                if vehicle:
                    trip_num = vehicle_trip_counter.get(vehicle, 0) + 1
                    vehicle_trip_counter[vehicle] = trip_num
                    clusters[i]['trip_name'] = f"C{i}-T{trip_num}"
                else:
                    clusters[i]['trip_name'] = f"C{i}-T1"
        
        return clusters