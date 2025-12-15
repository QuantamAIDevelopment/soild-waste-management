#!/usr/bin/env python3
"""Test script for the cluster endpoint."""

import requests
import json

def test_cluster_endpoint():
    """Test the GET /cluster/{cluster_id} endpoint."""
    
    base_url = "http://127.0.0.1:8000"
    
    # Test cluster endpoint
    cluster_id = 0
    url = f"{base_url}/cluster/{cluster_id}"
    
    try:
        print(f"Testing GET {url}")
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Success! Cluster endpoint response:")
            print(f"Cluster ID: {data['cluster_id']}")
            print(f"Vehicle: {data['vehicle_info']['vehicle_id']} ({data['vehicle_info']['vehicle_type']})")
            print(f"Buildings: {data['buildings_count']}")
            print(f"Road segments: {data['total_road_segments']}")
            
            if data['roads']:
                print("\nFirst road segment:")
                road = data['roads'][0]
                print(f"  Start: ({road['start_coordinate']['longitude']}, {road['start_coordinate']['latitude']})")
                print(f"  End: ({road['end_coordinate']['longitude']}, {road['end_coordinate']['latitude']})")
                print(f"  Distance: {road['distance_meters']:.2f} meters")
            
            print(f"\nCluster bounds:")
            bounds = data['cluster_bounds']
            print(f"  Longitude: {bounds['min_longitude']:.6f} to {bounds['max_longitude']:.6f}")
            print(f"  Latitude: {bounds['min_latitude']:.6f} to {bounds['max_latitude']:.6f}")
            
        elif response.status_code == 404:
            print("❌ Cluster data not found. Run /optimize-routes first.")
            print("Response:", response.json())
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection failed. Make sure the server is running on port 8000")
        print("Start server with: python main.py --api --port 8000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_cluster_endpoint()