"""Main orchestrator for geospatial AI garbage collection route optimization."""
import os
import sys
import argparse
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modular components
from src.data_processing.load_road_network import RoadNetworkLoader
from src.data_processing.snap_buildings import BuildingSnapper
from src.clustering.assign_buildings import BuildingClusterer
from src.routing.compute_routes import RouteComputer
from src.routing.get_osrm_directions import OSRMDirectionsProvider
from src.visualization.export_to_geojson import RouteExporter
from src.visualization.folium_map import FoliumMapGenerator

class GeospatialRouteOptimizer:
    """Complete geospatial AI routing system for garbage collection."""
    
    def __init__(self, osrm_url: str = "http://router.project-osrm.org"):
        self.road_loader = RoadNetworkLoader()
        self.building_snapper = None
        self.clusterer = BuildingClusterer()
        self.route_computer = None
        self.directions_provider = OSRMDirectionsProvider(osrm_url)
        self.exporter = RouteExporter()
        self.map_generator = FoliumMapGenerator()
        
        logger.info("🎯 Geospatial AI Route Optimizer initialized")
        logger.info("🚗 OSRM: Real-world driving directions")
        logger.info("🗺️ NetworkX: Road network graph construction")
        logger.info("🔧 OR-Tools: VRP optimization")
        logger.info("📊 K-means/DBSCAN: Geographic clustering")
        logger.info("🌐 Live API: Real-time vehicle data integration")
    
    def process_ward_data(self, roads_geojson: str, buildings_geojson: str, 
                         vehicles_csv: str = None, output_dir: str = "output") -> dict:
        """Complete pipeline: load → cluster → route → directions → export."""
        
        logger.info(f"🚀 Starting route optimization pipeline")
        
        # Create output directory
        Path(output_dir).mkdir(exist_ok=True)
        
        # 1️⃣ Load and build road network graph
        logger.info("1️⃣ Loading road network...")
        road_gdf = self.road_loader.load_geojson(roads_geojson)
        road_graph = self.road_loader.build_networkx_graph()
        
        # 2️⃣ Load and snap buildings to road network
        logger.info("2️⃣ Loading and snapping buildings...")
        self.building_snapper = BuildingSnapper(road_graph)
        buildings_gdf = self.building_snapper.load_buildings(buildings_geojson)
        snapped_buildings = self.building_snapper.snap_to_road_network(buildings_gdf)
        
        # 3️⃣ Load vehicles from live API and determine clustering
        logger.info("3️⃣ Loading vehicles from live API and clustering buildings...")
        vehicles_df = self.clusterer.load_vehicles(vehicles_csv)  # vehicles_csv is now optional
        num_vehicles = len(vehicles_df)
        clustered_buildings = self.clusterer.cluster_buildings(snapped_buildings, num_vehicles)
        
        # 4️⃣ Compute optimal routes with trip assignments
        logger.info("4️⃣ Computing optimal routes with trip assignments...")
        self.route_computer = RouteComputer(road_graph)
        routes = self.route_computer.compute_cluster_routes(clustered_buildings)
        
        # Log trip statistics
        if hasattr(self.clusterer, 'trip_assignments'):
            trip_info = self.clusterer.trip_assignments
            logger.info(f"📊 Trip Summary: {trip_info['num_trips']} trips, {trip_info['total_houses']} houses")
            logger.info(f"🏠 Houses per trip capacity: {trip_info['houses_per_trip']}")
        
        # 5️⃣ Get OSRM directions with color coding
        logger.info("5️⃣ Getting turn-by-turn directions...")
        directions = {}
        for cluster_id, route_data in routes.items():
            if route_data['nodes']:
                # Convert nodes to (lon, lat) for OSRM
                coordinates = [(node[0], node[1]) for node in route_data['nodes']]
                directions[cluster_id] = self.directions_provider.get_route_directions(coordinates)
        
        # 6️⃣ Export results and generate maps alternately
        logger.info("6️⃣ Exporting results and generating maps...")
        
        # Export GeoJSON
        routes_gdf = self.exporter.prepare_routes_geojson(routes, vehicles_df, directions)
        routes_path = os.path.join(output_dir, "routes.geojson")
        self.exporter.export_routes_geojson(routes_path)
        
        # Generate route map
        route_map = self.map_generator.create_route_map(routes_gdf, clustered_buildings)
        map_path = os.path.join(output_dir, "route_map.html")
        self.map_generator.save_map(route_map, map_path)
        
        # Export summary CSV
        summary_df = self.exporter.prepare_summary_csv(routes, vehicles_df, directions)
        summary_path = os.path.join(output_dir, "summary.csv")
        self.exporter.export_summary_csv(summary_path)
        
        # Generate cluster analysis map
        cluster_map = self.map_generator.create_cluster_analysis_map(clustered_buildings)
        cluster_map_path = os.path.join(output_dir, "cluster_analysis.html")
        self.map_generator.save_map(cluster_map, cluster_map_path)
        
        results = {
            'routes_geojson': routes_path,
            'summary_csv': summary_path,
            'route_map': map_path,
            'cluster_map': cluster_map_path,
            'num_vehicles': num_vehicles,
            'num_buildings': len(clustered_buildings),
            'total_distance': sum(route['total_distance'] for route in routes.values()),
            'total_duration': sum(directions.get(cid, {}).get('total_duration', 0) for cid in routes.keys())
        }
        
        logger.success(f"✅ Pipeline completed! Results saved to {output_dir}")
        logger.info(f"📊 {num_vehicles} vehicles, {len(clustered_buildings)} buildings")
        logger.info(f"📏 Total distance: {results['total_distance']:.0f}m")
        logger.info(f"⏱️ Total duration: {results['total_duration']/60:.1f} min")
        
        # Add trip-specific results
        if hasattr(self.clusterer, 'trip_assignments'):
            trip_info = self.clusterer.trip_assignments
            results['num_trips'] = trip_info['num_trips']
            results['houses_per_trip_capacity'] = trip_info['houses_per_trip']
            logger.info(f"🚛 Trips assigned: {trip_info['num_trips']} (max 2 per day)")
        
        return results

def main():
    """Command line interface for the route optimizer."""
    parser = argparse.ArgumentParser(description="Geospatial AI Garbage Collection Route Optimizer")
    parser.add_argument("--roads", help="Path to roads GeoJSON file")
    parser.add_argument("--buildings", help="Path to buildings GeoJSON file")
    parser.add_argument("--vehicles", help="Path to vehicles CSV file (optional - will use live API if not provided)")
    parser.add_argument("--output", default="output", help="Output directory")
    parser.add_argument("--osrm-url", default="http://router.project-osrm.org", help="OSRM server URL")
    parser.add_argument("--api", action="store_true", help="Start FastAPI server instead")
    parser.add_argument("--port", type=int, default=8001, help="Port for FastAPI server (default: 8081)")
    
    args = parser.parse_args()
    
    if args.api:
        # Refresh token before starting API server
        from src.services.auth_service import AuthService
        auth_service = AuthService()
        logger.info("🔑 Refreshing authentication token...")
        if auth_service.refresh_token():
            logger.success("✅ Token refreshed successfully")
        else:
            logger.warning("⚠️ Token refresh failed, using existing token")
        
        # Start FastAPI server
        import uvicorn
        from src.api.geospatial_routes import app
        logger.info(f"🚀 Starting FastAPI server on port {args.port}...")
        try:
            uvicorn.run(app, host="127.0.0.1", port=args.port)
        except OSError as e:
            if "Address already in use" in str(e) or "10048" in str(e):
                logger.error(f"❌ Port {args.port} is already in use. Try a different port with --port <number>")
                logger.info("💡 Example: python main.py --api --port 8081")
            else:
                logger.error(f"❌ Server startup failed: {e}")
            sys.exit(1)
    else:
        # Validate required arguments for CLI mode
        if not all([args.roads, args.buildings]):
            parser.error("--roads and --buildings are required when not using --api")
        # Run optimization pipeline
        optimizer = GeospatialRouteOptimizer(args.osrm_url)
        
        try:
            results = optimizer.process_ward_data(
                roads_geojson=args.roads,
                buildings_geojson=args.buildings,
                vehicles_csv=args.vehicles,
                output_dir=args.output
            )
            
            print("\nRoute optimization completed successfully!")
            print(f"Results saved to: {args.output}")
            print(f"Interactive map: {results['route_map']}")
            print(f"Summary report: {results['summary_csv']}")
            
            # Open the map automatically
            import webbrowser
            import os
            map_path = os.path.abspath(results['route_map'])
            print(f"\n🌐 Opening map in browser: {map_path}")
            webbrowser.open(f"file://{map_path}")
            
            print(f"Vehicle data source: {'Live API' if not args.vehicles else 'CSV file'}")
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            sys.exit(1)

if __name__ == "__main__":
    main()