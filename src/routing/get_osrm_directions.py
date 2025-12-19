"""Get OSRM directions with color-coded turn instructions."""
import requests
from loguru import logger
from shapely.geometry import LineString
import json
from typing import Dict, List, Tuple

class OSRMDirectionsProvider:
    def __init__(self, osrm_url: str = "http://router.project-osrm.org"):
        self.osrm_url = osrm_url.rstrip('/')
        
    def get_route_directions(self, coordinates: List[Tuple[float, float]]) -> Dict:
        """Get turn-by-turn directions from OSRM."""
        if len(coordinates) < 2:
            return self._create_empty_response()
        
        # Format coordinates for OSRM (lon,lat)
        coord_string = ";".join([f"{lon},{lat}" for lon, lat in coordinates])
        
        url = f"{self.osrm_url}/route/v1/driving/{coord_string}"
        params = {
            'steps': 'true',
            'geometries': 'geojson',
            'overview': 'full'
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if data['code'] == 'Ok':
                return self._process_osrm_response(data)
            else:
                logger.warning(f"OSRM returned error: {data.get('message', 'Unknown error')}")
                return self._create_fallback_response(coordinates)
                
        except requests.RequestException as e:
            logger.error(f"OSRM request failed: {e}")
            return self._create_fallback_response(coordinates)
    
    def _process_osrm_response(self, data: Dict) -> Dict:
        """Process OSRM response and add color coding."""
        if not data.get('routes') or len(data['routes']) == 0:
            logger.warning("No routes found in OSRM response")
            return self._create_empty_response()
            
        route = data['routes'][0]
        legs = route['legs']
        
        colored_steps = []
        total_distance = 0
        total_duration = 0
        
        for leg in legs:
            for step in leg['steps']:
                maneuver = step['maneuver']
                instruction = step['name'] or "Continue"
                
                # Color code based on maneuver type
                color = self._get_maneuver_color(maneuver['type'], maneuver.get('modifier', ''))
                
                colored_step = {
                    'instruction': instruction,
                    'maneuver_type': maneuver['type'],
                    'modifier': maneuver.get('modifier', ''),
                    'color': color,
                    'emoji': self._get_maneuver_emoji(color),
                    'distance': step['distance'],
                    'duration': step['duration'],
                    'geometry': step['geometry']
                }
                
                colored_steps.append(colored_step)
                total_distance += step['distance']
                total_duration += step['duration']
        
        return {
            'steps': colored_steps,
            'total_distance': total_distance,
            'total_duration': total_duration,
            'geometry': route['geometry'],
            'overview_geometry': LineString(route['geometry']['coordinates'])
        }
    
    def _get_maneuver_color(self, maneuver_type: str, modifier: str) -> str:
        """Get color based on maneuver type and modifier."""
        if maneuver_type in ['depart', 'arrive']:
            return 'green'
        elif maneuver_type == 'turn':
            if modifier in ['left', 'right']:
                return 'yellow'
            elif modifier in ['sharp left', 'sharp right']:
                return 'red'
            elif modifier in ['slight left', 'slight right']:
                return 'blue'
        elif maneuver_type == 'continue' or maneuver_type == 'straight':
            return 'green'
        elif maneuver_type in ['uturn', 'u-turn']:
            return 'red'
        elif maneuver_type in ['merge', 'on ramp', 'off ramp']:
            return 'blue'
        else:
            return 'green'  # Default
    
    def _get_maneuver_emoji(self, color: str) -> str:
        """Get emoji based on color."""
        emoji_map = {
            'green': '🟢',
            'blue': '🔵', 
            'yellow': '🟡',
            'red': '🔴'
        }
        return emoji_map.get(color, '🟢')
    
    def _create_empty_response(self) -> Dict:
        """Create empty response for invalid input."""
        return {
            'steps': [],
            'total_distance': 0,
            'total_duration': 0,
            'geometry': None,
            'overview_geometry': None
        }
    
    def _create_fallback_response(self, coordinates: List[Tuple[float, float]]) -> Dict:
        """Create fallback response when OSRM fails."""
        if len(coordinates) < 2:
            return self._create_empty_response()
        
        # Create simple straight-line response
        geometry = LineString([(lon, lat) for lon, lat in coordinates])
        distance = self._calculate_haversine_distance(coordinates[0], coordinates[-1])
        
        steps = [{
            'instruction': f"Head to destination",
            'maneuver_type': 'depart',
            'modifier': '',
            'color': 'green',
            'emoji': '🟢',
            'distance': distance,
            'duration': distance / 50 * 3600,  # Assume 50 km/h
            'geometry': {
                'type': 'LineString',
                'coordinates': coordinates
            }
        }]
        
        return {
            'steps': steps,
            'total_distance': distance,
            'total_duration': distance / 50 * 3600,
            'geometry': {
                'type': 'LineString', 
                'coordinates': coordinates
            },
            'overview_geometry': geometry
        }
    
    def _calculate_haversine_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """Calculate haversine distance between two coordinates."""
        import math
        
        lon1, lat1 = coord1
        lon2, lat2 = coord2
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        r = 6371000  # Earth radius in meters
        
        return c * r