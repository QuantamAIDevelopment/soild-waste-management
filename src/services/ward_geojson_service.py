"""Service for managing ward GeoJSON data with external API."""
import os
import requests
from typing import Optional, Dict, Any
from loguru import logger
from dotenv import load_dotenv
from ..configurations.config import Config

load_dotenv()

class WardGeoJSONService:
    def __init__(self):
        self.base_url = Config.SWM_API_BASE_URL
        self.token = Config.SWM_TOKEN
        
    def _get_headers(self) -> Dict[str, str]:
        """Get authorization headers."""
        token = self.token.strip("'")
        return {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
    
    def upload_ward_geojson(self, ward_no: str, geojson_data: Dict[str, Any], name: str = "Wardboundary") -> Dict[str, Any]:
        """
        Upload ward boundary GeoJSON to external API.
        
        Args:
            ward_no: Ward number (e.g., "Ward 29")
            geojson_data: GeoJSON data as dictionary
            name: Name parameter for the boundary (default: "Wardboundary")
            
        Returns:
            Response data from API
        """
        try:
            url = f"{self.base_url}/api/ward-geojson/create/{ward_no}"
            params = {'name': name}
            
            logger.info(f"Uploading ward GeoJSON for {ward_no}")
            
            response = requests.post(
                url,
                json=geojson_data,
                params=params,
                headers=self._get_headers(),
                timeout=30
            )
            
            if response.status_code in [200, 201]:
                logger.success(f"Successfully uploaded ward GeoJSON for {ward_no}")
                return {
                    "status": "success",
                    "ward_no": ward_no,
                    "message": "Ward boundary uploaded successfully",
                    "data": response.json() if response.text else None
                }
            elif response.status_code == 409:
                logger.warning(f"Ward GeoJSON for {ward_no} already exists")
                return {
                    "status": "exists",
                    "ward_no": ward_no,
                    "message": "Ward boundary already exists"
                }
            else:
                logger.error(f"Failed to upload ward GeoJSON: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "ward_no": ward_no,
                    "message": f"Upload failed with status {response.status_code}",
                    "error": response.text
                }
                
        except Exception as e:
            logger.error(f"Error uploading ward GeoJSON: {e}")
            return {
                "status": "error",
                "ward_no": ward_no,
                "message": str(e)
            }
    
    def get_ward_geojson(self, ward_no: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve ward boundary GeoJSON from external API.
        
        Args:
            ward_no: Ward number (e.g., "Ward 29" or "29")
            
        Returns:
            GeoJSON data or None if not found
        """
        # Try multiple ward number formats
        ward_formats = [
            ward_no,
            f"Ward {ward_no}",
            ward_no.replace("Ward ", ""),
            f"ward{ward_no}"
        ]
        
        for ward_format in ward_formats:
            try:
                url = f"{self.base_url}/api/ward-geojson/{ward_format}"
                logger.info(f"Trying: {url}")
                
                response = requests.get(
                    url,
                    headers=self._get_headers(),
                    timeout=30
                )
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        logger.success(f"Found ward data with format: {ward_format}")
                        logger.debug(f"Response type: {type(data).__name__}")
                        
                        if isinstance(data, dict):
                            logger.debug(f"Keys: {list(data.keys())}")
                            return data
                        elif isinstance(data, list):
                            logger.warning(f"API returned list, wrapping as FeatureCollection")
                            return {"type": "FeatureCollection", "features": data}
                        elif isinstance(data, str):
                            logger.warning(f"API returned string, parsing JSON")
                            parsed = json.loads(data)
                            return parsed if isinstance(parsed, dict) else {"type": "FeatureCollection", "features": [parsed]}
                        else:
                            logger.error(f"Unexpected type: {type(data)}")
                            continue
                    except Exception as parse_error:
                        logger.error(f"Parse error: {parse_error}")
                        logger.debug(f"Response: {response.text[:500]}")
                        continue
                elif response.status_code == 404:
                    logger.debug(f"Not found with format: {ward_format}")
                    continue
                else:
                    logger.debug(f"Status {response.status_code} for format: {ward_format}")
                    continue
                    
            except Exception as e:
                logger.debug(f"Error with format {ward_format}: {e}")
                continue
        
        logger.error(f"Ward GeoJSON not found for any format of: {ward_no}")
        return None
    
    def get_ward_data(self, ward_no: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve complete ward data (buildings, roads, boundaries) from external API.
        
        Args:
            ward_no: Ward number (e.g., "Ward 29" or "29")
            
        Returns:
            Dictionary with 'buildings', 'roads', 'ward_boundary' GeoJSON data or None
        """
        try:
            import json
            api_response = self.get_ward_geojson(ward_no)
            if not api_response:
                logger.warning(f"No data returned from API for {ward_no}")
                return None
            
            logger.info(f"Parsing ward data structure for {ward_no}")
            logger.info(f"API response keys: {list(api_response.keys()) if isinstance(api_response, dict) else 'not a dict'}")
            
            if not isinstance(api_response, dict):
                logger.error(f"API response is not a dictionary: {type(api_response)}")
                return None
            
            # Check if response is FeatureCollection with features array
            if api_response.get('type') == 'FeatureCollection' and 'features' in api_response:
                features = api_response['features']
                logger.info(f"Found FeatureCollection with {len(features)} features")
                
                buildings_data = None
                roads_data = None
                boundary_data = None
                
                # Extract each component by name field
                for feature in features:
                    name = feature.get('name', '').lower()
                    geojson_str = feature.get('geojson', '')
                    
                    if not geojson_str:
                        logger.warning(f"Feature {feature.get('id')} has no geojson field")
                        continue
                    
                    try:
                        geojson_data = json.loads(geojson_str) if isinstance(geojson_str, str) else geojson_str
                        
                        if 'building' in name:
                            buildings_data = geojson_data
                            logger.info(f"Found buildings: {len(geojson_data.get('features', []))} features")
                        elif 'road' in name:
                            roads_data = geojson_data
                            logger.info(f"Found roads: {len(geojson_data.get('features', []))} features")
                        elif 'ward' in name or 'boundary' in name:
                            boundary_data = geojson_data
                            logger.info(f"Found boundary: {len(geojson_data.get('features', []))} features")
                    except Exception as parse_err:
                        logger.error(f"Failed to parse geojson for {name}: {parse_err}")
                        continue
                
                if buildings_data and roads_data and boundary_data:
                    logger.success(f"Successfully extracted all ward components")
                    return {
                        'buildings': buildings_data,
                        'roads': roads_data,
                        'ward_boundary': boundary_data
                    }
                else:
                    logger.warning(f"Missing components - buildings: {buildings_data is not None}, roads: {roads_data is not None}, boundary: {boundary_data is not None}")
                    return None
            
            # Check if API returns separate fields for buildings, roads, boundary
            elif 'buildings' in api_response and 'roads' in api_response:
                result = {
                    'ward_boundary': api_response.get('wardBoundary') or api_response.get('boundary') or api_response.get('ward_boundary') or api_response,
                    'buildings': api_response.get('buildings'),
                    'roads': api_response.get('roads')
                }
                logger.success(f"Found separate fields in API response")
                return result
            else:
                logger.warning(f"Unrecognized response structure")
                return None
            
        except Exception as e:
            logger.error(f"Error retrieving ward data for {ward_no}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
