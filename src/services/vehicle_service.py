"""Vehicle service for fetching live vehicle data from SWM API."""
import os
import requests
import pandas as pd
from typing import List, Dict, Optional
from loguru import logger
from dotenv import load_dotenv
from .auth_service import AuthService

# Load environment variables
load_dotenv()

class VehicleService:
    def __init__(self):
        self.base_url = os.getenv('SWM_API_BASE_URL', 'https://uat-swm-main-service-hdaqcdcscbfedhhn.centralindia-01.azurewebsites.net')
        self.api_key = os.getenv('SWM_API_KEY', '')
        self.session = requests.Session()
        self.auth_service = AuthService()
        
        logger.info(f"VehicleService initialized with base URL: {self.base_url}")
        logger.info("Using automatic token management")
    
    def get_live_vehicles(self) -> pd.DataFrame:
        """Fetch live vehicle data from SWM API."""
        try:
            # Get valid token (automatically refreshes if needed)
            token = self.auth_service.get_valid_token()
            if not token:
                logger.error("Could not get valid authentication token")
                return self._create_fallback_vehicles()
            
            # Use the correct vehicle endpoint with pagination
            from datetime import datetime
            today = datetime.now().strftime('%Y-%m-%d')
            
            endpoint = f'/api/vehicles/paginated?date={today}&size=542&sortBy=vehicleNo'
            url = f"{self.base_url}{endpoint}"
            
            logger.info(f"Fetching vehicles from: {url}")
            
            headers = {
                'accept': '*/*',
                'Authorization': f'Bearer {token}'
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                vehicles_data = response.json()
                logger.success(f"Successfully fetched vehicle data")
                return self._process_vehicle_data(vehicles_data)
            elif response.status_code == 401:
                # Token might be invalid, try to refresh
                logger.warning("Token appears invalid, attempting refresh...")
                if self.auth_service.refresh_token():
                    # Retry with new token
                    new_token = self.auth_service.get_valid_token()
                    if new_token:
                        headers['Authorization'] = f'Bearer {new_token}'
                        response = requests.get(url, headers=headers, timeout=30)
                        if response.status_code == 200:
                            vehicles_data = response.json()
                            logger.success(f"Successfully fetched vehicle data after token refresh")
                            return self._process_vehicle_data(vehicles_data)
                
                logger.error(f"Authentication failed even after token refresh")
                return self._create_fallback_vehicles()
            else:
                logger.error(f"API returned status {response.status_code}: {response.text[:200]}")
                return self._create_fallback_vehicles()
            
        except Exception as e:
            logger.error(f"Error fetching live vehicle data: {e}")
            return self._create_fallback_vehicles()
    
    def get_vehicles_by_ward(self, ward_no: str, include_all_status: bool = True) -> pd.DataFrame:
        """Get ALL vehicles filtered by ward number (default: all statuses).
        
        Args:
            ward_no: Ward number to filter
            include_all_status: If True, returns all vehicles regardless of status (default: True)
        """
        try:
            # Always get all vehicles by default
            all_vehicles = self.get_live_vehicles_all_status()
            
            # Filter by ward - check multiple possible ward field names
            ward_fields = ['ward', 'wardNo', 'ward_no', 'wardNumber', 'zone', 'area']
            
            filtered_vehicles = None
            for field in ward_fields:
                if field in all_vehicles.columns:
                    filtered_vehicles = all_vehicles[all_vehicles[field].astype(str) == str(ward_no)]
                    if len(filtered_vehicles) > 0:
                        logger.info(f"Found {len(filtered_vehicles)} vehicles (all statuses) in ward {ward_no} using field '{field}'")
                        break
            
            # If no ward field found or no matches, return empty DataFrame
            if filtered_vehicles is None or len(filtered_vehicles) == 0:
                logger.warning(f"No vehicles found in ward {ward_no}")
                return pd.DataFrame(columns=all_vehicles.columns)
            
            return filtered_vehicles
            
        except Exception as e:
            logger.error(f"Error filtering vehicles by ward {ward_no}: {e}")
            return self._create_fallback_vehicles(ward_no)
    
    def get_live_vehicles_all_status(self) -> pd.DataFrame:
        """Fetch ALL vehicles from API regardless of status."""
        try:
            token = self.auth_service.get_valid_token()
            if not token:
                logger.error("Could not get valid authentication token")
                return self._create_fallback_vehicles()
            
            from datetime import datetime
            today = datetime.now().strftime('%Y-%m-%d')
            endpoint = f'/api/vehicles/paginated?date={today}&size=542&sortBy=vehicleNo'
            url = f"{self.base_url}{endpoint}"
            
            headers = {'accept': '*/*', 'Authorization': f'Bearer {token}'}
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                vehicles_data = response.json()
                # Process without status filtering
                if isinstance(vehicles_data, list):
                    df = pd.DataFrame(vehicles_data)
                elif isinstance(vehicles_data, dict):
                    if 'content' in vehicles_data:
                        df = pd.DataFrame(vehicles_data['content'])
                    elif 'data' in vehicles_data:
                        df = pd.DataFrame(vehicles_data['data'])
                    else:
                        df = pd.DataFrame([vehicles_data])
                else:
                    return self._create_fallback_vehicles()
                
                df = self._standardize_vehicle_data(df)
                logger.success(f"Loaded {len(df)} vehicles (all statuses) from live API")
                return df
            else:
                logger.error(f"API returned status {response.status_code}")
                return self._create_fallback_vehicles()
                
        except Exception as e:
            logger.error(f"Error fetching vehicles: {e}")
            return self._create_fallback_vehicles()
    
    def _get_auth_methods(self):
        """Get all possible authentication methods to try."""
        methods = [{}]  # No auth first
        
        # Bearer token from env
        if self.token:
            methods.append({'headers': {'Authorization': f'Bearer {self.token}'}})
        
        # API key variations
        if self.api_key:
            methods.extend([
                {'headers': {'Authorization': f'Bearer {self.api_key}'}},
                {'headers': {'X-API-Key': self.api_key}},
                {'headers': {'api-key': self.api_key}},
                {'params': {'api_key': self.api_key}}
            ])
        
        # Basic auth
        if self.username and self.password:
            methods.append({'auth': (self.username, self.password)})
        
        # Use auth token if available
        if self.auth_token:
            methods.append({'headers': {'Authorization': f'Bearer {self.auth_token}'}})
        
        return methods
    

    
    def _process_vehicle_data(self, vehicles_data) -> pd.DataFrame:
        """Process and standardize vehicle data from API response."""
        # Convert to DataFrame - handle paginated response
        if isinstance(vehicles_data, list):
            df = pd.DataFrame(vehicles_data)
        elif isinstance(vehicles_data, dict):
            if 'content' in vehicles_data:  # Paginated response
                df = pd.DataFrame(vehicles_data['content'])
            elif 'data' in vehicles_data:
                df = pd.DataFrame(vehicles_data['data'])
            elif 'vehicles' in vehicles_data:
                df = pd.DataFrame(vehicles_data['vehicles'])
            else:
                df = pd.DataFrame([vehicles_data])
        else:
            return self._create_fallback_vehicles()
        
        df = self._standardize_vehicle_data(df)
        
        # Filter active vehicles - handle different status formats
        if 'status' in df.columns:
            active_vehicles = df[df['status'].str.upper().isin(['ACTIVE', 'AVAILABLE', 'ONLINE', 'OPERATIONAL'])].copy()
        else:
            # If no status column, assume all are active
            df['status'] = 'active'
            active_vehicles = df.copy()
        
        if len(active_vehicles) == 0:
            active_vehicles = df.copy()
        
        logger.success(f"Loaded {len(active_vehicles)} active vehicles from live API")
        return active_vehicles
    
    def _standardize_vehicle_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize vehicle data column names and format."""
        # Map vehicleType from API to vehicle_type
        if 'vehicleType' in df.columns and 'vehicle_type' not in df.columns:
            df['vehicle_type'] = df['vehicleType']
        
        # Add legacy mappings for compatibility
        column_mappings = {
            'id': 'vehicle_id',
            'vehicle_number': 'vehicle_id',
            'registration_number': 'vehicle_id',
            'wardNumber': 'ward_no',
            'wardNo': 'ward_no',
            'name': 'vehicle_name',
            'vehicleName': 'vehicle_name',
            'type': 'vehicle_type',
            'capacity': 'capacity',
            'vehicleCapacity': 'capacity',
            'location': 'location',
            'currentLocation': 'location',
            'latitude': 'lat',
            'longitude': 'lon',
            'lng': 'lon'
        }
        
        # Rename columns
        for old_name, new_name in column_mappings.items():
            if old_name in df.columns and new_name not in df.columns:
                df = df.rename(columns={old_name: new_name})
        
        # Ensure vehicle_id exists
        if 'vehicle_id' not in df.columns:
            if 'vehicleNo' in df.columns:
                df['vehicle_id'] = df['vehicleNo']
            elif 'vehicleId' in df.columns:
                df['vehicle_id'] = df['vehicleId']
            else:
                df['vehicle_id'] = [f"vehicle_{i+1}" for i in range(len(df))]
        
        # Ensure status exists
        if 'status' not in df.columns:
            df['status'] = 'active'
        
        # Add default values only if missing
        if 'vehicle_type' not in df.columns:
            df['vehicle_type'] = 'garbage_truck'
        if 'capacity' not in df.columns:
            df['capacity'] = 500
        if 'ward_no' not in df.columns and 'wardNo' not in df.columns:
            df['ward_no'] = '1'
        
        return df
    
    def _create_fallback_vehicles(self, ward_no: str = None) -> pd.DataFrame:
        """Create fallback vehicle data when API is unavailable."""
        logger.warning("Creating fallback vehicle data")
        
        # If specific ward requested and it's not ward 1, return empty DataFrame
        if ward_no and str(ward_no) != '1':
            logger.warning(f"No fallback vehicles available for ward {ward_no}")
            return pd.DataFrame()
        
        fallback_data = [
            {'vehicle_id': 'SWM001', 'vehicleId': 'SWM001', 'vehicleNo': 'SWM001', 'driverName': 'Driver1', 'imeiN': '123456789', 'phoneNo': '9876543210', 'wardNo': '1', 'vehicleType': 'garbage_truck', 'department': 'SWM', 'timestamp': '2024-01-01T00:00:00Z', 'status': 'active', 'capacity': 500},
            {'vehicle_id': 'SWM002', 'vehicleId': 'SWM002', 'vehicleNo': 'SWM002', 'driverName': 'Driver2', 'imeiN': '123456790', 'phoneNo': '9876543211', 'wardNo': '1', 'vehicleType': 'garbage_truck', 'department': 'SWM', 'timestamp': '2024-01-01T00:00:00Z', 'status': 'active', 'capacity': 500},
            {'vehicle_id': 'SWM003', 'vehicleId': 'SWM003', 'vehicleNo': 'SWM003', 'driverName': 'Driver3', 'imeiN': '123456791', 'phoneNo': '9876543212', 'wardNo': '1', 'vehicleType': 'garbage_truck', 'department': 'SWM', 'timestamp': '2024-01-01T00:00:00Z', 'status': 'active', 'capacity': 500},
            {'vehicle_id': 'SWM004', 'vehicleId': 'SWM004', 'vehicleNo': 'SWM004', 'driverName': 'Driver4', 'imeiN': '123456792', 'phoneNo': '9876543213', 'wardNo': '1', 'vehicleType': 'garbage_truck', 'department': 'SWM', 'timestamp': '2024-01-01T00:00:00Z', 'status': 'active', 'capacity': 500},
            {'vehicle_id': 'SWM005', 'vehicleId': 'SWM005', 'vehicleNo': 'SWM005', 'driverName': 'Driver5', 'imeiN': '123456793', 'phoneNo': '9876543214', 'wardNo': '1', 'vehicleType': 'garbage_truck', 'department': 'SWM', 'timestamp': '2024-01-01T00:00:00Z', 'status': 'active', 'capacity': 500}
        ]
        
        return pd.DataFrame(fallback_data)
    
    def get_vehicle_by_id(self, vehicle_id: str) -> Optional[Dict]:
        """Get specific vehicle data by ID."""
        try:
            url = f"{self.base_url}/api/vehicles/{vehicle_id}"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Vehicle {vehicle_id} not found: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching vehicle {vehicle_id}: {e}")
            return None
    
    def update_vehicle_status(self, vehicle_id: str, status: str) -> bool:
        """Update vehicle status via API."""
        try:
            url = f"{self.base_url}/api/vehicles/{vehicle_id}/status"
            data = {'status': status}
            
            response = self.session.put(url, json=data, timeout=30)
            
            if response.status_code in [200, 204]:
                logger.info(f"Updated vehicle {vehicle_id} status to {status}")
                return True
            else:
                logger.error(f"Failed to update vehicle status: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating vehicle status: {e}")
            return False