"""Authentication service for local token management."""
import os
import jwt
import requests
from datetime import datetime, timedelta
from typing import Optional
from dotenv import load_dotenv, set_key
from loguru import logger

# Load environment variables
load_dotenv()

class AuthService:
    def __init__(self):
        self.base_url = os.getenv('SWM_API_BASE_URL')
        self.username = os.getenv('SWM_USERNAME')
        self.password = os.getenv('SWM_PASSWORD')
        self.env_file = '.env'
        self._current_token = os.getenv('SWM_TOKEN')
        self._token_expiry = self._get_token_expiry(self._current_token) if self._current_token else None
        logger.info("AuthService initialized for local use only")
    
    def _get_token_expiry(self, token: str) -> Optional[datetime]:
        """Extract expiry time from JWT token."""
        try:
            # Decode without verification to get expiry
            decoded = jwt.decode(token, options={"verify_signature": False})
            exp_timestamp = decoded.get('exp')
            if exp_timestamp:
                return datetime.fromtimestamp(exp_timestamp)
        except Exception as e:
            logger.warning(f"Could not decode token expiry: {e}")
        return None
    
    def _is_token_expired(self) -> bool:
        """Check if current token is expired or will expire soon."""
        if not self._current_token or not self._token_expiry:
            return True
        
        # Consider token expired if it expires within 5 minutes
        buffer_time = timedelta(minutes=5)
        return datetime.now() + buffer_time >= self._token_expiry
    
    def _login_and_get_token(self) -> Optional[str]:
        """Login and get fresh token from API."""
        try:
            url = f"{self.base_url}/auth/login"
            login_data = {
                'loginId': self.username,
                'password': self.password
            }
            
            headers = {
                'accept': 'application/json',
                'Content-Type': 'application/json'
            }
            
            logger.info("Requesting new authentication token...")
            response = requests.post(url, json=login_data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                # Look for token in response
                token_fields = ['token', 'access_token', 'accessToken', 'authToken', 'jwt', 'bearerToken']
                for field in token_fields:
                    if field in data:
                        return data[field]
                    elif isinstance(data, dict) and 'data' in data and isinstance(data['data'], dict) and field in data['data']:
                        return data['data'][field]
                
                logger.error("Token not found in login response")
                return None
            else:
                logger.error(f"Login failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Login error: {e}")
            return None
    
    def _update_env_token(self, new_token: str) -> bool:
        """Update token in memory (container-safe)."""
        try:
            # Update current instance
            self._current_token = new_token
            self._token_expiry = self._get_token_expiry(new_token)
            
            # Update environment variable for current session
            os.environ['SWM_TOKEN'] = new_token
            
            # Try to update .env file (works locally, fails silently in container)
            try:
                if os.path.exists(self.env_file) and os.access(self.env_file, os.W_OK):
                    set_key(self.env_file, 'SWM_TOKEN', new_token)
                    logger.success("Token updated in .env file")
                else:
                    logger.info("Token updated in memory (container mode)")
            except:
                logger.info("Token updated in memory (container mode)")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update token: {e}")
            return False
    
    def get_valid_token(self) -> Optional[str]:
        """Get a valid token, refreshing if necessary."""
        # Check if current token is still valid
        if not self._is_token_expired():
            logger.info("Using existing valid token")
            return self._current_token
        
        # Token is expired or missing, get a new one
        logger.info("Token expired or missing, getting new token...")
        new_token = self._login_and_get_token()
        
        if new_token:
            # Update .env file with new token
            if self._update_env_token(new_token):
                logger.success("Token automatically refreshed")
                return new_token
            else:
                logger.error("Failed to save new token")
                return None
        else:
            logger.error("Failed to get new token")
            return None
    
    def refresh_token(self) -> bool:
        """Force refresh the token."""
        logger.info("Force refreshing token...")
        new_token = self._login_and_get_token()
        
        if new_token:
            return self._update_env_token(new_token)
        return False
    
    def get_token_info(self) -> dict:
        """Get information about current token."""
        if not self._current_token:
            return {"status": "no_token", "message": "No token available"}
        
        if not self._token_expiry:
            return {"status": "unknown_expiry", "token_length": len(self._current_token)}
        
        now = datetime.now()
        if now >= self._token_expiry:
            return {
                "status": "expired",
                "expired_at": self._token_expiry.isoformat(),
                "expired_ago": str(now - self._token_expiry)
            }
        else:
            return {
                "status": "valid",
                "expires_at": self._token_expiry.isoformat(),
                "expires_in": str(self._token_expiry - now)
            }