#!/usr/bin/env python3
"""Test script for automatic token management."""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from services.auth_service import AuthService
from services.vehicle_service import VehicleService

def test_auth_service():
    """Test the authentication service."""
    print("🔐 Testing Authentication Service")
    print("=" * 50)
    
    auth_service = AuthService()
    
    # Check current token info
    print("1. Current token info:")
    token_info = auth_service.get_token_info()
    print(f"   Status: {token_info.get('status')}")
    if 'expires_at' in token_info:
        print(f"   Expires at: {token_info['expires_at']}")
    if 'expires_in' in token_info:
        print(f"   Expires in: {token_info['expires_in']}")
    
    print("\n2. Getting valid token (auto-refresh if needed):")
    token = auth_service.get_valid_token()
    if token:
        print(f"   ✅ Got valid token (length: {len(token)})")
        print(f"   Token preview: {token[:50]}...")
    else:
        print("   ❌ Failed to get valid token")
        return False
    
    print("\n3. Updated token info:")
    token_info = auth_service.get_token_info()
    print(f"   Status: {token_info.get('status')}")
    if 'expires_at' in token_info:
        print(f"   Expires at: {token_info['expires_at']}")
    if 'expires_in' in token_info:
        print(f"   Expires in: {token_info['expires_in']}")
    
    return True

def test_vehicle_service():
    """Test the vehicle service with automatic token management."""
    print("\n🚛 Testing Vehicle Service with Auto-Auth")
    print("=" * 50)
    
    vehicle_service = VehicleService()
    
    print("1. Fetching live vehicles (will auto-refresh token if needed):")
    vehicles_df = vehicle_service.get_live_vehicles()
    
    if vehicles_df is not None and len(vehicles_df) > 0:
        print(f"   ✅ Successfully fetched {len(vehicles_df)} vehicles")
        print(f"   Active vehicles: {len(vehicles_df[vehicles_df['status'].str.upper().isin(['ACTIVE', 'AVAILABLE', 'ONLINE'])])}")
        
        # Show first few vehicles
        print("\n   Sample vehicles:")
        for i, (_, vehicle) in enumerate(vehicles_df.head(3).iterrows()):
            print(f"   - {vehicle.get('vehicle_id', 'N/A')}: {vehicle.get('status', 'N/A')} ({vehicle.get('vehicle_type', 'N/A')})")
        
        return True
    else:
        print("   ❌ Failed to fetch vehicles or no vehicles found")
        return False

def main():
    """Main test function."""
    print("🧪 Automatic Token Management Test")
    print("=" * 60)
    
    # Test auth service
    auth_success = test_auth_service()
    
    if auth_success:
        # Test vehicle service
        vehicle_success = test_vehicle_service()
        
        if vehicle_success:
            print("\n🎉 All tests passed! Automatic token management is working.")
            print("\nNext steps:")
            print("- Your token will now refresh automatically when needed")
            print("- Check the .env file to see the updated token")
            print("- Use the API endpoints to monitor token status:")
            print("  • GET /api/auth/token/info - Check token status")
            print("  • POST /api/auth/token/refresh - Force refresh token")
            print("  • GET /api/auth/token/validate - Validate and refresh if needed")
        else:
            print("\n❌ Vehicle service test failed")
    else:
        print("\n❌ Authentication service test failed")

if __name__ == "__main__":
    main()