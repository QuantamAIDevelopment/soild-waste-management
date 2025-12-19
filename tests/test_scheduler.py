"""Test script for the SWM Agent scheduler functionality."""
import asyncio
import os
import requests
import time
from datetime import datetime
from loguru import logger

# Configuration
BASE_URL = "http://127.0.0.1:8000"
API_KEY = os.getenv("SWM_API_KEY", "swm-2024-secure-key")

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

def test_scheduler_endpoints():
    """Test all scheduler-related endpoints."""
    
    print("🧪 Testing SWM Agent Scheduler Functionality")
    print("=" * 50)
    
    # Test 1: Check scheduler status
    print("\n1️⃣ Testing scheduler status...")
    try:
        response = requests.get(f"{BASE_URL}/scheduler/status", headers=headers)
        if response.status_code == 200:
            status_data = response.json()
            print(f"✅ Status: {status_data}")
        else:
            print(f"❌ Status check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Status check error: {e}")
    
    # Test 2: Manual trigger
    print("\n2️⃣ Testing manual vehicle data fetch...")
    try:
        response = requests.post(f"{BASE_URL}/scheduler/trigger", headers=headers)
        if response.status_code == 200:
            trigger_data = response.json()
            print(f"✅ Manual trigger: {trigger_data}")
        else:
            print(f"❌ Manual trigger failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Manual trigger error: {e}")
    
    # Test 3: Check if scheduler is running
    print("\n3️⃣ Checking scheduler status after trigger...")
    try:
        response = requests.get(f"{BASE_URL}/scheduler/status", headers=headers)
        if response.status_code == 200:
            status_data = response.json()
            print(f"✅ Current status: {status_data}")
            
            if status_data.get("status") == "running":
                print(f"📅 Next scheduled run: {status_data.get('next_run_time', 'Unknown')}")
            
        else:
            print(f"❌ Status check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Status check error: {e}")
    
    # Test 4: Check vehicle API
    print("\n4️⃣ Testing live vehicle API...")
    try:
        response = requests.get(f"{BASE_URL}/api/vehicles/live", headers=headers)
        if response.status_code == 200:
            vehicles_data = response.json()
            print(f"✅ Live vehicles: {len(vehicles_data.get('vehicles', []))} vehicles found")
        else:
            print(f"❌ Vehicle API failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Vehicle API error: {e}")
    
    print("\n" + "=" * 50)
    print("🎯 Test Summary:")
    print("- Scheduler automatically starts with FastAPI")
    print("- Daily fetch scheduled for 5:30 AM")
    print("- Manual trigger available for testing")
    print("- All fetched data saved with timestamps")
    print("- Check output/ directory for saved files")

def check_output_files():
    """Check for generated output files."""
    import os
    import glob
    
    print("\n📁 Checking output files...")
    
    if os.path.exists("output"):
        vehicle_files = glob.glob("output/vehicles_daily_*.csv")
        if vehicle_files:
            print(f"✅ Found {len(vehicle_files)} daily vehicle files:")
            for file in sorted(vehicle_files)[-3:]:  # Show last 3 files
                print(f"   - {file}")
        else:
            print("ℹ️ No daily vehicle files found yet")
    else:
        print("ℹ️ Output directory not found")

if __name__ == "__main__":
    print(f"🚀 Starting SWM Agent Scheduler Tests at {datetime.now()}")
    print(f"🌐 Base URL: {BASE_URL}")
    print("📝 Make sure the FastAPI server is running with: python main.py --api --port 8000")
    
    # Wait a moment for server to be ready
    print("\n⏳ Waiting 3 seconds for server...")
    time.sleep(3)
    
    # Run tests
    test_scheduler_endpoints()
    
    # Check output files
    check_output_files()
    
    print(f"\n✅ Tests completed at {datetime.now()}")
    print("\n💡 Usage Tips:")
    print("- Use GET /scheduler/status to monitor the scheduler")
    print("- Use POST /scheduler/trigger for manual testing")
    print("- Check logs for detailed scheduling information")
    print("- Vehicle data is automatically fetched daily at 5:30 AM")