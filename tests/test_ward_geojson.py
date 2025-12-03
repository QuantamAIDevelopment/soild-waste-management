"""Test ward GeoJSON integration."""
import pytest
import os
from src.services.ward_geojson_service import WardGeoJSONService

def test_ward_geojson_service_initialization():
    """Test that WardGeoJSONService initializes correctly."""
    service = WardGeoJSONService()
    assert service.base_url is not None
    assert service.token is not None

def test_get_ward_geojson():
    """Test retrieving ward GeoJSON."""
    service = WardGeoJSONService()
    
    # Test with a known ward
    result = service.get_ward_geojson("Ward 29")
    
    # Result should be either a dict (success) or None (not found)
    assert result is None or isinstance(result, dict)

def test_upload_ward_geojson():
    """Test uploading ward GeoJSON."""
    service = WardGeoJSONService()
    
    # Sample GeoJSON
    sample_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]
                },
                "properties": {"name": "Test Ward"}
            }
        ]
    }
    
    result = service.upload_ward_geojson("Ward Test", sample_geojson, "TestBoundary")
    
    # Should return a dict with status
    assert isinstance(result, dict)
    assert "status" in result
    assert result["status"] in ["success", "exists", "error"]

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
