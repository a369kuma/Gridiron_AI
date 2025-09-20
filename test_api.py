#!/usr/bin/env python3
"""
Simple test script to verify the API endpoints are working correctly.
"""

import requests
import json

def test_api():
    """Test the API endpoints."""
    base_url = "http://localhost:5003/api"
    
    print("🏈 Testing Gridiron AI API...")
    print("=" * 50)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health")
        print(f"✅ Health Check: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
    except requests.exceptions.ConnectionError:
        print("❌ Health Check: API server is not running")
        print("   Please start the API server first with: python backend/run_api.py")
        return
    except Exception as e:
        print(f"❌ Health Check: {e}")
        return
    
    # Test teams endpoint
    try:
        response = requests.get(f"{base_url}/teams")
        print(f"✅ Teams Endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            teams = data.get('teams', [])
            print(f"   Number of teams: {len(teams)}")
            print(f"   Teams: {teams}")
            
            if len(teams) != 32:
                print(f"⚠️  Warning: Expected 32 teams, got {len(teams)}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Teams Endpoint: {e}")
    
    # Test team stats endpoint
    try:
        response = requests.get(f"{base_url}/team/KC/stats")
        print(f"✅ Team Stats Endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   KC Stats: {data.get('stats', {})}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Team Stats Endpoint: {e}")
    
    # Test predictions endpoint
    try:
        response = requests.get(f"{base_url}/predictions")
        print(f"✅ Predictions Endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            predictions = data.get('predictions', [])
            print(f"   Number of predictions: {len(predictions)}")
        else:
            print(f"   Error: {response.text}")
    except Exception as e:
        print(f"❌ Predictions Endpoint: {e}")
    
    print("=" * 50)
    print("🏈 API Test Complete!")

if __name__ == "__main__":
    test_api()
