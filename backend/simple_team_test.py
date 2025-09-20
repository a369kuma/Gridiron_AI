#!/usr/bin/env python3
"""
Simple test for team data system.
Tests the core functionality without complex database operations.
"""

import sys
import os
import logging
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.nfl_predictor import NFLPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_predictor_with_real_data():
    """Test the predictor with real team data functionality."""
    print("🏈 Testing Predictor with Real Team Data")
    print("=" * 50)
    
    # Initialize predictor
    predictor = NFLPredictor()
    
    # Test getting real team stats
    print("📊 Testing Real Team Stats:")
    sample_teams = ['KC', 'BUF', 'MIA', 'NE', 'BAL']
    
    for team in sample_teams:
        stats = predictor.get_real_team_stats(team)
        print(f"{team}: Off={stats['off_rating']:.1f}, Def={stats['def_rating']:.1f}, Win%={stats['win_pct']:.1%}")
    
    # Test team data update
    print("\n🔄 Testing Team Data Update:")
    update_success = predictor.update_team_data()
    print(f"Update successful: {update_success}")
    
    return predictor

def test_prediction_with_real_data():
    """Test prediction using real team data."""
    print("\n🎯 Testing Prediction with Real Data")
    print("=" * 50)
    
    predictor = NFLPredictor()
    
    # Load model if available
    try:
        predictor.load_model()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Model load failed: {e}")
        return
    
    # Test prediction with real team data
    home_team = 'KC'
    away_team = 'BUF'
    
    print(f"\n🏈 Predicting: {home_team} vs {away_team}")
    
    # Get real team stats
    home_stats = predictor.get_real_team_stats(home_team)
    away_stats = predictor.get_real_team_stats(away_team)
    
    print(f"\n📊 Team Statistics:")
    print(f"{home_team}: Off={home_stats['off_rating']:.1f}, Def={home_stats['def_rating']:.1f}, Win%={home_stats['win_pct']:.1%}")
    print(f"{away_team}: Off={away_stats['off_rating']:.1f}, Def={away_stats['def_rating']:.1f}, Win%={away_stats['win_pct']:.1%}")
    
    # Make prediction
    try:
        prediction = predictor.predict_game(
            home_stats, away_stats,
            {'temp': 45, 'wind': 10, 'humidity': 60},
            {'home': 7, 'away': 7}
        )
        
        print(f"\n🎯 Prediction Result:")
        print(f"Winner: {prediction['predicted_winner'].upper()}")
        print(f"Confidence: {prediction['confidence']:.1%}")
        print(f"Home Win Probability: {prediction['home_win_probability']:.1%}")
        print(f"Away Win Probability: {prediction['away_win_probability']:.1%}")
        
        if 'analysis' in prediction:
            analysis = prediction['analysis']
            print(f"\n📈 Analysis:")
            print(f"Confidence Level: {analysis['confidence_level']}")
            print(f"Key Factors: {len(analysis['key_factors'])}")
            
            for i, factor in enumerate(analysis['key_factors'][:3], 1):
                print(f"  {i}. {factor['factor']} ({factor['impact']} Impact)")
                print(f"     {factor['description']}")
            
            print(f"\n📝 Reasoning:")
            print(f"  {analysis['prediction_reasoning']}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")

def test_api_integration():
    """Test API integration with real team data."""
    print("\n🌐 Testing API Integration")
    print("=" * 50)
    
    import requests
    import json
    
    # Test team stats endpoint
    print("📊 Testing Team Stats Endpoint:")
    
    sample_teams = ['KC', 'BUF', 'MIA']
    
    for team in sample_teams:
        try:
            response = requests.get(f'http://localhost:5003/api/team/{team}/stats')
            if response.status_code == 200:
                data = response.json()
                stats = data['stats']
                source = data.get('data_source', 'unknown')
                print(f"{team}: Off={stats['off_rating']:.1f}, Def={stats['def_rating']:.1f} (Source: {source})")
            else:
                print(f"{team}: API error - {response.status_code}")
        except Exception as e:
            print(f"{team}: Connection error - {e}")
    
    # Test team data update endpoint
    print("\n🔄 Testing Team Data Update Endpoint:")
    try:
        response = requests.post('http://localhost:5003/api/update-team-data')
        if response.status_code == 200:
            data = response.json()
            print(f"Update result: {data['success']} - {data['message']}")
        else:
            print(f"Update failed: {response.status_code}")
    except Exception as e:
        print(f"Update error: {e}")

def main():
    """Main test function."""
    print("🏈 Gridiron AI - Simple Team Data Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Test predictor integration
        predictor = test_predictor_with_real_data()
        
        # Test prediction with real data
        test_prediction_with_real_data()
        
        # Test API integration
        test_api_integration()
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.exception("Test failed")
    
    print(f"\nTest completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
