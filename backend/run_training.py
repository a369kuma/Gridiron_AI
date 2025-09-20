#!/usr/bin/env python3
"""
Training script for Gridiron AI NFL prediction model.
This script handles the complete training pipeline from data loading to model saving.
"""

import os
import sys
import logging
from datetime import datetime

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processor import NFLDataProcessor
from models.nfl_predictor import NFLPredictor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main training pipeline."""
    logger.info("Starting Gridiron AI model training pipeline...")
    
    try:
        # Initialize components
        processor = NFLDataProcessor()
        predictor = NFLPredictor()
        
        # Step 1: Load and process data
        logger.info("Step 1: Loading and processing data...")
        df = processor.load_data()
        logger.info(f"Loaded {len(df)} records")
        
        # Step 2: Clean data
        logger.info("Step 2: Cleaning data...")
        df = processor.clean_data(df)
        logger.info(f"After cleaning: {len(df)} records")
        
        # Step 3: Engineer features
        logger.info("Step 3: Engineering features...")
        df = processor.engineer_features(df)
        
        # Step 4: Prepare training data
        logger.info("Step 4: Preparing training data...")
        X, y = processor.prepare_training_data(df)
        
        # Step 5: Train model
        logger.info("Step 5: Training model...")
        metrics = predictor.train(X, y, epochs=50, verbose=1)
        
        # Step 6: Save model
        logger.info("Step 6: Saving model...")
        predictor.save_model()
        
        # Step 7: Display results
        logger.info("Training completed successfully!")
        logger.info("Model Performance:")
        logger.info(f"  Validation Accuracy: {metrics['val_accuracy']:.4f}")
        logger.info(f"  Validation F1-Score: {metrics['val_f1']:.4f}")
        logger.info(f"  Validation AUC: {metrics['val_auc']:.4f}")
        
        # Step 8: Test prediction
        logger.info("Step 8: Testing prediction...")
        test_prediction(predictor)
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def test_prediction(predictor):
    """Test the trained model with sample data."""
    logger.info("Testing model with sample prediction...")
    
    # Sample team stats
    home_stats = {
        'win_pct': 0.7,
        'off_rating': 110,
        'def_rating': 95,
        'avg_points': 28,
        'avg_yards': 380,
        'turnover_diff': 0.5,
        'momentum': 0.3,
        'injuries': 2
    }
    
    away_stats = {
        'win_pct': 0.6,
        'off_rating': 105,
        'def_rating': 100,
        'avg_points': 25,
        'avg_yards': 360,
        'turnover_diff': -0.2,
        'momentum': -0.1,
        'injuries': 4
    }
    
    weather = {'temp': 45, 'wind': 8, 'humidity': 60}
    rest = {'home': 7, 'away': 10}
    
    try:
        prediction = predictor.predict_game(home_stats, away_stats, weather, rest)
        logger.info("Sample Prediction Result:")
        logger.info(f"  Home Win Probability: {prediction['home_win_probability']:.4f}")
        logger.info(f"  Away Win Probability: {prediction['away_win_probability']:.4f}")
        logger.info(f"  Predicted Winner: {prediction['predicted_winner']}")
        logger.info(f"  Confidence: {prediction['confidence']:.4f}")
    except Exception as e:
        logger.error(f"Prediction test failed: {e}")

if __name__ == "__main__":
    main()
