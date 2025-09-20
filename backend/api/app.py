"""
Flask API for Gridiron AI NFL prediction system.
Provides endpoints for predictions, team stats, and model management.
"""

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime, timedelta
import os
import logging
import numpy as np
from typing import Dict, List, Optional
import json

# Import our custom modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.nfl_predictor import NFLPredictor
from data_processor import NFLDataProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Database configuration
import os
db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '..', 'data', 'nfl_data.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database
db = SQLAlchemy(app)

# Initialize ML components
predictor = NFLPredictor()
data_processor = NFLDataProcessor()

# Database Models
class Game(db.Model):
    """Database model for NFL games."""
    __tablename__ = 'games'
    
    id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.String(50), unique=True, nullable=False)
    season = db.Column(db.Integer, nullable=False)
    week = db.Column(db.Integer, nullable=False)
    home_team = db.Column(db.String(10), nullable=False)
    away_team = db.Column(db.String(10), nullable=False)
    home_wins = db.Column(db.Integer, nullable=False)
    date = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'game_id': self.game_id,
            'season': self.season,
            'week': self.week,
            'home_team': self.home_team,
            'away_team': self.away_team,
            'home_wins': self.home_wins,
            'date': self.date.isoformat() if self.date else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class TeamStats(db.Model):
    """Database model for team statistics."""
    __tablename__ = 'team_stats'
    
    id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.String(50), db.ForeignKey('games.game_id'), nullable=False)
    team = db.Column(db.String(10), nullable=False)
    is_home = db.Column(db.Boolean, nullable=False)
    win_pct = db.Column(db.Float, nullable=False)
    off_rating = db.Column(db.Float, nullable=False)
    def_rating = db.Column(db.Float, nullable=False)
    avg_points = db.Column(db.Float, nullable=False)
    avg_yards = db.Column(db.Float, nullable=False)
    turnover_diff = db.Column(db.Float, nullable=False)
    rest_days = db.Column(db.Integer, nullable=False)
    injuries = db.Column(db.Integer, nullable=False)
    momentum = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'game_id': self.game_id,
            'team': self.team,
            'is_home': self.is_home,
            'win_pct': self.win_pct,
            'off_rating': self.off_rating,
            'def_rating': self.def_rating,
            'avg_points': self.avg_points,
            'avg_yards': self.avg_yards,
            'turnover_diff': self.turnover_diff,
            'rest_days': self.rest_days,
            'injuries': self.injuries,
            'momentum': self.momentum,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Weather(db.Model):
    """Database model for weather conditions."""
    __tablename__ = 'weather'
    
    id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.String(50), db.ForeignKey('games.game_id'), nullable=False)
    temperature = db.Column(db.Float, nullable=False)
    wind_speed = db.Column(db.Float, nullable=False)
    humidity = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'game_id': self.game_id,
            'temperature': self.temperature,
            'wind_speed': self.wind_speed,
            'humidity': self.humidity,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class Prediction(db.Model):
    """Database model for storing predictions."""
    __tablename__ = 'predictions'
    
    id = db.Column(db.Integer, primary_key=True)
    game_id = db.Column(db.String(50), nullable=False)
    home_team = db.Column(db.String(10), nullable=False)
    away_team = db.Column(db.String(10), nullable=False)
    home_win_probability = db.Column(db.Float, nullable=False)
    away_win_probability = db.Column(db.Float, nullable=False)
    predicted_winner = db.Column(db.String(10), nullable=False)
    confidence = db.Column(db.Float, nullable=False)
    actual_winner = db.Column(db.String(10), nullable=True)
    is_correct = db.Column(db.Boolean, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        return {
            'id': self.id,
            'game_id': self.game_id,
            'home_team': self.home_team,
            'away_team': self.away_team,
            'home_win_probability': self.home_win_probability,
            'away_win_probability': self.away_win_probability,
            'predicted_winner': self.predicted_winner,
            'confidence': self.confidence,
            'actual_winner': self.actual_winner,
            'is_correct': self.is_correct,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# API Routes
@app.route('/')
def index():
    """Serve the main dashboard."""
    return render_template('index.html')

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'model_loaded': predictor.model is not None
    })

@app.route('/api/predict', methods=['POST'])
def predict_game():
    """Predict the outcome of an NFL game."""
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['home_team', 'away_team', 'home_stats', 'away_stats']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        # Extract data
        home_team = data['home_team']
        away_team = data['away_team']
        home_stats = data['home_stats']
        away_stats = data['away_stats']
        weather = data.get('weather', {'temp': 50, 'wind': 5, 'humidity': 50})
        rest_days = data.get('rest_days', {'home': 7, 'away': 7})
        
        # Make prediction
        prediction = predictor.predict_game(home_stats, away_stats, weather, rest_days)
        
        # Add basic analysis if available
        try:
            if hasattr(predictor, '_generate_prediction_analysis'):
                analysis = predictor._generate_prediction_analysis(
                    home_stats, away_stats, weather, rest_days, 
                    predictor._calculate_game_features(home_stats, away_stats, weather, rest_days),
                    prediction['home_win_probability']
                )
                prediction['analysis'] = analysis
        except Exception as e:
            logger.warning(f"Could not generate analysis: {e}")
            # Continue without analysis
        
        # Store prediction in database
        game_id = f"{home_team}_{away_team}_{datetime.now().strftime('%Y%m%d')}"
        pred_record = Prediction(
            game_id=game_id,
            home_team=home_team,
            away_team=away_team,
            home_win_probability=prediction['home_win_probability'],
            away_win_probability=prediction['away_win_probability'],
            predicted_winner=prediction['predicted_winner'],
            confidence=prediction['confidence']
        )
        
        db.session.add(pred_record)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'prediction': prediction,
            'game_id': game_id
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/teams', methods=['GET'])
def get_teams():
    """Get list of NFL teams."""
    teams = [
        'NE', 'KC', 'BUF', 'MIA', 'NYJ', 'PIT', 'BAL', 'CLE', 'CIN', 'HOU',
        'IND', 'TEN', 'JAX', 'DEN', 'LV', 'LAC', 'DAL', 'PHI', 'WAS', 'NYG',
        'GB', 'MIN', 'CHI', 'DET', 'TB', 'NO', 'ATL', 'CAR', 'SF', 'SEA', 'LAR', 'ARI'
    ]
    return jsonify({'teams': teams})

@app.route('/api/team/<team_name>/stats', methods=['GET'])
def get_team_stats(team_name):
    """Get current statistics for a specific team."""
    try:
        # Try to get real team stats from the predictor
        if hasattr(predictor, 'get_real_team_stats'):
            real_stats = predictor.get_real_team_stats(team_name)
            if real_stats:
                return jsonify({
                    'success': True,
                    'team': team_name,
                    'stats': real_stats,
                    'data_source': 'real'
                })
        
        # Fallback to database
        latest_stats = TeamStats.query.filter_by(team=team_name).order_by(TeamStats.created_at.desc()).first()
        
        if latest_stats:
            return jsonify({
                'success': True,
                'team': team_name,
                'stats': latest_stats.to_dict(),
                'data_source': 'database'
            })
        else:
            # Return default stats if no data found
            default_stats = {
                'win_pct': 0.5,
                'off_rating': 100.0,
                'def_rating': 100.0,
                'avg_points': 25.0,
                'avg_yards': 350.0,
                'turnover_diff': 0.0,
                'injuries': 3,
                'momentum': 0.0
            }
            return jsonify({
                'success': True,
                'team': team_name,
                'stats': default_stats,
                'data_source': 'default'
            })
            
    except Exception as e:
        logger.error(f"Error getting team stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions', methods=['GET'])
def get_predictions():
    """Get recent predictions."""
    try:
        limit = request.args.get('limit', 10, type=int)
        predictions = Prediction.query.order_by(Prediction.created_at.desc()).limit(limit).all()
        
        return jsonify({
            'success': True,
            'predictions': [pred.to_dict() for pred in predictions]
        })
        
    except Exception as e:
        logger.error(f"Error getting predictions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/update-team-data', methods=['POST'])
def update_team_data():
    """Update team data from external sources."""
    try:
        if hasattr(predictor, 'update_team_data'):
            success = predictor.update_team_data()
            return jsonify({
                'success': success,
                'message': 'Team data updated successfully' if success else 'Failed to update team data'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Team data update not available'
            }), 400
            
    except Exception as e:
        logger.error(f"Error updating team data: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions/<prediction_id>', methods=['PUT'])
def update_prediction_result(prediction_id):
    """Update prediction with actual game result."""
    try:
        data = request.get_json()
        actual_winner = data.get('actual_winner')
        
        if not actual_winner:
            return jsonify({'error': 'actual_winner is required'}), 400
        
        prediction = Prediction.query.get(prediction_id)
        if not prediction:
            return jsonify({'error': 'Prediction not found'}), 404
        
        prediction.actual_winner = actual_winner
        prediction.is_correct = (prediction.predicted_winner == actual_winner)
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'prediction': prediction.to_dict()
        })
        
    except Exception as e:
        logger.error(f"Error updating prediction: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/accuracy', methods=['GET'])
def get_model_accuracy():
    """Get model accuracy statistics."""
    try:
        # Get predictions with actual results
        completed_predictions = Prediction.query.filter(Prediction.is_correct.isnot(None)).all()
        
        if not completed_predictions:
            return jsonify({
                'success': True,
                'accuracy': None,
                'total_predictions': 0,
                'correct_predictions': 0
            })
        
        correct_predictions = sum(1 for pred in completed_predictions if pred.is_correct)
        total_predictions = len(completed_predictions)
        accuracy = correct_predictions / total_predictions
        
        return jsonify({
            'success': True,
            'accuracy': accuracy,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions
        })
        
    except Exception as e:
        logger.error(f"Error getting model accuracy: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/model/retrain', methods=['POST'])
def retrain_model():
    """Retrain the model with new data."""
    try:
        # Load and process data
        df = data_processor.load_data()
        df = data_processor.clean_data(df)
        df = data_processor.engineer_features(df)
        X, y = data_processor.prepare_training_data(df)
        
        # Train model
        metrics = predictor.train(X, y, epochs=50)
        
        # Save model
        predictor.save_model()
        
        return jsonify({
            'success': True,
            'message': 'Model retrained successfully',
            'metrics': metrics
        })
        
    except Exception as e:
        logger.error(f"Error retraining model: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/data/upload', methods=['POST'])
def upload_data():
    """Upload new game data."""
    try:
        data = request.get_json()
        
        # Validate data structure
        if 'games' not in data:
            return jsonify({'error': 'games data is required'}), 400
        
        games_added = 0
        for game_data in data['games']:
            # Create game record
            game = Game(
                game_id=game_data['game_id'],
                season=game_data['season'],
                week=game_data['week'],
                home_team=game_data['home_team'],
                away_team=game_data['away_team'],
                home_wins=game_data['home_wins'],
                date=datetime.fromisoformat(game_data['date'])
            )
            
            db.session.add(game)
            games_added += 1
        
        db.session.commit()
        
        return jsonify({
            'success': True,
            'message': f'Successfully uploaded {games_added} games'
        })
        
    except Exception as e:
        logger.error(f"Error uploading data: {e}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    db.session.rollback()
    return jsonify({'error': 'Internal server error'}), 500

def init_db():
    """Initialize the database with tables."""
    with app.app_context():
        db.create_all()
        logger.info("Database initialized")

def load_model():
    """Load the trained model."""
    try:
        model_path = "models/nfl_predictor_model"
        if os.path.exists(f"{model_path}.h5"):
            predictor.load_model()
            logger.info("Model loaded successfully")
        else:
            logger.warning("No trained model found. Train model first.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

if __name__ == '__main__':
    # Initialize database
    init_db()
    
    # Load model
    load_model()
    
    # Run app
    app.run(debug=True, host='0.0.0.0', port=5000)
