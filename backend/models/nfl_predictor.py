"""
TensorFlow-based NFL game outcome prediction model.
Implements neural network with hyperparameter optimization and feature engineering.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, callbacks
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
import joblib
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import optuna

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from data.team_data_manager import TeamDataManager
    TEAM_DATA_AVAILABLE = True
except ImportError:
    TEAM_DATA_AVAILABLE = False
    logging.warning("Team data manager not available, using fallback data")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NFLPredictor:
    """Neural network model for predicting NFL game outcomes."""
    
    def __init__(self, model_path: str = "models/nfl_predictor_model"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_names = None
        self.training_history = None
        
        # Initialize team data manager
        if TEAM_DATA_AVAILABLE:
            self.team_data_manager = TeamDataManager()
            logger.info("Team data manager initialized")
        else:
            self.team_data_manager = None
            logger.warning("Team data manager not available")
        
    def build_model(self, input_dim: int, hidden_layers: List[int] = [128, 64, 32], 
                   dropout_rate: float = 0.3, learning_rate: float = 0.001) -> keras.Model:
        """Build the neural network architecture."""
        model = keras.Sequential()
        
        # Input layer
        model.add(layers.Dense(hidden_layers[0], activation='relu', input_shape=(input_dim,)))
        model.add(layers.BatchNormalization())
        model.add(layers.Dropout(dropout_rate))
        
        # Hidden layers
        for units in hidden_layers[1:]:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.BatchNormalization())
            model.add(layers.Dropout(dropout_rate))
        
        # Output layer
        model.add(layers.Dense(1, activation='sigmoid'))
        
        # Compile model
        optimizer = optimizers.Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              validation_split: float = 0.2, epochs: int = 100, 
              batch_size: int = 32, verbose: int = 1) -> Dict:
        """Train the model with early stopping and learning rate reduction."""
        logger.info("Starting model training...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Build model
        self.model = self.build_model(X.shape[1])
        
        # Callbacks
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                filepath=f"{self.model_path}_best.h5",
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        self.training_history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=verbose
        )
        
        # Evaluate model
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_pred_binary = (train_pred > 0.5).astype(int).flatten()
        val_pred_binary = (val_pred > 0.5).astype(int).flatten()
        
        metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred_binary),
            'val_accuracy': accuracy_score(y_val, val_pred_binary),
            'train_precision': precision_score(y_train, train_pred_binary),
            'val_precision': precision_score(y_val, val_pred_binary),
            'train_recall': recall_score(y_train, train_pred_binary),
            'val_recall': recall_score(y_val, val_pred_binary),
            'train_f1': f1_score(y_train, train_pred_binary),
            'val_f1': f1_score(y_val, val_pred_binary),
            'train_auc': roc_auc_score(y_train, train_pred),
            'val_auc': roc_auc_score(y_val, val_pred)
        }
        
        logger.info("Training completed!")
        logger.info(f"Validation Accuracy: {metrics['val_accuracy']:.4f}")
        logger.info(f"Validation F1-Score: {metrics['val_f1']:.4f}")
        logger.info(f"Validation AUC: {metrics['val_auc']:.4f}")
        
        return metrics
    
    def optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                                n_trials: int = 50) -> Dict:
        """Optimize hyperparameters using Optuna."""
        logger.info("Starting hyperparameter optimization...")
        
        def objective(trial):
            # Suggest hyperparameters
            hidden_layers = [
                trial.suggest_int('layer1', 64, 256),
                trial.suggest_int('layer2', 32, 128),
                trial.suggest_int('layer3', 16, 64)
            ]
            dropout_rate = trial.suggest_float('dropout', 0.1, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            
            # Build and train model
            model = self.build_model(X.shape[1], hidden_layers, dropout_rate, learning_rate)
            
            # Split data
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Train with early stopping
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True
            )
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=50,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=0
            )
            
            # Return validation loss
            return min(history.history['val_loss'])
        
        # Run optimization
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)
        
        best_params = study.best_params
        logger.info(f"Best hyperparameters: {best_params}")
        
        return best_params
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        predictions = self.model.predict(X)
        return predictions.flatten()
    
    def predict_game(self, home_team_stats: Dict, away_team_stats: Dict, 
                    weather_conditions: Dict, rest_days: Dict) -> Dict:
        """Predict outcome for a specific game with detailed analysis."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Calculate features
        features = self._calculate_game_features(
            home_team_stats, away_team_stats, weather_conditions, rest_days
        )
        
        # Make prediction
        prediction = self.predict(features.reshape(1, -1))[0]
        
        # Generate detailed analysis
        analysis = self._generate_prediction_analysis(
            home_team_stats, away_team_stats, weather_conditions, rest_days, features, prediction
        )
        
        return {
            'home_win_probability': float(prediction),
            'away_win_probability': float(1 - prediction),
            'predicted_winner': 'home' if prediction > 0.5 else 'away',
            'confidence': float(abs(prediction - 0.5) * 2),  # 0-1 scale
            'analysis': analysis
        }
    
    def _calculate_game_features(self, home_stats: Dict, away_stats: Dict,
                               weather: Dict, rest: Dict) -> np.ndarray:
        """Calculate features for a specific game."""
        features = np.array([
            home_stats.get('off_rating', 100) - away_stats.get('off_rating', 100),  # off_rating_diff
            away_stats.get('def_rating', 100) - home_stats.get('def_rating', 100),  # def_rating_diff
            home_stats.get('win_pct', 0.5) - away_stats.get('win_pct', 0.5),       # win_pct_diff
            home_stats.get('avg_points', 25) - away_stats.get('avg_points', 25),   # points_diff
            home_stats.get('avg_yards', 350) - away_stats.get('avg_yards', 350),   # yards_diff
            home_stats.get('turnover_diff', 0) - away_stats.get('turnover_diff', 0), # turnover_diff
            rest.get('home', 7) - rest.get('away', 7),                              # rest_advantage
            away_stats.get('injuries', 3) - home_stats.get('injuries', 3),         # injury_advantage
            home_stats.get('momentum', 0) - away_stats.get('momentum', 0),         # momentum_diff
            self._calculate_weather_impact(weather),                                # weather_impact
            self._calculate_strength_diff(home_stats, away_stats),                 # strength_diff
            rest.get('home', 7),                                                   # home_rest_days
            rest.get('away', 7),                                                   # away_rest_days
            weather.get('temp', 50),                                               # weather_temp
            weather.get('wind', 5),                                                # weather_wind
            weather.get('humidity', 50)                                            # weather_humidity
        ])
        
        return features
    
    def _generate_prediction_analysis(self, home_stats: Dict, away_stats: Dict,
                                    weather: Dict, rest: Dict, features: np.ndarray, 
                                    prediction: float) -> Dict:
        """Generate detailed analysis of the prediction."""
        
        # Calculate key metrics
        off_rating_diff = home_stats.get('off_rating', 100) - away_stats.get('off_rating', 100)
        def_rating_diff = away_stats.get('def_rating', 100) - home_stats.get('def_rating', 100)
        win_pct_diff = home_stats.get('win_pct', 0.5) - away_stats.get('win_pct', 0.5)
        points_diff = home_stats.get('avg_points', 25) - away_stats.get('avg_points', 25)
        yards_diff = home_stats.get('avg_yards', 350) - away_stats.get('avg_yards', 350)
        turnover_diff = home_stats.get('turnover_diff', 0) - away_stats.get('turnover_diff', 0)
        rest_advantage = rest.get('home', 7) - rest.get('away', 7)
        injury_advantage = away_stats.get('injuries', 3) - home_stats.get('injuries', 3)
        momentum_diff = home_stats.get('momentum', 0) - away_stats.get('momentum', 0)
        
        # Determine key factors
        key_factors = []
        
        # Offensive advantage
        if abs(off_rating_diff) > 10:
            if off_rating_diff > 0:
                key_factors.append({
                    'factor': 'Offensive Advantage',
                    'impact': 'High',
                    'description': f'Home team has significantly stronger offense (+{off_rating_diff:.1f} rating)',
                    'value': off_rating_diff,
                    'favor': 'home'
                })
            else:
                key_factors.append({
                    'factor': 'Offensive Advantage',
                    'impact': 'High',
                    'description': f'Away team has significantly stronger offense (+{abs(off_rating_diff):.1f} rating)',
                    'value': off_rating_diff,
                    'favor': 'away'
                })
        
        # Defensive advantage
        if abs(def_rating_diff) > 10:
            if def_rating_diff > 0:
                key_factors.append({
                    'factor': 'Defensive Advantage',
                    'impact': 'High',
                    'description': f'Home team has significantly stronger defense (+{def_rating_diff:.1f} rating)',
                    'value': def_rating_diff,
                    'favor': 'home'
                })
            else:
                key_factors.append({
                    'factor': 'Defensive Advantage',
                    'impact': 'High',
                    'description': f'Away team has significantly stronger defense (+{abs(def_rating_diff):.1f} rating)',
                    'value': def_rating_diff,
                    'favor': 'away'
                })
        
        # Win percentage
        if abs(win_pct_diff) > 0.2:
            if win_pct_diff > 0:
                key_factors.append({
                    'factor': 'Season Performance',
                    'impact': 'Medium',
                    'description': f'Home team has better season record ({home_stats.get("win_pct", 0.5):.1%} vs {away_stats.get("win_pct", 0.5):.1%})',
                    'value': win_pct_diff,
                    'favor': 'home'
                })
            else:
                key_factors.append({
                    'factor': 'Season Performance',
                    'impact': 'Medium',
                    'description': f'Away team has better season record ({away_stats.get("win_pct", 0.5):.1%} vs {home_stats.get("win_pct", 0.5):.1%})',
                    'value': win_pct_diff,
                    'favor': 'away'
                })
        
        # Rest advantage
        if abs(rest_advantage) > 2:
            if rest_advantage > 0:
                key_factors.append({
                    'factor': 'Rest Advantage',
                    'impact': 'Medium',
                    'description': f'Home team has more rest days ({rest.get("home", 7)} vs {rest.get("away", 7)} days)',
                    'value': rest_advantage,
                    'favor': 'home'
                })
            else:
                key_factors.append({
                    'factor': 'Rest Advantage',
                    'impact': 'Medium',
                    'description': f'Away team has more rest days ({rest.get("away", 7)} vs {rest.get("home", 7)} days)',
                    'value': rest_advantage,
                    'favor': 'away'
                })
        
        # Injury advantage
        if abs(injury_advantage) > 2:
            if injury_advantage > 0:
                key_factors.append({
                    'factor': 'Injury Advantage',
                    'impact': 'Medium',
                    'description': f'Home team has fewer injuries ({home_stats.get("injuries", 3)} vs {away_stats.get("injuries", 3)})',
                    'value': injury_advantage,
                    'favor': 'home'
                })
            else:
                key_factors.append({
                    'factor': 'Injury Advantage',
                    'impact': 'Medium',
                    'description': f'Away team has fewer injuries ({away_stats.get("injuries", 3)} vs {home_stats.get("injuries", 3)})',
                    'value': injury_advantage,
                    'favor': 'away'
                })
        
        # Weather impact
        weather_impact = self._calculate_weather_impact(weather)
        if weather_impact > 0.1:
            key_factors.append({
                'factor': 'Weather Conditions',
                'impact': 'Low',
                'description': f'Adverse weather conditions (Temp: {weather.get("temp", 50)}°F, Wind: {weather.get("wind", 5)} mph)',
                'value': weather_impact,
                'favor': 'neutral'
            })
        
        # Momentum
        if abs(momentum_diff) > 0.3:
            if momentum_diff > 0:
                key_factors.append({
                    'factor': 'Recent Momentum',
                    'impact': 'Medium',
                    'description': f'Home team has positive momentum (+{momentum_diff:.2f})',
                    'value': momentum_diff,
                    'favor': 'home'
                })
            else:
                key_factors.append({
                    'factor': 'Recent Momentum',
                    'impact': 'Medium',
                    'description': f'Away team has positive momentum (+{abs(momentum_diff):.2f})',
                    'value': momentum_diff,
                    'favor': 'away'
                })
        
        # Sort factors by impact
        impact_order = {'High': 3, 'Medium': 2, 'Low': 1}
        key_factors.sort(key=lambda x: impact_order.get(x['impact'], 0), reverse=True)
        
        # Calculate team comparison
        home_advantages = sum(1 for factor in key_factors if factor['favor'] == 'home')
        away_advantages = sum(1 for factor in key_factors if factor['favor'] == 'away')
        
        # Generate summary
        confidence_level = "Very High" if abs(prediction - 0.5) > 0.3 else "High" if abs(prediction - 0.5) > 0.2 else "Medium" if abs(prediction - 0.5) > 0.1 else "Low"
        
        return {
            'key_factors': key_factors[:5],  # Top 5 factors
            'team_comparison': {
                'home_advantages': home_advantages,
                'away_advantages': away_advantages,
                'offensive_rating': {
                    'home': home_stats.get('off_rating', 100),
                    'away': away_stats.get('off_rating', 100),
                    'difference': off_rating_diff
                },
                'defensive_rating': {
                    'home': home_stats.get('def_rating', 100),
                    'away': away_stats.get('def_rating', 100),
                    'difference': -def_rating_diff
                },
                'win_percentage': {
                    'home': home_stats.get('win_pct', 0.5),
                    'away': away_stats.get('win_pct', 0.5),
                    'difference': win_pct_diff
                }
            },
            'confidence_level': confidence_level,
            'prediction_reasoning': self._generate_reasoning_text(prediction, key_factors, confidence_level)
        }
    
    def _generate_reasoning_text(self, prediction: float, key_factors: List[Dict], confidence_level: str) -> str:
        """Generate human-readable reasoning for the prediction."""
        winner = "home team" if prediction > 0.5 else "away team"
        confidence_pct = abs(prediction - 0.5) * 200
        
        if not key_factors:
            return f"The model predicts the {winner} will win with {confidence_level.lower()} confidence ({confidence_pct:.1f}%). The teams appear evenly matched based on current statistics."
        
        top_factor = key_factors[0]
        reasoning = f"The model predicts the {winner} will win with {confidence_level.lower()} confidence ({confidence_pct:.1f}%). "
        
        if top_factor['favor'] == 'home' and prediction > 0.5:
            reasoning += f"The primary factor is {top_factor['factor'].lower()}: {top_factor['description']}. "
        elif top_factor['favor'] == 'away' and prediction < 0.5:
            reasoning += f"The primary factor is {top_factor['factor'].lower()}: {top_factor['description']}. "
        else:
            reasoning += f"Despite {top_factor['factor'].lower()} favoring the other team, other factors suggest the {winner} will prevail. "
        
        if len(key_factors) > 1:
            secondary_factors = [f['factor'] for f in key_factors[1:3]]
            reasoning += f"Additional factors include {', '.join(secondary_factors).lower()}. "
        
        return reasoning.strip()
    
    def get_real_team_stats(self, team: str) -> Dict:
        """Get real team statistics from data manager."""
        if self.team_data_manager:
            try:
                # Get current team stats
                current_stats = self.team_data_manager.get_team_current_stats(team)
                if current_stats:
                    return {
                        'win_pct': current_stats['win_percentage'],
                        'off_rating': current_stats['offensive_rating'],
                        'def_rating': current_stats['defensive_rating'],
                        'avg_points': current_stats['points_for'],
                        'avg_yards': current_stats['yards_for'],
                        'turnover_diff': current_stats['turnover_differential'],
                        'injuries': np.random.randint(0, 8),  # Placeholder - would need injury data
                        'momentum': self._calculate_team_momentum(current_stats)
                    }
            except Exception as e:
                logger.error(f"Error getting real team stats for {team}: {e}")
        
        # Fallback to default stats
        return self._get_default_team_stats(team)
    
    def _calculate_team_momentum(self, stats: Dict) -> float:
        """Calculate team momentum from real stats."""
        win_pct = stats['win_percentage']
        point_diff = stats.get('point_differential', 0)
        
        # Momentum based on win percentage and point differential
        momentum = (win_pct - 0.5) * 2  # -1 to 1 range
        momentum += (point_diff / 100) * 0.5  # Adjust for point differential
        
        return max(-1, min(1, momentum))  # Clamp between -1 and 1
    
    def _get_default_team_stats(self, team: str) -> Dict:
        """Get default team stats when real data is not available."""
        # Team-specific default ratings based on recent performance
        team_defaults = {
            'KC': {'off_rating': 125, 'def_rating': 95, 'win_pct': 0.75},
            'BUF': {'off_rating': 115, 'def_rating': 100, 'win_pct': 0.65},
            'MIA': {'off_rating': 120, 'def_rating': 105, 'win_pct': 0.60},
            'NE': {'off_rating': 90, 'def_rating': 110, 'win_pct': 0.45},
            'BAL': {'off_rating': 110, 'def_rating': 95, 'win_pct': 0.70},
            'CIN': {'off_rating': 115, 'def_rating': 100, 'win_pct': 0.65},
            'SF': {'off_rating': 110, 'def_rating': 90, 'win_pct': 0.70},
            'PHI': {'off_rating': 115, 'def_rating': 100, 'win_pct': 0.65},
            'DAL': {'off_rating': 110, 'def_rating': 105, 'win_pct': 0.60},
            'GB': {'off_rating': 120, 'def_rating': 110, 'win_pct': 0.55},
            'LAR': {'off_rating': 105, 'def_rating': 100, 'win_pct': 0.60},
            'TB': {'off_rating': 100, 'def_rating': 105, 'win_pct': 0.50},
            'MIN': {'off_rating': 110, 'def_rating': 110, 'win_pct': 0.55},
            'DET': {'off_rating': 115, 'def_rating': 105, 'win_pct': 0.60},
            'NO': {'off_rating': 105, 'def_rating': 100, 'win_pct': 0.55},
            'ATL': {'off_rating': 100, 'def_rating': 110, 'win_pct': 0.50},
            'CAR': {'off_rating': 85, 'def_rating': 105, 'win_pct': 0.35},
            'NYG': {'off_rating': 90, 'def_rating': 110, 'win_pct': 0.40},
            'WAS': {'off_rating': 95, 'def_rating': 105, 'win_pct': 0.45},
            'CHI': {'off_rating': 85, 'def_rating': 100, 'win_pct': 0.35},
            'GB': {'off_rating': 120, 'def_rating': 110, 'win_pct': 0.55},
            'MIN': {'off_rating': 110, 'def_rating': 110, 'win_pct': 0.55},
            'DET': {'off_rating': 115, 'def_rating': 105, 'win_pct': 0.60},
            'TB': {'off_rating': 100, 'def_rating': 105, 'win_pct': 0.50},
            'NO': {'off_rating': 105, 'def_rating': 100, 'win_pct': 0.55},
            'ATL': {'off_rating': 100, 'def_rating': 110, 'win_pct': 0.50},
            'CAR': {'off_rating': 85, 'def_rating': 105, 'win_pct': 0.35},
            'SF': {'off_rating': 110, 'def_rating': 90, 'win_pct': 0.70},
            'SEA': {'off_rating': 105, 'def_rating': 105, 'win_pct': 0.55},
            'LAR': {'off_rating': 105, 'def_rating': 100, 'win_pct': 0.60},
            'ARI': {'off_rating': 95, 'def_rating': 110, 'win_pct': 0.45},
            'DEN': {'off_rating': 90, 'def_rating': 100, 'win_pct': 0.40},
            'LV': {'off_rating': 100, 'def_rating': 110, 'win_pct': 0.50},
            'LAC': {'off_rating': 110, 'def_rating': 105, 'win_pct': 0.60},
            'KC': {'off_rating': 125, 'def_rating': 95, 'win_pct': 0.75},
            'HOU': {'off_rating': 105, 'def_rating': 100, 'win_pct': 0.55},
            'IND': {'off_rating': 100, 'def_rating': 105, 'win_pct': 0.50},
            'TEN': {'off_rating': 95, 'def_rating': 105, 'win_pct': 0.45},
            'JAX': {'off_rating': 110, 'def_rating': 100, 'win_pct': 0.60},
            'CLE': {'off_rating': 100, 'def_rating': 95, 'win_pct': 0.55},
            'PIT': {'off_rating': 95, 'def_rating': 100, 'win_pct': 0.50},
            'BAL': {'off_rating': 110, 'def_rating': 95, 'win_pct': 0.70},
            'CIN': {'off_rating': 115, 'def_rating': 100, 'win_pct': 0.65},
            'NYJ': {'off_rating': 85, 'def_rating': 95, 'win_pct': 0.35},
            'BUF': {'off_rating': 115, 'def_rating': 100, 'win_pct': 0.65},
            'MIA': {'off_rating': 120, 'def_rating': 105, 'win_pct': 0.60},
            'NE': {'off_rating': 90, 'def_rating': 110, 'win_pct': 0.45}
        }
        
        defaults = team_defaults.get(team, {
            'off_rating': 100, 'def_rating': 100, 'win_pct': 0.50
        })
        
        return {
            'win_pct': defaults['win_pct'],
            'off_rating': defaults['off_rating'],
            'def_rating': defaults['def_rating'],
            'avg_points': defaults['off_rating'] * 0.25,  # Rough conversion
            'avg_yards': defaults['off_rating'] * 3.5,    # Rough conversion
            'turnover_diff': np.random.uniform(-1, 1),
            'injuries': np.random.randint(0, 8),
            'momentum': np.random.uniform(-0.5, 0.5)
        }
    
    def update_team_data(self) -> bool:
        """Update team data from external sources."""
        if self.team_data_manager:
            try:
                results = self.team_data_manager.update_all_team_data()
                logger.info(f"Team data update results: {results}")
                return any(results.values())
            except Exception as e:
                logger.error(f"Error updating team data: {e}")
                return False
        return False
    
    def _calculate_weather_impact(self, weather: Dict) -> float:
        """Calculate weather impact on the game."""
        temp_impact = abs(weather.get('temp', 50) - 70) * 0.01
        wind_impact = weather.get('wind', 5) * 0.02
        humidity_impact = abs(weather.get('humidity', 50) - 50) * 0.005
        
        return temp_impact + wind_impact + humidity_impact
    
    def _calculate_strength_diff(self, home_stats: Dict, away_stats: Dict) -> float:
        """Calculate overall team strength difference."""
        home_strength = (
            home_stats.get('win_pct', 0.5) * 0.3 +
            (home_stats.get('off_rating', 100) / 100) * 0.3 +
            (home_stats.get('def_rating', 100) / 100) * 0.2 +
            (home_stats.get('avg_points', 25) / 30) * 0.2
        )
        
        away_strength = (
            away_stats.get('win_pct', 0.5) * 0.3 +
            (away_stats.get('off_rating', 100) / 100) * 0.3 +
            (away_stats.get('def_rating', 100) / 100) * 0.2 +
            (away_stats.get('avg_points', 25) / 30) * 0.2
        )
        
        return home_strength - away_strength
    
    def save_model(self, scaler=None):
        """Save the trained model and scaler."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Save model
        self.model.save(f"{self.model_path}.h5")
        
        # Save scaler if provided
        if scaler is not None:
            joblib.dump(scaler, f"{self.model_path}_scaler.pkl")
        
        # Save model metadata
        metadata = {
            'feature_names': self.feature_names,
            'model_architecture': self.model.to_json(),
            'training_date': datetime.now().isoformat(),
            'model_path': self.model_path
        }
        
        joblib.dump(metadata, f"{self.model_path}_metadata.pkl")
        logger.info(f"Model saved to {self.model_path}")
    
    def load_model(self, scaler_path: str = None):
        """Load a trained model."""
        # Load model
        self.model = keras.models.load_model(f"{self.model_path}.h5")
        
        # Load scaler if provided
        if scaler_path and os.path.exists(scaler_path):
            self.scaler = joblib.load(scaler_path)
        
        # Load metadata
        metadata_path = f"{self.model_path}_metadata.pkl"
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.feature_names = metadata.get('feature_names')
        
        logger.info(f"Model loaded from {self.model_path}")
    
    def get_feature_importance(self, X: np.ndarray, feature_names: List[str]) -> Dict:
        """Calculate feature importance using permutation importance."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        baseline_score = self.model.evaluate(X, np.zeros(len(X)), verbose=0)[0]
        importance_scores = {}
        
        for i, feature_name in enumerate(feature_names):
            X_permuted = X.copy()
            np.random.shuffle(X_permuted[:, i])
            
            permuted_score = self.model.evaluate(X_permuted, np.zeros(len(X)), verbose=0)[0]
            importance = permuted_score - baseline_score
            importance_scores[feature_name] = importance
        
        # Sort by importance
        sorted_importance = dict(sorted(importance_scores.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return sorted_importance

if __name__ == "__main__":
    # Example usage
    from data_processor import NFLDataProcessor
    
    # Load and process data
    processor = NFLDataProcessor()
    df = processor.load_data()
    df = processor.clean_data(df)
    df = processor.engineer_features(df)
    X, y = processor.prepare_training_data(df)
    
    # Train model
    predictor = NFLPredictor()
    
    # Optional: Optimize hyperparameters
    # best_params = predictor.optimize_hyperparameters(X, y, n_trials=20)
    
    # Train with best parameters or defaults
    metrics = predictor.train(X, y, epochs=50)
    
    # Save model
    predictor.save_model()
    
    # Example prediction
    home_stats = {
        'win_pct': 0.7, 'off_rating': 110, 'def_rating': 95,
        'avg_points': 28, 'avg_yards': 380, 'turnover_diff': 0.5,
        'momentum': 0.3, 'injuries': 2
    }
    
    away_stats = {
        'win_pct': 0.6, 'off_rating': 105, 'def_rating': 100,
        'avg_points': 25, 'avg_yards': 360, 'turnover_diff': -0.2,
        'momentum': -0.1, 'injuries': 4
    }
    
    weather = {'temp': 45, 'wind': 8, 'humidity': 60}
    rest = {'home': 7, 'away': 10}
    
    prediction = predictor.predict_game(home_stats, away_stats, weather, rest)
    print(f"Game Prediction: {prediction}")
