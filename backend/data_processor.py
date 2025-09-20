"""
Data processing pipeline for NFL statistics and game outcomes.
Handles data cleaning, feature engineering, and preparation for ML models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import sqlite3
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NFLDataProcessor:
    """Processes NFL data for machine learning model training and prediction."""
    
    def __init__(self, db_path: str = "data/nfl_data.db"):
        self.db_path = db_path
        self.features = [
            'home_team_win_pct', 'away_team_win_pct',
            'home_team_off_rating', 'away_team_off_rating',
            'home_team_def_rating', 'away_team_def_rating',
            'home_team_avg_points', 'away_team_avg_points',
            'home_team_avg_yards', 'away_team_avg_yards',
            'home_team_turnover_diff', 'away_team_turnover_diff',
            'home_team_rest_days', 'away_team_rest_days',
            'weather_temp', 'weather_wind', 'weather_humidity',
            'home_team_injuries', 'away_team_injuries',
            'home_team_momentum', 'away_team_momentum'
        ]
    
    def load_data(self) -> pd.DataFrame:
        """Load NFL data from database or CSV files."""
        try:
            if os.path.exists(self.db_path):
                conn = sqlite3.connect(self.db_path)
                query = """
                SELECT * FROM games 
                JOIN team_stats ON games.game_id = team_stats.game_id
                JOIN weather ON games.game_id = weather.game_id
                """
                df = pd.read_sql_query(query, conn)
                conn.close()
                logger.info(f"Loaded {len(df)} records from database")
            else:
                # Create sample data for demonstration
                df = self._create_sample_data()
                logger.info(f"Created {len(df)} sample records")
            
            return df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """Create sample NFL data for demonstration purposes."""
        np.random.seed(42)
        n_games = 1000
        
        teams = ['NE', 'KC', 'BUF', 'MIA', 'NYJ', 'PIT', 'BAL', 'CLE', 'CIN', 'HOU', 
                'IND', 'TEN', 'JAX', 'DEN', 'LV', 'LAC', 'DAL', 'PHI', 'WAS', 'NYG',
                'GB', 'MIN', 'CHI', 'DET', 'TB', 'NO', 'ATL', 'CAR', 'SF', 'SEA', 'LAR', 'ARI']
        
        data = []
        for i in range(n_games):
            home_team = np.random.choice(teams)
            away_team = np.random.choice([t for t in teams if t != home_team])
            
            # Generate realistic team statistics
            home_win_pct = np.random.beta(2, 2)  # Win percentage between 0-1
            away_win_pct = np.random.beta(2, 2)
            
            home_off_rating = np.random.normal(100, 15)
            away_off_rating = np.random.normal(100, 15)
            home_def_rating = np.random.normal(100, 15)
            away_def_rating = np.random.normal(100, 15)
            
            home_avg_points = np.random.normal(25, 5)
            away_avg_points = np.random.normal(25, 5)
            home_avg_yards = np.random.normal(350, 50)
            away_avg_yards = np.random.normal(350, 50)
            
            home_turnover_diff = np.random.normal(0, 1)
            away_turnover_diff = np.random.normal(0, 1)
            
            home_rest_days = np.random.choice([3, 7, 10, 14])
            away_rest_days = np.random.choice([3, 7, 10, 14])
            
            # Weather conditions
            weather_temp = np.random.normal(50, 20)
            weather_wind = np.random.exponential(5)
            weather_humidity = np.random.beta(2, 2) * 100
            
            # Injury reports
            home_injuries = np.random.poisson(3)
            away_injuries = np.random.poisson(3)
            
            # Team momentum (recent performance)
            home_momentum = np.random.normal(0, 1)
            away_momentum = np.random.normal(0, 1)
            
            # Calculate game outcome based on features
            home_advantage = 0.05  # Home field advantage
            win_prob = self._calculate_win_probability(
                home_win_pct, away_win_pct, home_off_rating, away_off_rating,
                home_def_rating, away_def_rating, home_advantage, home_rest_days, away_rest_days
            )
            
            home_wins = 1 if np.random.random() < win_prob else 0
            
            data.append({
                'game_id': f'game_{i:04d}',
                'season': 2023,
                'week': (i % 18) + 1,
                'home_team': home_team,
                'away_team': away_team,
                'home_team_win_pct': home_win_pct,
                'away_team_win_pct': away_win_pct,
                'home_team_off_rating': home_off_rating,
                'away_team_off_rating': away_off_rating,
                'home_team_def_rating': home_def_rating,
                'away_team_def_rating': away_def_rating,
                'home_team_avg_points': home_avg_points,
                'away_team_avg_points': away_avg_points,
                'home_team_avg_yards': home_avg_yards,
                'away_team_avg_yards': away_avg_yards,
                'home_team_turnover_diff': home_turnover_diff,
                'away_team_turnover_diff': away_turnover_diff,
                'home_team_rest_days': home_rest_days,
                'away_team_rest_days': away_rest_days,
                'weather_temp': weather_temp,
                'weather_wind': weather_wind,
                'weather_humidity': weather_humidity,
                'home_team_injuries': home_injuries,
                'away_team_injuries': away_injuries,
                'home_team_momentum': home_momentum,
                'away_team_momentum': away_momentum,
                'home_wins': home_wins,
                'date': datetime.now() - timedelta(days=np.random.randint(1, 365))
            })
        
        return pd.DataFrame(data)
    
    def _calculate_win_probability(self, home_win_pct: float, away_win_pct: float,
                                 home_off_rating: float, away_off_rating: float,
                                 home_def_rating: float, away_def_rating: float,
                                 home_advantage: float, home_rest: int, away_rest: int) -> float:
        """Calculate win probability based on team statistics."""
        # Base probability from win percentages
        base_prob = home_win_pct / (home_win_pct + away_win_pct)
        
        # Adjust for offensive/defensive ratings
        off_diff = (home_off_rating - away_off_rating) / 100
        def_diff = (away_def_rating - home_def_rating) / 100  # Higher away def rating helps home team
        
        # Rest advantage
        rest_advantage = (home_rest - away_rest) * 0.01
        
        # Combine factors
        win_prob = base_prob + home_advantage + off_diff * 0.1 + def_diff * 0.1 + rest_advantage
        
        return np.clip(win_prob, 0.1, 0.9)  # Keep probabilities reasonable
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the dataset."""
        logger.info("Cleaning data...")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['game_id'])
        
        # Handle missing values
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        # Remove outliers (beyond 3 standard deviations)
        for col in numeric_columns:
            if col not in ['home_wins', 'game_id', 'season', 'week']:
                mean = df[col].mean()
                std = df[col].std()
                df = df[np.abs(df[col] - mean) <= 3 * std]
        
        logger.info(f"Data cleaned. Remaining records: {len(df)}")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create additional features for better model performance."""
        logger.info("Engineering features...")
        
        # Create relative strength features
        df['off_rating_diff'] = df['home_team_off_rating'] - df['away_team_off_rating']
        df['def_rating_diff'] = df['away_team_def_rating'] - df['home_team_def_rating']  # Inverted for home team advantage
        df['win_pct_diff'] = df['home_team_win_pct'] - df['away_team_win_pct']
        df['points_diff'] = df['home_team_avg_points'] - df['away_team_avg_points']
        df['yards_diff'] = df['home_team_avg_yards'] - df['away_team_avg_yards']
        df['turnover_diff'] = df['home_team_turnover_diff'] - df['away_team_turnover_diff']
        df['rest_advantage'] = df['home_team_rest_days'] - df['away_team_rest_days']
        df['injury_advantage'] = df['away_team_injuries'] - df['home_team_injuries']  # More away injuries = home advantage
        df['momentum_diff'] = df['home_team_momentum'] - df['away_team_momentum']
        
        # Weather impact features
        df['weather_impact'] = (
            np.abs(df['weather_temp'] - 70) * 0.01 +  # Temperature deviation from optimal
            df['weather_wind'] * 0.02 +  # Wind impact
            np.abs(df['weather_humidity'] - 50) * 0.005  # Humidity deviation
        )
        
        # Team strength composite
        df['home_strength'] = (
            df['home_team_win_pct'] * 0.3 +
            (df['home_team_off_rating'] / 100) * 0.3 +
            (df['home_team_def_rating'] / 100) * 0.2 +
            (df['home_team_avg_points'] / 30) * 0.2
        )
        
        df['away_strength'] = (
            df['away_team_win_pct'] * 0.3 +
            (df['away_team_off_rating'] / 100) * 0.3 +
            (df['away_team_def_rating'] / 100) * 0.2 +
            (df['away_team_avg_points'] / 30) * 0.2
        )
        
        df['strength_diff'] = df['home_strength'] - df['away_strength']
        
        logger.info("Feature engineering completed")
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training."""
        logger.info("Preparing training data...")
        
        # Select features for training
        feature_columns = [
            'off_rating_diff', 'def_rating_diff', 'win_pct_diff', 'points_diff',
            'yards_diff', 'turnover_diff', 'rest_advantage', 'injury_advantage',
            'momentum_diff', 'weather_impact', 'strength_diff',
            'home_team_rest_days', 'away_team_rest_days',
            'weather_temp', 'weather_wind', 'weather_humidity'
        ]
        
        X = df[feature_columns].values
        y = df['home_wins'].values
        
        # Normalize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        logger.info(f"Training data prepared: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        return X_scaled, y
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_nfl_data.csv"):
        """Save processed data to file."""
        filepath = os.path.join("data/processed", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df.to_csv(filepath, index=False)
        logger.info(f"Processed data saved to {filepath}")
    
    def get_team_stats(self, team: str, df: pd.DataFrame) -> Dict:
        """Get current statistics for a specific team."""
        team_data = df[(df['home_team'] == team) | (df['away_team'] == team)]
        
        if team_data.empty:
            return {}
        
        # Calculate average stats for the team
        home_games = team_data[team_data['home_team'] == team]
        away_games = team_data[team_data['away_team'] == team]
        
        stats = {
            'win_pct': (home_games['home_wins'].sum() + (len(away_games) - away_games['home_wins'].sum())) / len(team_data),
            'avg_points': (home_games['home_team_avg_points'].mean() + away_games['away_team_avg_points'].mean()) / 2,
            'avg_yards': (home_games['home_team_avg_yards'].mean() + away_games['away_team_avg_yards'].mean()) / 2,
            'off_rating': (home_games['home_team_off_rating'].mean() + away_games['away_team_off_rating'].mean()) / 2,
            'def_rating': (home_games['home_team_def_rating'].mean() + away_games['away_team_def_rating'].mean()) / 2,
            'turnover_diff': (home_games['home_team_turnover_diff'].mean() + away_games['away_team_turnover_diff'].mean()) / 2,
            'momentum': (home_games['home_team_momentum'].mean() + away_games['away_team_momentum'].mean()) / 2
        }
        
        return stats

if __name__ == "__main__":
    # Example usage
    processor = NFLDataProcessor()
    df = processor.load_data()
    df = processor.clean_data(df)
    df = processor.engineer_features(df)
    X, y = processor.prepare_training_data(df)
    processor.save_processed_data(df)
    
    print(f"Data processing complete!")
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {len(processor.features)}")
    print(f"Sample team stats: {processor.get_team_stats('KC', df)}")
