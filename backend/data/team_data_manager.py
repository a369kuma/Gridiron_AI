"""
Team Data Manager for Gridiron AI
Manages real-time team ratings and integrates with the prediction model.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import json

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.nfl_api_client import NFLAPIClient
from data.team_ratings_scraper import TeamRatingsScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class TeamData(Base):
    """Database model for comprehensive team data."""
    __tablename__ = 'team_data'
    
    id = Column(Integer, primary_key=True)
    team = Column(String(10), nullable=False)
    season = Column(Integer, nullable=False)
    week = Column(Integer, nullable=True)
    
    # Core ratings
    offensive_rating = Column(Float, nullable=False)
    defensive_rating = Column(Float, nullable=False)
    
    # Statistical data
    points_for = Column(Float, nullable=False)
    points_against = Column(Float, nullable=False)
    yards_for = Column(Float, nullable=False)
    yards_against = Column(Float, nullable=False)
    turnovers_for = Column(Integer, nullable=False)
    turnovers_against = Column(Integer, nullable=False)
    
    # Record data
    wins = Column(Integer, nullable=False)
    losses = Column(Integer, nullable=False)
    ties = Column(Integer, nullable=False, default=0)
    win_percentage = Column(Float, nullable=False)
    
    # Advanced metrics
    strength_of_schedule = Column(Float, nullable=True)
    pythagorean_wins = Column(Float, nullable=True)
    point_differential = Column(Float, nullable=True)
    yard_differential = Column(Float, nullable=True)
    turnover_differential = Column(Float, nullable=True)
    
    # Metadata
    data_source = Column(String(50), nullable=False, default='api')
    last_updated = Column(DateTime, default=datetime.utcnow)
    is_current = Column(Boolean, default=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'team': self.team,
            'season': self.season,
            'week': self.week,
            'offensive_rating': self.offensive_rating,
            'defensive_rating': self.defensive_rating,
            'points_for': self.points_for,
            'points_against': self.points_against,
            'yards_for': self.yards_for,
            'yards_against': self.yards_against,
            'turnovers_for': self.turnovers_for,
            'turnovers_against': self.turnovers_against,
            'wins': self.wins,
            'losses': self.losses,
            'ties': self.ties,
            'win_percentage': self.win_percentage,
            'strength_of_schedule': self.strength_of_schedule,
            'pythagorean_wins': self.pythagorean_wins,
            'point_differential': self.point_differential,
            'yard_differential': self.yard_differential,
            'turnover_differential': self.turnover_differential,
            'data_source': self.data_source,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'is_current': self.is_current
        }

class TeamDataManager:
    """Manages team data collection, storage, and retrieval."""
    
    def __init__(self, db_path: str = "team_data.db"):
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Initialize data sources
        self.api_client = NFLAPIClient()
        self.scraper = TeamRatingsScraper()
        
        # Team list
        self.teams = [
            'ARI', 'ATL', 'BAL', 'BUF', 'CAR', 'CHI', 'CIN', 'CLE', 'DAL', 'DEN',
            'DET', 'GB', 'HOU', 'IND', 'JAX', 'KC', 'LV', 'LAC', 'LAR', 'MIA',
            'MIN', 'NE', 'NO', 'NYG', 'NYJ', 'PHI', 'PIT', 'SF', 'SEA', 'TB',
            'TEN', 'WAS'
        ]
    
    def update_all_team_data(self, season: int = 2024) -> Dict[str, bool]:
        """Update data for all teams."""
        logger.info(f"Updating team data for season {season}")
        
        results = {}
        
        # Try API first
        try:
            api_data = self.api_client.fetch_all_team_data(season)
            if api_data:
                self._save_team_data(api_data, 'api')
                results['api'] = True
                logger.info("Successfully updated data via API")
            else:
                results['api'] = False
        except Exception as e:
            logger.error(f"API update failed: {e}")
            results['api'] = False
        
        # Fallback to scraping
        if not results.get('api', False):
            try:
                scraper_data = self.scraper.scrape_pro_football_reference(season)
                if scraper_data:
                    self._save_team_data(scraper_data, 'scraper')
                    results['scraper'] = True
                    logger.info("Successfully updated data via scraper")
                else:
                    results['scraper'] = False
            except Exception as e:
                logger.error(f"Scraper update failed: {e}")
                results['scraper'] = False
        
        # Generate fallback data if both fail
        if not any(results.values()):
            logger.warning("Both API and scraper failed, generating fallback data")
            fallback_data = self._generate_fallback_data(season)
            self._save_team_data(fallback_data, 'fallback')
            results['fallback'] = True
        
        return results
    
    def _save_team_data(self, data: List[Dict], source: str) -> None:
        """Save team data to database."""
        logger.info(f"Saving {len(data)} records from {source}")
        
        for record in data:
            try:
                # Calculate additional metrics
                record = self._calculate_advanced_metrics(record)
                record['data_source'] = source
                record['last_updated'] = datetime.utcnow()
                record['is_current'] = True
                
                # Ensure win_percentage is not None
                if record.get('win_percentage') is None:
                    wins = record.get('wins', 0)
                    losses = record.get('losses', 0)
                    ties = record.get('ties', 0)
                    total_games = wins + losses + ties
                    record['win_percentage'] = wins / total_games if total_games > 0 else 0.0
                
                # Check if record exists
                existing = self.session.query(TeamData).filter_by(
                    team=record['team'],
                    season=record['season'],
                    week=record.get('week')
                ).first()
                
                if existing:
                    # Update existing record
                    for key, value in record.items():
                        if hasattr(existing, key):
                            setattr(existing, key, value)
                else:
                    # Create new record
                    team_data = TeamData(**record)
                    self.session.add(team_data)
                
            except Exception as e:
                logger.error(f"Error saving record {record}: {e}")
                self.session.rollback()  # Rollback on error
                continue
        
        try:
            self.session.commit()
            logger.info("Database save completed")
        except Exception as e:
            logger.error(f"Error committing to database: {e}")
            self.session.rollback()
    
    def _calculate_advanced_metrics(self, record: Dict) -> Dict:
        """Calculate advanced metrics for team data."""
        # Point differential
        record['point_differential'] = record.get('points_for', 0) - record.get('points_against', 0)
        
        # Yard differential
        record['yard_differential'] = record.get('yards_for', 0) - record.get('yards_against', 0)
        
        # Turnover differential
        record['turnover_differential'] = record.get('turnovers_against', 0) - record.get('turnovers_for', 0)
        
        # Pythagorean wins (simplified)
        points_for = record.get('points_for', 1)
        points_against = record.get('points_against', 1)
        games = record.get('wins', 0) + record.get('losses', 0) + record.get('ties', 0)
        
        if games > 0 and points_for > 0 and points_against > 0:
            pythagorean_win_pct = (points_for ** 2.37) / (points_for ** 2.37 + points_against ** 2.37)
            record['pythagorean_wins'] = pythagorean_win_pct * games
        else:
            record['pythagorean_wins'] = 0
        
        # Strength of schedule (placeholder - would need opponent data)
        record['strength_of_schedule'] = 0.0
        
        return record
    
    def _generate_fallback_data(self, season: int) -> List[Dict]:
        """Generate fallback data when external sources fail."""
        logger.info("Generating fallback team data")
        
        fallback_data = []
        
        for team in self.teams:
            # Generate realistic but varied data
            base_off_rating = np.random.normal(100, 15)
            base_def_rating = np.random.normal(100, 15)
            
            # Add some team-specific adjustments (simplified)
            team_adjustments = {
                'KC': {'off': 15, 'def': -5},   # Strong offense
                'BUF': {'off': 10, 'def': 5},   # Balanced
                'MIA': {'off': 8, 'def': -8},   # Offense-heavy
                'NE': {'off': -5, 'def': 10},   # Defense-heavy
                'BAL': {'off': 5, 'def': 8},    # Balanced
                'CIN': {'off': 12, 'def': -3},  # Offense-heavy
                'SF': {'off': 8, 'def': 8},     # Balanced
                'PHI': {'off': 10, 'def': 5},   # Balanced
                'DAL': {'off': 8, 'def': 3},    # Slightly offense
                'GB': {'off': 12, 'def': -5},   # Offense-heavy
            }
            
            adj = team_adjustments.get(team, {'off': 0, 'def': 0})
            off_rating = max(50, min(150, base_off_rating + adj['off']))
            def_rating = max(50, min(150, base_def_rating + adj['def']))
            
            # Generate correlated stats
            points_for = max(15, min(40, (off_rating - 50) * 0.3 + 20))
            points_against = max(15, min(40, (def_rating - 50) * 0.3 + 20))
            yards_for = max(250, min(500, (off_rating - 50) * 2 + 300))
            yards_against = max(250, min(500, (def_rating - 50) * 2 + 300))
            
            wins = np.random.randint(5, 13)
            losses = 17 - wins
            ties = np.random.randint(0, 2)
            
            total_games = wins + losses + ties
            win_percentage = wins / total_games if total_games > 0 else 0.0
            
            team_data = {
                'team': team,
                'season': season,
                'week': None,
                'offensive_rating': off_rating,
                'defensive_rating': def_rating,
                'points_for': points_for,
                'points_against': points_against,
                'yards_for': yards_for,
                'yards_against': yards_against,
                'turnovers_for': np.random.randint(15, 35),
                'turnovers_against': np.random.randint(15, 35),
                'wins': wins,
                'losses': losses,
                'ties': ties,
                'win_percentage': win_percentage
            }
            
            fallback_data.append(team_data)
        
        return fallback_data
    
    def get_team_current_stats(self, team: str) -> Optional[Dict]:
        """Get current stats for a team."""
        current = self.session.query(TeamData).filter_by(
            team=team,
            is_current=True
        ).order_by(TeamData.season.desc(), TeamData.week.desc()).first()
        
        return current.to_dict() if current else None
    
    def get_all_teams_current_stats(self) -> Dict[str, Dict]:
        """Get current stats for all teams."""
        teams_stats = {}
        
        for team in self.teams:
            stats = self.get_team_current_stats(team)
            if stats:
                teams_stats[team] = stats
        
        return teams_stats
    
    def get_team_historical_stats(self, team: str, seasons: int = 5) -> List[Dict]:
        """Get historical stats for a team."""
        current_year = datetime.now().year
        start_year = current_year - seasons + 1
        
        historical = self.session.query(TeamData).filter(
            TeamData.team == team,
            TeamData.season >= start_year,
            TeamData.week.is_(None)  # Season totals only
        ).order_by(TeamData.season.desc()).all()
        
        return [record.to_dict() for record in historical]
    
    def get_team_average_ratings(self, team: str, seasons: int = 3) -> Dict[str, float]:
        """Get average ratings for a team over multiple seasons."""
        historical = self.get_team_historical_stats(team, seasons)
        
        if not historical:
            return {'offensive_rating': 100.0, 'defensive_rating': 100.0}
        
        avg_off = sum(record['offensive_rating'] for record in historical) / len(historical)
        avg_def = sum(record['defensive_rating'] for record in historical) / len(historical)
        
        return {
            'offensive_rating': avg_off,
            'defensive_rating': avg_def
        }
    
    def update_team_ratings_for_model(self) -> Dict[str, Dict]:
        """Get team ratings formatted for the prediction model."""
        logger.info("Updating team ratings for model")
        
        # First, try to update data
        self.update_all_team_data()
        
        # Get current stats for all teams
        teams_stats = self.get_all_teams_current_stats()
        
        # Format for model
        model_ratings = {}
        
        for team, stats in teams_stats.items():
            model_ratings[team] = {
                'win_pct': stats['win_percentage'],
                'off_rating': stats['offensive_rating'],
                'def_rating': stats['defensive_rating'],
                'avg_points': stats['points_for'],
                'avg_yards': stats['yards_for'],
                'turnover_diff': stats['turnover_differential'],
                'injuries': np.random.randint(0, 8),  # Placeholder - would need injury data
                'momentum': self._calculate_momentum(team, stats)
            }
        
        return model_ratings
    
    def _calculate_momentum(self, team: str, stats: Dict) -> float:
        """Calculate team momentum based on recent performance."""
        # Simplified momentum calculation
        win_pct = stats['win_percentage']
        point_diff = stats['point_differential']
        
        # Momentum based on win percentage and point differential
        momentum = (win_pct - 0.5) * 2  # -1 to 1 range
        momentum += (point_diff / 100) * 0.5  # Adjust for point differential
        
        return max(-1, min(1, momentum))  # Clamp between -1 and 1
    
    def export_team_data(self, filename: str = None) -> str:
        """Export team data to JSON file."""
        if not filename:
            filename = f"team_data_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        all_data = self.session.query(TeamData).all()
        data_list = [record.to_dict() for record in all_data]
        
        with open(filename, 'w') as f:
            json.dump(data_list, f, indent=2, default=str)
        
        logger.info(f"Exported {len(data_list)} records to {filename}")
        return filename

def main():
    """Main function to test the team data manager."""
    manager = TeamDataManager()
    
    print("🏈 Gridiron AI - Team Data Manager")
    print("=" * 50)
    
    # Update team data
    print("Updating team data...")
    results = manager.update_all_team_data(2024)
    print(f"Update results: {results}")
    
    # Get sample team stats
    print("\n📊 Sample Team Stats:")
    sample_teams = ['KC', 'BUF', 'MIA', 'NE']
    
    for team in sample_teams:
        stats = manager.get_team_current_stats(team)
        if stats:
            print(f"{team}: Off={stats['offensive_rating']:.1f}, Def={stats['defensive_rating']:.1f}, W-L={stats['wins']}-{stats['losses']}")
    
    # Get model-ready ratings
    print("\n🤖 Model-Ready Ratings:")
    model_ratings = manager.update_team_ratings_for_model()
    
    for team in sample_teams:
        if team in model_ratings:
            ratings = model_ratings[team]
            print(f"{team}: Off={ratings['off_rating']:.1f}, Def={ratings['def_rating']:.1f}, Win%={ratings['win_pct']:.1%}")

if __name__ == "__main__":
    main()
