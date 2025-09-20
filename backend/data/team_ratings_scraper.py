"""
NFL Team Ratings Data Scraper
Fetches offensive and defensive ratings for all NFL teams from the last 5 years.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple
import json
import os
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

class TeamRating(Base):
    """Database model for team ratings."""
    __tablename__ = 'team_ratings'
    
    id = Column(Integer, primary_key=True)
    team = Column(String(10), nullable=False)
    season = Column(Integer, nullable=False)
    week = Column(Integer, nullable=True)  # None for season totals
    offensive_rating = Column(Float, nullable=False)
    defensive_rating = Column(Float, nullable=False)
    points_for = Column(Float, nullable=True)
    points_against = Column(Float, nullable=True)
    yards_for = Column(Float, nullable=True)
    yards_against = Column(Float, nullable=True)
    turnovers_for = Column(Integer, nullable=True)
    turnovers_against = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    
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
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class TeamRatingsScraper:
    """Scraper for NFL team ratings data."""
    
    def __init__(self, db_path: str = "team_ratings.db"):
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Team name mappings
        self.team_mappings = {
            'ARI': ['Arizona Cardinals', 'Cardinals'],
            'ATL': ['Atlanta Falcons', 'Falcons'],
            'BAL': ['Baltimore Ravens', 'Ravens'],
            'BUF': ['Buffalo Bills', 'Bills'],
            'CAR': ['Carolina Panthers', 'Panthers'],
            'CHI': ['Chicago Bears', 'Bears'],
            'CIN': ['Cincinnati Bengals', 'Bengals'],
            'CLE': ['Cleveland Browns', 'Browns'],
            'DAL': ['Dallas Cowboys', 'Cowboys'],
            'DEN': ['Denver Broncos', 'Broncos'],
            'DET': ['Detroit Lions', 'Lions'],
            'GB': ['Green Bay Packers', 'Packers'],
            'HOU': ['Houston Texans', 'Texans'],
            'IND': ['Indianapolis Colts', 'Colts'],
            'JAX': ['Jacksonville Jaguars', 'Jaguars'],
            'KC': ['Kansas City Chiefs', 'Chiefs'],
            'LV': ['Las Vegas Raiders', 'Raiders', 'Oakland Raiders'],
            'LAC': ['Los Angeles Chargers', 'Chargers'],
            'LAR': ['Los Angeles Rams', 'Rams'],
            'MIA': ['Miami Dolphins', 'Dolphins'],
            'MIN': ['Minnesota Vikings', 'Vikings'],
            'NE': ['New England Patriots', 'Patriots'],
            'NO': ['New Orleans Saints', 'Saints'],
            'NYG': ['New York Giants', 'Giants'],
            'NYJ': ['New York Jets', 'Jets'],
            'PHI': ['Philadelphia Eagles', 'Eagles'],
            'PIT': ['Pittsburgh Steelers', 'Steelers'],
            'SF': ['San Francisco 49ers', '49ers'],
            'SEA': ['Seattle Seahawks', 'Seahawks'],
            'TB': ['Tampa Bay Buccaneers', 'Buccaneers'],
            'TEN': ['Tennessee Titans', 'Titans'],
            'WAS': ['Washington Commanders', 'Commanders', 'Washington Football Team']
        }
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def scrape_pro_football_reference(self, season: int) -> List[Dict]:
        """Scrape team ratings from Pro Football Reference."""
        logger.info(f"Scraping Pro Football Reference for season {season}")
        
        try:
            url = f"https://www.pro-football-reference.com/years/{season}/"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            teams_data = []
            
            # Find team stats table
            team_stats_table = soup.find('table', {'id': 'team_stats'})
            if not team_stats_table:
                logger.warning(f"No team stats table found for season {season}")
                return teams_data
            
            rows = team_stats_table.find('tbody').find_all('tr')
            
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) < 20:  # Ensure we have enough columns
                    continue
                
                try:
                    team_name = cells[0].get_text().strip()
                    team_code = self._get_team_code(team_name)
                    
                    if not team_code:
                        continue
                    
                    # Extract key statistics
                    points_for = self._safe_float(cells[2].get_text())
                    points_against = self._safe_float(cells[3].get_text())
                    yards_for = self._safe_float(cells[4].get_text())
                    yards_against = self._safe_float(cells[5].get_text())
                    turnovers_for = self._safe_int(cells[6].get_text())
                    turnovers_against = self._safe_int(cells[7].get_text())
                    
                    # Calculate ratings (simplified DVOA-like calculation)
                    offensive_rating = self._calculate_offensive_rating(
                        points_for, yards_for, turnovers_for, season
                    )
                    defensive_rating = self._calculate_defensive_rating(
                        points_against, yards_against, turnovers_against, season
                    )
                    
                    team_data = {
                        'team': team_code,
                        'season': season,
                        'week': None,  # Season totals
                        'offensive_rating': offensive_rating,
                        'defensive_rating': defensive_rating,
                        'points_for': points_for,
                        'points_against': points_against,
                        'yards_for': yards_for,
                        'yards_against': yards_against,
                        'turnovers_for': turnovers_for,
                        'turnovers_against': turnovers_against
                    }
                    
                    teams_data.append(team_data)
                    
                except Exception as e:
                    logger.warning(f"Error processing row for season {season}: {e}")
                    continue
            
            logger.info(f"Successfully scraped {len(teams_data)} teams for season {season}")
            return teams_data
            
        except Exception as e:
            logger.error(f"Error scraping Pro Football Reference for season {season}: {e}")
            return []
    
    def scrape_espn_ratings(self, season: int) -> List[Dict]:
        """Scrape team ratings from ESPN."""
        logger.info(f"Scraping ESPN for season {season}")
        
        try:
            # ESPN team stats URL
            url = f"https://www.espn.com/nfl/stats/team/_/season/{season}/seasontype/2"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            teams_data = []
            
            # This is a simplified version - ESPN's structure may vary
            # In practice, you might need to use their API or handle dynamic content
            
            logger.info(f"ESPN scraping completed for season {season}")
            return teams_data
            
        except Exception as e:
            logger.error(f"Error scraping ESPN for season {season}: {e}")
            return []
    
    def _get_team_code(self, team_name: str) -> Optional[str]:
        """Convert team name to team code."""
        team_name = team_name.strip()
        
        for code, names in self.team_mappings.items():
            for name in names:
                if name.lower() in team_name.lower() or team_name.lower() in name.lower():
                    return code
        
        return None
    
    def _safe_float(self, value: str) -> Optional[float]:
        """Safely convert string to float."""
        try:
            return float(value.replace(',', ''))
        except (ValueError, AttributeError):
            return None
    
    def _safe_int(self, value: str) -> Optional[int]:
        """Safely convert string to int."""
        try:
            return int(value.replace(',', ''))
        except (ValueError, AttributeError):
            return None
    
    def _calculate_offensive_rating(self, points_for: float, yards_for: float, 
                                  turnovers_for: int, season: int) -> float:
        """Calculate offensive rating based on key metrics."""
        if not all([points_for, yards_for, turnovers_for]):
            return 100.0  # Default rating
        
        # Base rating calculation (simplified DVOA-like)
        # Higher points and yards = better rating
        # Fewer turnovers = better rating
        
        points_factor = (points_for / 25.0) * 20  # Normalize around 25 points
        yards_factor = (yards_for / 350.0) * 20   # Normalize around 350 yards
        turnover_factor = max(0, 20 - (turnovers_for * 2))  # Penalty for turnovers
        
        rating = 100 + points_factor + yards_factor + turnover_factor - 60
        return max(50, min(150, rating))  # Clamp between 50-150
    
    def _calculate_defensive_rating(self, points_against: float, yards_against: float,
                                  turnovers_against: int, season: int) -> float:
        """Calculate defensive rating based on key metrics."""
        if not all([points_against, yards_against, turnovers_against]):
            return 100.0  # Default rating
        
        # For defense, lower is better
        points_factor = max(0, 20 - (points_against / 25.0) * 20)
        yards_factor = max(0, 20 - (yards_against / 350.0) * 20)
        turnover_factor = (turnovers_against * 2)  # Bonus for forcing turnovers
        
        rating = 100 + points_factor + yards_factor + turnover_factor - 60
        return max(50, min(150, rating))  # Clamp between 50-150
    
    def scrape_all_seasons(self, start_year: int = 2020, end_year: int = 2024) -> None:
        """Scrape data for all seasons."""
        logger.info(f"Starting to scrape team ratings from {start_year} to {end_year}")
        
        all_data = []
        
        for season in range(start_year, end_year + 1):
            logger.info(f"Scraping season {season}...")
            
            # Scrape from Pro Football Reference
            pfr_data = self.scrape_pro_football_reference(season)
            all_data.extend(pfr_data)
            
            # Add delay to be respectful
            time.sleep(2)
        
        # Save to database
        self._save_to_database(all_data)
        logger.info(f"Completed scraping. Total records: {len(all_data)}")
    
    def _save_to_database(self, data: List[Dict]) -> None:
        """Save scraped data to database."""
        logger.info(f"Saving {len(data)} records to database")
        
        for record in data:
            try:
                # Check if record already exists
                existing = self.session.query(TeamRating).filter_by(
                    team=record['team'],
                    season=record['season'],
                    week=record['week']
                ).first()
                
                if existing:
                    # Update existing record
                    for key, value in record.items():
                        if key not in ['team', 'season', 'week']:
                            setattr(existing, key, value)
                else:
                    # Create new record
                    team_rating = TeamRating(**record)
                    self.session.add(team_rating)
                
            except Exception as e:
                logger.error(f"Error saving record {record}: {e}")
                continue
        
        self.session.commit()
        logger.info("Database save completed")
    
    def get_team_ratings(self, team: str, season: int = None, 
                        week: int = None) -> List[Dict]:
        """Get team ratings from database."""
        query = self.session.query(TeamRating).filter_by(team=team)
        
        if season:
            query = query.filter_by(season=season)
        if week is not None:
            query = query.filter_by(week=week)
        
        ratings = query.order_by(TeamRating.season.desc(), TeamRating.week.desc()).all()
        return [rating.to_dict() for rating in ratings]
    
    def get_latest_ratings(self, team: str) -> Optional[Dict]:
        """Get the latest ratings for a team."""
        latest = self.session.query(TeamRating).filter_by(team=team).order_by(
            TeamRating.season.desc(), TeamRating.week.desc()
        ).first()
        
        return latest.to_dict() if latest else None
    
    def get_all_teams_latest_ratings(self) -> Dict[str, Dict]:
        """Get latest ratings for all teams."""
        teams = {}
        
        for team_code in self.team_mappings.keys():
            latest = self.get_latest_ratings(team_code)
            if latest:
                teams[team_code] = latest
        
        return teams
    
    def update_current_season(self) -> None:
        """Update ratings for the current season."""
        current_year = datetime.now().year
        logger.info(f"Updating ratings for current season {current_year}")
        
        # Scrape current season data
        current_data = self.scrape_pro_football_reference(current_year)
        self._save_to_database(current_data)
        
        logger.info("Current season update completed")

def main():
    """Main function to run the scraper."""
    scraper = TeamRatingsScraper()
    
    # Scrape historical data (2020-2024)
    scraper.scrape_all_seasons(2020, 2024)
    
    # Update current season
    scraper.update_current_season()
    
    # Display sample results
    print("\n🏈 Sample Team Ratings:")
    print("=" * 50)
    
    sample_teams = ['KC', 'BUF', 'MIA', 'NE']
    for team in sample_teams:
        latest = scraper.get_latest_ratings(team)
        if latest:
            print(f"{team}: Off={latest['offensive_rating']:.1f}, Def={latest['defensive_rating']:.1f}")

if __name__ == "__main__":
    main()
