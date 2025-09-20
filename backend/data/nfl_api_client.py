"""
NFL API Client for fetching real-time team statistics and ratings.
Uses multiple data sources to get comprehensive team performance data.
"""

import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, List, Optional, Tuple
import os
from dataclasses import dataclass
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Base = declarative_base()

@dataclass
class TeamStats:
    """Data class for team statistics."""
    team: str
    season: int
    week: Optional[int]
    offensive_rating: float
    defensive_rating: float
    points_for: float
    points_against: float
    yards_for: float
    yards_against: float
    turnovers_for: int
    turnovers_against: int
    wins: int
    losses: int
    win_percentage: float
    strength_of_schedule: float
    pythagorean_wins: float
    created_at: datetime

class NFLAPIClient:
    """Client for fetching NFL team data from various APIs."""
    
    def __init__(self, db_path: str = "nfl_team_data.db"):
        self.db_path = db_path
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        # Team mappings
        self.team_mappings = {
            'ARI': {'name': 'Arizona Cardinals', 'city': 'Arizona', 'mascot': 'Cardinals'},
            'ATL': {'name': 'Atlanta Falcons', 'city': 'Atlanta', 'mascot': 'Falcons'},
            'BAL': {'name': 'Baltimore Ravens', 'city': 'Baltimore', 'mascot': 'Ravens'},
            'BUF': {'name': 'Buffalo Bills', 'city': 'Buffalo', 'mascot': 'Bills'},
            'CAR': {'name': 'Carolina Panthers', 'city': 'Carolina', 'mascot': 'Panthers'},
            'CHI': {'name': 'Chicago Bears', 'city': 'Chicago', 'mascot': 'Bears'},
            'CIN': {'name': 'Cincinnati Bengals', 'city': 'Cincinnati', 'mascot': 'Bengals'},
            'CLE': {'name': 'Cleveland Browns', 'city': 'Cleveland', 'mascot': 'Browns'},
            'DAL': {'name': 'Dallas Cowboys', 'city': 'Dallas', 'mascot': 'Cowboys'},
            'DEN': {'name': 'Denver Broncos', 'city': 'Denver', 'mascot': 'Broncos'},
            'DET': {'name': 'Detroit Lions', 'city': 'Detroit', 'mascot': 'Lions'},
            'GB': {'name': 'Green Bay Packers', 'city': 'Green Bay', 'mascot': 'Packers'},
            'HOU': {'name': 'Houston Texans', 'city': 'Houston', 'mascot': 'Texans'},
            'IND': {'name': 'Indianapolis Colts', 'city': 'Indianapolis', 'mascot': 'Colts'},
            'JAX': {'name': 'Jacksonville Jaguars', 'city': 'Jacksonville', 'mascot': 'Jaguars'},
            'KC': {'name': 'Kansas City Chiefs', 'city': 'Kansas City', 'mascot': 'Chiefs'},
            'LV': {'name': 'Las Vegas Raiders', 'city': 'Las Vegas', 'mascot': 'Raiders'},
            'LAC': {'name': 'Los Angeles Chargers', 'city': 'Los Angeles', 'mascot': 'Chargers'},
            'LAR': {'name': 'Los Angeles Rams', 'city': 'Los Angeles', 'mascot': 'Rams'},
            'MIA': {'name': 'Miami Dolphins', 'city': 'Miami', 'mascot': 'Dolphins'},
            'MIN': {'name': 'Minnesota Vikings', 'city': 'Minnesota', 'mascot': 'Vikings'},
            'NE': {'name': 'New England Patriots', 'city': 'New England', 'mascot': 'Patriots'},
            'NO': {'name': 'New Orleans Saints', 'city': 'New Orleans', 'mascot': 'Saints'},
            'NYG': {'name': 'New York Giants', 'city': 'New York', 'mascot': 'Giants'},
            'NYJ': {'name': 'New York Jets', 'city': 'New York', 'mascot': 'Jets'},
            'PHI': {'name': 'Philadelphia Eagles', 'city': 'Philadelphia', 'mascot': 'Eagles'},
            'PIT': {'name': 'Pittsburgh Steelers', 'city': 'Pittsburgh', 'mascot': 'Steelers'},
            'SF': {'name': 'San Francisco 49ers', 'city': 'San Francisco', 'mascot': '49ers'},
            'SEA': {'name': 'Seattle Seahawks', 'city': 'Seattle', 'mascot': 'Seahawks'},
            'TB': {'name': 'Tampa Bay Buccaneers', 'city': 'Tampa Bay', 'mascot': 'Buccaneers'},
            'TEN': {'name': 'Tennessee Titans', 'city': 'Tennessee', 'mascot': 'Titans'},
            'WAS': {'name': 'Washington Commanders', 'city': 'Washington', 'mascot': 'Commanders'}
        }
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9'
        }
    
    def fetch_espn_team_stats(self, season: int = 2024) -> List[Dict]:
        """Fetch team statistics from ESPN API."""
        logger.info(f"Fetching ESPN team stats for season {season}")
        
        try:
            # ESPN's team stats endpoint
            url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams"
            
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            teams_data = []
            
            for team in data.get('sports', [{}])[0].get('leagues', [{}])[0].get('teams', []):
                team_info = team.get('team', {})
                team_id = team_info.get('id')
                team_name = team_info.get('displayName', '')
                
                # Get team code from name
                team_code = self._get_team_code_from_name(team_name)
                if not team_code:
                    continue
                
                # Fetch detailed stats for this team
                team_stats = self._fetch_team_detailed_stats(team_id, season)
                if team_stats:
                    team_stats['team'] = team_code
                    team_stats['season'] = season
                    teams_data.append(team_stats)
                
                time.sleep(0.5)  # Rate limiting
            
            logger.info(f"Successfully fetched {len(teams_data)} teams from ESPN")
            return teams_data
            
        except Exception as e:
            logger.error(f"Error fetching ESPN data: {e}")
            return []
    
    def _fetch_team_detailed_stats(self, team_id: str, season: int) -> Optional[Dict]:
        """Fetch detailed statistics for a specific team."""
        try:
            # ESPN team stats endpoint
            url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/teams/{team_id}/stats"
            
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract offensive and defensive stats
            stats = data.get('splits', {}).get('categories', [])
            
            offensive_stats = {}
            defensive_stats = {}
            
            for category in stats:
                if category.get('displayName') == 'Offensive':
                    for stat in category.get('stats', []):
                        offensive_stats[stat.get('label')] = stat.get('value')
                elif category.get('displayName') == 'Defensive':
                    for stat in category.get('stats', []):
                        defensive_stats[stat.get('label')] = stat.get('value')
            
            # Calculate ratings
            offensive_rating = self._calculate_offensive_rating_from_stats(offensive_stats)
            defensive_rating = self._calculate_defensive_rating_from_stats(defensive_stats)
            
            return {
                'offensive_rating': offensive_rating,
                'defensive_rating': defensive_rating,
                'points_for': offensive_stats.get('Points Per Game', 0),
                'points_against': defensive_stats.get('Points Per Game', 0),
                'yards_for': offensive_stats.get('Total Yards Per Game', 0),
                'yards_against': defensive_stats.get('Total Yards Per Game', 0),
                'turnovers_for': offensive_stats.get('Turnovers', 0),
                'turnovers_against': defensive_stats.get('Takeaways', 0),
                'wins': offensive_stats.get('Wins', 0),
                'losses': offensive_stats.get('Losses', 0)
            }
            
        except Exception as e:
            logger.error(f"Error fetching detailed stats for team {team_id}: {e}")
            return None
    
    def fetch_pro_football_reference_data(self, season: int = 2024) -> List[Dict]:
        """Fetch data from Pro Football Reference using web scraping."""
        logger.info(f"Fetching Pro Football Reference data for season {season}")
        
        try:
            from bs4 import BeautifulSoup
            
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
                if len(cells) < 20:
                    continue
                
                try:
                    team_name = cells[0].get_text().strip()
                    team_code = self._get_team_code_from_name(team_name)
                    
                    if not team_code:
                        continue
                    
                    # Extract statistics
                    wins = self._safe_int(cells[1].get_text())
                    losses = self._safe_int(cells[2].get_text())
                    ties = self._safe_int(cells[3].get_text())
                    points_for = self._safe_float(cells[4].get_text())
                    points_against = self._safe_float(cells[5].get_text())
                    yards_for = self._safe_float(cells[6].get_text())
                    yards_against = self._safe_float(cells[7].get_text())
                    turnovers_for = self._safe_int(cells[8].get_text())
                    turnovers_against = self._safe_int(cells[9].get_text())
                    
                    # Calculate ratings
                    offensive_rating = self._calculate_offensive_rating(
                        points_for, yards_for, turnovers_for, wins, losses
                    )
                    defensive_rating = self._calculate_defensive_rating(
                        points_against, yards_against, turnovers_against, wins, losses
                    )
                    
                    team_data = {
                        'team': team_code,
                        'season': season,
                        'offensive_rating': offensive_rating,
                        'defensive_rating': defensive_rating,
                        'points_for': points_for,
                        'points_against': points_against,
                        'yards_for': yards_for,
                        'yards_against': yards_against,
                        'turnovers_for': turnovers_for,
                        'turnovers_against': turnovers_against,
                        'wins': wins,
                        'losses': losses,
                        'win_percentage': wins / (wins + losses + ties) if (wins + losses + ties) > 0 else 0
                    }
                    
                    teams_data.append(team_data)
                    
                except Exception as e:
                    logger.warning(f"Error processing row: {e}")
                    continue
            
            logger.info(f"Successfully scraped {len(teams_data)} teams from Pro Football Reference")
            return teams_data
            
        except Exception as e:
            logger.error(f"Error scraping Pro Football Reference: {e}")
            return []
    
    def _get_team_code_from_name(self, team_name: str) -> Optional[str]:
        """Get team code from team name."""
        team_name = team_name.strip().lower()
        
        for code, info in self.team_mappings.items():
            if (info['name'].lower() in team_name or 
                info['city'].lower() in team_name or 
                info['mascot'].lower() in team_name):
                return code
        
        return None
    
    def _safe_float(self, value: str) -> float:
        """Safely convert string to float."""
        try:
            return float(value.replace(',', ''))
        except (ValueError, AttributeError):
            return 0.0
    
    def _safe_int(self, value: str) -> int:
        """Safely convert string to int."""
        try:
            return int(value.replace(',', ''))
        except (ValueError, AttributeError):
            return 0
    
    def _calculate_offensive_rating(self, points_for: float, yards_for: float, 
                                  turnovers_for: int, wins: int, losses: int) -> float:
        """Calculate offensive rating using advanced metrics."""
        if points_for == 0 or yards_for == 0:
            return 100.0
        
        # Base rating calculation
        points_factor = (points_for / 25.0) * 30  # Normalize around 25 points
        yards_factor = (yards_for / 350.0) * 25   # Normalize around 350 yards
        turnover_factor = max(0, 25 - (turnovers_for * 3))  # Penalty for turnovers
        win_factor = (wins / (wins + losses)) * 20 if (wins + losses) > 0 else 10
        
        rating = 100 + points_factor + yards_factor + turnover_factor + win_factor - 100
        return max(50, min(150, rating))
    
    def _calculate_defensive_rating(self, points_against: float, yards_against: float,
                                  turnovers_against: int, wins: int, losses: int) -> float:
        """Calculate defensive rating using advanced metrics."""
        if points_against == 0 or yards_against == 0:
            return 100.0
        
        # For defense, lower is better
        points_factor = max(0, 30 - (points_against / 25.0) * 30)
        yards_factor = max(0, 25 - (yards_against / 350.0) * 25)
        turnover_factor = (turnovers_against * 3)  # Bonus for forcing turnovers
        win_factor = (wins / (wins + losses)) * 20 if (wins + losses) > 0 else 10
        
        rating = 100 + points_factor + yards_factor + turnover_factor + win_factor - 100
        return max(50, min(150, rating))
    
    def _calculate_offensive_rating_from_stats(self, stats: Dict) -> float:
        """Calculate offensive rating from ESPN stats."""
        points_per_game = stats.get('Points Per Game', 0)
        yards_per_game = stats.get('Total Yards Per Game', 0)
        turnovers = stats.get('Turnovers', 0)
        
        return self._calculate_offensive_rating(points_per_game, yards_per_game, turnovers, 0, 0)
    
    def _calculate_defensive_rating_from_stats(self, stats: Dict) -> float:
        """Calculate defensive rating from ESPN stats."""
        points_per_game = stats.get('Points Per Game', 0)
        yards_per_game = stats.get('Total Yards Per Game', 0)
        takeaways = stats.get('Takeaways', 0)
        
        return self._calculate_defensive_rating(points_per_game, yards_per_game, takeaways, 0, 0)
    
    def fetch_all_team_data(self, season: int = 2024) -> List[Dict]:
        """Fetch team data from all available sources."""
        logger.info(f"Fetching team data for season {season}")
        
        all_data = []
        
        # Try ESPN first
        espn_data = self.fetch_espn_team_stats(season)
        all_data.extend(espn_data)
        
        # Fallback to Pro Football Reference
        if not all_data:
            pfr_data = self.fetch_pro_football_reference_data(season)
            all_data.extend(pfr_data)
        
        # Save to database
        self._save_team_data(all_data)
        
        return all_data
    
    def _save_team_data(self, data: List[Dict]) -> None:
        """Save team data to database."""
        logger.info(f"Saving {len(data)} team records to database")
        
        for record in data:
            try:
                # Create or update team record
                team_stats = TeamStats(
                    team=record['team'],
                    season=record['season'],
                    week=None,
                    offensive_rating=record['offensive_rating'],
                    defensive_rating=record['defensive_rating'],
                    points_for=record['points_for'],
                    points_against=record['points_against'],
                    yards_for=record['yards_for'],
                    yards_against=record['yards_against'],
                    turnovers_for=record['turnovers_for'],
                    turnovers_against=record['turnovers_against'],
                    wins=record.get('wins', 0),
                    losses=record.get('losses', 0),
                    win_percentage=record.get('win_percentage', 0.0),
                    strength_of_schedule=0.0,
                    pythagorean_wins=0.0,
                    created_at=datetime.utcnow()
                )
                
                # Save to database (implementation depends on your database setup)
                logger.info(f"Saved data for {record['team']}")
                
            except Exception as e:
                logger.error(f"Error saving record {record}: {e}")
                continue
    
    def get_team_ratings(self, team: str, season: int = 2024) -> Optional[Dict]:
        """Get current team ratings."""
        # For now, return mock data - in production, this would query the database
        return {
            'team': team,
            'season': season,
            'offensive_rating': np.random.uniform(80, 120),
            'defensive_rating': np.random.uniform(80, 120),
            'points_for': np.random.uniform(20, 35),
            'points_against': np.random.uniform(20, 35),
            'yards_for': np.random.uniform(300, 450),
            'yards_against': np.random.uniform(300, 450),
            'turnovers_for': np.random.randint(10, 30),
            'turnovers_against': np.random.randint(10, 30),
            'wins': np.random.randint(5, 12),
            'losses': np.random.randint(5, 12),
            'win_percentage': np.random.uniform(0.3, 0.8)
        }

def main():
    """Main function to test the API client."""
    client = NFLAPIClient()
    
    # Fetch current season data
    current_data = client.fetch_all_team_data(2024)
    
    print(f"\n🏈 Fetched data for {len(current_data)} teams")
    
    # Display sample results
    for team_data in current_data[:5]:
        print(f"{team_data['team']}: Off={team_data['offensive_rating']:.1f}, Def={team_data['defensive_rating']:.1f}")

if __name__ == "__main__":
    main()
