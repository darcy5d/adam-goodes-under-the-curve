#!/usr/bin/env python3
"""
AFL Data Pipeline
Handles data extraction, validation, and storage for AFL prediction model.
"""

import os
import sys
import logging
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, date
import git
import requests
from tqdm import tqdm
import json
import re
from pydantic import BaseModel, ValidationError
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('afl_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DataQualityMetrics:
    """Data quality metrics for validation reporting."""
    total_records: int
    missing_values: Dict[str, int]
    invalid_values: Dict[str, int]
    outliers: Dict[str, int]
    duplicates: int
    date_range: Tuple[date, date]
    team_count: int
    venue_count: int

class AFLDataPipeline:
    """Main pipeline class for AFL data processing."""
    
    def __init__(self, data_dir: str = "afl_data"):
        self.data_dir = Path(data_dir)
        self.repo_url = "https://github.com/akareen/AFL-Data-Analysis"
        self.db_path = self.data_dir / "afl_database.db"
        self.parquet_dir = self.data_dir / "parquet"
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        self.parquet_dir.mkdir(exist_ok=True)
        
        # Initialize database
        self.init_database()
        
        # Data quality metrics
        self.match_metrics = None
        self.player_metrics = None
        
    def init_database(self):
        """Initialize SQLite database with proper schema and indexing."""
        logger.info("Initializing database...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create matches table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                year INTEGER NOT NULL,
                ground TEXT,
                venue TEXT,
                date TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_team_goals_by_quarter TEXT,
                home_team_behinds_by_quarter TEXT,
                away_team_goals_by_quarter TEXT,
                away_team_behinds_by_quarter TEXT,
                home_total_goals INTEGER,
                home_total_behinds INTEGER,
                away_total_goals INTEGER,
                away_total_behinds INTEGER,
                winning_team TEXT,
                margin INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create players table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                team TEXT,
                year INTEGER,
                games_played INTEGER,
                opponent TEXT,
                round TEXT,
                result TEXT,
                jersey_number INTEGER,
                kicks INTEGER,
                marks INTEGER,
                handballs INTEGER,
                disposals INTEGER,
                goals INTEGER,
                behinds INTEGER,
                hit_outs INTEGER,
                tackles INTEGER,
                rebound_50s INTEGER,
                inside_50s INTEGER,
                clearances INTEGER,
                clangers INTEGER,
                free_kicks_for INTEGER,
                free_kicks_against INTEGER,
                brownlow_votes INTEGER,
                contested_possessions INTEGER,
                uncontested_possessions INTEGER,
                contested_marks INTEGER,
                marks_inside_50 INTEGER,
                one_percenters INTEGER,
                bounces INTEGER,
                goal_assist INTEGER,
                percentage_of_game_played REAL,
                first_name TEXT,
                last_name TEXT,
                born_date TEXT,
                debut_date TEXT,
                height INTEGER,
                weight INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_matches_year ON matches(year)",
            "CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date)",
            "CREATE INDEX IF NOT EXISTS idx_matches_home_team ON matches(home_team)",
            "CREATE INDEX IF NOT EXISTS idx_matches_away_team ON matches(away_team)",
            "CREATE INDEX IF NOT EXISTS idx_matches_venue ON matches(venue)",
            "CREATE INDEX IF NOT EXISTS idx_players_year ON players(year)",
            "CREATE INDEX IF NOT EXISTS idx_players_team ON players(team)",
            "CREATE INDEX IF NOT EXISTS idx_players_name ON players(first_name, last_name)",
            "CREATE INDEX IF NOT EXISTS idx_players_disposals ON players(disposals)",
            "CREATE INDEX IF NOT EXISTS idx_players_goals ON players(goals)"
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def clone_repository(self):
        """Clone the AFL data repository."""
        repo_path = self.data_dir / "AFL-Data-Analysis"
        
        if repo_path.exists():
            logger.info("Repository already exists, pulling latest changes...")
            try:
                repo = git.Repo(repo_path)
                repo.remotes.origin.pull()
                logger.info("Repository updated successfully")
            except Exception as e:
                logger.warning(f"Could not update repository: {e}")
        else:
            logger.info("Cloning AFL data repository...")
            try:
                git.Repo.clone_from(self.repo_url, repo_path)
                logger.info("Repository cloned successfully")
            except Exception as e:
                logger.error(f"Failed to clone repository: {e}")
                raise
        
        return repo_path
    
    def load_match_data(self, repo_path: Path) -> pd.DataFrame:
        """Load and process match data from the repository."""
        logger.info("Loading match data...")
        
        match_files = []
        data_dir = repo_path / "data" / "matches"
        
        # Find all match data files
        for file_path in data_dir.rglob("*.csv"):
            if "match" in file_path.name.lower():
                match_files.append(file_path)
        
        if not match_files:
            logger.warning("No match data files found")
            return pd.DataFrame()
        
        all_matches = []
        
        for file_path in tqdm(match_files, desc="Processing match files"):
            try:
                df = pd.read_csv(file_path)
                logger.info(f"Loaded {len(df)} records from {file_path.name}")
                all_matches.append(df)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        if not all_matches:
            logger.error("No match data could be loaded")
            return pd.DataFrame()
        
        # Combine all match data
        matches_df = pd.concat(all_matches, ignore_index=True)
        
        # Standardize column names
        column_mapping = {
            'Year': 'year',
            'Ground': 'ground',
            'Venue': 'venue',
            'Date': 'date',
            'Home Team': 'home_team',
            'Away Team': 'away_team',
            'Home Team Goals by Quarter': 'home_team_goals_by_quarter',
            'Home Team Behinds by Quarter': 'home_team_behinds_by_quarter',
            'Away Team Goals by Quarter': 'away_team_goals_by_quarter',
            'Away Team Behinds by Quarter': 'away_team_behinds_by_quarter',
            'Home Total Goals': 'home_total_goals',
            'Home Total Behinds': 'home_total_behinds',
            'Away Total Goals': 'away_total_goals',
            'Away Total Behinds': 'away_total_behinds',
            'Winning Team': 'winning_team',
            'Margin': 'margin'
        }
        
        matches_df = matches_df.rename(columns=column_mapping)
        
        # Ensure required columns exist
        required_columns = ['year', 'home_team', 'away_team', 'date']
        missing_columns = [col for col in required_columns if col not in matches_df.columns]
        
        if missing_columns:
            logger.error(f"Missing required columns: {missing_columns}")
            return pd.DataFrame()
        
        logger.info(f"Loaded {len(matches_df)} total match records")
        return matches_df
    
    def load_player_data(self, repo_path: Path) -> pd.DataFrame:
        """Load and process player data from the repository."""
        logger.info("Loading player data...")
        
        player_files = []
        data_dir = repo_path / "data" / "players"
        
        # Find all player data files
        for file_path in data_dir.rglob("*_performance_details.csv"):
            player_files.append(file_path)
        
        if not player_files:
            logger.warning("No player performance data files found")
            return pd.DataFrame()
        
        all_players = []
        
        for file_path in tqdm(player_files, desc="Processing player files"):
            try:
                df = pd.read_csv(file_path)
                logger.info(f"Loaded {len(df)} records from {file_path.name}")
                all_players.append(df)
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        
        if not all_players:
            logger.error("No player data could be loaded")
            return pd.DataFrame()
        
        # Combine all player data
        players_df = pd.concat(all_players, ignore_index=True)
        
        # Standardize column names
        column_mapping = {
            'Team': 'team',
            'Year': 'year',
            'Games Played': 'games_played',
            'Opponent': 'opponent',
            'Round': 'round',
            'Result': 'result',
            'Jersey Number': 'jersey_number',
            'Kicks': 'kicks',
            'Marks': 'marks',
            'Handballs': 'handballs',
            'Disposals': 'disposals',
            'Goals': 'goals',
            'Behinds': 'behinds',
            'Hit Outs': 'hit_outs',
            'Tackles': 'tackles',
            'Rebound 50s': 'rebound_50s',
            'Inside 50s': 'inside_50s',
            'Clearances': 'clearances',
            'Clangers': 'clangers',
            'Free Kicks For': 'free_kicks_for',
            'Free Kicks Against': 'free_kicks_against',
            'Brownlow Votes': 'brownlow_votes',
            'Contested Possessions': 'contested_possessions',
            'Uncontested Possessions': 'uncontested_possessions',
            'Contested Marks': 'contested_marks',
            'Marks Inside 50': 'marks_inside_50',
            'One Percenters': 'one_percenters',
            'Bounces': 'bounces',
            'Goal Assist': 'goal_assist',
            'Percentage of Game Played': 'percentage_of_game_played',
            'First Name': 'first_name',
            'Last Name': 'last_name',
            'Born Date': 'born_date',
            'Debut Date': 'debut_date',
            'Height': 'height',
            'Weight': 'weight'
        }
        
        players_df = players_df.rename(columns=column_mapping)
        
        logger.info(f"Loaded {len(players_df)} total player records")
        return players_df
    
    def validate_match_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, DataQualityMetrics]:
        """Validate match data and return cleaned dataframe with metrics."""
        logger.info("Validating match data...")
        
        original_count = len(df)
        missing_values = {}
        invalid_values = {}
        outliers = {}
        
        # Check for missing values
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                missing_values[column] = missing_count
        
        # Data type conversions and validation
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            invalid_years = df[df['year'] < 1897]['year'].count()
            if invalid_years > 0:
                invalid_values['year'] = invalid_years
        
        if 'home_total_goals' in df.columns:
            df['home_total_goals'] = pd.to_numeric(df['home_total_goals'], errors='coerce')
            outliers['home_total_goals'] = df[df['home_total_goals'] > 50]['home_total_goals'].count()
        
        if 'away_total_goals' in df.columns:
            df['away_total_goals'] = pd.to_numeric(df['away_total_goals'], errors='coerce')
            outliers['away_total_goals'] = df[df['away_total_goals'] > 50]['away_total_goals'].count()
        
        if 'margin' in df.columns:
            df['margin'] = pd.to_numeric(df['margin'], errors='coerce')
            outliers['margin'] = df[df['margin'].abs() > 200]['margin'].count()
        
        # Remove duplicates
        df = df.drop_duplicates()
        duplicates = original_count - len(df)
        
        # Date range
        if 'date' in df.columns:
            try:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                date_range = (df['date'].min().date(), df['date'].max().date())
            except:
                date_range = (date(1897, 1, 1), date(2025, 12, 31))
        else:
            date_range = (date(1897, 1, 1), date(2025, 12, 31))
        
        # Team and venue counts
        team_count = len(df['home_team'].unique()) if 'home_team' in df.columns else 0
        venue_count = len(df['venue'].unique()) if 'venue' in df.columns else 0
        
        metrics = DataQualityMetrics(
            total_records=len(df),
            missing_values=missing_values,
            invalid_values=invalid_values,
            outliers=outliers,
            duplicates=duplicates,
            date_range=date_range,
            team_count=team_count,
            venue_count=venue_count
        )
        
        self.match_metrics = metrics
        logger.info(f"Match data validation complete. {len(df)} valid records.")
        
        return df, metrics
    
    def validate_player_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, DataQualityMetrics]:
        """Validate player data and return cleaned dataframe with metrics."""
        logger.info("Validating player data...")
        
        original_count = len(df)
        missing_values = {}
        invalid_values = {}
        outliers = {}
        
        # Check for missing values
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                missing_values[column] = missing_count
        
        # Data type conversions and validation
        numeric_columns = ['year', 'games_played', 'jersey_number', 'kicks', 'marks', 
                          'handballs', 'disposals', 'goals', 'behinds', 'hit_outs', 
                          'tackles', 'rebound_50s', 'inside_50s', 'clearances', 
                          'clangers', 'free_kicks_for', 'free_kicks_against', 
                          'brownlow_votes', 'contested_possessions', 'uncontested_possessions',
                          'contested_marks', 'marks_inside_50', 'one_percenters', 
                          'bounces', 'goal_assist', 'height', 'weight']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Validate percentage of game played
        if 'percentage_of_game_played' in df.columns:
            df['percentage_of_game_played'] = pd.to_numeric(df['percentage_of_game_played'], errors='coerce')
            outliers['percentage_of_game_played'] = df[df['percentage_of_game_played'] > 100]['percentage_of_game_played'].count()
        
        # Remove duplicates
        df = df.drop_duplicates()
        duplicates = original_count - len(df)
        
        # Date range (using year)
        if 'year' in df.columns:
            date_range = (date(df['year'].min(), 1, 1), date(df['year'].max(), 12, 31))
        else:
            date_range = (date(1897, 1, 1), date(2025, 12, 31))
        
        # Team count
        team_count = len(df['team'].unique()) if 'team' in df.columns else 0
        
        metrics = DataQualityMetrics(
            total_records=len(df),
            missing_values=missing_values,
            invalid_values=invalid_values,
            outliers=outliers,
            duplicates=duplicates,
            date_range=date_range,
            team_count=team_count,
            venue_count=0  # Not applicable for player data
        )
        
        self.player_metrics = metrics
        logger.info(f"Player data validation complete. {len(df)} valid records.")
        
        return df, metrics
    
    def store_data(self, matches_df: pd.DataFrame, players_df: pd.DataFrame):
        """Store data in SQLite database and Parquet files."""
        logger.info("Storing data...")
        
        # Store in SQLite
        conn = sqlite3.connect(self.db_path)
        
        # Store matches
        if not matches_df.empty:
            matches_df.to_sql('matches', conn, if_exists='replace', index=False)
            logger.info(f"Stored {len(matches_df)} match records in database")
        
        # Store players
        if not players_df.empty:
            players_df.to_sql('players', conn, if_exists='replace', index=False)
            logger.info(f"Stored {len(players_df)} player records in database")
        
        conn.close()
        
        # Store as Parquet files for backup and analytics
        if not matches_df.empty:
            matches_df.to_parquet(self.parquet_dir / "matches.parquet", index=False)
            logger.info("Stored match data as Parquet file")
        
        if not players_df.empty:
            players_df.to_parquet(self.parquet_dir / "players.parquet", index=False)
            logger.info("Stored player data as Parquet file")
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        logger.info("Generating data quality report...")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "database_info": {
                "path": str(self.db_path),
                "size_mb": self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0
            },
            "match_data": {
                "metrics": self.match_metrics.__dict__ if self.match_metrics else None,
                "sample_queries": {
                    "total_matches": "SELECT COUNT(*) FROM matches",
                    "matches_by_year": "SELECT year, COUNT(*) FROM matches GROUP BY year ORDER BY year",
                    "teams": "SELECT DISTINCT home_team FROM matches ORDER BY home_team"
                }
            },
            "player_data": {
                "metrics": self.player_metrics.__dict__ if self.player_metrics else None,
                "sample_queries": {
                    "total_players": "SELECT COUNT(*) FROM players",
                    "players_by_year": "SELECT year, COUNT(*) FROM players GROUP BY year ORDER BY year",
                    "top_goal_kickers": "SELECT first_name, last_name, SUM(goals) as total_goals FROM players GROUP BY first_name, last_name ORDER BY total_goals DESC LIMIT 10"
                }
            }
        }
        
        # Save report
        with open(self.data_dir / "quality_report.json", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info("Quality report generated and saved")
        return report
    
    def run_pipeline(self):
        """Run the complete data pipeline."""
        logger.info("Starting AFL data pipeline...")
        
        try:
            # Step 1: Clone repository
            repo_path = self.clone_repository()
            
            # Step 2: Load data
            matches_df = self.load_match_data(repo_path)
            players_df = self.load_player_data(repo_path)
            
            # Step 3: Validate data
            if not matches_df.empty:
                matches_df, match_metrics = self.validate_match_data(matches_df)
            
            if not players_df.empty:
                players_df, player_metrics = self.validate_player_data(players_df)
            
            # Step 4: Store data
            if matches_df.empty and players_df.empty:
                logger.error("No data loaded, aborting storage and quality report.")
                return False

            self.store_data(matches_df, players_df)
            
            # Step 5: Generate quality report
            report = self.generate_quality_report()
            
            logger.info("Pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            return False

def main():
    """Main function to run the pipeline."""
    pipeline = AFLDataPipeline()
    success = pipeline.run_pipeline()
    
    if success:
        print("\n" + "="*50)
        print("AFL DATA PIPELINE COMPLETED SUCCESSFULLY")
        print("="*50)
        print(f"Database: {pipeline.db_path}")
        print(f"Parquet files: {pipeline.parquet_dir}")
        print(f"Quality report: {pipeline.data_dir}/quality_report.json")
        print(f"Log file: afl_pipeline.log")
        print("="*50)
    else:
        print("\nPipeline failed. Check logs for details.")

if __name__ == "__main__":
    main()
