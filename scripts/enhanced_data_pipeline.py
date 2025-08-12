#!/usr/bin/env python3
"""
Enhanced AFL Data Pipeline with Integrated Scrapers
Builds upon akareen's excellent scraping work while maintaining our clean database structure.
"""

import pandas as pd
import numpy as np
import sqlite3
import requests
from bs4 import BeautifulSoup
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import re
import os
import git
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AFLScrapingPipeline:
    """Enhanced data pipeline that scrapes directly from AFL Tables and other sources."""
    
    def __init__(self, data_dir: str = "afl_data"):
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / "afl_database.db"
        self.backup_repo_url = "https://github.com/akareen/AFL-Data-Analysis"
        
        # AFL Tables base URLs (primary sources)
        self.base_urls = {
            'matches': 'https://afltables.com',
            'players': 'https://afltables.com',
            'odds': 'https://www.aussportsbetting.com'
        }
        
        # Create directories
        self.data_dir.mkdir(exist_ok=True)
        
        # Rate limiting
        self.request_delay = 1.0  # seconds between requests
        
        # Initialize database
        self.init_database()
        
    def init_database(self):
        """Initialize SQLite database with comprehensive schema."""
        logger.info("Initializing enhanced database...")
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced matches table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                year INTEGER NOT NULL,
                round TEXT,
                venue TEXT,
                date TEXT NOT NULL,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_q1_g INTEGER DEFAULT 0,
                home_q1_b INTEGER DEFAULT 0,
                home_q2_g INTEGER DEFAULT 0,
                home_q2_b INTEGER DEFAULT 0,
                home_q3_g INTEGER DEFAULT 0,
                home_q3_b INTEGER DEFAULT 0,
                home_q4_g INTEGER DEFAULT 0,
                home_q4_b INTEGER DEFAULT 0,
                away_q1_g INTEGER DEFAULT 0,
                away_q1_b INTEGER DEFAULT 0,
                away_q2_g INTEGER DEFAULT 0,
                away_q2_b INTEGER DEFAULT 0,
                away_q3_g INTEGER DEFAULT 0,
                away_q3_b INTEGER DEFAULT 0,
                away_q4_g INTEGER DEFAULT 0,
                away_q4_b INTEGER DEFAULT 0,
                home_total_goals INTEGER,
                home_total_behinds INTEGER,
                away_total_goals INTEGER,
                away_total_behinds INTEGER,
                winning_team TEXT,
                margin INTEGER,
                attendance INTEGER,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source TEXT DEFAULT 'scraper'
            )
        ''')
        
        # Enhanced players table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS players (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_name TEXT,
                last_name TEXT,
                born_date TEXT,
                debut_date TEXT,
                height INTEGER,
                weight INTEGER,
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
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source TEXT DEFAULT 'scraper'
            )
        ''')
        
        # Odds table for betting data
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS odds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                year INTEGER NOT NULL,
                round TEXT,
                date TEXT,
                home_team TEXT NOT NULL,
                away_team TEXT NOT NULL,
                home_odds REAL,
                away_odds REAL,
                draw_odds REAL,
                bookmaker TEXT,
                scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_year ON matches(year)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team, away_team)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_matches_date ON matches(date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_players_team_year ON players(team, year)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_players_name ON players(first_name, last_name)')
        
        conn.commit()
        conn.close()
        
    def scrape_matches_for_year(self, year: int) -> List[Dict]:
        """Scrape match data for a specific year from AFL Tables."""
        logger.info(f"Scraping match data for {year}...")
        
        matches = []
        
        try:
            # AFL Tables season URL pattern
            url = f"https://afltables.com/afl/seas/{year}.html"
            
            response = requests.get(url)
            time.sleep(self.request_delay)
            
            if response.status_code != 200:
                logger.warning(f"Failed to fetch data for {year}: {response.status_code}")
                return matches
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find match tables (this is simplified - actual implementation would be more complex)
            # The real scraper would parse the specific table structure of AFL Tables
            match_tables = soup.find_all('table')
            
            for table in match_tables:
                rows = table.find_all('tr')
                for row in rows[1:]:  # Skip header
                    cells = row.find_all('td')
                    if len(cells) >= 10:  # Minimum columns for match data
                        try:
                            match_data = self._parse_match_row(cells, year)
                            if match_data:
                                matches.append(match_data)
                        except Exception as e:
                            logger.warning(f"Error parsing match row: {e}")
                            continue
            
        except Exception as e:
            logger.error(f"Error scraping matches for {year}: {e}")
        
        logger.info(f"Scraped {len(matches)} matches for {year}")
        return matches
    
    def _parse_match_row(self, cells, year: int) -> Optional[Dict]:
        """Parse a single match row from AFL Tables."""
        try:
            # This is a simplified parser - the real implementation would handle
            # the specific AFL Tables format which includes:
            # Round, Date, Venue, Teams, Scores by quarter, etc.
            
            if len(cells) < 8:
                return None
            
            # Extract basic match information
            # Note: Actual AFL Tables parsing would be more complex
            match = {
                'year': year,
                'round': cells[0].get_text(strip=True) if cells[0] else '',
                'date': cells[1].get_text(strip=True) if cells[1] else '',
                'venue': cells[2].get_text(strip=True) if cells[2] else '',
                'home_team': self._standardize_team_name(cells[3].get_text(strip=True)) if cells[3] else '',
                'away_team': self._standardize_team_name(cells[4].get_text(strip=True)) if cells[4] else '',
                'source': 'scraper'
            }
            
            # Parse scores if available
            if len(cells) >= 10:
                # Extract quarter by quarter scores and totals
                # This would need to be customized based on AFL Tables format
                score_text = cells[5].get_text() if cells[5] else ''
                match.update(self._parse_scores(score_text))
            
            return match
            
        except Exception as e:
            logger.warning(f"Error parsing match row: {e}")
            return None
    
    def _parse_scores(self, score_text: str) -> Dict:
        """Parse score text into structured data."""
        # Simplified score parsing - real implementation would handle
        # AFL Tables specific format like "3.2 5.4 8.7 12.10.82"
        scores = {}
        
        try:
            # Extract basic totals for now
            # Real implementation would parse quarter by quarter
            if 'def' in score_text.lower():
                parts = score_text.split('def')
                if len(parts) == 2:
                    winner_score = parts[0].strip()
                    loser_score = parts[1].strip()
                    
                    # Parse final scores (goals.behinds.points format)
                    scores.update(self._parse_final_score(winner_score, 'home'))
                    scores.update(self._parse_final_score(loser_score, 'away'))
        
        except Exception as e:
            logger.warning(f"Error parsing scores: {e}")
        
        return scores
    
    def _parse_final_score(self, score_str: str, team: str) -> Dict:
        """Parse final score string like '12.10.82'."""
        scores = {}
        
        try:
            # Extract goals.behinds.points
            parts = score_str.strip().split('.')
            if len(parts) >= 2:
                scores[f'{team}_total_goals'] = int(parts[0])
                scores[f'{team}_total_behinds'] = int(parts[1])
                
                # Calculate margin if both teams are present
                if len(parts) >= 3:
                    total_points = int(parts[2])
                    # Margin calculation would be done later when both teams are processed
        
        except (ValueError, IndexError) as e:
            logger.warning(f"Error parsing final score {score_str}: {e}")
        
        return scores
    
    def _standardize_team_name(self, team_name: str) -> str:
        """Standardize team names to match our database format."""
        # Common team name mappings
        name_mappings = {
            'Brisbane Lions': 'Brisbane',
            'West Coast Eagles': 'West Coast',
            'Greater Western Sydney Giants': 'Greater Western Sydney',
            'GWS Giants': 'Greater Western Sydney',
            'Port Adelaide Power': 'Port Adelaide',
            'Gold Coast Suns': 'Gold Coast',
            # Add more mappings as needed
        }
        
        return name_mappings.get(team_name, team_name)
    
    def scrape_player_data_for_year(self, year: int) -> List[Dict]:
        """Scrape player performance data for a specific year."""
        logger.info(f"Scraping player data for {year}...")
        
        players = []
        
        try:
            # This would iterate through team pages and player statistics
            # For now, return empty list as placeholder
            # Real implementation would scrape individual player pages from AFL Tables
            pass
            
        except Exception as e:
            logger.error(f"Error scraping player data for {year}: {e}")
        
        return players
    
    def compare_with_existing_data(self, new_matches: List[Dict]) -> Dict:
        """Compare new scraped data with existing database."""
        logger.info("Comparing scraped data with existing database...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Get existing data for comparison
        existing_df = pd.read_sql_query("SELECT * FROM matches", conn)
        conn.close()
        
        new_df = pd.DataFrame(new_matches)
        
        comparison = {
            'existing_matches': len(existing_df),
            'new_matches': len(new_df),
            'overlapping_matches': 0,
            'truly_new_matches': 0,
            'data_consistency_score': 0.0
        }
        
        if not existing_df.empty and not new_df.empty:
            # Find overlaps based on year, teams, and date
            merge_cols = ['year', 'home_team', 'away_team', 'date']
            available_cols = [col for col in merge_cols if col in existing_df.columns and col in new_df.columns]
            
            if available_cols:
                merged = existing_df.merge(new_df, on=available_cols, how='inner', suffixes=('_existing', '_new'))
                comparison['overlapping_matches'] = len(merged)
                comparison['truly_new_matches'] = len(new_df) - len(merged)
                
                # Calculate consistency score
                if len(merged) > 0:
                    # Compare scores for overlapping matches
                    score_consistency = []
                    for col in ['home_total_goals', 'away_total_goals', 'margin']:
                        if f'{col}_existing' in merged.columns and f'{col}_new' in merged.columns:
                            consistency = (merged[f'{col}_existing'] == merged[f'{col}_new']).mean()
                            score_consistency.append(consistency)
                    
                    if score_consistency:
                        comparison['data_consistency_score'] = np.mean(score_consistency)
        
        logger.info(f"Data comparison: {comparison}")
        return comparison
    
    def fallback_to_repository_data(self):
        """Fallback to cloning akareen's repository if scraping fails."""
        logger.info("Falling back to repository data...")
        
        repo_dir = self.data_dir / "AFL-Data-Analysis"
        
        try:
            if repo_dir.exists():
                # Update existing repository
                repo = git.Repo(repo_dir)
                repo.remotes.origin.pull()
                logger.info("Updated existing repository")
            else:
                # Clone repository
                git.Repo.clone_from(self.backup_repo_url, repo_dir)
                logger.info("Cloned AFL-Data-Analysis repository")
            
            # Load data from repository using our existing pipeline
            from scripts.data_pipeline import AFLDataPipeline
            backup_pipeline = AFLDataPipeline(self.data_dir)
            return backup_pipeline.load_match_data(repo_dir)
            
        except Exception as e:
            logger.error(f"Failed to use repository fallback: {e}")
            return pd.DataFrame()
    
    def run_enhanced_pipeline(self, start_year: int = 1897, end_year: int = None, 
                            scrape_matches: bool = True, scrape_players: bool = False,
                            verify_with_existing: bool = True):
        """Run the enhanced data pipeline with scraping and verification."""
        
        if end_year is None:
            end_year = datetime.now().year
        
        logger.info(f"Starting enhanced pipeline for years {start_year}-{end_year}")
        
        all_matches = []
        all_players = []
        
        # Step 1: Scrape new data
        if scrape_matches:
            for year in tqdm(range(start_year, end_year + 1), desc="Scraping matches"):
                try:
                    year_matches = self.scrape_matches_for_year(year)
                    all_matches.extend(year_matches)
                    
                    # Be respectful to the server
                    time.sleep(self.request_delay)
                    
                except Exception as e:
                    logger.error(f"Failed to scrape matches for {year}: {e}")
                    continue
        
        if scrape_players:
            for year in tqdm(range(start_year, end_year + 1), desc="Scraping players"):
                try:
                    year_players = self.scrape_player_data_for_year(year)
                    all_players.extend(year_players)
                    
                    time.sleep(self.request_delay)
                    
                except Exception as e:
                    logger.error(f"Failed to scrape players for {year}: {e}")
                    continue
        
        # Step 2: Verify data quality and completeness
        if verify_with_existing and all_matches:
            comparison = self.compare_with_existing_data(all_matches)
            
            # If consistency is low, fall back to repository data
            if comparison['data_consistency_score'] < 0.95:
                logger.warning(f"Data consistency score {comparison['data_consistency_score']:.2f} is low. Using repository fallback.")
                backup_data = self.fallback_to_repository_data()
                if not backup_data.empty:
                    # Convert to our format and merge with scraped data
                    all_matches = self._merge_scraped_and_backup_data(all_matches, backup_data)
        
        # Step 3: Store the data
        self._store_scraped_data(all_matches, all_players)
        
        # Step 4: Generate comprehensive report
        report = self._generate_scraping_report(all_matches, all_players)
        
        logger.info("Enhanced pipeline completed successfully!")
        return report
    
    def _merge_scraped_and_backup_data(self, scraped_matches: List[Dict], backup_df: pd.DataFrame) -> List[Dict]:
        """Intelligently merge scraped data with backup repository data."""
        # Implementation would prioritize recent scraped data over backup data
        # and use backup data to fill gaps
        return scraped_matches  # Simplified for now
    
    def _store_scraped_data(self, matches: List[Dict], players: List[Dict]):
        """Store scraped data in the database."""
        if not matches and not players:
            logger.warning("No data to store")
            return
        
        conn = sqlite3.connect(self.db_path)
        
        try:
            if matches:
                matches_df = pd.DataFrame(matches)
                matches_df.to_sql('matches', conn, if_exists='append', index=False)
                logger.info(f"Stored {len(matches)} matches")
            
            if players:
                players_df = pd.DataFrame(players)
                players_df.to_sql('players', conn, if_exists='append', index=False)
                logger.info(f"Stored {len(players)} player records")
            
            conn.commit()
            
        except Exception as e:
            logger.error(f"Error storing data: {e}")
            conn.rollback()
        finally:
            conn.close()
    
    def _generate_scraping_report(self, matches: List[Dict], players: List[Dict]) -> Dict:
        """Generate a comprehensive report of the scraping process."""
        
        conn = sqlite3.connect(self.db_path)
        
        # Get database statistics
        matches_count = pd.read_sql_query("SELECT COUNT(*) as count FROM matches", conn).iloc[0]['count']
        players_count = pd.read_sql_query("SELECT COUNT(*) as count FROM players", conn).iloc[0]['count']
        
        year_range = pd.read_sql_query("""
            SELECT MIN(year) as min_year, MAX(year) as max_year 
            FROM matches WHERE year IS NOT NULL
        """, conn)
        
        conn.close()
        
        report = {
            'scraping_timestamp': datetime.now().isoformat(),
            'scraped_matches': len(matches),
            'scraped_players': len(players),
            'total_matches_in_db': matches_count,
            'total_players_in_db': players_count,
            'year_range': {
                'min_year': year_range.iloc[0]['min_year'] if not year_range.empty else None,
                'max_year': year_range.iloc[0]['max_year'] if not year_range.empty else None
            },
            'data_sources': ['scraper', 'repository_backup'],
            'pipeline_version': '2.0_enhanced_scraping'
        }
        
        # Save report
        report_path = self.data_dir / "scraping_report.json"
        import json
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Scraping report saved to {report_path}")
        return report

def main():
    """Main function to run the enhanced scraping pipeline."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced AFL Data Scraping Pipeline')
    parser.add_argument('--start_year', type=int, default=2020, help='Start year for scraping')
    parser.add_argument('--end_year', type=int, default=None, help='End year for scraping')
    parser.add_argument('--scrape_matches', action='store_true', help='Scrape match data')
    parser.add_argument('--scrape_players', action='store_true', help='Scrape player data')
    parser.add_argument('--data_dir', type=str, default='afl_data', help='Data directory')
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = AFLScrapingPipeline(data_dir=args.data_dir)
    
    # Run enhanced pipeline
    report = pipeline.run_enhanced_pipeline(
        start_year=args.start_year,
        end_year=args.end_year,
        scrape_matches=args.scrape_matches,
        scrape_players=args.scrape_players
    )
    
    print("\n" + "="*60)
    print("ENHANCED AFL SCRAPING PIPELINE COMPLETE")
    print("="*60)
    print(f"Scraped {report['scraped_matches']} matches")
    print(f"Scraped {report['scraped_players']} player records")
    print(f"Total in database: {report['total_matches_in_db']} matches, {report['total_players_in_db']} players")
    print(f"Data range: {report['year_range']['min_year']}-{report['year_range']['max_year']}")
    print("="*60)

if __name__ == "__main__":
    main()
