#!/usr/bin/env python3
"""
Hybrid AFL Data Pipeline - Evergreen Solution
Combines repository historical data with real-time scraping for current season.
Automatically detects data cutoff and switches to scraping for newer matches.
"""

import pandas as pd
import sqlite3
import requests
from bs4 import BeautifulSoup
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import re
import json
import git
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HybridAFLPipeline:
    """Evergreen AFL data pipeline that combines repository data with real-time scraping."""
    
    def __init__(self, data_dir: str = "afl_data"):
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / "afl_database.db"
        self.repo_url = "https://github.com/akareen/AFL-Data-Analysis"
        self.current_year = datetime.now().year
        
        # AFL season typically runs March-October
        self.afl_season_start = 3  # March
        self.afl_season_end = 10   # October
        
        # For testing purposes, assume we're checking for future AFL seasons
        # In real use, this would detect actual AFL season timing
        
        # Rate limiting for respectful scraping
        self.request_delay = 2.0  # seconds between requests
        
        self.data_dir.mkdir(exist_ok=True)
        
    def detect_repository_data_cutoff(self) -> Dict:
        """Detect what year and round the repository data goes up to."""
        logger.info("Detecting repository data cutoff...")
        
        if not self.db_path.exists():
            logger.warning("No existing database found")
            return {'year': 1897, 'round': None, 'last_match_date': None, 'total_matches': 0}
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get overall stats
            overall_result = pd.read_sql_query("""
                SELECT MAX(year) as max_year, COUNT(*) as total_matches
                FROM matches 
                WHERE year IS NOT NULL
            """, conn)
            
            # Get current year specific stats
            current_year_result = pd.read_sql_query(f"""
                SELECT 
                    COUNT(*) as matches_2025,
                    MAX(date) as last_match_date,
                    MAX(CAST(round AS INTEGER)) as last_round
                FROM matches 
                WHERE year = {self.current_year}
                AND round IS NOT NULL 
                AND round != ''
            """, conn)
            
            conn.close()
            
            max_year = overall_result.iloc[0]['max_year']
            total_matches = overall_result.iloc[0]['total_matches']
            
            cutoff_info = {
                'year': int(max_year) if max_year else self.current_year - 1,
                'total_matches': total_matches,
                'current_year_matches': 0,
                'last_round': None,
                'last_match_date': None
            }
            
            if not current_year_result.empty and current_year_result.iloc[0]['matches_2025'] > 0:
                cutoff_info.update({
                    'current_year_matches': current_year_result.iloc[0]['matches_2025'],
                    'last_round': current_year_result.iloc[0]['last_round'],
                    'last_match_date': current_year_result.iloc[0]['last_match_date']
                })
            
            logger.info(f"Repository data: {total_matches} total matches, {cutoff_info['current_year_matches']} matches in {self.current_year} up to round {cutoff_info['last_round']}")
            return cutoff_info
            
        except Exception as e:
            logger.error(f"Error detecting data cutoff: {e}")
            return {'year': self.current_year - 1, 'round': None, 'last_match_date': None, 'total_matches': 0}
    
    def determine_scraping_strategy(self) -> Dict:
        """Determine what data needs to be scraped based on current date and existing data."""
        
        cutoff_info = self.detect_repository_data_cutoff()
        current_month = datetime.now().month
        current_date = datetime.now()
        
        strategy = {
            'cutoff_info': cutoff_info,
            'current_year': self.current_year,
            'current_month': current_month,
            'current_date': current_date.strftime('%Y-%m-%d'),
            'needs_repository_update': False,
            'needs_current_season_scraping': False,
            'scraping_years': [],
            'estimated_missing_rounds': 0,
            'strategy_description': ''
        }
        
        # Determine if we're in AFL season (March to October)
        in_afl_season = self.afl_season_start <= current_month <= self.afl_season_end
        
        repo_cutoff_year = cutoff_info['year']
        current_year_matches = cutoff_info['current_year_matches']
        last_round = cutoff_info['last_round']
        last_match_date = cutoff_info['last_match_date']
        
        if repo_cutoff_year < self.current_year:
            # Repository data is behind current year completely
            if self.current_year > repo_cutoff_year + 1:
                strategy['needs_repository_update'] = True
                strategy['strategy_description'] = f"Repository data ends at {repo_cutoff_year}, current year is {self.current_year}. Update repository first."
            else:
                strategy['needs_current_season_scraping'] = True
                strategy['scraping_years'] = [self.current_year]
                strategy['strategy_description'] = f"Repository current to {repo_cutoff_year}. Scrape entire {self.current_year} season."
                
        elif repo_cutoff_year == self.current_year and current_year_matches > 0:
            # Repository includes current year but may be incomplete
            if last_match_date:
                try:
                    last_match_datetime = datetime.strptime(last_match_date.split()[0], '%Y-%m-%d')
                    days_since_last_match = (current_date - last_match_datetime).days
                    
                    # If it's been more than 7 days since last match and we're in AFL season
                    if days_since_last_match > 7 and in_afl_season:
                        strategy['needs_current_season_scraping'] = True
                        strategy['scraping_years'] = [self.current_year]
                        
                        # Estimate missing rounds (AFL typically has ~23 rounds, ~9 matches per round)
                        estimated_total_rounds = 23
                        if last_round:
                            strategy['estimated_missing_rounds'] = max(0, estimated_total_rounds - last_round)
                        
                        strategy['strategy_description'] = f"Repository has {current_year_matches} matches through round {last_round} (last match: {last_match_date.split()[0]}). {days_since_last_match} days old - scraping needed for rounds {last_round + 1}+."
                    else:
                        strategy['strategy_description'] = f"Repository current with {current_year_matches} matches through round {last_round}. Last match {days_since_last_match} days ago - likely up to date."
                except:
                    # Fallback if date parsing fails
                    strategy['needs_current_season_scraping'] = in_afl_season
                    strategy['scraping_years'] = [self.current_year] if in_afl_season else []
                    strategy['strategy_description'] = f"Repository has {current_year_matches} matches through round {last_round}. AFL season active - checking for updates."
            else:
                # No date info available
                strategy['needs_current_season_scraping'] = in_afl_season
                strategy['scraping_years'] = [self.current_year] if in_afl_season else []
                strategy['strategy_description'] = f"Repository has {current_year_matches} matches. AFL season active - checking for updates."
                
        elif repo_cutoff_year == self.current_year and current_year_matches == 0:
            # Repository year matches current year but no current year matches
            strategy['needs_current_season_scraping'] = True
            strategy['scraping_years'] = [self.current_year]
            strategy['strategy_description'] = f"Repository covers up to {repo_cutoff_year} but no {self.current_year} matches found. Scrape entire {self.current_year} season."
            
        else:
            # Repository is ahead of current year (shouldn't happen)
            strategy['strategy_description'] = f"Repository data extends to {repo_cutoff_year}, ahead of current year {self.current_year}."
        
        logger.info(f"Strategy: {strategy['strategy_description']}")
        return strategy
    
    def update_repository_data(self) -> bool:
        """Update data from akareen's repository."""
        logger.info("Updating repository data...")
        
        try:
            repo_dir = self.data_dir / "AFL-Data-Analysis"
            
            if repo_dir.exists():
                # Update existing repository
                repo = git.Repo(repo_dir)
                repo.remotes.origin.pull()
                logger.info("Updated existing repository")
            else:
                # Clone repository
                git.Repo.clone_from(self.repo_url, repo_dir)
                logger.info("Cloned AFL-Data-Analysis repository")
            
            # Load data using existing pipeline
            from scripts.data_pipeline import AFLDataPipeline
            pipeline = AFLDataPipeline(self.data_dir)
            success = pipeline.run_pipeline()
            
            if success:
                logger.info("✅ Repository data updated successfully")
                return True
            else:
                logger.error("❌ Repository data update failed")
                return False
                
        except Exception as e:
            logger.error(f"Repository update failed: {e}")
            return False
    
    def scrape_afl_tables_season(self, year: int) -> List[Dict]:
        """Scrape match data for a specific year from AFL Tables."""
        logger.info(f"Scraping {year} season from AFL Tables...")
        
        matches = []
        
        try:
            # Use modern parser for 2025+ seasons
            if year >= 2025:
                matches = self._scrape_modern_afl_season(year)
            else:
                matches = self._scrape_legacy_afl_season(year)
            
            logger.info(f"Scraped {len(matches)} matches for {year}")
            return matches
            
        except Exception as e:
            logger.error(f"Error scraping {year}: {e}")
            return matches
    
    def _scrape_modern_afl_season(self, year: int) -> List[Dict]:
        """Scrape modern AFL Tables format (2025+)."""
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from modern_afl_parser import ModernAFLParser
        
        parser = ModernAFLParser()
        url = f"https://afltables.com/afl/seas/{year}.html"
        matches = parser.parse_afl_2025_page(url)
        
        # Filter to only matches we don't already have
        if hasattr(self, '_last_round_in_db'):
            filtered_matches = []
            for match in matches:
                try:
                    match_round = int(match.get('round', 0))
                    if match_round > self._last_round_in_db:
                        filtered_matches.append(match)
                except (ValueError, TypeError):
                    # Include match if round parsing fails
                    filtered_matches.append(match)
            
            logger.info(f"Filtered to {len(filtered_matches)} new matches (after round {self._last_round_in_db})")
            return filtered_matches
        
        return matches
    
    def _scrape_legacy_afl_season(self, year: int) -> List[Dict]:
        """Scrape legacy AFL Tables format (pre-2025)."""
        matches = []
        url = f"https://afltables.com/afl/seas/{year}.html"
        
        response = requests.get(url, timeout=15)
        time.sleep(self.request_delay)
        
        if response.status_code != 200:
            logger.warning(f"Failed to fetch {year} season: HTTP {response.status_code}")
            return matches
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the main results table for legacy format
        tables = soup.find_all('table')
        
        for table in tables:
            rows = table.find_all('tr')
            
            # Skip if table is too small to contain match data
            if len(rows) < 5:
                continue
            
            # Look for tables that contain AFL match indicators
            table_text = table.get_text().lower()
            if not ('round' in table_text and ('def' in table_text or 'drew' in table_text)):
                continue
            
            logger.info(f"Found potential match table with {len(rows)} rows")
            
            for row in rows[1:]:  # Skip header row
                match_data = self._parse_afl_tables_row(row, year)
                if match_data:
                    matches.append(match_data)
        
        return matches
    
    def _parse_afl_tables_row(self, row, year: int) -> Optional[Dict]:
        """Parse a single row from AFL Tables match data."""
        try:
            cells = row.find_all(['td', 'th'])
            
            if len(cells) < 6:
                return None
            
            # Extract text from cells
            cell_texts = [cell.get_text(strip=True) for cell in cells]
            
            # Look for typical AFL Tables format
            # This is a simplified parser - real AFL Tables parsing would be more complex
            match_data = None
            
            # Try to find round, date, teams, and scores
            for i, text in enumerate(cell_texts):
                if 'round' in text.lower() or re.match(r'r\d+', text.lower()):
                    # Found a round indicator
                    if i + 5 < len(cell_texts):
                        # Try to extract match info
                        round_text = text
                        date_text = cell_texts[i + 1] if i + 1 < len(cell_texts) else ''
                        venue_text = cell_texts[i + 2] if i + 2 < len(cell_texts) else ''
                        
                        # Look for team vs team pattern
                        for j in range(i + 3, min(i + 6, len(cell_texts))):
                            if 'def' in cell_texts[j].lower() or 'drew' in cell_texts[j].lower():
                                score_text = cell_texts[j]
                                match_data = self._extract_match_from_score_text(
                                    score_text, year, round_text, date_text, venue_text
                                )
                                break
                
                if match_data:
                    break
            
            return match_data
            
        except Exception as e:
            logger.debug(f"Error parsing row: {e}")
            return None
    
    def _extract_match_from_score_text(self, score_text: str, year: int, 
                                     round_text: str, date_text: str, venue_text: str) -> Optional[Dict]:
        """Extract match details from score text like 'Team1 12.8.80 def Team2 10.5.65' or modern format."""
        try:
            # Handle modern AFL Tables format (2025+) 
            # Example: "Collingwood won by 17 pts" with scores in separate lines
            if 'won by' in score_text.lower():
                winner_match = re.search(r'(\w+(?:\s+\w+)*)\s+won by\s+(\d+)\s+pts?', score_text, re.IGNORECASE)
                if winner_match:
                    winning_team = winner_match.group(1).strip()
                    margin = int(winner_match.group(2))
                    
                    return {
                        'year': year,
                        'round': self._clean_round_text(round_text),
                        'date': self._clean_date_text(date_text),
                        'venue': self._clean_venue_text(venue_text),
                        'winning_team': self._standardize_team_name(winning_team),
                        'margin': margin,
                        'source': 'scraper_afl_tables_modern'
                    }
            
            # Handle legacy format for historical data
            elif 'def' in score_text.lower():
                parts = re.split(r'\s+def\s+', score_text, flags=re.IGNORECASE)
                if len(parts) == 2:
                    winner_part = parts[0].strip()
                    loser_part = parts[1].strip()
                    
                    # Extract team names and scores
                    winner_match = re.match(r'(.+?)\s+(\d+\.\d+\.\d+)$', winner_part)
                    loser_match = re.match(r'(.+?)\s+(\d+\.\d+\.\d+)$', loser_part)
                    
                    if winner_match and loser_match:
                        home_team = winner_match.group(1).strip()
                        home_score = winner_match.group(2)
                        away_team = loser_match.group(1).strip()
                        away_score = loser_match.group(2)
                        
                        # Parse scores (goals.behinds.points format)
                        home_goals, home_behinds = self._parse_score(home_score)
                        away_goals, away_behinds = self._parse_score(away_score)
                        
                        if home_goals is not None and away_goals is not None:
                            return {
                                'year': year,
                                'round': self._clean_round_text(round_text),
                                'date': self._clean_date_text(date_text),
                                'venue': self._clean_venue_text(venue_text),
                                'home_team': self._standardize_team_name(home_team),
                                'away_team': self._standardize_team_name(away_team),
                                'home_total_goals': home_goals,
                                'home_total_behinds': home_behinds,
                                'away_total_goals': away_goals,
                                'away_total_behinds': away_behinds,
                                'winning_team': self._standardize_team_name(home_team),
                                'margin': (home_goals * 6 + home_behinds) - (away_goals * 6 + away_behinds),
                                'source': 'scraper_afl_tables'
                            }
            
            elif 'drew' in score_text.lower():
                # Handle draws for both formats
                if 'drew with' in score_text.lower():
                    # Modern format: "Team1 drew with Team2"
                    teams = re.findall(r'(\w+(?:\s+\w+)*)', score_text)
                    if len(teams) >= 2:
                        return {
                            'year': year,
                            'round': self._clean_round_text(round_text),
                            'date': self._clean_date_text(date_text), 
                            'venue': self._clean_venue_text(venue_text),
                            'home_team': self._standardize_team_name(teams[0]),
                            'away_team': self._standardize_team_name(teams[1]),
                            'margin': 0,
                            'source': 'scraper_afl_tables_modern'
                        }
            
        except Exception as e:
            logger.debug(f"Error extracting match from score text '{score_text}': {e}")
        
        return None
    
    def _parse_score(self, score_str: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse score string like '12.8.80' into goals and behinds."""
        try:
            parts = score_str.split('.')
            if len(parts) >= 2:
                goals = int(parts[0])
                behinds = int(parts[1])
                return goals, behinds
        except (ValueError, IndexError):
            pass
        return None, None
    
    def _clean_round_text(self, round_text: str) -> str:
        """Clean and standardize round text."""
        if not round_text:
            return ''
        # Remove extra whitespace and standardize format
        cleaned = re.sub(r'\s+', ' ', round_text.strip())
        return cleaned
    
    def _clean_date_text(self, date_text: str) -> str:
        """Clean and standardize date text."""
        if not date_text:
            return ''
        # Basic date cleaning
        return re.sub(r'\s+', ' ', date_text.strip())
    
    def _clean_venue_text(self, venue_text: str) -> str:
        """Clean and standardize venue text."""
        if not venue_text:
            return ''
        return re.sub(r'\s+', ' ', venue_text.strip())
    
    def _standardize_team_name(self, team_name: str) -> str:
        """Standardize team names to match database format."""
        if not team_name:
            return ''
        
        # Team name mappings to match existing database
        name_mappings = {
            'Brisbane Lions': 'Brisbane',
            'West Coast Eagles': 'West Coast',
            'Greater Western Sydney Giants': 'Greater Western Sydney',
            'GWS Giants': 'Greater Western Sydney', 
            'GWS': 'Greater Western Sydney',
            'Port Adelaide Power': 'Port Adelaide',
            'Gold Coast Suns': 'Gold Coast',
            'North Melbourne Kangaroos': 'North Melbourne',
            'St Kilda Saints': 'St Kilda',
            # Add more as needed
        }
        
        cleaned_name = team_name.strip()
        return name_mappings.get(cleaned_name, cleaned_name)
    
    def store_scraped_matches(self, matches: List[Dict]) -> bool:
        """Store scraped matches in database, avoiding duplicates."""
        if not matches:
            logger.info("No matches to store")
            return True
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Convert to DataFrame for easier handling
            new_matches_df = pd.DataFrame(matches)
            
            # Check for existing matches to avoid duplicates
            # Try to use source column if it exists, otherwise use all matches
            try:
                existing_matches = pd.read_sql_query("""
                    SELECT year, home_team, away_team, date, round
                    FROM matches
                    WHERE source = 'scraper_afl_tables'
                """, conn)
            except:
                # Fallback if source column doesn't exist
                existing_matches = pd.read_sql_query("""
                    SELECT year, home_team, away_team, date, round
                    FROM matches
                    WHERE year = 2025
                """, conn)
            
            if not existing_matches.empty:
                # Remove duplicates based on key fields
                merge_cols = ['year', 'home_team', 'away_team']
                
                # Simple duplicate removal by checking if combination exists
                existing_keys = set()
                for _, row in existing_matches.iterrows():
                    key = (row.get('year'), row.get('home_team'), row.get('away_team'))
                    existing_keys.add(key)
                
                # Filter out duplicates
                filtered_matches = []
                duplicates_found = 0
                
                for _, match in new_matches_df.iterrows():
                    match_key = (match.get('year'), match.get('home_team'), match.get('away_team'))
                    if match_key not in existing_keys:
                        filtered_matches.append(match.to_dict())
                    else:
                        duplicates_found += 1
                
                if duplicates_found > 0:
                    logger.info(f"Filtered out {duplicates_found} duplicate matches")
                
                new_matches_df = pd.DataFrame(filtered_matches)
            
            if not new_matches_df.empty:
                # Map scraped data to existing database schema
                new_matches_df = self._map_to_database_schema(new_matches_df)
                
                # Store new matches
                new_matches_df.to_sql('matches', conn, if_exists='append', index=False)
                logger.info(f"✅ Stored {len(new_matches_df)} new scraped matches")
                conn.commit()
                return True
            else:
                logger.info("No new matches to store (all were duplicates)")
                return True
                
        except Exception as e:
            logger.error(f"Error storing scraped matches: {e}")
            return False
        finally:
            conn.close()
    
    def _map_to_database_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map scraped data columns to existing database schema."""
        
        # Create a new dataframe with the correct schema
        mapped_df = pd.DataFrame()
        
        # Direct mappings
        column_mappings = {
            'year': 'year',
            'round': 'round',
            'venue': 'venue', 
            'date': 'date',
            'home_team': 'home_team',
            'away_team': 'away_team',
            'home_total_goals': 'home_total_goals',
            'home_total_behinds': 'home_total_behinds',
            'away_total_goals': 'away_total_goals',
            'away_total_behinds': 'away_total_behinds'
        }
        
        # Map existing columns
        for db_col, scraped_col in column_mappings.items():
            if scraped_col in df.columns:
                mapped_df[db_col] = df[scraped_col]
            else:
                # Set default values for missing columns
                if db_col in ['home_total_goals', 'away_total_goals', 'home_total_behinds', 'away_total_behinds']:
                    mapped_df[db_col] = 0
                else:
                    mapped_df[db_col] = None
        
        # Handle attendance (might not be present in scraped data)
        if 'attendance' not in mapped_df.columns:
            mapped_df['attendance'] = None
        
        # Handle quarter-by-quarter data (not available in modern scraper, fill with None)
        quarter_cols = [
            'team_1_q1_goals', 'team_1_q1_behinds', 'team_1_q2_goals', 'team_1_q2_behinds',
            'team_1_q3_goals', 'team_1_q3_behinds', 'team_2_q1_goals', 'team_2_q1_behinds', 
            'team_2_q2_goals', 'team_2_q2_behinds', 'team_2_q3_goals', 'team_2_q3_behinds'
        ]
        
        for col in quarter_cols:
            mapped_df[col] = None
        
        logger.info(f"Mapped {len(df)} matches to database schema with {len(mapped_df.columns)} columns")
        return mapped_df
    
    def run_hybrid_pipeline(self) -> Dict:
        """Run the complete hybrid pipeline."""
        logger.info("🚀 Starting Hybrid AFL Data Pipeline")
        
        start_time = datetime.now()
        
        # Step 1: Analyze current situation
        strategy = self.determine_scraping_strategy()
        
        results = {
            'start_time': start_time.isoformat(),
            'strategy': strategy,
            'repository_updated': False,
            'seasons_scraped': [],
            'total_new_matches': 0,
            'errors': [],
            'success': False
        }
        
        try:
            # Step 2: Update repository data if needed
            if strategy['needs_repository_update']:
                logger.info("📥 Updating repository data...")
                repo_success = self.update_repository_data()
                results['repository_updated'] = repo_success
                
                if not repo_success:
                    results['errors'].append("Repository update failed")
                    logger.warning("Repository update failed, continuing with scraping...")
            
            # Step 3: Scrape current season data if needed
            if strategy['needs_current_season_scraping']:
                for year in strategy['scraping_years']:
                    logger.info(f"🕷️ Scraping {year} season...")
                    
                    # Pass the last round info for filtering
                    if year == self.current_year and strategy['cutoff_info'].get('last_round'):
                        self._last_round_in_db = strategy['cutoff_info']['last_round']
                        logger.info(f"Filtering to matches after round {self._last_round_in_db}")
                    
                    scraped_matches = self.scrape_afl_tables_season(year)
                    
                    if scraped_matches:
                        stored_success = self.store_scraped_matches(scraped_matches)
                        
                        if stored_success:
                            results['seasons_scraped'].append(year)
                            results['total_new_matches'] += len(scraped_matches)
                            logger.info(f"✅ Successfully scraped and stored {len(scraped_matches)} matches for {year}")
                        else:
                            results['errors'].append(f"Failed to store {year} scraped data")
                    else:
                        results['errors'].append(f"No matches scraped for {year}")
            
            # Step 4: Generate final report
            end_time = datetime.now()
            results['end_time'] = end_time.isoformat()
            results['duration_seconds'] = (end_time - start_time).total_seconds()
            results['success'] = len(results['errors']) == 0
            
            # Save detailed report
            report_path = self.data_dir / "hybrid_pipeline_report.json"
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"📊 Pipeline report saved to {report_path}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            results['errors'].append(str(e))
            results['success'] = False
            return results

def main():
    """Main function to run the hybrid pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid AFL Data Pipeline')
    parser.add_argument('--data_dir', type=str, default='afl_data', help='Data directory')
    parser.add_argument('--force_scraping', action='store_true', help='Force scraping even if not needed')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("🏈 HYBRID AFL DATA PIPELINE - EVERGREEN SOLUTION")
    print("="*70)
    
    pipeline = HybridAFLPipeline(data_dir=args.data_dir)
    results = pipeline.run_hybrid_pipeline()
    
    # Print results summary
    print(f"\n📊 Pipeline Results:")
    print(f"- Strategy: {results['strategy']['strategy_description']}")
    print(f"- Repository Updated: {'✅ YES' if results['repository_updated'] else '❌ NO'}")
    print(f"- Seasons Scraped: {results['seasons_scraped'] if results['seasons_scraped'] else 'None'}")
    print(f"- New Matches Added: {results['total_new_matches']}")
    print(f"- Duration: {results.get('duration_seconds', 0):.1f} seconds")
    print(f"- Success: {'✅ YES' if results['success'] else '❌ NO'}")
    
    if results['errors']:
        print(f"\n⚠️ Errors encountered:")
        for error in results['errors']:
            print(f"  - {error}")
    
    print("="*70)
    
    return results

if __name__ == "__main__":
    main()
