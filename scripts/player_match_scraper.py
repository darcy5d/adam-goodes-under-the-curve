#!/usr/bin/env python3
"""
AFL Player Match Statistics Scraper for 2025+ seasons.
Scrapes individual player performance data from AFL Tables match stats pages.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import time
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PlayerMatchScraper:
    """Scraper for individual player match statistics from AFL Tables."""
    
    def __init__(self):
        self.base_url = "https://afltables.com"
        self.request_delay = 2.0  # seconds between requests
        
        # Column mapping from AFL Tables to our database
        self.column_mapping = {
            '#': 'jersey_num',
            'Player': 'player_name',
            'KI': 'kicks', 
            'MK': 'marks',
            'HB': 'handballs',
            'DI': 'disposals', 
            'GL': 'goals',
            'BH': 'behinds',
            'HO': 'hit_outs',
            'TK': 'tackles',
            'RB': 'rebound_50s',
            'IF': 'inside_50s', 
            'CL': 'clearances',
            'CG': 'clangers',
            'FF': 'free_kicks_for',
            'FA': 'free_kicks_against',
            'BR': 'brownlow_votes',
            'CP': 'contested_possessions',
            'UP': 'uncontested_possessions', 
            'CM': 'contested_marks',
            'MI': 'marks_inside_50',
            '1%': 'one_percenters',
            'BO': 'bounces',
            'GA': 'goal_assist',
            '%P': 'percentage_of_game_played'
        }
        
        self.team_name_mappings = {
            'Brisbane Lions': 'Brisbane',
            'West Coast Eagles': 'West Coast', 
            'Greater Western Sydney Giants': 'Greater Western Sydney',
            'GWS Giants': 'Greater Western Sydney',
            'GWS': 'Greater Western Sydney',
            'Port Adelaide Power': 'Port Adelaide',
            'Gold Coast Suns': 'Gold Coast',
            'North Melbourne Kangaroos': 'North Melbourne',
            'St Kilda Saints': 'St Kilda',
        }
    
    def get_match_stats_urls_for_rounds(self, year: int, start_round: int = 5, end_round: int = 23) -> List[Tuple[str, Dict]]:
        """Get all match stats URLs for specified rounds."""
        logger.info(f"Getting match stats URLs for {year} rounds {start_round}-{end_round}...")
        
        # Get the season page
        season_url = f"{self.base_url}/afl/seas/{year}.html"
        response = requests.get(season_url)
        
        if response.status_code != 200:
            logger.error(f"Failed to fetch season page: {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        match_urls = []
        
        # Find all match stats links
        for link in soup.find_all('a', href=True):
            href = link['href']
            if 'stats/games' in href and str(year) in href:
                # Extract match info from the page context
                match_info = self._extract_match_info_from_context(link, soup)
                if match_info and start_round <= match_info.get('round', 0) <= end_round:
                    full_url = f"{self.base_url}{href}" if href.startswith('../') else href
                    match_urls.append((full_url, match_info))
        
        logger.info(f"Found {len(match_urls)} match stats URLs for rounds {start_round}-{end_round}")
        return match_urls
    
    def _extract_match_info_from_context(self, link, soup) -> Optional[Dict]:
        """Extract match information from the page context around a match stats link."""
        try:
            # Find the parent context to get team names, round, etc.
            parent = link.parent
            while parent and parent.name != 'tr':
                parent = parent.parent
            
            if not parent:
                return None
            
            # Extract text from the row to find team names and round
            row_text = parent.get_text()
            
            # Look for round information
            round_match = re.search(r'round\s+(\d+)', row_text.lower())
            if not round_match:
                # Try alternative patterns
                round_match = re.search(r'rd\s+(\d+)', row_text.lower())
            
            if round_match:
                round_num = int(round_match.group(1))
                
                # Extract team names (this is approximate, will be refined from actual match page)
                return {
                    'round': round_num,
                    'context_text': row_text[:200]  # Store context for debugging
                }
        
        except Exception as e:
            logger.debug(f"Error extracting match info: {e}")
        
        return None
    
    def scrape_player_data_from_match(self, match_url: str, match_info: Dict) -> List[Dict]:
        """Scrape player data from a specific match stats page."""
        logger.info(f"Scraping player data from: {match_url}")
        
        try:
            response = requests.get(match_url, timeout=15)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {match_url}: HTTP {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract match details from the page
            match_details = self._extract_match_details(soup)
            
            # Find player statistics tables
            player_data = []
            tables = soup.find_all('table')
            
            for table in tables:
                table_players = self._parse_player_stats_table(table, match_details)
                player_data.extend(table_players)
            
            logger.info(f"Extracted {len(player_data)} player records from {match_url}")
            return player_data
        
        except Exception as e:
            logger.error(f"Error scraping {match_url}: {e}")
            return []
    
    def _extract_match_details(self, soup) -> Dict:
        """Extract match details from the stats page."""
        details = {}
        
        try:
            # Get page title and header info
            page_text = soup.get_text()
            
            # Extract round information
            round_match = re.search(r'Round:\s*(\d+)', page_text)
            if round_match:
                details['round'] = int(round_match.group(1))
            
            # Extract venue
            venue_match = re.search(r'Venue:\s*([^→]+)', page_text)
            if venue_match:
                details['venue'] = venue_match.group(1).strip()
            
            # Extract date
            date_match = re.search(r'Date:\s*([^→]+)', page_text)
            if date_match:
                details['date'] = date_match.group(1).strip()
            
            # Extract attendance
            att_match = re.search(r'Attendance:\s*(\d+)', page_text)
            if att_match:
                details['attendance'] = int(att_match.group(1))
            
            # Extract team names from the page structure
            teams = self._extract_team_names(soup)
            if teams:
                details.update(teams)
        
        except Exception as e:
            logger.debug(f"Error extracting match details: {e}")
        
        return details
    
    def _extract_team_names(self, soup) -> Dict:
        """Extract team names from the match page."""
        teams = {}
        
        try:
            # Look for team statistics headers
            for header in soup.find_all(['h2', 'h3', 'h4']):
                header_text = header.get_text().strip()
                if 'Match Statistics' in header_text:
                    team_name = header_text.replace('Match Statistics', '').strip()
                    if team_name:
                        # Determine if this is home or away team (simplified)
                        if not teams.get('home_team'):
                            teams['home_team'] = self._standardize_team_name(team_name)
                        else:
                            teams['away_team'] = self._standardize_team_name(team_name)
        
        except Exception as e:
            logger.debug(f"Error extracting team names: {e}")
        
        return teams
    
    def _parse_player_stats_table(self, table, match_details: Dict) -> List[Dict]:
        """Parse a player statistics table."""
        player_records = []
        
        try:
            # Check if this table contains player statistics
            table_text = table.get_text()
            if not ('Match Statistics' in table_text or 'Player' in table_text):
                return player_records
            
            # Check if table has the right structure
            rows = table.find_all('tr')
            if len(rows) < 3:  # Need header + at least 2 data rows
                return player_records
            
            # Find header row
            header_row = None
            for row in rows:
                row_text = row.get_text()
                if 'KI' in row_text and 'MK' in row_text and 'Player' in row_text:
                    header_row = row
                    break
            
            if not header_row:
                return player_records
            
            # Extract headers
            header_cells = header_row.find_all(['th', 'td'])
            headers = [cell.get_text(strip=True) for cell in header_cells]
            
            # Determine team name from table context
            team_name = self._determine_team_from_table(table, match_details)
            
            # Process data rows
            data_start_idx = rows.index(header_row) + 1
            for row in rows[data_start_idx:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) < len(headers) * 0.5:  # Skip rows with too few cells
                    continue
                
                cell_values = [cell.get_text(strip=True) for cell in cells]
                
                # Skip totals, rushed, and other summary rows
                if (len(cell_values) > 0 and 
                    cell_values[0].lower() in ['totals', 'rushed', 'opposition']):
                    continue
                
                # Parse player record
                player_record = self._parse_player_row(headers, cell_values, team_name, match_details)
                if player_record:
                    player_records.append(player_record)
        
        except Exception as e:
            logger.debug(f"Error parsing player stats table: {e}")
        
        return player_records
    
    def _determine_team_from_table(self, table, match_details: Dict) -> str:
        """Determine which team this table represents."""
        
        # Look for team name in table headers or nearby text
        table_text = table.get_text()
        
        # Check for team names in the table
        for team in ['Adelaide', 'Brisbane', 'Carlton', 'Collingwood', 'Essendon',
                     'Fremantle', 'Geelong', 'Gold Coast', 'Greater Western Sydney', 
                     'Hawthorn', 'Melbourne', 'North Melbourne', 'Port Adelaide', 
                     'Richmond', 'St Kilda', 'Sydney', 'West Coast', 'Western Bulldogs']:
            if team in table_text:
                return self._standardize_team_name(team)
        
        # Fallback: check match details
        if 'home_team' in match_details:
            return match_details['home_team']
        
        return 'Unknown'
    
    def _parse_player_row(self, headers: List[str], values: List[str], team_name: str, match_details: Dict) -> Optional[Dict]:
        """Parse a single player data row."""
        
        try:
            if len(values) < 3:
                return None
            
            # Basic record structure
            record = {
                'team': team_name,
                'year': 2025,  # Hardcoded for now
                'round': match_details.get('round'),
                'opponent': self._get_opponent(team_name, match_details),
                'result': None,  # Will be determined later
                'games_played': None  # Not available in match data
            }
            
            # Map values to database columns
            for i, header in enumerate(headers):
                if i < len(values):
                    value = values[i].strip()
                    
                    # Map header to database column
                    db_column = self.column_mapping.get(header)
                    if db_column:
                        # Convert to appropriate type
                        if db_column == 'player_name':
                            record[db_column] = value
                        elif db_column in ['jersey_num', 'kicks', 'marks', 'handballs', 'disposals', 
                                         'goals', 'behinds', 'hit_outs', 'tackles', 'rebound_50s', 
                                         'inside_50s', 'clearances', 'clangers', 'free_kicks_for',
                                         'free_kicks_against', 'brownlow_votes', 'contested_possessions', 
                                         'uncontested_possessions', 'contested_marks', 'marks_inside_50', 
                                         'one_percenters', 'bounces', 'goal_assist', 'percentage_of_game_played']:
                            try:
                                # Handle empty values
                                if value == '' or value == ' ':
                                    record[db_column] = None
                                else:
                                    record[db_column] = int(value) if value.isdigit() else None
                            except ValueError:
                                record[db_column] = None
            
            # Only return record if we have essential data
            if record.get('player_name') and record.get('team'):
                return record
        
        except Exception as e:
            logger.debug(f"Error parsing player row: {e}")
        
        return None
    
    def _get_opponent(self, team_name: str, match_details: Dict) -> str:
        """Determine the opponent team."""
        home_team = match_details.get('home_team')
        away_team = match_details.get('away_team')
        
        if team_name == home_team:
            return away_team
        elif team_name == away_team:
            return home_team
        else:
            return 'Unknown'
    
    def _standardize_team_name(self, team_name: str) -> str:
        """Standardize team names to match database format."""
        if not team_name:
            return ''
        
        cleaned = team_name.strip()
        return self.team_name_mappings.get(cleaned, cleaned)
    
    def scrape_all_missing_player_data(self, year: int = 2025, start_round: int = 5, end_round: int = 23) -> List[Dict]:
        """Scrape all missing player data for specified rounds."""
        logger.info(f"Starting comprehensive player data scraping for {year} rounds {start_round}-{end_round}")
        
        all_player_data = []
        
        # Get all match URLs
        match_urls = self.get_match_stats_urls_for_rounds(year, start_round, end_round)
        
        if not match_urls:
            logger.error("No match URLs found")
            return all_player_data
        
        # Scrape each match
        for i, (match_url, match_info) in enumerate(match_urls):
            logger.info(f"Processing match {i+1}/{len(match_urls)}: {match_url}")
            
            match_player_data = self.scrape_player_data_from_match(match_url, match_info)
            all_player_data.extend(match_player_data)
            
            # Be respectful to the server
            time.sleep(self.request_delay)
        
        logger.info(f"Completed scraping: {len(all_player_data)} total player records")
        return all_player_data
    
    def store_player_data(self, player_data: List[Dict], db_path: str = 'afl_data/afl_database.db'):
        """Store scraped player data in the database."""
        logger.info(f"Storing {len(player_data)} player records in database...")
        
        if not player_data:
            logger.warning("No player data to store")
            return
        
        try:
            conn = sqlite3.connect(db_path)
            
            # Convert to DataFrame
            df = pd.DataFrame(player_data)
            
            # Remove any existing data for these rounds to avoid duplicates
            if not df.empty and 'round' in df.columns:
                rounds = df['round'].dropna().unique()
                year = df['year'].iloc[0] if 'year' in df.columns else 2025
                
                for round_num in rounds:
                    conn.execute('DELETE FROM players WHERE year = ? AND round = ?', (year, round_num))
                
                logger.info(f"Cleared existing data for rounds: {list(rounds)}")
            
            # Insert new data
            df.to_sql('players', conn, if_exists='append', index=False)
            conn.commit()
            conn.close()
            
            logger.info(f"Successfully stored {len(player_data)} player records")
        
        except Exception as e:
            logger.error(f"Error storing player data: {e}")
            raise

def main():
    """Main function to run the player data scraper."""
    scraper = PlayerMatchScraper()
    
    print("🏈 AFL Player Match Statistics Scraper")
    print("=" * 50)
    
    # Scrape missing 2025 player data
    player_data = scraper.scrape_all_missing_player_data(
        year=2025, 
        start_round=5, 
        end_round=23
    )
    
    if player_data:
        print(f"\n📊 Scraped {len(player_data)} player records")
        
        # Show sample
        print("\n🏈 Sample player records:")
        for i, record in enumerate(player_data[:3]):
            print(f"  {i+1}. {record.get('player_name', 'Unknown')} - {record.get('team', 'Unknown')} vs {record.get('opponent', 'Unknown')}")
            print(f"     Round {record.get('round', 'N/A')}: {record.get('kicks', 0)} kicks, {record.get('marks', 0)} marks, {record.get('goals', 0)} goals")
        
        # Store in database
        scraper.store_player_data(player_data)
        
        print(f"\n✅ Player data scraping complete!")
        print(f"   {len(player_data)} records added to database")
    else:
        print("\n❌ No player data scraped")

if __name__ == "__main__":
    main()
