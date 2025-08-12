#!/usr/bin/env python3
"""
Modern AFL Player Data Scraper for 2025+ seasons.
Extends the hybrid pipeline to handle player statistics scraping.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import time
import logging
from typing import List, Dict, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModernPlayerScraper:
    """Scraper for modern AFL player statistics from AFL Tables."""
    
    def __init__(self):
        self.base_url = "https://afltables.com"
        self.request_delay = 2.0  # seconds between requests
        
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
    
    def scrape_player_data_for_rounds(self, year: int, start_round: int = 5, end_round: int = 23) -> List[Dict]:
        """Scrape player data for specific rounds of a season."""
        logger.info(f"Scraping player data for {year} rounds {start_round}-{end_round}...")
        
        all_player_data = []
        
        # Get team list for the year
        teams = self._get_teams_for_year(year)
        
        for team in teams:
            logger.info(f"Scraping {team} player data...")
            team_player_data = self._scrape_team_player_data(team, year, start_round, end_round)
            all_player_data.extend(team_player_data)
            
            # Be respectful to the server
            time.sleep(self.request_delay)
        
        logger.info(f"Scraped {len(all_player_data)} player records for {year} rounds {start_round}-{end_round}")
        return all_player_data
    
    def _get_teams_for_year(self, year: int) -> List[str]:
        """Get list of AFL teams for a specific year."""
        # Standard AFL teams for 2025
        teams = [
            'Adelaide', 'Brisbane', 'Carlton', 'Collingwood', 'Essendon',
            'Fremantle', 'Geelong', 'Gold Coast', 'Greater Western Sydney', 'Hawthorn',
            'Melbourne', 'North Melbourne', 'Port Adelaide', 'Richmond', 'St Kilda',
            'Sydney', 'West Coast', 'Western Bulldogs'
        ]
        return teams
    
    def _scrape_team_player_data(self, team: str, year: int, start_round: int, end_round: int) -> List[Dict]:
        """Scrape player data for a specific team and round range."""
        
        player_data = []
        
        try:
            # AFL Tables team page URL pattern
            team_url_name = self._get_team_url_name(team)
            url = f"{self.base_url}/afl/stats/{year}/{team_url_name}.html"
            
            response = requests.get(url, timeout=15)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {team} data: HTTP {response.status_code}")
                return player_data
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find player statistics tables
            tables = soup.find_all('table')
            
            for table in tables:
                # Look for player stats tables (they typically have player names and stats)
                if self._is_player_stats_table(table):
                    round_player_data = self._parse_player_stats_table(table, team, year, start_round, end_round)
                    player_data.extend(round_player_data)
            
        except Exception as e:
            logger.error(f"Error scraping {team} player data: {e}")
        
        return player_data
    
    def _get_team_url_name(self, team: str) -> str:
        """Convert team name to AFL Tables URL format."""
        url_mappings = {
            'Adelaide': 'adelaide',
            'Brisbane': 'brisbanel',
            'Carlton': 'carlton',
            'Collingwood': 'collingwood',
            'Essendon': 'essendon',
            'Fremantle': 'fremantle',
            'Geelong': 'geelong',
            'Gold Coast': 'goldcoast',
            'Greater Western Sydney': 'gws',
            'Hawthorn': 'hawthorn',
            'Melbourne': 'melbourne',
            'North Melbourne': 'kangaroos',
            'Port Adelaide': 'padelaide',
            'Richmond': 'richmond',
            'St Kilda': 'stkilda',
            'Sydney': 'swans',
            'West Coast': 'westcoast',
            'Western Bulldogs': 'bullldogs'
        }
        return url_mappings.get(team, team.lower().replace(' ', ''))
    
    def _is_player_stats_table(self, table) -> bool:
        """Check if a table contains player statistics."""
        table_text = table.get_text().lower()
        
        # Look for common player stats indicators
        player_indicators = [
            'kicks', 'marks', 'handballs', 'disposals', 'goals', 'behinds',
            'tackles', 'hit outs', 'inside 50s', 'clangers'
        ]
        
        indicator_count = sum(1 for indicator in player_indicators if indicator in table_text)
        return indicator_count >= 4  # If we find at least 4 indicators, likely a player stats table
    
    def _parse_player_stats_table(self, table, team: str, year: int, start_round: int, end_round: int) -> List[Dict]:
        """Parse a player statistics table."""
        
        player_records = []
        
        try:
            rows = table.find_all('tr')
            
            # Try to find header row to understand column structure
            header_row = None
            for row in rows:
                row_text = row.get_text().lower()
                if 'kicks' in row_text and 'marks' in row_text:
                    header_row = row
                    break
            
            if not header_row:
                return player_records
            
            # Extract column headers
            headers = [th.get_text(strip=True) for th in header_row.find_all(['th', 'td'])]
            
            # Process data rows
            for row in rows[rows.index(header_row) + 1:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) < len(headers) * 0.5:  # Skip rows with too few cells
                    continue
                
                cell_values = [cell.get_text(strip=True) for cell in cells]
                
                # Try to extract player information
                player_record = self._extract_player_record(headers, cell_values, team, year)
                if player_record:
                    # Check if this record is from the rounds we want
                    if self._is_record_in_round_range(player_record, start_round, end_round):
                        player_records.append(player_record)
        
        except Exception as e:
            logger.debug(f"Error parsing player stats table for {team}: {e}")
        
        return player_records
    
    def _extract_player_record(self, headers: List[str], values: List[str], team: str, year: int) -> Optional[Dict]:
        """Extract a single player record from table row."""
        
        if len(values) < 3:
            return None
        
        try:
            # Basic player record structure
            record = {
                'team': self._standardize_team_name(team),
                'year': year
            }
            
            # Map common columns
            column_mappings = {
                'player': ['player', 'name', 'player name'],
                'round': ['round', 'rnd', 'r'],
                'opponent': ['opponent', 'opp', 'vs'],
                'kicks': ['kicks', 'k'],
                'marks': ['marks', 'mk'],
                'handballs': ['handballs', 'hb'],
                'disposals': ['disposals', 'disp', 'd'],
                'goals': ['goals', 'g'],
                'behinds': ['behinds', 'b'],
                'hit_outs': ['hit outs', 'ho', 'hitouts'],
                'tackles': ['tackles', 't'],
                'rebound_50s': ['rebound 50s', 'r50'],
                'inside_50s': ['inside 50s', 'i50'],
                'clearances': ['clearances', 'cl'],
                'clangers': ['clangers', 'clang'],
                'free_kicks_for': ['free kicks for', 'ff'],
                'free_kicks_against': ['free kicks against', 'fa'],
                'contested_possessions': ['contested possessions', 'cp'],
                'uncontested_possessions': ['uncontested possessions', 'up'],
                'contested_marks': ['contested marks', 'cm'],
                'marks_inside_50': ['marks inside 50', 'mi50'],
                'one_percenters': ['one percenters', '1%'],
                'bounces': ['bounces', 'bo'],
                'goal_assist': ['goal assist', 'ga']
            }
            
            # Map values to record
            for i, header in enumerate(headers):
                if i < len(values):
                    header_clean = header.lower().strip()
                    value = values[i].strip()
                    
                    # Find matching column
                    for record_key, possible_headers in column_mappings.items():
                        if any(ph in header_clean for ph in possible_headers):
                            # Convert numeric values
                            if record_key in ['year', 'round', 'kicks', 'marks', 'handballs', 'disposals', 
                                            'goals', 'behinds', 'hit_outs', 'tackles', 'rebound_50s', 
                                            'inside_50s', 'clearances', 'clangers', 'free_kicks_for',
                                            'free_kicks_against', 'contested_possessions', 'uncontested_possessions',
                                            'contested_marks', 'marks_inside_50', 'one_percenters', 'bounces',
                                            'goal_assist']:
                                try:
                                    record[record_key] = int(value) if value.isdigit() else None
                                except ValueError:
                                    record[record_key] = None
                            else:
                                record[record_key] = value
            
            # Only return record if we have essential data
            if 'player' in record or len([k for k, v in record.items() if v is not None]) > 3:
                return record
        
        except Exception as e:
            logger.debug(f"Error extracting player record: {e}")
        
        return None
    
    def _is_record_in_round_range(self, record: Dict, start_round: int, end_round: int) -> bool:
        """Check if a player record is from the desired round range."""
        
        record_round = record.get('round')
        if record_round is None:
            return True  # Include if round info is missing
        
        try:
            round_num = int(record_round)
            return start_round <= round_num <= end_round
        except (ValueError, TypeError):
            return True  # Include if round parsing fails
    
    def _standardize_team_name(self, team_name: str) -> str:
        """Standardize team names to match database format."""
        if not team_name:
            return ''
        
        cleaned = team_name.strip()
        return self.team_name_mappings.get(cleaned, cleaned)

def test_player_scraper():
    """Test the player scraper on a small sample."""
    
    print("🧪 Testing Modern Player Scraper...")
    
    scraper = ModernPlayerScraper()
    
    # Test with one team for a few rounds
    test_data = scraper._scrape_team_player_data('Richmond', 2025, 5, 7)
    
    print(f"📊 Test Results: {len(test_data)} player records found")
    
    if test_data:
        print("\n🏈 Sample player records:")
        for i, record in enumerate(test_data[:3]):
            print(f"  {i+1}. {record.get('player', 'Unknown')} - Round {record.get('round', 'N/A')}")
            print(f"     Stats: {record.get('kicks', 0)} kicks, {record.get('marks', 0)} marks, {record.get('goals', 0)} goals")
    
    return test_data

if __name__ == "__main__":
    test_player_scraper()
