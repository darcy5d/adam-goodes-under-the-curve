#!/usr/bin/env python3
"""
Simple AFL Player Scraper using direct URLs.
Scrapes individual player performance data from AFL Tables match stats pages.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import time
import logging
from typing import List, Dict, Optional
import sys
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimplePlayerScraper:
    """Simple scraper for individual player match statistics."""
    
    def __init__(self):
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
    
    def load_match_urls(self, filename: str = 'final_match_urls.txt') -> List[str]:
        """Load match URLs from file."""
        try:
            with open(filename, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(urls)} match URLs from {filename}")
            return urls
        except FileNotFoundError:
            logger.error(f"File {filename} not found")
            return []
    
    def scrape_player_data_from_match(self, match_url: str) -> List[Dict]:
        """Scrape player data from a specific match stats page."""
        logger.info(f"Scraping: {match_url}")
        
        try:
            response = requests.get(match_url, timeout=15)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch {match_url}: HTTP {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract match details
            match_details = self._extract_match_details(soup, match_url)
            
            # Find and parse player statistics tables
            player_data = []
            tables = soup.find_all('table')
            
            for table in tables:
                table_players = self._parse_player_stats_table(table, match_details)
                player_data.extend(table_players)
            
            logger.info(f"Extracted {len(player_data)} player records")
            return player_data
        
        except Exception as e:
            logger.error(f"Error scraping {match_url}: {e}")
            return []
    
    def _extract_match_details(self, soup, match_url: str) -> Dict:
        """Extract match details from the stats page."""
        details = {'url': match_url}
        
        try:
            page_text = soup.get_text()
            
            # Extract round information - look for "Round: X" pattern
            import re
            round_match = re.search(r'Round:\s*(\d+)', page_text)
            if round_match:
                details['round'] = int(round_match.group(1))
            else:
                # Try alternative patterns
                round_match = re.search(r'Rd\s+(\d+)', page_text)
                if round_match:
                    details['round'] = int(round_match.group(1))
            
            # Extract venue
            venue_match = re.search(r'Venue:\s*([^→\n]+)', page_text)
            if venue_match:
                details['venue'] = venue_match.group(1).strip()
            
            # Extract date information
            date_match = re.search(r'Date:\s*([^→\n]+)', page_text)
            if date_match:
                details['date'] = date_match.group(1).strip()
            
            # Try to extract team names from headers
            team_headers = []
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4']):
                text = element.get_text().strip()
                if 'Match Statistics' in text:
                    team_name = text.replace('Match Statistics', '').strip()
                    if team_name:
                        team_headers.append(team_name)
            
            if len(team_headers) >= 2:
                details['home_team'] = team_headers[0]
                details['away_team'] = team_headers[1]
        
        except Exception as e:
            logger.debug(f"Error extracting match details: {e}")
        
        return details
    
    def _parse_player_stats_table(self, table, match_details: Dict) -> List[Dict]:
        """Parse a player statistics table."""
        player_records = []
        
        try:
            # Check if this table contains player statistics
            table_text = table.get_text()
            if not any(keyword in table_text for keyword in ['Match Statistics', 'Player', 'KI', 'MK']):
                return player_records
            
            rows = table.find_all('tr')
            if len(rows) < 3:  # Need header + data rows
                return player_records
            
            # Find header row with player stats columns
            header_row = None
            for row in rows:
                row_text = row.get_text()
                if 'Player' in row_text and ('KI' in row_text or 'kicks' in row_text.lower()):
                    header_row = row
                    break
            
            if not header_row:
                return player_records
            
            # Extract headers
            header_cells = header_row.find_all(['th', 'td'])
            headers = [cell.get_text(strip=True) for cell in header_cells]
            
            # Determine team name
            team_name = self._determine_team_from_table(table, match_details)
            
            # Process data rows
            data_start_idx = rows.index(header_row) + 1
            for row in rows[data_start_idx:]:
                cells = row.find_all(['td', 'th'])
                if len(cells) < len(headers) * 0.3:  # Skip rows with too few cells
                    continue
                
                cell_values = [cell.get_text(strip=True) for cell in cells]
                
                # Skip summary rows
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
        
        # Look for team name in table context
        table_text = table.get_text()
        
        # Check for explicit team mentions
        teams = ['Adelaide', 'Brisbane', 'Carlton', 'Collingwood', 'Essendon',
                'Fremantle', 'Geelong', 'Gold Coast', 'Greater Western Sydney', 
                'Hawthorn', 'Melbourne', 'North Melbourne', 'Port Adelaide', 
                'Richmond', 'St Kilda', 'Sydney', 'West Coast', 'Western Bulldogs']
        
        for team in teams:
            if team in table_text:
                return team
        
        # Fallback to match details
        return match_details.get('home_team', 'Unknown')
    
    def _parse_player_row(self, headers: List[str], values: List[str], team_name: str, match_details: Dict) -> Optional[Dict]:
        """Parse a single player data row."""
        
        try:
            if len(values) < 2:
                return None
            
            # Basic record structure
            record = {
                'team': team_name,
                'year': 2025,
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
                        if db_column == 'player_name':
                            # Clean up player name
                            record[db_column] = value if value and value != '' else None
                        elif db_column in ['jersey_num', 'kicks', 'marks', 'handballs', 'disposals', 
                                         'goals', 'behinds', 'hit_outs', 'tackles', 'rebound_50s', 
                                         'inside_50s', 'clearances', 'clangers', 'free_kicks_for',
                                         'free_kicks_against', 'brownlow_votes', 'contested_possessions', 
                                         'uncontested_possessions', 'contested_marks', 'marks_inside_50', 
                                         'one_percenters', 'bounces', 'goal_assist', 'percentage_of_game_played']:
                            try:
                                if value == '' or value == ' ' or value == '↑' or value == '↓':
                                    record[db_column] = None
                                elif value.isdigit():
                                    record[db_column] = int(value)
                                else:
                                    record[db_column] = None
                            except (ValueError, TypeError):
                                record[db_column] = None
            
            # Only return record if we have essential data
            if record.get('player_name') and record.get('team') and record.get('round'):
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
            return None
    
    def scrape_all_matches(self, url_file: str = 'final_match_urls.txt') -> List[Dict]:
        """Scrape all matches from the URL file."""
        logger.info("Starting comprehensive player data scraping")
        
        urls = self.load_match_urls(url_file)
        if not urls:
            logger.error("No URLs loaded")
            return []
        
        all_player_data = []
        
        for i, url in enumerate(urls):
            logger.info(f"Processing match {i+1}/{len(urls)}")
            
            match_player_data = self.scrape_player_data_from_match(url)
            all_player_data.extend(match_player_data)
            
            # Be respectful to the server
            time.sleep(self.request_delay)
            
            # Progress update every 10 matches
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i+1}/{len(urls)} matches, {len(all_player_data)} player records so far")
        
        logger.info(f"Scraping complete: {len(all_player_data)} total player records")
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
    """Main function to run the simple player scraper."""
    scraper = SimplePlayerScraper()
    
    print("🏈 Simple AFL Player Match Statistics Scraper")
    print("=" * 60)
    
    # Test with just first few URLs
    print("Starting with a test run (first 5 matches)...")
    
    urls = scraper.load_match_urls()
    if not urls:
        print("❌ No URLs found")
        return
    
    print(f"Found {len(urls)} total URLs")
    
    # Test with first 5 matches
    test_urls = urls[:5]
    all_player_data = []
    
    for i, url in enumerate(test_urls):
        print(f"\nProcessing test match {i+1}/{len(test_urls)}: {url}")
        match_data = scraper.scrape_player_data_from_match(url)
        all_player_data.extend(match_data)
        time.sleep(1)  # Reduced delay for testing
    
    if all_player_data:
        print(f"\n📊 Test Results: {len(all_player_data)} player records")
        
        # Show sample
        print("\n🏈 Sample player records:")
        for i, record in enumerate(all_player_data[:5]):
            print(f"  {i+1}. {record.get('player_name', 'Unknown')} - {record.get('team', 'Unknown')} vs {record.get('opponent', 'Unknown')}")
            print(f"     Round {record.get('round', 'N/A')}: {record.get('kicks', 0)} kicks, {record.get('marks', 0)} marks")
        
        # Ask user if they want to proceed with full scraping
        response = input(f"\nTest successful! Proceed with full scraping of all {len(urls)} matches? (y/n): ")
        
        if response.lower() in ['y', 'yes']:
            print("\n🚀 Starting full scraping...")
            full_data = scraper.scrape_all_matches()
            
            if full_data:
                scraper.store_player_data(full_data)
                print(f"\n✅ Complete! {len(full_data)} player records stored in database")
            else:
                print("\n❌ Full scraping failed")
        else:
            print("\n⏸️  Full scraping cancelled by user")
    else:
        print("\n❌ Test failed - no player data scraped")

if __name__ == "__main__":
    main()
