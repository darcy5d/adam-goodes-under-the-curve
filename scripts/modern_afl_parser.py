#!/usr/bin/env python3
"""
Modern AFL Tables Parser for 2025+ season format.
Handles the new structure where matches are presented differently.
"""

import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModernAFLParser:
    """Parser for modern AFL Tables format (2025+)."""
    
    def __init__(self):
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
    
    def parse_afl_2025_page(self, url: str = "https://afltables.com/afl/seas/2025.html") -> list:
        """Parse the AFL 2025 page and extract match data."""
        logger.info(f"Parsing AFL 2025 page: {url}")
        
        try:
            response = requests.get(url, timeout=15)
            if response.status_code != 200:
                logger.error(f"Failed to fetch page: {response.status_code}")
                return []
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Get the full text and split by rounds
            page_text = soup.get_text()
            
            matches = []
            
            # Find all rounds in the page
            for round_num in range(1, 25):  # AFL typically has up to 23 rounds + finals
                round_matches = self._extract_round_matches(page_text, round_num)
                matches.extend(round_matches)
            
            logger.info(f"Extracted {len(matches)} matches from 2025 season")
            return matches
            
        except Exception as e:
            logger.error(f"Error parsing AFL 2025 page: {e}")
            return []
    
    def _extract_round_matches(self, page_text: str, round_num: int) -> list:
        """Extract matches for a specific round."""
        
        # Find the round section
        round_pattern = rf'round {round_num}(?!\d)(.*?)(?=round {round_num+1}(?!\d)|$)'
        round_match = re.search(round_pattern, page_text, re.IGNORECASE | re.DOTALL)
        
        if not round_match:
            return []
        
        round_text = round_match.group(1)
        matches = []
        
        # Modern AFL Tables format analysis from debug output:
        # Format appears to be:
        # Team1  Q1 Q2 Q3 Q4 Total
        # Team2  Q1 Q2 Q3 Q4 Total
        # Winner won by X pts [Match stats]
        # Date, Time, Venue info
        
        # Split by match blocks - look for "won by" patterns
        match_results = re.findall(
            r'([^\n]+?)\s+won by\s+(\d+)\s+pts?',
            round_text,
            re.IGNORECASE
        )
        
        for winner, margin in match_results:
            # For each "won by" result, try to find the associated match data
            match_data = self._extract_match_details_around_result(
                round_text, winner, int(margin), round_num
            )
            if match_data:
                matches.append(match_data)
        
        logger.info(f"Round {round_num}: Found {len(matches)} matches")
        return matches
    
    def _extract_match_details_around_result(self, round_text: str, winner: str, margin: int, round_num: int) -> dict:
        """Extract full match details around a 'won by' result."""
        
        try:
            # Clean winner name
            winner_clean = self._standardize_team_name(winner.strip())
            
            # Find the position of this result in the text
            result_pattern = rf'{re.escape(winner)}\s+won by\s+{margin}\s+pts?'
            result_match = re.search(result_pattern, round_text, re.IGNORECASE)
            
            if not result_match:
                return None
            
            result_pos = result_match.start()
            
            # Look backwards to find the teams and scores
            text_before = round_text[:result_pos]
            
            # Find team score lines before this result
            # Look for patterns like: "Team  1.2  3.4  5.6  7.8  50"
            score_lines = re.findall(
                r'([A-Za-z][A-Za-z\s]+?)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)',
                text_before[-500:],  # Look at last 500 chars before result
                re.MULTILINE
            )
            
            if len(score_lines) >= 2:
                # Take the last two score lines (should be the two teams for this match)
                team1_data = score_lines[-2]
                team2_data = score_lines[-1]
                
                team1_name = self._standardize_team_name(team1_data[0].strip())
                team1_total = int(team1_data[5])
                
                team2_name = self._standardize_team_name(team2_data[0].strip())
                team2_total = int(team2_data[5])
                
                # Determine home/away based on winner
                if winner_clean == team1_name:
                    home_team = team1_name
                    away_team = team2_name
                    home_score = team1_total
                    away_score = team2_total
                elif winner_clean == team2_name:
                    home_team = team2_name
                    away_team = team1_name
                    home_score = team2_total
                    away_score = team1_total
                else:
                    # Winner name doesn't exactly match, try fuzzy matching
                    if winner_clean.lower() in team1_name.lower() or team1_name.lower() in winner_clean.lower():
                        home_team = team1_name
                        away_team = team2_name
                        home_score = team1_total
                        away_score = team2_total
                    else:
                        home_team = team2_name
                        away_team = team1_name
                        home_score = team2_total
                        away_score = team1_total
                
                # Extract date and venue info
                date_venue_info = self._extract_date_venue_info(round_text, result_pos)
                
                # Convert total points back to approximate goals/behinds
                home_goals, home_behinds = self._points_to_goals_behinds(home_score)
                away_goals, away_behinds = self._points_to_goals_behinds(away_score)
                
                match_data = {
                    'year': 2025,
                    'round': str(round_num),
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_total_goals': home_goals,
                    'home_total_behinds': home_behinds,
                    'away_total_goals': away_goals,
                    'away_total_behinds': away_behinds,
                    'winning_team': winner_clean,
                    'margin': margin,
                    'source': 'scraper_modern_afl_tables',
                    **date_venue_info
                }
                
                return match_data
        
        except Exception as e:
            logger.debug(f"Error extracting match details for {winner}: {e}")
        
        return None
    
    def _extract_date_venue_info(self, round_text: str, result_pos: int) -> dict:
        """Extract date and venue information near the match result."""
        
        # Look for date patterns after the result
        text_after = round_text[result_pos:result_pos+200]
        
        # Look for venue patterns like "Venue: M.C.G."
        venue_match = re.search(r'Venue:\s*([^\n\r]+)', text_after, re.IGNORECASE)
        venue = venue_match.group(1).strip() if venue_match else ''
        
        # Look for date patterns like "Thu 03-Apr-2025"
        date_match = re.search(r'(\w{3}\s+\d{2}-\w{3}-2025)', text_after)
        date_str = date_match.group(1) if date_match else ''
        
        return {
            'venue': venue,
            'date': date_str
        }
    
    def _points_to_goals_behinds(self, total_points: int) -> tuple:
        """Convert total points back to approximate goals and behinds."""
        # This is an approximation since we don't have the exact breakdown
        # Assume average of 1.5 behinds per goal (realistic AFL ratio)
        
        # Goals are worth 6 points, behinds are worth 1 point
        # If we assume 1.5 behinds per goal: total = goals * 6 + goals * 1.5 = goals * 7.5
        estimated_goals = round(total_points / 7.5)
        estimated_behinds = total_points - (estimated_goals * 6)
        
        # Ensure behinds is non-negative
        if estimated_behinds < 0:
            estimated_goals = total_points // 6
            estimated_behinds = total_points % 6
        
        return estimated_goals, estimated_behinds
    
    def _standardize_team_name(self, team_name: str) -> str:
        """Standardize team names to match database format."""
        if not team_name:
            return ''
        
        cleaned = team_name.strip()
        return self.team_name_mappings.get(cleaned, cleaned)

def test_modern_parser():
    """Test the modern parser on AFL 2025 data."""
    
    print("🧪 Testing Modern AFL Parser...")
    
    parser = ModernAFLParser()
    matches = parser.parse_afl_2025_page()
    
    print(f"📊 Results: {len(matches)} matches found")
    
    if matches:
        print("\n🏈 Sample matches:")
        for i, match in enumerate(matches[:5]):  # Show first 5 matches
            print(f"  {i+1}. Round {match['round']}: {match['home_team']} vs {match['away_team']}")
            print(f"     Score: {match['home_total_goals']}.{match['home_total_behinds']} to {match['away_total_goals']}.{match['away_total_behinds']}")
            print(f"     Winner: {match['winning_team']} by {match['margin']} pts")
            print(f"     Venue: {match.get('venue', 'Unknown')}")
            print()
    
    # Group by round
    if matches:
        rounds = {}
        for match in matches:
            round_num = match['round']
            if round_num not in rounds:
                rounds[round_num] = 0
            rounds[round_num] += 1
        
        print("📈 Matches by round:")
        for round_num in sorted(rounds.keys(), key=lambda x: int(x) if x.isdigit() else 999):
            print(f"  Round {round_num}: {rounds[round_num]} matches")
    
    return matches

if __name__ == "__main__":
    test_modern_parser()
