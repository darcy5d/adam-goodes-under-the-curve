#!/usr/bin/env python3
"""
Debug script to understand AFL Tables structure for 2025 season.
"""

import requests
from bs4 import BeautifulSoup
import re
import time

def debug_afl_tables_2025():
    """Debug the AFL Tables 2025 page to understand its structure."""
    
    print("🔍 Debugging AFL Tables 2025 page structure...")
    
    url = "https://afltables.com/afl/seas/2025.html"
    
    try:
        response = requests.get(url, timeout=15)
        print(f"✅ Page loaded: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ Failed to load page: {response.status_code}")
            return
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Look for tables that might contain match data
        tables = soup.find_all('table')
        print(f"📊 Found {len(tables)} tables")
        
        match_tables = []
        
        for i, table in enumerate(tables):
            table_text = table.get_text().lower()
            
            # Look for AFL match indicators
            has_round = 'round' in table_text or any(f'r{j}' in table_text for j in range(1, 25))
            has_teams = any(team in table_text for team in ['collingwood', 'richmond', 'essendon', 'carlton', 'melbourne'])
            has_scores = 'def' in table_text or 'drew' in table_text
            
            if has_round and (has_teams or has_scores):
                match_tables.append((i, table))
                print(f"🏈 Table {i}: Potential match table (round: {has_round}, teams: {has_teams}, scores: {has_scores})")
        
        print(f"\n🎯 Found {len(match_tables)} potential match tables")
        
        # Examine the most promising tables
        for table_idx, (i, table) in enumerate(match_tables[:3]):  # Look at first 3 promising tables
            print(f"\n🔍 Analyzing Table {i}:")
            
            rows = table.find_all('tr')
            print(f"   Rows: {len(rows)}")
            
            for row_idx, row in enumerate(rows[:10]):  # First 10 rows
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 3:
                    cell_texts = [cell.get_text(strip=True) for cell in cells[:8]]  # First 8 cells
                    
                    # Look for round indicators
                    round_pattern = re.search(r'r(\d+)|round\s*(\d+)', ' '.join(cell_texts).lower())
                    if round_pattern:
                        round_num = round_pattern.group(1) or round_pattern.group(2)
                        print(f"   Row {row_idx}: Round {round_num} - {cell_texts}")
                    
                    # Look for score patterns
                    score_pattern = re.search(r'(\w+.*?)\s+def\s+(\w+.*?)(?:\s|$)', ' '.join(cell_texts), re.IGNORECASE)
                    if score_pattern:
                        print(f"   Row {row_idx}: MATCH - {score_pattern.group(1)} def {score_pattern.group(2)}")
        
        # Look for specific recent rounds
        print(f"\n🔍 Looking for Round 5+ matches...")
        
        page_text = soup.get_text()
        
        for round_num in [5, 6, 7, 8, 9, 10]:
            round_matches = re.findall(f'round {round_num}.*?(?=round {round_num+1}|$)', page_text, re.IGNORECASE | re.DOTALL)
            if round_matches:
                print(f"   Round {round_num}: Found section ({len(round_matches[0])} chars)")
                
                # Look for match results in this round
                round_text = round_matches[0][:500]  # First 500 chars
                def_matches = re.findall(r'(\w+[\w\s]*?)\s+def\s+(\w+[\w\s]*?)(?:\s|$)', round_text, re.IGNORECASE)
                if def_matches:
                    print(f"      Matches found: {len(def_matches)}")
                    for match in def_matches[:3]:  # Show first 3 matches
                        print(f"         {match[0]} def {match[1]}")
                else:
                    print(f"      No 'def' matches found in round text")
    
    except Exception as e:
        print(f"❌ Error debugging AFL Tables: {e}")

if __name__ == "__main__":
    debug_afl_tables_2025()
