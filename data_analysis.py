#!/usr/bin/env python3
"""
AFL Data Analysis
Provides insights, benchmarks, and data quality statistics for the AFL dataset.
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AFLDataAnalyzer:
    """Analyzer class for AFL data insights and benchmarks."""
    
    def __init__(self, data_dir: str = "afl_data"):
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / "afl_database.db"
        self.parquet_dir = self.data_dir / "parquet"
        
    def load_data(self):
        """Load data from database or parquet files."""
        # Try database first
        if self.db_path.exists():
            conn = sqlite3.connect(self.db_path)
            self.matches_df = pd.read_sql_query("SELECT * FROM matches", conn)
            self.players_df = pd.read_sql_query("SELECT * FROM players", conn)
            conn.close()
            print(f"Loaded {len(self.matches_df)} matches and {len(self.players_df)} player records from database")
        else:
            # Fall back to parquet files
            matches_file = self.parquet_dir / "matches.parquet"
            players_file = self.parquet_dir / "players.parquet"
            
            if matches_file.exists():
                self.matches_df = pd.read_parquet(matches_file)
            else:
                self.matches_df = pd.DataFrame()
                
            if players_file.exists():
                self.players_df = pd.read_parquet(players_file)
            else:
                self.players_df = pd.DataFrame()
                
            print(f"Loaded {len(self.matches_df)} matches and {len(self.players_df)} player records from parquet files")
    
    def analyze_match_data(self):
        """Analyze match data and generate insights."""
        if self.matches_df.empty:
            print("No match data available for analysis")
            return {}
        
        print("\n" + "="*50)
        print("MATCH DATA ANALYSIS")
        print("="*50)
        
        analysis = {}
        
        # Basic statistics
        analysis['total_matches'] = len(self.matches_df)
        analysis['date_range'] = {
            'start': self.matches_df['date'].min() if 'date' in self.matches_df.columns else 'Unknown',
            'end': self.matches_df['date'].max() if 'date' in self.matches_df.columns else 'Unknown'
        }
        
        # Year distribution
        if 'year' in self.matches_df.columns:
            year_counts = self.matches_df['year'].value_counts().sort_index()
            analysis['matches_by_year'] = year_counts.to_dict()
            analysis['year_range'] = {
                'min': int(self.matches_df['year'].min()),
                'max': int(self.matches_df['year'].max())
            }
            print(f"Year range: {analysis['year_range']['min']} - {analysis['year_range']['max']}")
        
        # Team analysis
        if 'home_team' in self.matches_df.columns:
            teams = pd.concat([self.matches_df['home_team'], self.matches_df['away_team']]).unique()
            analysis['total_teams'] = len(teams)
            analysis['teams'] = sorted(teams.tolist())
            print(f"Total teams: {len(teams)}")
        
        # Venue analysis
        if 'venue' in self.matches_df.columns:
            venues = self.matches_df['venue'].value_counts()
            analysis['total_venues'] = len(venues)
            analysis['top_venues'] = venues.head(10).to_dict()
            print(f"Total venues: {len(venues)}")
        
        # Score analysis
        score_columns = ['home_total_goals', 'home_total_behinds', 'away_total_goals', 'away_total_behinds']
        for col in score_columns:
            if col in self.matches_df.columns:
                analysis[f'{col}_stats'] = {
                    'mean': float(self.matches_df[col].mean()),
                    'median': float(self.matches_df[col].median()),
                    'std': float(self.matches_df[col].std()),
                    'min': float(self.matches_df[col].min()),
                    'max': float(self.matches_df[col].max())
                }
        
        # Margin analysis
        if 'margin' in self.matches_df.columns:
            analysis['margin_stats'] = {
                'mean': float(self.matches_df['margin'].mean()),
                'median': float(self.matches_df['margin'].median()),
                'std': float(self.matches_df['margin'].std()),
                'min': float(self.matches_df['margin'].min()),
                'max': float(self.matches_df['margin'].max())
            }
            print(f"Average margin: {analysis['margin_stats']['mean']:.1f} points")
        
        return analysis
    
    def analyze_player_data(self):
        """Analyze player data and generate insights."""
        if self.players_df.empty:
            print("No player data available for analysis")
            return {}
        
        print("\n" + "="*50)
        print("PLAYER DATA ANALYSIS")
        print("="*50)
        
        analysis = {}
        
        # Basic statistics
        analysis['total_player_records'] = len(self.players_df)
        
        # Year distribution
        if 'year' in self.players_df.columns:
            year_counts = self.players_df['year'].value_counts().sort_index()
            analysis['players_by_year'] = year_counts.to_dict()
            analysis['year_range'] = {
                'min': int(self.players_df['year'].min()),
                'max': int(self.players_df['year'].max())
            }
            print(f"Year range: {analysis['year_range']['min']} - {analysis['year_range']['max']}")
        
        # Team analysis
        if 'team' in self.players_df.columns:
            teams = self.players_df['team'].value_counts()
            analysis['total_teams'] = len(teams)
            analysis['teams'] = teams.to_dict()
            print(f"Total teams: {len(teams)}")
        
        # Player statistics
        stat_columns = ['kicks', 'marks', 'handballs', 'disposals', 'goals', 'behinds', 
                       'tackles', 'hit_outs', 'clearances', 'brownlow_votes']
        
        for col in stat_columns:
            if col in self.players_df.columns:
                analysis[f'{col}_stats'] = {
                    'mean': float(self.players_df[col].mean()),
                    'median': float(self.players_df[col].median()),
                    'std': float(self.players_df[col].std()),
                    'min': float(self.players_df[col].min()),
                    'max': float(self.players_df[col].max())
                }
        
        # Top performers
        if 'goals' in self.players_df.columns and 'first_name' in self.players_df.columns and 'last_name' in self.players_df.columns:
            top_goal_kickers = self.players_df.groupby(['first_name', 'last_name'])['goals'].sum().sort_values(ascending=False).head(10)
            analysis['top_goal_kickers'] = top_goal_kickers.to_dict()
            print("Top 5 goal kickers:")
            for i, (name, goals) in enumerate(top_goal_kickers.head(5).items()):
                print(f"  {i+1}. {' '.join(name)}: {goals} goals")
        
        if 'brownlow_votes' in self.players_df.columns and 'first_name' in self.players_df.columns and 'last_name' in self.players_df.columns:
            top_vote_getters = self.players_df.groupby(['first_name', 'last_name'])['brownlow_votes'].sum().sort_values(ascending=False).head(10)
            analysis['top_vote_getters'] = top_vote_getters.to_dict()
        
        return analysis
    
    def generate_visualizations(self):
        """Generate data visualizations."""
        if self.matches_df.empty and self.players_df.empty:
            print("No data available for visualizations")
            return
        
        # Create plots directory
        plots_dir = self.data_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Match data visualizations
        if not self.matches_df.empty:
            # Matches by year
            if 'year' in self.matches_df.columns:
                plt.figure(figsize=(12, 6))
                year_counts = self.matches_df['year'].value_counts().sort_index()
                plt.plot(year_counts.index, year_counts.values, marker='o', linewidth=2, markersize=4)
                plt.title('AFL Matches by Year', fontsize=16, fontweight='bold')
                plt.xlabel('Year', fontsize=12)
                plt.ylabel('Number of Matches', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(plots_dir / "matches_by_year.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Score distribution
            if 'home_total_goals' in self.matches_df.columns and 'away_total_goals' in self.matches_df.columns:
                plt.figure(figsize=(12, 5))
                
                plt.subplot(1, 2, 1)
                plt.hist(self.matches_df['home_total_goals'], bins=30, alpha=0.7, label='Home Goals')
                plt.hist(self.matches_df['away_total_goals'], bins=30, alpha=0.7, label='Away Goals')
                plt.title('Goal Distribution', fontsize=14, fontweight='bold')
                plt.xlabel('Goals', fontsize=12)
                plt.ylabel('Frequency', fontsize=12)
                plt.legend()
                
                plt.subplot(1, 2, 2)
                if 'margin' in self.matches_df.columns:
                    plt.hist(self.matches_df['margin'], bins=50, alpha=0.7, color='green')
                    plt.title('Margin Distribution', fontsize=14, fontweight='bold')
                    plt.xlabel('Margin (points)', fontsize=12)
                    plt.ylabel('Frequency', fontsize=12)
                
                plt.tight_layout()
                plt.savefig(plots_dir / "score_distributions.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        # Player data visualizations
        if not self.players_df.empty:
            # Player stats by year
            if 'year' in self.players_df.columns and 'disposals' in self.players_df.columns:
                plt.figure(figsize=(12, 6))
                yearly_disposals = self.players_df.groupby('year')['disposals'].mean()
                plt.plot(yearly_disposals.index, yearly_disposals.values, marker='o', linewidth=2, markersize=4)
                plt.title('Average Disposals per Player by Year', fontsize=16, fontweight='bold')
                plt.xlabel('Year', fontsize=12)
                plt.ylabel('Average Disposals', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(plots_dir / "disposals_by_year.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Player stats correlation
            if all(col in self.players_df.columns for col in ['kicks', 'marks', 'handballs', 'goals', 'tackles']):
                plt.figure(figsize=(10, 8))
                stats_cols = ['kicks', 'marks', 'handballs', 'goals', 'tackles']
                correlation_matrix = self.players_df[stats_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                           square=True, linewidths=0.5)
                plt.title('Player Statistics Correlation Matrix', fontsize=16, fontweight='bold')
                plt.tight_layout()
                plt.savefig(plots_dir / "player_stats_correlation.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"Visualizations saved to {plots_dir}")
    
    def generate_benchmarks(self):
        """Generate performance benchmarks."""
        benchmarks = {
            "timestamp": datetime.now().isoformat(),
            "data_loading": {},
            "query_performance": {},
            "storage_efficiency": {}
        }
        
        # Data loading benchmarks
        if self.db_path.exists():
            import time
            
            # Database query benchmarks
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            queries = [
                ("count_matches", "SELECT COUNT(*) FROM matches"),
                ("count_players", "SELECT COUNT(*) FROM players"),
                ("matches_by_year", "SELECT year, COUNT(*) FROM matches GROUP BY year ORDER BY year"),
                ("top_teams", "SELECT home_team, COUNT(*) FROM matches GROUP BY home_team ORDER BY COUNT(*) DESC LIMIT 10"),
                ("recent_matches", "SELECT * FROM matches WHERE year >= 2020 ORDER BY date DESC LIMIT 100")
            ]
            
            for query_name, query in queries:
                start_time = time.time()
                cursor.execute(query)
                result = cursor.fetchall()
                end_time = time.time()
                benchmarks["query_performance"][query_name] = {
                    "execution_time_ms": (end_time - start_time) * 1000,
                    "result_count": len(result)
                }
            
            conn.close()
            
            # Storage efficiency
            db_size = self.db_path.stat().st_size / (1024 * 1024)  # MB
            benchmarks["storage_efficiency"]["database_size_mb"] = db_size
            
            if not self.matches_df.empty and not self.players_df.empty:
                matches_size = len(self.matches_df) * len(self.matches_df.columns) * 8  # Rough estimate
                players_size = len(self.players_df) * len(self.players_df.columns) * 8
                total_data_size = (matches_size + players_size) / (1024 * 1024)  # MB
                benchmarks["storage_efficiency"]["compression_ratio"] = total_data_size / db_size if db_size > 0 else 0
        
        return benchmarks
    
    def run_analysis(self):
        """Run complete analysis and generate reports."""
        print("Starting AFL data analysis...")
        
        # Load data
        self.load_data()
        
        # Analyze data
        match_analysis = self.analyze_match_data()
        player_analysis = self.analyze_player_data()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Generate benchmarks
        benchmarks = self.generate_benchmarks()
        
        # Combine all analysis
        complete_analysis = {
            "timestamp": datetime.now().isoformat(),
            "match_analysis": match_analysis,
            "player_analysis": player_analysis,
            "benchmarks": benchmarks
        }
        
        # Save analysis report
        with open(self.data_dir / "analysis_report.json", 'w') as f:
            json.dump(complete_analysis, f, indent=2, default=str)
        
        print(f"\nAnalysis complete! Report saved to {self.data_dir}/analysis_report.json")
        return complete_analysis

def main():
    """Main function to run the analysis."""
    analyzer = AFLDataAnalyzer()
    analysis = analyzer.run_analysis()
    
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    
    if analysis['match_analysis']:
        print(f"Total matches: {analysis['match_analysis'].get('total_matches', 0)}")
        if 'year_range' in analysis['match_analysis']:
            year_range = analysis['match_analysis']['year_range']
            print(f"Year range: {year_range['min']} - {year_range['max']}")
    
    if analysis['player_analysis']:
        print(f"Total player records: {analysis['player_analysis'].get('total_player_records', 0)}")
        if 'year_range' in analysis['player_analysis']:
            year_range = analysis['player_analysis']['year_range']
            print(f"Year range: {year_range['min']} - {year_range['max']}")
    
    print("="*50)

if __name__ == "__main__":
    main() 