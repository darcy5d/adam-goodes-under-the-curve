#!/usr/bin/env python3
"""
Comprehensive AFL Data Exploratory Analysis (Phase 1B)
=====================================================

This script implements a multi-faceted EDA approach:
1. Statistical summary approach (descriptive stats, distributions)
2. Visual exploration approach (plotting trends, relationships)
3. Time series analysis approach (temporal patterns, seasonality)
4. Data quality assessment approach (missing data, outliers, consistency)

Author: AFL Prediction Model Team
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from pathlib import Path
import warnings
from datetime import datetime
import json

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
warnings.filterwarnings('ignore')

class ComprehensiveEDA:
    """Comprehensive Exploratory Data Analysis for AFL Data."""
    
    def __init__(self, db_path="afl_data/afl_database.db"):
        """Initialize EDA with database connection."""
        self.db_path = db_path
        self.output_dir = Path("eda_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        self.load_data()
        
        # Initialize analysis results
        self.analysis_results = {
            "timestamp": datetime.now().isoformat(),
            "data_overview": {},
            "temporal_analysis": {},
            "match_analysis": {},
            "player_analysis": {},
            "data_quality": {},
            "recommendations": {}
        }
    
    def load_data(self):
        """Load data from SQLite database."""
        print("Loading data from database...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Load match data
        self.matches_df = pd.read_sql_query("SELECT * FROM matches", conn)
        print(f"Loaded {len(self.matches_df):,} match records")
        
        # Load player data
        self.players_df = pd.read_sql_query("SELECT * FROM players", conn)
        print(f"Loaded {len(self.players_df):,} player records")
        
        conn.close()
        
        # Basic data cleaning
        self.clean_data()
    
    def clean_data(self):
        """Basic data cleaning and preprocessing."""
        print("Cleaning and preprocessing data...")
        
        # Convert date columns
        if 'date' in self.matches_df.columns:
            self.matches_df['date'] = pd.to_datetime(self.matches_df['date'], errors='coerce')
        
        # Convert numeric columns
        numeric_cols = ['year', 'home_total_goals', 'away_total_goals', 'home_total_behinds', 'away_total_behinds']
        for col in numeric_cols:
            if col in self.matches_df.columns:
                self.matches_df[col] = pd.to_numeric(self.matches_df[col], errors='coerce')
        
        # Player data numeric conversion
        player_numeric = ['year', 'kicks', 'marks', 'handballs', 'disposals', 'goals', 'behinds', 
                         'hit_outs', 'tackles', 'rebound_50s', 'inside_50s', 'clearances', 
                         'clangers', 'free_kicks_for', 'free_kicks_against', 'brownlow_votes',
                         'contested_possessions', 'uncontested_possessions', 'contested_marks',
                         'marks_inside_50', 'one_percenters', 'bounces', 'goal_assist',
                         'percentage_of_game_played', 'height', 'weight', 'round']
        
        for col in player_numeric:
            if col in self.players_df.columns:
                self.players_df[col] = pd.to_numeric(self.players_df[col], errors='coerce')
    
    def approach_1_statistical_summary(self):
        """Approach 1: Statistical Summary Analysis."""
        print("\n=== APPROACH 1: STATISTICAL SUMMARY ANALYSIS ===")
        
        # Match data summary
        match_summary = {
            "total_matches": len(self.matches_df),
            "year_range": (self.matches_df['year'].min(), self.matches_df['year'].max()),
            "unique_teams": len(self.matches_df['home_team'].unique()) if 'home_team' in self.matches_df.columns else 0,
            "unique_venues": len(self.matches_df['venue'].unique()) if 'venue' in self.matches_df.columns else 0
        }
        
        # Player data summary
        player_summary = {
            "total_players": len(self.players_df),
            "year_range": (self.players_df['year'].min(), self.players_df['year'].max()),
            "unique_teams": len(self.players_df['team'].unique()) if 'team' in self.players_df.columns else 0,
            "unique_players": len(self.players_df[['first_name', 'last_name']].drop_duplicates()) if 'first_name' in self.players_df.columns else 0
        }
        
        # Statistical distributions
        if 'home_total_goals' in self.matches_df.columns and 'away_total_goals' in self.matches_df.columns:
            scoring_stats = {
                "home_goals_mean": self.matches_df['home_total_goals'].mean(),
                "away_goals_mean": self.matches_df['away_total_goals'].mean(),
                "home_goals_std": self.matches_df['home_total_goals'].std(),
                "away_goals_std": self.matches_df['away_total_goals'].std(),
                "total_goals_mean": (self.matches_df['home_total_goals'] + self.matches_df['away_total_goals']).mean()
            }
        else:
            scoring_stats = {}
        
        self.analysis_results["data_overview"] = {
            "matches": match_summary,
            "players": player_summary,
            "scoring": scoring_stats
        }
        
        print(f"Match Summary: {match_summary}")
        print(f"Player Summary: {player_summary}")
        if scoring_stats:
            print(f"Scoring Summary: {scoring_stats}")
    
    def approach_2_visual_exploration(self):
        """Approach 2: Visual Exploration Analysis."""
        print("\n=== APPROACH 2: VISUAL EXPLORATION ANALYSIS ===")
        
        # Create comprehensive visualizations
        self.create_temporal_visualizations()
        self.create_match_analysis_visualizations()
        self.create_player_analysis_visualizations()
        self.create_data_quality_visualizations()
    
    def approach_3_time_series_analysis(self):
        """Approach 3: Time Series Analysis."""
        print("\n=== APPROACH 3: TIME SERIES ANALYSIS ===")
        
        # Temporal patterns analysis
        self.analyze_temporal_patterns()
        self.analyze_seasonal_patterns()
        self.analyze_era_changes()
    
    def approach_4_data_quality_assessment(self):
        """Approach 4: Data Quality Assessment."""
        print("\n=== APPROACH 4: DATA QUALITY ASSESSMENT ===")
        
        # Missing data analysis
        self.analyze_missing_data()
        self.analyze_outliers()
        self.analyze_data_consistency()
    
    def create_temporal_visualizations(self):
        """Create temporal analysis visualizations."""
        print("Creating temporal visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Temporal Analysis: Data Completeness and Trends (1897-2025)', fontsize=16)
        
        # 1. Data volume over time
        if 'year' in self.matches_df.columns:
            match_counts = self.matches_df['year'].value_counts().sort_index()
            axes[0, 0].plot(match_counts.index, match_counts.values, marker='o', linewidth=2)
            axes[0, 0].set_title('Match Data Volume Over Time')
            axes[0, 0].set_xlabel('Year')
            axes[0, 0].set_ylabel('Number of Matches')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Player data volume over time
        if 'year' in self.players_df.columns:
            player_counts = self.players_df['year'].value_counts().sort_index()
            axes[0, 1].plot(player_counts.index, player_counts.values, marker='s', color='orange', linewidth=2)
            axes[0, 1].set_title('Player Performance Records Over Time')
            axes[0, 1].set_xlabel('Year')
            axes[0, 1].set_ylabel('Number of Player Records')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Scoring trends over time
        if 'home_total_goals' in self.matches_df.columns and 'away_total_goals' in self.matches_df.columns:
            yearly_scoring = self.matches_df.groupby('year').agg({
                'home_total_goals': 'mean',
                'away_total_goals': 'mean'
            }).reset_index()
            
            axes[1, 0].plot(yearly_scoring['year'], yearly_scoring['home_total_goals'], 
                           label='Home Goals', marker='o', linewidth=2)
            axes[1, 0].plot(yearly_scoring['year'], yearly_scoring['away_total_goals'], 
                           label='Away Goals', marker='s', linewidth=2)
            axes[1, 0].set_title('Average Goals per Team Over Time')
            axes[1, 0].set_xlabel('Year')
            axes[1, 0].set_ylabel('Average Goals')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Data completeness by era
        if 'year' in self.matches_df.columns:
            eras = {
                'Early Era (1897-1950)': (1897, 1950),
                'Mid Era (1951-1990)': (1951, 1990),
                'Modern Era (1991-2025)': (1991, 2025)
            }
            
            era_completeness = []
            era_labels = []
            
            for era_name, (start_year, end_year) in eras.items():
                era_matches = self.matches_df[(self.matches_df['year'] >= start_year) & 
                                            (self.matches_df['year'] <= end_year)]
                era_players = self.players_df[(self.players_df['year'] >= start_year) & 
                                            (self.players_df['year'] <= end_year)]
                
                era_completeness.append({
                    'matches': len(era_matches),
                    'players': len(era_players)
                })
                era_labels.append(era_name)
            
            match_counts = [e['matches'] for e in era_completeness]
            player_counts = [e['players'] for e in era_completeness]
            
            x = np.arange(len(era_labels))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, match_counts, width, label='Matches', alpha=0.8)
            axes[1, 1].bar(x + width/2, player_counts, width, label='Player Records', alpha=0.8)
            axes[1, 1].set_title('Data Volume by Historical Era')
            axes[1, 1].set_xlabel('Era')
            axes[1, 1].set_ylabel('Number of Records')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(era_labels, rotation=45)
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_match_analysis_visualizations(self):
        """Create match analysis visualizations."""
        print("Creating match analysis visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Match Analysis: Home Advantage, Scoring, and Patterns', fontsize=16)
        
        # 1. Home vs Away advantage over time
        if 'home_total_goals' in self.matches_df.columns and 'away_total_goals' in self.matches_df.columns:
            yearly_advantage = self.matches_df.groupby('year').agg({
                'home_total_goals': 'mean',
                'away_total_goals': 'mean'
            }).reset_index()
            yearly_advantage['home_advantage'] = yearly_advantage['home_total_goals'] - yearly_advantage['away_total_goals']
            
            axes[0, 0].plot(yearly_advantage['year'], yearly_advantage['home_advantage'], 
                           marker='o', linewidth=2, color='red')
            axes[0, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
            axes[0, 0].set_title('Home Team Advantage Over Time')
            axes[0, 0].set_xlabel('Year')
            axes[0, 0].set_ylabel('Home Goals - Away Goals')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Scoring distribution
        if 'home_total_goals' in self.matches_df.columns and 'away_total_goals' in self.matches_df.columns:
            total_goals = self.matches_df['home_total_goals'] + self.matches_df['away_total_goals']
            axes[0, 1].hist(total_goals.dropna(), bins=30, alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].set_title('Distribution of Total Goals per Match')
            axes[0, 1].set_xlabel('Total Goals')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Margin distribution
        if 'home_total_goals' in self.matches_df.columns and 'away_total_goals' in self.matches_df.columns:
            margins = self.matches_df['home_total_goals'] - self.matches_df['away_total_goals']
            axes[1, 0].hist(margins.dropna(), bins=30, alpha=0.7, color='blue', edgecolor='black')
            axes[1, 0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
            axes[1, 0].set_title('Distribution of Match Margins')
            axes[1, 0].set_xlabel('Margin (Home - Away)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Seasonal patterns (by round if available)
        if 'round' in self.matches_df.columns:
            round_scoring = self.matches_df.groupby('round').agg({
                'home_total_goals': 'mean',
                'away_total_goals': 'mean'
            }).reset_index()
            round_scoring['total_goals'] = round_scoring['home_total_goals'] + round_scoring['away_total_goals']
            
            axes[1, 1].plot(round_scoring['round'], round_scoring['total_goals'], 
                           marker='o', linewidth=2, color='purple')
            axes[1, 1].set_title('Average Goals by Round')
            axes[1, 1].set_xlabel('Round')
            axes[1, 1].set_ylabel('Average Total Goals')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'match_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_player_analysis_visualizations(self):
        """Create player analysis visualizations."""
        print("Creating player analysis visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Player Analysis: Performance Distributions and Patterns', fontsize=16)
        
        # 1. Performance distributions for key stats
        key_stats = ['kicks', 'marks', 'handballs', 'disposals', 'goals', 'tackles']
        available_stats = [stat for stat in key_stats if stat in self.players_df.columns]
        
        if available_stats:
            for i, stat in enumerate(available_stats[:4]):  # Plot first 4 available stats
                row, col = i // 2, i % 2
                data = self.players_df[stat].dropna()
                if len(data) > 0:
                    axes[row, col].hist(data, bins=30, alpha=0.7, edgecolor='black')
                    axes[row, col].set_title(f'Distribution of {stat.title()}')
                    axes[row, col].set_xlabel(stat.title())
                    axes[row, col].set_ylabel('Frequency')
                    axes[row, col].grid(True, alpha=0.3)
        
        # 2. Player longevity analysis
        if 'first_name' in self.players_df.columns and 'last_name' in self.players_df.columns:
            player_careers = self.players_df.groupby(['first_name', 'last_name']).agg({
                'year': ['min', 'max', 'count']
            }).reset_index()
            player_careers.columns = ['first_name', 'last_name', 'debut_year', 'last_year', 'games_played']
            player_careers['career_length'] = player_careers['last_year'] - player_careers['debut_year']
            
            axes[1, 0].hist(player_careers['career_length'].dropna(), bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].set_title('Player Career Length Distribution')
            axes[1, 0].set_xlabel('Career Length (Years)')
            axes[1, 0].set_ylabel('Number of Players')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 3. Performance by era
        if 'year' in self.players_df.columns and 'disposals' in self.players_df.columns:
            era_performance = self.players_df.groupby('year')['disposals'].mean().reset_index()
            axes[1, 1].plot(era_performance['year'], era_performance['disposals'], 
                           marker='o', linewidth=2, color='green')
            axes[1, 1].set_title('Average Disposals Over Time')
            axes[1, 1].set_xlabel('Year')
            axes[1, 1].set_ylabel('Average Disposals')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'player_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_data_quality_visualizations(self):
        """Create data quality assessment visualizations."""
        print("Creating data quality visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Quality Assessment: Missing Data and Outliers', fontsize=16)
        
        # 1. Missing data by column
        if 'home_total_goals' in self.matches_df.columns:
            match_missing = self.matches_df.isnull().sum()
            match_missing = match_missing[match_missing > 0]
            
            if len(match_missing) > 0:
                axes[0, 0].bar(range(len(match_missing)), match_missing.values)
                axes[0, 0].set_title('Missing Data in Match Records')
                axes[0, 0].set_xlabel('Columns')
                axes[0, 0].set_ylabel('Missing Values')
                axes[0, 0].set_xticks(range(len(match_missing)))
                axes[0, 0].set_xticklabels(match_missing.index, rotation=45)
                axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Missing data by era
        if 'year' in self.players_df.columns:
            eras = [(1897, 1950), (1951, 1990), (1991, 2025)]
            era_missing = []
            era_labels = ['Early Era', 'Mid Era', 'Modern Era']
            
            for start_year, end_year in eras:
                era_data = self.players_df[(self.players_df['year'] >= start_year) & 
                                         (self.players_df['year'] <= end_year)]
                missing_pct = (era_data.isnull().sum() / len(era_data)) * 100
                era_missing.append(missing_pct.mean())
            
            axes[0, 1].bar(era_labels, era_missing, alpha=0.7, color='red')
            axes[0, 1].set_title('Average Missing Data by Era')
            axes[0, 1].set_xlabel('Era')
            axes[0, 1].set_ylabel('Missing Data (%)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Outlier analysis for key stats
        if 'disposals' in self.players_df.columns:
            data = self.players_df['disposals'].dropna()
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            axes[1, 0].hist(data, bins=50, alpha=0.7, color='blue', edgecolor='black')
            axes[1, 0].axvline(x=lower_bound, color='red', linestyle='--', alpha=0.7, label='Outlier Boundary')
            axes[1, 0].axvline(x=upper_bound, color='red', linestyle='--', alpha=0.7)
            axes[1, 0].set_title('Disposals Distribution with Outlier Boundaries')
            axes[1, 0].set_xlabel('Disposals')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Data consistency check
        if 'year' in self.matches_df.columns and 'year' in self.players_df.columns:
            match_years = self.matches_df['year'].value_counts().sort_index()
            player_years = self.players_df['year'].value_counts().sort_index()
            
            common_years = set(match_years.index) & set(player_years.index)
            if common_years:
                common_years = sorted(list(common_years))
                match_counts = [match_years.get(year, 0) for year in common_years]
                player_counts = [player_years.get(year, 0) for year in common_years]
                
                axes[1, 1].plot(common_years, match_counts, label='Matches', marker='o')
                axes[1, 1].plot(common_years, player_counts, label='Player Records', marker='s')
                axes[1, 1].set_title('Data Consistency: Matches vs Player Records')
                axes[1, 1].set_xlabel('Year')
                axes[1, 1].set_ylabel('Number of Records')
                axes[1, 1].legend()
                axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'data_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in the data."""
        print("Analyzing temporal patterns...")
        
        temporal_insights = {}
        
        # Analyze data completeness over time
        if 'year' in self.matches_df.columns:
            yearly_completeness = self.matches_df.groupby('year').size()
            temporal_insights['match_data_growth'] = {
                'early_era_avg': yearly_completeness[1897:1950].mean(),
                'mid_era_avg': yearly_completeness[1951:1990].mean(),
                'modern_era_avg': yearly_completeness[1991:2025].mean()
            }
        
        if 'year' in self.players_df.columns:
            yearly_player_completeness = self.players_df.groupby('year').size()
            temporal_insights['player_data_growth'] = {
                'early_era_avg': yearly_player_completeness[1897:1950].mean(),
                'mid_era_avg': yearly_player_completeness[1951:1990].mean(),
                'modern_era_avg': yearly_player_completeness[1991:2025].mean()
            }
        
        self.analysis_results["temporal_analysis"] = temporal_insights
    
    def analyze_seasonal_patterns(self):
        """Analyze seasonal patterns in the data."""
        print("Analyzing seasonal patterns...")
        
        seasonal_insights = {}
        
        # Analyze scoring patterns by round
        if 'round' in self.matches_df.columns and 'home_total_goals' in self.matches_df.columns:
            round_scoring = self.matches_df.groupby('round').agg({
                'home_total_goals': 'mean',
                'away_total_goals': 'mean'
            }).reset_index()
            round_scoring['total_goals'] = round_scoring['home_total_goals'] + round_scoring['away_total_goals']
            
            seasonal_insights['round_scoring_patterns'] = {
                'highest_scoring_round': round_scoring.loc[round_scoring['total_goals'].idxmax(), 'round'],
                'lowest_scoring_round': round_scoring.loc[round_scoring['total_goals'].idxmin(), 'round'],
                'avg_goals_per_round': round_scoring['total_goals'].mean()
            }
        
        self.analysis_results["temporal_analysis"].update(seasonal_insights)
    
    def analyze_era_changes(self):
        """Analyze changes between different eras."""
        print("Analyzing era changes...")
        
        era_insights = {}
        
        # Define eras
        eras = {
            'early': (1897, 1950),
            'mid': (1951, 1990),
            'modern': (1991, 2025)
        }
        
        for era_name, (start_year, end_year) in eras.items():
            era_matches = self.matches_df[(self.matches_df['year'] >= start_year) & 
                                        (self.matches_df['year'] <= end_year)]
            
            if 'home_total_goals' in era_matches.columns and 'away_total_goals' in era_matches.columns:
                era_insights[f'{era_name}_era'] = {
                    'avg_home_goals': era_matches['home_total_goals'].mean(),
                    'avg_away_goals': era_matches['away_total_goals'].mean(),
                    'home_advantage': era_matches['home_total_goals'].mean() - era_matches['away_total_goals'].mean(),
                    'total_matches': len(era_matches)
                }
        
        self.analysis_results["temporal_analysis"].update(era_insights)
    
    def analyze_missing_data(self):
        """Analyze missing data patterns."""
        print("Analyzing missing data patterns...")
        
        missing_analysis = {}
        
        # Match data missing analysis
        if len(self.matches_df) > 0:
            match_missing = self.matches_df.isnull().sum()
            match_missing_pct = (match_missing / len(self.matches_df)) * 100
            missing_analysis['match_data'] = {
                'columns_with_missing': len(match_missing[match_missing > 0]),
                'avg_missing_pct': match_missing_pct.mean(),
                'worst_columns': match_missing_pct.nlargest(3).to_dict()
            }
        
        # Player data missing analysis
        if len(self.players_df) > 0:
            player_missing = self.players_df.isnull().sum()
            player_missing_pct = (player_missing / len(self.players_df)) * 100
            missing_analysis['player_data'] = {
                'columns_with_missing': len(player_missing[player_missing > 0]),
                'avg_missing_pct': player_missing_pct.mean(),
                'worst_columns': player_missing_pct.nlargest(5).to_dict()
            }
        
        self.analysis_results["data_quality"]["missing_data"] = missing_analysis
    
    def analyze_outliers(self):
        """Analyze statistical outliers."""
        print("Analyzing outliers...")
        
        outlier_analysis = {}
        
        # Analyze outliers in key player statistics
        key_stats = ['kicks', 'marks', 'handballs', 'disposals', 'goals', 'tackles']
        
        for stat in key_stats:
            if stat in self.players_df.columns:
                data = self.players_df[stat].dropna()
                if len(data) > 0:
                    Q1 = data.quantile(0.25)
                    Q3 = data.quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = data[(data < lower_bound) | (data > upper_bound)]
                    outlier_analysis[stat] = {
                        'outlier_count': len(outliers),
                        'outlier_pct': (len(outliers) / len(data)) * 100,
                        'min_value': data.min(),
                        'max_value': data.max(),
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    }
        
        self.analysis_results["data_quality"]["outliers"] = outlier_analysis
    
    def analyze_data_consistency(self):
        """Analyze consistency between match and player data."""
        print("Analyzing data consistency...")
        
        consistency_analysis = {}
        
        # Check for overlapping years
        if 'year' in self.matches_df.columns and 'year' in self.players_df.columns:
            match_years = set(self.matches_df['year'].unique())
            player_years = set(self.players_df['year'].unique())
            
            overlap_years = match_years & player_years
            consistency_analysis['year_overlap'] = {
                'match_years': len(match_years),
                'player_years': len(player_years),
                'overlap_years': len(overlap_years),
                'overlap_pct': (len(overlap_years) / len(match_years)) * 100
            }
        
        # Check for team consistency
        if 'home_team' in self.matches_df.columns and 'team' in self.players_df.columns:
            match_teams = set(self.matches_df['home_team'].unique())
            player_teams = set(self.players_df['team'].unique())
            
            team_overlap = match_teams & player_teams
            consistency_analysis['team_consistency'] = {
                'match_teams': len(match_teams),
                'player_teams': len(player_teams),
                'overlap_teams': len(team_overlap),
                'overlap_pct': (len(team_overlap) / len(match_teams)) * 100
            }
        
        self.analysis_results["data_quality"]["consistency"] = consistency_analysis
    
    def generate_recommendations(self):
        """Generate recommendations based on EDA findings."""
        print("Generating recommendations...")
        
        recommendations = {
            "reliable_eras": [],
            "predictive_statistics": [],
            "preprocessing_strategies": [],
            "model_training_timeline": []
        }
        
        # Determine reliable eras for training
        if 'year' in self.matches_df.columns:
            yearly_completeness = self.matches_df.groupby('year').size()
            modern_era_avg = yearly_completeness[1991:2025].mean()
            mid_era_avg = yearly_completeness[1951:1990].mean()
            
            if modern_era_avg > mid_era_avg * 1.5:
                recommendations["reliable_eras"].append("Modern Era (1991-2025) - Most complete data")
                recommendations["reliable_eras"].append("Mid Era (1951-1990) - Good for historical patterns")
            else:
                recommendations["reliable_eras"].append("Mid Era (1951-1990) - Most balanced data")
                recommendations["reliable_eras"].append("Modern Era (1991-2025) - Good for recent trends")
        
        # Identify most predictive statistics
        key_stats = ['kicks', 'marks', 'handballs', 'disposals', 'goals', 'tackles', 'hit_outs']
        available_stats = [stat for stat in key_stats if stat in self.players_df.columns]
        
        if available_stats:
            # Check completeness of each stat
            stat_completeness = {}
            for stat in available_stats:
                completeness = (self.players_df[stat].notna().sum() / len(self.players_df)) * 100
                stat_completeness[stat] = completeness
            
            # Recommend stats with >50% completeness
            reliable_stats = [stat for stat, pct in stat_completeness.items() if pct > 50]
            recommendations["predictive_statistics"] = reliable_stats
        
        # Preprocessing strategies
        recommendations["preprocessing_strategies"].extend([
            "Handle missing values with era-specific imputation",
            "Remove statistical outliers using IQR method",
            "Normalize statistics by era to account for rule changes",
            "Create derived features (efficiency ratios, etc.)"
        ])
        
        # Model training timeline
        recommendations["model_training_timeline"] = [
            "Use 1991-2020 for training (30 years of modern data)",
            "Use 2021-2023 for validation",
            "Use 2024-2025 for testing",
            "Consider separate models for different eras"
        ]
        
        self.analysis_results["recommendations"] = recommendations
    
    def run_comprehensive_analysis(self):
        """Run the complete comprehensive EDA analysis."""
        print("="*60)
        print("COMPREHENSIVE AFL DATA EXPLORATORY ANALYSIS")
        print("="*60)
        
        # Run all four approaches
        self.approach_1_statistical_summary()
        self.approach_2_visual_exploration()
        self.approach_3_time_series_analysis()
        self.approach_4_data_quality_assessment()
        
        # Generate recommendations
        self.generate_recommendations()
        
        # Save analysis results
        self.save_analysis_results()
        
        print("\n" + "="*60)
        print("COMPREHENSIVE EDA ANALYSIS COMPLETED!")
        print("="*60)
        print(f"Visualizations saved to: {self.output_dir}")
        print(f"Analysis results saved to: {self.output_dir}/eda_analysis_results.json")
        print("="*60)
    
    def save_analysis_results(self):
        """Save analysis results to JSON file."""
        output_file = self.output_dir / "eda_analysis_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.analysis_results, f, indent=2, default=str)
        print(f"Analysis results saved to: {output_file}")

def main():
    """Main function to run comprehensive EDA."""
    eda = ComprehensiveEDA()
    eda.run_comprehensive_analysis()

if __name__ == "__main__":
    main() 