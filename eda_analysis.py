#!/usr/bin/env python3
"""
AFL Data Exploratory Analysis (Phase 1B)
Comprehensive EDA combining statistical, visual, temporal, and quality assessment approaches.

Author: AFL Data Analysis Team
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from pathlib import Path
import warnings
from datetime import datetime, date
import json
from typing import Dict, List, Tuple, Any
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.offline as pyo

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AFLEDA:
    """Comprehensive Exploratory Data Analysis for AFL Data."""
    
    def __init__(self, db_path: str = "afl_data/afl_database.db"):
        """Initialize EDA with database connection."""
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.output_dir = Path("afl_data/eda_output")
        self.output_dir.mkdir(exist_ok=True)
        
        # Load data
        self.players_df = None
        self.matches_df = None
        self.load_data()
        
        # Analysis results storage
        self.analysis_results = {
            "temporal_analysis": {},
            "match_analysis": {},
            "player_analysis": {},
            "data_quality": {},
            "recommendations": {}
        }
        
    def load_data(self):
        """Load data from SQLite database."""
        print("Loading data from database...")
        
        # Load player data
        self.players_df = pd.read_sql_query("SELECT * FROM players", self.conn)
        print(f"Loaded {len(self.players_df):,} player records")
        
        # Load match data (if available)
        try:
            self.matches_df = pd.read_sql_query("SELECT * FROM matches", self.conn)
            print(f"Loaded {len(self.matches_df):,} match records")
        except:
            print("No match data available - focusing on player analysis")
            self.matches_df = pd.DataFrame()
            
    def temporal_analysis(self) -> Dict[str, Any]:
        """Analyze temporal patterns and data quality changes over time."""
        print("\n=== TEMPORAL ANALYSIS ===")
        
        results = {}
        
        # 1. Data completeness over time
        yearly_counts = self.players_df.groupby('year').size().reset_index(name='count')
        results['yearly_counts'] = yearly_counts
        
        # Identify eras
        eras = {
            'Early Era (1897-1945)': (1897, 1945),
            'Post-War Era (1946-1975)': (1946, 1975),
            'Modern Era (1976-2000)': (1976, 2000),
            'Contemporary Era (2001-2025)': (2001, 2025)
        }
        
        era_stats = {}
        for era_name, (start_year, end_year) in eras.items():
            era_data = self.players_df[
                (self.players_df['year'] >= start_year) & 
                (self.players_df['year'] <= end_year)
            ]
            
            if len(era_data) > 0:
                era_stats[era_name] = {
                    'total_records': len(era_data),
                    'unique_players': era_data[['first_name', 'last_name']].drop_duplicates().shape[0],
                    'teams_count': era_data['team'].nunique(),
                    'avg_games_per_player': era_data.groupby(['first_name', 'last_name'])['games_played'].sum().mean(),
                    'data_completeness': self._calculate_completeness(era_data)
                }
        
        results['era_analysis'] = era_stats
        
        # 2. Statistical evolution over time
        numeric_cols = ['kicks', 'marks', 'handballs', 'disposals', 'goals', 'behinds', 
                       'hit_outs', 'tackles', 'rebound_50s', 'inside_50s', 'clearances']
        
        yearly_stats = {}
        for col in numeric_cols:
            if col in self.players_df.columns:
                yearly_avg = self.players_df.groupby('year')[col].mean().reset_index()
                yearly_stats[col] = yearly_avg
        
        results['yearly_statistics'] = yearly_stats
        
        # 3. Data quality timeline
        quality_timeline = self._analyze_data_quality_timeline()
        results['quality_timeline'] = quality_timeline
        
        self.analysis_results['temporal_analysis'] = results
        return results
    
    def match_analysis(self) -> Dict[str, Any]:
        """Analyze match data patterns and trends."""
        print("\n=== MATCH ANALYSIS ===")
        
        results = {}
        
        if self.matches_df.empty:
            print("No match data available for analysis")
            results['status'] = 'no_data'
            self.analysis_results['match_analysis'] = results
            return results
        
        # 1. Home vs Away advantage
        if 'home_total_goals' in self.matches_df.columns and 'away_total_goals' in self.matches_df.columns:
            self.matches_df['home_advantage'] = (
                self.matches_df['home_total_goals'] - self.matches_df['away_total_goals']
            )
            
            yearly_advantage = self.matches_df.groupby('year')['home_advantage'].mean().reset_index()
            results['home_advantage_trend'] = yearly_advantage
        
        # 2. Scoring trends
        if 'home_total_goals' in self.matches_df.columns:
            yearly_scoring = self.matches_df.groupby('year').agg({
                'home_total_goals': 'mean',
                'away_total_goals': 'mean'
            }).reset_index()
            yearly_scoring['total_scoring'] = (
                yearly_scoring['home_total_goals'] + yearly_scoring['away_total_goals']
            )
            results['scoring_trends'] = yearly_scoring
        
        # 3. Margin distributions
        if 'margin' in self.matches_df.columns:
            margin_stats = {
                'mean_margin': self.matches_df['margin'].mean(),
                'median_margin': self.matches_df['margin'].median(),
                'std_margin': self.matches_df['margin'].std(),
                'margin_distribution': self.matches_df['margin'].value_counts().sort_index()
            }
            results['margin_analysis'] = margin_stats
        
        self.analysis_results['match_analysis'] = results
        return results
    
    def player_analysis(self) -> Dict[str, Any]:
        """Analyze player performance patterns and distributions."""
        print("\n=== PLAYER ANALYSIS ===")
        
        results = {}
        
        # 1. Performance distributions
        numeric_cols = ['kicks', 'marks', 'handballs', 'disposals', 'goals', 'behinds', 
                       'hit_outs', 'tackles', 'rebound_50s', 'inside_50s', 'clearances']
        
        performance_stats = {}
        for col in numeric_cols:
            if col in self.players_df.columns:
                stats = {
                    'mean': self.players_df[col].mean(),
                    'median': self.players_df[col].median(),
                    'std': self.players_df[col].std(),
                    'min': self.players_df[col].min(),
                    'max': self.players_df[col].max(),
                    'q25': self.players_df[col].quantile(0.25),
                    'q75': self.players_df[col].quantile(0.75)
                }
                performance_stats[col] = stats
        
        results['performance_distributions'] = performance_stats
        
        # 2. Player longevity analysis
        player_careers = self.players_df.groupby(['first_name', 'last_name']).agg({
            'year': ['min', 'max', 'count'],
            'games_played': 'sum',
            'goals': 'sum',
            'disposals': 'sum'
        }).reset_index()
        
        player_careers.columns = ['first_name', 'last_name', 'debut_year', 'last_year', 
                                'seasons', 'total_games', 'total_goals', 'total_disposals']
        player_careers['career_length'] = player_careers['last_year'] - player_careers['debut_year'] + 1
        
        results['career_analysis'] = {
            'avg_career_length': player_careers['career_length'].mean(),
            'avg_games_per_career': player_careers['total_games'].mean(),
            'longest_careers': player_careers.nlargest(20, 'career_length'),
            'most_games': player_careers.nlargest(20, 'total_games'),
            'most_goals': player_careers.nlargest(20, 'total_goals')
        }
        
        # 3. Position-based analysis (infer from stats)
        position_patterns = self._analyze_position_patterns()
        results['position_analysis'] = position_patterns
        
        # 4. Statistical correlations
        correlation_matrix = self.players_df[numeric_cols].corr()
        results['statistical_correlations'] = correlation_matrix
        
        self.analysis_results['player_analysis'] = results
        return results
    
    def data_quality_assessment(self) -> Dict[str, Any]:
        """Comprehensive data quality assessment."""
        print("\n=== DATA QUALITY ASSESSMENT ===")
        
        results = {}
        
        # 1. Missing data analysis
        missing_data = {}
        for col in self.players_df.columns:
            missing_count = self.players_df[col].isnull().sum()
            missing_pct = (missing_count / len(self.players_df)) * 100
            missing_data[col] = {'count': missing_count, 'percentage': missing_pct}
        
        results['missing_data'] = missing_data
        
        # 2. Outlier detection
        outliers = self._detect_outliers()
        results['outliers'] = outliers
        
        # 3. Data consistency checks
        consistency_checks = self._check_data_consistency()
        results['consistency_checks'] = consistency_checks
        
        # 4. Era reliability assessment
        era_reliability = self._assess_era_reliability()
        results['era_reliability'] = era_reliability
        
        self.analysis_results['data_quality'] = results
        return results
    
    def _calculate_completeness(self, df: pd.DataFrame) -> float:
        """Calculate data completeness percentage."""
        total_cells = df.shape[0] * df.shape[1]
        non_null_cells = df.count().sum()
        return (non_null_cells / total_cells) * 100
    
    def _analyze_data_quality_timeline(self) -> Dict[str, Any]:
        """Analyze how data quality changes over time."""
        timeline = {}
        
        for year in range(1897, 2026):
            year_data = self.players_df[self.players_df['year'] == year]
            if len(year_data) > 0:
                timeline[year] = {
                    'record_count': len(year_data),
                    'completeness': self._calculate_completeness(year_data),
                    'teams': year_data['team'].nunique(),
                    'players': year_data[['first_name', 'last_name']].drop_duplicates().shape[0]
                }
        
        return timeline
    
    def _analyze_position_patterns(self) -> Dict[str, Any]:
        """Infer player positions from statistical patterns."""
        patterns = {}
        
        # Define position indicators based on statistical patterns
        position_indicators = {
            'Forward': ['goals', 'marks_inside_50', 'goal_assist'],
            'Midfielder': ['disposals', 'clearances', 'inside_50s'],
            'Defender': ['rebound_50s', 'one_percenters'],
            'Ruck': ['hit_outs']
        }
        
        for position, indicators in position_indicators.items():
            available_indicators = [ind for ind in indicators if ind in self.players_df.columns]
            if available_indicators:
                # Calculate average stats for this position
                avg_stats = self.players_df[available_indicators].mean()
                patterns[position] = avg_stats.to_dict()
        
        return patterns
    
    def _detect_outliers(self) -> Dict[str, Any]:
        """Detect statistical outliers in the data."""
        outliers = {}
        
        numeric_cols = ['kicks', 'marks', 'handballs', 'disposals', 'goals', 'behinds', 
                       'hit_outs', 'tackles', 'rebound_50s', 'inside_50s', 'clearances']
        
        for col in numeric_cols:
            if col in self.players_df.columns:
                Q1 = self.players_df[col].quantile(0.25)
                Q3 = self.players_df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_count = len(self.players_df[
                    (self.players_df[col] < lower_bound) | 
                    (self.players_df[col] > upper_bound)
                ])
                
                outliers[col] = {
                    'outlier_count': outlier_count,
                    'outlier_percentage': (outlier_count / len(self.players_df)) * 100,
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        return outliers
    
    def _check_data_consistency(self) -> Dict[str, Any]:
        """Check data consistency and logical relationships."""
        consistency = {}
        
        # Check if disposals = kicks + handballs
        if all(col in self.players_df.columns for col in ['disposals', 'kicks', 'handballs']):
            disposal_check = self.players_df['disposals'] == (self.players_df['kicks'] + self.players_df['handballs'])
            consistency['disposal_consistency'] = {
                'consistent_records': disposal_check.sum(),
                'inconsistent_records': (~disposal_check).sum(),
                'consistency_percentage': (disposal_check.sum() / len(self.players_df)) * 100
            }
        
        # Check for negative values in positive-only stats
        positive_stats = ['kicks', 'marks', 'handballs', 'disposals', 'goals', 'behinds', 
                         'hit_outs', 'tackles', 'rebound_50s', 'inside_50s', 'clearances']
        
        negative_values = {}
        for stat in positive_stats:
            if stat in self.players_df.columns:
                negative_count = (self.players_df[stat] < 0).sum()
                negative_values[stat] = negative_count
        
        consistency['negative_values'] = negative_values
        
        return consistency
    
    def _assess_era_reliability(self) -> Dict[str, Any]:
        """Assess which eras have reliable data for modeling."""
        eras = {
            'Early Era (1897-1945)': (1897, 1945),
            'Post-War Era (1946-1975)': (1946, 1975),
            'Modern Era (1976-2000)': (1976, 2000),
            'Contemporary Era (2001-2025)': (2001, 2025)
        }
        
        reliability = {}
        for era_name, (start_year, end_year) in eras.items():
            era_data = self.players_df[
                (self.players_df['year'] >= start_year) & 
                (self.players_df['year'] <= end_year)
            ]
            
            if len(era_data) > 0:
                # Calculate reliability score based on multiple factors
                completeness = self._calculate_completeness(era_data)
                data_volume = len(era_data)
                team_coverage = era_data['team'].nunique()
                
                # Simple reliability score (0-100)
                reliability_score = (
                    (completeness * 0.4) + 
                    (min(data_volume / 1000, 100) * 0.3) + 
                    (min(team_coverage / 18, 100) * 0.3)
                )
                
                reliability[era_name] = {
                    'reliability_score': reliability_score,
                    'completeness': completeness,
                    'data_volume': data_volume,
                    'team_coverage': team_coverage,
                    'recommended_for_modeling': reliability_score > 70
                }
        
        return reliability
    
    def generate_visualizations(self):
        """Generate comprehensive visualizations."""
        print("\n=== GENERATING VISUALIZATIONS ===")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Temporal Analysis Visualizations
        self._create_temporal_plots()
        
        # 2. Player Analysis Visualizations
        self._create_player_plots()
        
        # 3. Data Quality Visualizations
        self._create_quality_plots()
        
        # 4. Match Analysis Visualizations (if data available)
        if not self.matches_df.empty:
            self._create_match_plots()
    
    def _create_temporal_plots(self):
        """Create temporal analysis visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Temporal Analysis: AFL Data Evolution (1897-2025)', fontsize=16)
        
        # 1. Data volume over time
        yearly_counts = self.players_df.groupby('year').size()
        axes[0, 0].plot(yearly_counts.index, yearly_counts.values, linewidth=2)
        axes[0, 0].set_title('Player Records per Year')
        axes[0, 0].set_xlabel('Year')
        axes[0, 0].set_ylabel('Number of Records')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Data completeness over time
        completeness_by_year = []
        years = []
        for year in range(1897, 2026):
            year_data = self.players_df[self.players_df['year'] == year]
            if len(year_data) > 0:
                completeness = self._calculate_completeness(year_data)
                completeness_by_year.append(completeness)
                years.append(year)
        
        axes[0, 1].plot(years, completeness_by_year, linewidth=2, color='orange')
        axes[0, 1].set_title('Data Completeness Over Time')
        axes[0, 1].set_xlabel('Year')
        axes[0, 1].set_ylabel('Completeness (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Team coverage over time
        teams_by_year = self.players_df.groupby('year')['team'].nunique()
        axes[1, 0].plot(teams_by_year.index, teams_by_year.values, linewidth=2, color='green')
        axes[1, 0].set_title('Number of Teams Over Time')
        axes[1, 0].set_xlabel('Year')
        axes[1, 0].set_ylabel('Number of Teams')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Era comparison
        era_data = self.analysis_results['temporal_analysis']['era_analysis']
        eras = list(era_data.keys())
        completeness_scores = [era_data[era]['data_completeness'] for era in eras]
        
        axes[1, 1].bar(eras, completeness_scores, color=['red', 'orange', 'green', 'blue'])
        axes[1, 1].set_title('Data Completeness by Era')
        axes[1, 1].set_xlabel('Era')
        axes[1, 1].set_ylabel('Completeness (%)')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'temporal_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_player_plots(self):
        """Create player analysis visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Player Performance Analysis', fontsize=16)
        
        # 1. Performance distributions
        numeric_cols = ['kicks', 'marks', 'handballs', 'disposals', 'goals']
        available_cols = [col for col in numeric_cols if col in self.players_df.columns]
        
        if available_cols:
            # Sample data for visualization (take 1000 random records)
            sample_data = self.players_df[available_cols].sample(n=min(1000, len(self.players_df)))
            sample_data.boxplot(ax=axes[0, 0])
            axes[0, 0].set_title('Performance Statistics Distribution')
            axes[0, 0].set_ylabel('Value')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Career length distribution
        player_careers = self.players_df.groupby(['first_name', 'last_name']).agg({
            'year': ['min', 'max']
        }).reset_index()
        player_careers.columns = ['first_name', 'last_name', 'debut_year', 'last_year']
        player_careers['career_length'] = player_careers['last_year'] - player_careers['debut_year'] + 1
        
        axes[0, 1].hist(player_careers['career_length'], bins=30, alpha=0.7, color='skyblue')
        axes[0, 1].set_title('Player Career Length Distribution')
        axes[0, 1].set_xlabel('Career Length (Years)')
        axes[0, 1].set_ylabel('Number of Players')
        
        # 3. Goals vs Disposals scatter
        if 'goals' in self.players_df.columns and 'disposals' in self.players_df.columns:
            sample_data = self.players_df[['goals', 'disposals']].sample(n=min(1000, len(self.players_df)))
            axes[1, 0].scatter(sample_data['disposals'], sample_data['goals'], alpha=0.6)
            axes[1, 0].set_title('Goals vs Disposals')
            axes[1, 0].set_xlabel('Disposals')
            axes[1, 0].set_ylabel('Goals')
        
        # 4. Top performers by era
        era_performers = self._get_top_performers_by_era()
        if era_performers:
            era_names = list(era_performers.keys())
            avg_goals = [era_performers[era]['avg_goals'] for era in era_names]
            
            axes[1, 1].bar(era_names, avg_goals, color=['red', 'orange', 'green', 'blue'])
            axes[1, 1].set_title('Average Goals per Player by Era')
            axes[1, 1].set_xlabel('Era')
            axes[1, 1].set_ylabel('Average Goals')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'player_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_quality_plots(self):
        """Create data quality visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Data Quality Assessment', fontsize=16)
        
        # 1. Missing data heatmap
        missing_data = self.analysis_results['data_quality']['missing_data']
        missing_df = pd.DataFrame.from_dict(missing_data, orient='index')
        
        if not missing_df.empty:
            # Select top 15 columns with most missing data
            top_missing = missing_df.nlargest(15, 'percentage')
            sns.heatmap(top_missing[['percentage']].T, annot=True, fmt='.1f', 
                       cmap='Reds', ax=axes[0, 0])
            axes[0, 0].set_title('Missing Data Percentage (Top 15 Columns)')
        
        # 2. Outlier analysis
        outliers = self.analysis_results['data_quality']['outliers']
        if outliers:
            outlier_percentages = [outliers[col]['outlier_percentage'] 
                                 for col in outliers.keys()]
            outlier_cols = list(outliers.keys())
            
            axes[0, 1].bar(range(len(outlier_cols)), outlier_percentages)
            axes[0, 1].set_title('Outlier Percentage by Statistic')
            axes[0, 1].set_xlabel('Statistics')
            axes[0, 1].set_ylabel('Outlier Percentage (%)')
            axes[0, 1].set_xticks(range(len(outlier_cols)))
            axes[0, 1].set_xticklabels(outlier_cols, rotation=45)
        
        # 3. Era reliability scores
        era_reliability = self.analysis_results['data_quality']['era_reliability']
        if era_reliability:
            eras = list(era_reliability.keys())
            reliability_scores = [era_reliability[era]['reliability_score'] 
                                for era in eras]
            
            colors = ['red' if score < 50 else 'orange' if score < 70 else 'green' 
                     for score in reliability_scores]
            
            axes[1, 0].bar(eras, reliability_scores, color=colors)
            axes[1, 0].set_title('Era Reliability Scores')
            axes[1, 0].set_xlabel('Era')
            axes[1, 0].set_ylabel('Reliability Score')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].axhline(y=70, color='red', linestyle='--', alpha=0.7, 
                              label='Modeling Threshold')
            axes[1, 0].legend()
        
        # 4. Data consistency check
        consistency = self.analysis_results['data_quality']['consistency_checks']
        if 'disposal_consistency' in consistency:
            disposal_consistency = consistency['disposal_consistency']
            labels = ['Consistent', 'Inconsistent']
            sizes = [disposal_consistency['consistent_records'], 
                    disposal_consistency['inconsistent_records']]
            
            axes[1, 1].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Disposal Data Consistency')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'data_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_match_plots(self):
        """Create match analysis visualizations."""
        if self.matches_df.empty:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Match Analysis', fontsize=16)
        
        # 1. Home advantage over time
        if 'home_advantage_trend' in self.analysis_results['match_analysis']:
            home_adv = self.analysis_results['match_analysis']['home_advantage_trend']
            axes[0, 0].plot(home_adv['year'], home_adv['home_advantage'], linewidth=2)
            axes[0, 0].set_title('Home Advantage Over Time')
            axes[0, 0].set_xlabel('Year')
            axes[0, 0].set_ylabel('Home Advantage (Goals)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Scoring trends
        if 'scoring_trends' in self.analysis_results['match_analysis']:
            scoring = self.analysis_results['match_analysis']['scoring_trends']
            axes[0, 1].plot(scoring['year'], scoring['total_scoring'], linewidth=2, color='green')
            axes[0, 1].set_title('Total Scoring Over Time')
            axes[0, 1].set_xlabel('Year')
            axes[0, 1].set_ylabel('Total Goals per Game')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Margin distribution
        if 'margin_analysis' in self.analysis_results['match_analysis']:
            margin_stats = self.analysis_results['match_analysis']['margin_analysis']
            if 'margin_distribution' in margin_stats:
                margin_dist = margin_stats['margin_distribution']
                axes[1, 0].hist(margin_dist.index, weights=margin_dist.values, bins=30, alpha=0.7)
                axes[1, 0].set_title('Margin Distribution')
                axes[1, 0].set_xlabel('Margin')
                axes[1, 0].set_ylabel('Frequency')
        
        # 4. Seasonal patterns
        if 'date' in self.matches_df.columns:
            try:
                self.matches_df['date'] = pd.to_datetime(self.matches_df['date'])
                self.matches_df['month'] = self.matches_df['date'].dt.month
                monthly_scoring = self.matches_df.groupby('month')['home_total_goals'].mean()
                
                axes[1, 1].plot(monthly_scoring.index, monthly_scoring.values, marker='o')
                axes[1, 1].set_title('Scoring by Month')
                axes[1, 1].set_xlabel('Month')
                axes[1, 1].set_ylabel('Average Goals')
                axes[1, 1].set_xticks(range(1, 13))
            except:
                pass
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'match_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _get_top_performers_by_era(self) -> Dict[str, Dict[str, float]]:
        """Get top performers by era for visualization."""
        eras = {
            'Early Era (1897-1945)': (1897, 1945),
            'Post-War Era (1946-1975)': (1946, 1975),
            'Modern Era (1976-2000)': (1976, 2000),
            'Contemporary Era (2001-2025)': (2001, 2025)
        }
        
        era_performers = {}
        for era_name, (start_year, end_year) in eras.items():
            era_data = self.players_df[
                (self.players_df['year'] >= start_year) & 
                (self.players_df['year'] <= end_year)
            ]
            
            if len(era_data) > 0 and 'goals' in era_data.columns:
                era_performers[era_name] = {
                    'avg_goals': era_data['goals'].mean(),
                    'avg_disposals': era_data['disposals'].mean() if 'disposals' in era_data.columns else 0
                }
        
        return era_performers
    
    def generate_recommendations(self) -> Dict[str, Any]:
        """Generate recommendations based on EDA findings."""
        print("\n=== GENERATING RECOMMENDATIONS ===")
        
        recommendations = {
            'reliable_eras': [],
            'predictive_statistics': [],
            'preprocessing_strategies': [],
            'modeling_approach': {},
            'data_limitations': []
        }
        
        # 1. Identify reliable eras for modeling
        era_reliability = self.analysis_results['data_quality']['era_reliability']
        for era, data in era_reliability.items():
            if data['recommended_for_modeling']:
                recommendations['reliable_eras'].append({
                    'era': era,
                    'reliability_score': data['reliability_score'],
                    'data_volume': data['data_volume']
                })
        
        # 2. Identify most predictive statistics
        if 'statistical_correlations' in self.analysis_results['player_analysis']:
            correlations = self.analysis_results['player_analysis']['statistical_correlations']
            if 'goals' in correlations.columns:
                goal_correlations = correlations['goals'].abs().sort_values(ascending=False)
                top_predictors = goal_correlations.head(10).index.tolist()
                recommendations['predictive_statistics'] = top_predictors
        
        # 3. Preprocessing strategies
        missing_data = self.analysis_results['data_quality']['missing_data']
        high_missing_cols = [col for col, data in missing_data.items() 
                           if data['percentage'] > 20]
        
        recommendations['preprocessing_strategies'] = [
            'Remove columns with >20% missing data',
            'Use median imputation for numeric columns',
            'Create era-specific models to handle rule changes',
            'Standardize statistics across different time periods',
            'Handle outliers using IQR method'
        ]
        
        # 4. Modeling approach recommendations
        recommendations['modeling_approach'] = {
            'time_periods': 'Use era-specific models (1976-2025 recommended)',
            'validation': 'Use time-based cross-validation',
            'features': 'Focus on core statistics: disposals, goals, marks, tackles',
            'target': 'Goals scored (most reliable target variable)',
            'ensemble': 'Combine era-specific models for predictions'
        }
        
        # 5. Data limitations
        recommendations['data_limitations'] = [
            'Limited data before 1976 for reliable modeling',
            'Rule changes affect statistical patterns over time',
            'Position information not explicitly available',
            'Some statistics introduced in later eras'
        ]
        
        self.analysis_results['recommendations'] = recommendations
        return recommendations
    
    def generate_report(self):
        """Generate comprehensive EDA report."""
        print("\n=== GENERATING COMPREHENSIVE REPORT ===")
        
        # Generate all analyses
        self.temporal_analysis()
        self.match_analysis()
        self.player_analysis()
        self.data_quality_assessment()
        self.generate_recommendations()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Create report
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_player_records': len(self.players_df),
                'total_match_records': len(self.matches_df),
                'data_years': f"{self.players_df['year'].min()}-{self.players_df['year'].max()}",
                'teams_covered': self.players_df['team'].nunique(),
                'unique_players': self.players_df[['first_name', 'last_name']].drop_duplicates().shape[0]
            },
            'key_findings': self._extract_key_findings(),
            'analysis_results': self.analysis_results,
            'visualizations': [
                'temporal_analysis.png',
                'player_analysis.png', 
                'data_quality.png',
                'match_analysis.png'
            ]
        }
        
        # Save report
        with open(self.output_dir / 'eda_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create markdown summary
        self._create_markdown_summary(report)
        
        print(f"\nEDA Report saved to: {self.output_dir / 'eda_report.json'}")
        print(f"Visualizations saved to: {self.output_dir}")
        
        return report
    
    def _extract_key_findings(self) -> Dict[str, Any]:
        """Extract key findings from the analysis."""
        findings = {
            'temporal_insights': [],
            'performance_patterns': [],
            'data_quality_insights': [],
            'modeling_implications': []
        }
        
        # Temporal insights
        era_analysis = self.analysis_results['temporal_analysis']['era_analysis']
        best_era = max(era_analysis.items(), key=lambda x: x[1]['data_completeness'])
        findings['temporal_insights'].append(f"Best data quality in {best_era[0]} ({best_era[1]['data_completeness']:.1f}% completeness)")
        
        # Performance patterns
        if 'performance_distributions' in self.analysis_results['player_analysis']:
            perf_stats = self.analysis_results['player_analysis']['performance_distributions']
            if 'goals' in perf_stats:
                avg_goals = perf_stats['goals']['mean']
                findings['performance_patterns'].append(f"Average goals per game: {avg_goals:.2f}")
        
        # Data quality insights
        era_reliability = self.analysis_results['data_quality']['era_reliability']
        reliable_eras = [era for era, data in era_reliability.items() 
                        if data['recommended_for_modeling']]
        findings['data_quality_insights'].append(f"Reliable eras for modeling: {', '.join(reliable_eras)}")
        
        # Modeling implications
        recommendations = self.analysis_results['recommendations']
        findings['modeling_implications'].append(f"Recommended modeling period: {recommendations['modeling_approach']['time_periods']}")
        
        return findings
    
    def _create_markdown_summary(self, report: Dict[str, Any]):
        """Create a markdown summary of the EDA."""
        markdown_content = f"""# AFL Data Exploratory Analysis Report

## Executive Summary
- **Total Player Records**: {report['summary']['total_player_records']:,}
- **Data Coverage**: {report['summary']['data_years']}
- **Teams Covered**: {report['summary']['teams_covered']}
- **Unique Players**: {report['summary']['unique_players']:,}

## Key Findings

### Temporal Analysis
{chr(10).join([f"- {finding}" for finding in report['key_findings']['temporal_insights']])}

### Performance Patterns
{chr(10).join([f"- {finding}" for finding in report['key_findings']['performance_patterns']])}

### Data Quality Insights
{chr(10).join([f"- {finding}" for finding in report['key_findings']['data_quality_insights']])}

### Modeling Implications
{chr(10).join([f"- {finding}" for finding in report['key_findings']['modeling_implications']])}

## Recommendations

### Reliable Eras for Modeling
{chr(10).join([f"- {era['era']}: Score {era['reliability_score']:.1f}, {era['data_volume']:,} records" for era in report['analysis_results']['recommendations']['reliable_eras']])}

### Predictive Statistics
Top predictive statistics for goals:
{chr(10).join([f"- {stat}" for stat in report['analysis_results']['recommendations']['predictive_statistics'][:5]])}

### Preprocessing Strategies
{chr(10).join([f"- {strategy}" for strategy in report['analysis_results']['recommendations']['preprocessing_strategies']])}

## Visualizations
Generated visualizations:
{chr(10).join([f"- {viz}" for viz in report['visualizations']])}

## Data Limitations
{chr(10).join([f"- {limitation}" for limitation in report['analysis_results']['recommendations']['data_limitations']])}

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        with open(self.output_dir / 'eda_summary.md', 'w') as f:
            f.write(markdown_content)
    
    def close(self):
        """Close database connection."""
        self.conn.close()

def main():
    """Main function to run the EDA."""
    print("="*60)
    print("AFL DATA EXPLORATORY ANALYSIS (Phase 1B)")
    print("="*60)
    
    # Initialize EDA
    eda = AFLEDA()
    
    try:
        print("EDA initialized successfully!")
        print(f"Player records: {len(eda.players_df):,}")
        print(f"Match records: {len(eda.matches_df):,}")
        
    except Exception as e:
        print(f"Error during EDA: {e}")
        raise
    finally:
        eda.close()

if __name__ == "__main__":
    main() 