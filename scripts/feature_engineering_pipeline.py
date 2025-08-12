"""
Phase 2A: Advanced Feature Engineering Pipeline
AFL Prediction Model - Feature Engineering Implementation

This script implements a sophisticated feature engineering pipeline that creates:
1. Team Performance Features (rolling averages, EWM, head-to-head)
2. Player Aggregation Features (team composition, experience, star impact)
3. Contextual Features (venue effects, rest days, situational factors)
4. Advanced Features (interactions, trends, momentum indicators)
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineeringPipeline:
    """
    Comprehensive feature engineering pipeline for AFL prediction modeling.
    """
    
    def __init__(self, db_path="afl_data/afl_database.db"):
        self.db_path = db_path
        self.matches_df = None
        self.players_df = None
        self.features_df = None
        self.feature_importance = {}
        self.correlation_matrix = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_data(self):
        """Load and prepare data for feature engineering."""
        print("Loading data for feature engineering...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Load match data with proper date parsing
        self.matches_df = pd.read_sql_query("""
            SELECT * FROM matches 
            WHERE year >= 1991 
            ORDER BY year, date
        """, conn)
        
        # Load player data
        self.players_df = pd.read_sql_query("""
            SELECT * FROM players 
            WHERE year >= 1991 
            ORDER BY year, team
        """, conn)
        
        conn.close()
        
        # Convert date column to datetime
        self.matches_df['date'] = pd.to_datetime(self.matches_df['date'], errors='coerce')
        
        print(f"Loaded {len(self.matches_df)} matches and {len(self.players_df)} player records")
        
    def create_team_performance_features(self):
        """Create team performance features including rolling averages and EWM."""
        print("Creating team performance features...")
        
        # Prepare team performance data
        team_stats = []
        
        for _, match in self.matches_df.iterrows():
            # Calculate margin based on goals
            home_goals = match['home_total_goals']
            away_goals = match['away_total_goals']
            margin = home_goals - away_goals
            winning_team = match['home_team'] if margin > 0 else match['away_team'] if margin < 0 else 'Draw'
            
            # Home team stats
            team_stats.append({
                'team': match['home_team'],
                'date': match['date'],
                'year': match['year'],
                'goals_for': match['home_total_goals'],
                'behinds_for': match['home_total_behinds'],
                'goals_against': match['away_total_goals'],
                'behinds_against': match['away_total_behinds'],
                'margin': margin,
                'venue': match['venue'],
                'is_home': True
            })
            
            # Away team stats
            team_stats.append({
                'team': match['away_team'],
                'date': match['date'],
                'year': match['year'],
                'goals_for': match['away_total_goals'],
                'behinds_for': match['away_total_behinds'],
                'goals_against': match['home_total_goals'],
                'behinds_against': match['home_total_behinds'],
                'margin': -margin,
                'venue': match['venue'],
                'is_home': False
            })
        
        team_stats_df = pd.DataFrame(team_stats)
        team_stats_df = team_stats_df.sort_values(['team', 'date']).reset_index(drop=True)
        
        # Calculate rolling averages for different windows
        windows = [5, 10, 20]
        for window in windows:
            team_stats_df[f'rolling_avg_goals_for_{window}'] = team_stats_df.groupby('team')['goals_for'].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
            team_stats_df[f'rolling_avg_goals_against_{window}'] = team_stats_df.groupby('team')['goals_against'].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
            team_stats_df[f'rolling_avg_margin_{window}'] = team_stats_df.groupby('team')['margin'].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
        
        # Calculate exponentially weighted moving averages
        alphas = [0.1, 0.3, 0.5]
        for alpha in alphas:
            team_stats_df[f'ewm_goals_for_alpha_{alpha}'] = team_stats_df.groupby('team')['goals_for'].ewm(alpha=alpha).mean().reset_index(0, drop=True)
            team_stats_df[f'ewm_goals_against_alpha_{alpha}'] = team_stats_df.groupby('team')['goals_against'].ewm(alpha=alpha).mean().reset_index(0, drop=True)
        
        # Calculate home/away specific performance
        home_stats = team_stats_df[team_stats_df['is_home'] == True].groupby('team').agg({
            'goals_for': 'mean',
            'goals_against': 'mean',
            'margin': 'mean'
        }).add_suffix('_home_avg')
        
        away_stats = team_stats_df[team_stats_df['is_home'] == False].groupby('team').agg({
            'goals_for': 'mean',
            'goals_against': 'mean',
            'margin': 'mean'
        }).add_suffix('_away_avg')
        
        # Calculate season averages
        season_stats = team_stats_df.groupby(['team', 'year']).agg({
            'goals_for': 'mean',
            'goals_against': 'mean',
            'margin': 'mean'
        }).add_suffix('_season_avg').reset_index()
        
        # Calculate career averages
        career_stats = team_stats_df.groupby('team').agg({
            'goals_for': 'mean',
            'goals_against': 'mean',
            'margin': 'mean'
        }).add_suffix('_career_avg')
        
        # Merge all team performance features
        team_performance_features = team_stats_df.merge(
            home_stats, left_on='team', right_index=True, how='left'
        ).merge(
            away_stats, left_on='team', right_index=True, how='left'
        ).merge(
            season_stats, on=['team', 'year'], how='left'
        ).merge(
            career_stats, left_on='team', right_index=True, how='left'
        )
        
        return team_performance_features
    
    def create_head_to_head_features(self):
        """Create head-to-head historical performance features."""
        print("Creating head-to-head features...")
        
        h2h_features = []
        
        for idx, match in self.matches_df.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            match_date = match['date']
            
            # Get historical matches between these teams
            historical_matches = self.matches_df[
                ((self.matches_df['home_team'] == home_team) & (self.matches_df['away_team'] == away_team)) |
                ((self.matches_df['home_team'] == away_team) & (self.matches_df['away_team'] == home_team))
            ]
            
            # Only consider matches before current match
            historical_matches = historical_matches[historical_matches['date'] < match_date]
            
            if len(historical_matches) > 0:
                # Calculate head-to-head statistics
                home_wins = 0
                away_wins = 0
                total_goals_home = 0
                total_goals_away = 0
                
                for _, h_match in historical_matches.iterrows():
                    if h_match['home_team'] == home_team:
                        # Home team in historical match is current home team
                        if h_match['home_total_goals'] > h_match['away_total_goals']:
                            home_wins += 1
                        elif h_match['away_total_goals'] > h_match['home_total_goals']:
                            away_wins += 1
                        total_goals_home += h_match['home_total_goals']
                        total_goals_away += h_match['away_total_goals']
                    else:
                        # Away team in historical match is current home team
                        if h_match['away_total_goals'] > h_match['home_total_goals']:
                            home_wins += 1
                        elif h_match['home_total_goals'] > h_match['away_total_goals']:
                            away_wins += 1
                        total_goals_home += h_match['away_total_goals']
                        total_goals_away += h_match['home_total_goals']
                
                h2h_features.append({
                    'match_id': idx,  # Use index as match_id
                    'h2h_total_matches': len(historical_matches),
                    'h2h_home_wins': home_wins,
                    'h2h_away_wins': away_wins,
                    'h2h_home_win_rate': home_wins / len(historical_matches) if len(historical_matches) > 0 else 0.5,
                    'h2h_avg_goals_home': total_goals_home / len(historical_matches) if len(historical_matches) > 0 else 0,
                    'h2h_avg_goals_away': total_goals_away / len(historical_matches) if len(historical_matches) > 0 else 0,
                    'h2h_recent_form': home_wins - away_wins  # Positive favors home team
                })
            else:
                # No historical data
                h2h_features.append({
                    'match_id': idx,  # Use index as match_id
                    'h2h_total_matches': 0,
                    'h2h_home_wins': 0,
                    'h2h_away_wins': 0,
                    'h2h_home_win_rate': 0.5,
                    'h2h_avg_goals_home': 0,
                    'h2h_avg_goals_away': 0,
                    'h2h_recent_form': 0
                })
        
        return pd.DataFrame(h2h_features)
    
    def create_player_aggregation_features(self):
        """Create player aggregation features for team composition analysis."""
        print("Creating player aggregation features...")
        
        player_features = []
        
        for idx, match in self.matches_df.iterrows():
            match_year = match['year']
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Get player data for both teams in the match year
            home_players = self.players_df[
                (self.players_df['team'] == home_team) & 
                (self.players_df['year'] == match_year)
            ]
            
            away_players = self.players_df[
                (self.players_df['team'] == away_team) & 
                (self.players_df['year'] == match_year)
            ]
            
            # Home team player features
            home_features = self._calculate_team_player_features(home_players, 'home')
            
            # Away team player features
            away_features = self._calculate_team_player_features(away_players, 'away')
            
            # Combine features
            match_player_features = {
                'match_id': idx,  # Use index as match_id
                **home_features,
                **away_features
            }
            
            player_features.append(match_player_features)
        
        return pd.DataFrame(player_features)
    
    def _calculate_team_player_features(self, team_players, team_type):
        """Calculate player-based features for a team."""
        features = {}
        
        if len(team_players) > 0:
            # Core statistics averages
            for stat in ['kicks', 'marks', 'handballs', 'disposals', 'goals', 'tackles']:
                if stat in team_players.columns:
                    features[f'{team_type}_avg_{stat}'] = team_players[stat].mean()
                    features[f'{team_type}_max_{stat}'] = team_players[stat].max()
                    features[f'{team_type}_std_{stat}'] = team_players[stat].std()
            
            # Experience metrics
            features[f'{team_type}_total_games'] = team_players['games_played'].sum() if 'games_played' in team_players.columns else 0
            features[f'{team_type}_avg_games_played'] = team_players['games_played'].mean() if 'games_played' in team_players.columns else 0
            
            # Team composition strength (aggregate player ratings)
            if 'disposals' in team_players.columns:
                features[f'{team_type}_team_strength'] = team_players['disposals'].sum()
                features[f'{team_type}_star_player_impact'] = team_players['disposals'].nlargest(3).sum()  # Top 3 players
            
            # Position balance (assuming different positions have different typical disposal counts)
            if 'disposals' in team_players.columns:
                disposal_quartiles = team_players['disposals'].quantile([0.25, 0.5, 0.75])
                features[f'{team_type}_position_balance'] = disposal_quartiles[0.75] - disposal_quartiles[0.25]
            
            # Team depth (number of players with above-average performance)
            if 'disposals' in team_players.columns:
                avg_disposals = team_players['disposals'].mean()
                features[f'{team_type}_depth_score'] = (team_players['disposals'] > avg_disposals).sum()
        else:
            # Default values if no player data
            for stat in ['kicks', 'marks', 'handballs', 'disposals', 'goals', 'tackles']:
                features[f'{team_type}_avg_{stat}'] = 0
                features[f'{team_type}_max_{stat}'] = 0
                features[f'{team_type}_std_{stat}'] = 0
            
            features[f'{team_type}_total_games'] = 0
            features[f'{team_type}_avg_games_played'] = 0
            features[f'{team_type}_team_strength'] = 0
            features[f'{team_type}_star_player_impact'] = 0
            features[f'{team_type}_position_balance'] = 0
            features[f'{team_type}_depth_score'] = 0
        
        return features
    
    def create_contextual_features(self):
        """Create contextual features including venue effects and situational factors."""
        print("Creating contextual features...")
        
        contextual_features = []
        
        for idx, match in self.matches_df.iterrows():
            venue = match['venue']
            match_date = match['date']
            home_team = match['home_team']
            away_team = match['away_team']
            
            # Venue-specific features
            venue_matches = self.matches_df[self.matches_df['venue'] == venue]
            venue_home_advantage = venue_matches.groupby('home_team').agg({
                'home_total_goals': 'mean',
                'away_total_goals': 'mean'
            }).reset_index()
            
            if len(venue_home_advantage) > 0:
                venue_home_advantage['venue_advantage'] = (
                    venue_home_advantage['home_total_goals'] - venue_home_advantage['away_total_goals']
                )
                venue_avg_advantage = venue_home_advantage['venue_advantage'].mean()
            else:
                venue_avg_advantage = 0
            
            # Rest days between games
            home_previous_match = self.matches_df[
                ((self.matches_df['home_team'] == home_team) | (self.matches_df['away_team'] == home_team)) &
                (self.matches_df['date'] < match_date)
            ].sort_values('date').tail(1)
            
            away_previous_match = self.matches_df[
                ((self.matches_df['home_team'] == away_team) | (self.matches_df['away_team'] == away_team)) &
                (self.matches_df['date'] < match_date)
            ].sort_values('date').tail(1)
            
            home_rest_days = (match_date - home_previous_match['date'].iloc[0]).days if len(home_previous_match) > 0 else 7
            away_rest_days = (match_date - away_previous_match['date'].iloc[0]).days if len(away_previous_match) > 0 else 7
            
            # Time of season effects
            season_matches = self.matches_df[self.matches_df['year'] == match['year']]
            match_week = len(season_matches[season_matches['date'] <= match_date])
            total_season_matches = len(season_matches)
            season_progress = match_week / total_season_matches if total_season_matches > 0 else 0.5
            
            # Historical matchup performance
            historical_matchups = self.matches_df[
                ((self.matches_df['home_team'] == home_team) & (self.matches_df['away_team'] == away_team)) |
                ((self.matches_df['home_team'] == away_team) & (self.matches_df['away_team'] == home_team))
            ]
            historical_matchups = historical_matchups[historical_matchups['date'] < match_date]
            
            if len(historical_matchups) > 0:
                recent_matchups = historical_matchups.tail(5)  # Last 5 matchups
                avg_goals_per_game = (
                    recent_matchups['home_total_goals'].sum() + recent_matchups['away_total_goals'].sum()
                ) / len(recent_matchups)
            else:
                avg_goals_per_game = 0
            
            contextual_features.append({
                'match_id': idx,  # Use index as match_id
                'venue_home_advantage': venue_avg_advantage,
                'home_rest_days': home_rest_days,
                'away_rest_days': away_rest_days,
                'rest_days_difference': home_rest_days - away_rest_days,
                'season_progress': season_progress,
                'is_late_season': 1 if season_progress > 0.8 else 0,
                'is_early_season': 1 if season_progress < 0.2 else 0,
                'historical_avg_goals': avg_goals_per_game,
                'venue_total_matches': len(venue_matches),
                'home_team_venue_experience': len(venue_matches[
                    (venue_matches['home_team'] == home_team) | (venue_matches['away_team'] == home_team)
                ]),
                'away_team_venue_experience': len(venue_matches[
                    (venue_matches['home_team'] == away_team) | (venue_matches['away_team'] == away_team)
                ])
            })
        
        return pd.DataFrame(contextual_features)
    
    def create_advanced_features(self):
        """Create advanced features including interactions and momentum indicators."""
        print("Creating advanced features...")
        
        # This will be implemented after basic features are created
        # For now, return empty DataFrame
        return pd.DataFrame()
    
    def create_feature_interactions(self, base_features):
        """Create feature interaction terms."""
        print("Creating feature interactions...")
        
        interaction_features = base_features.copy()
        
        # Create polynomial terms for key numerical features
        key_features = [
            'home_rolling_avg_goals_for_10', 'away_rolling_avg_goals_for_10',
            'home_rolling_avg_goals_against_10', 'away_rolling_avg_goals_against_10',
            'home_avg_disposals', 'away_avg_disposals'
        ]
        
        available_features = [f for f in key_features if f in base_features.columns]
        
        for feature in available_features:
            interaction_features[f'{feature}_squared'] = base_features[feature] ** 2
        
        # Create interaction terms between home and away features
        home_features = [f for f in base_features.columns if f.startswith('home_') and f.endswith('_10')]
        away_features = [f for f in base_features.columns if f.startswith('away_') and f.endswith('_10')]
        
        for home_feat in home_features[:3]:  # Limit to first 3 to avoid too many interactions
            for away_feat in away_features[:3]:
                if home_feat in base_features.columns and away_feat in base_features.columns:
                    interaction_name = f'interaction_{home_feat}_{away_feat}'
                    interaction_features[interaction_name] = base_features[home_feat] * base_features[away_feat]
        
        return interaction_features
    
    def create_momentum_indicators(self, base_features):
        """Create momentum and streak indicators."""
        print("Creating momentum indicators...")
        
        momentum_features = base_features.copy()
        
        # Calculate momentum based on recent performance vs longer-term performance
        if 'home_rolling_avg_goals_for_5' in base_features.columns and 'home_rolling_avg_goals_for_20' in base_features.columns:
            momentum_features['home_goals_momentum'] = (
                base_features['home_rolling_avg_goals_for_5'] - base_features['home_rolling_avg_goals_for_20']
            )
        
        if 'away_rolling_avg_goals_for_5' in base_features.columns and 'away_rolling_avg_goals_for_20' in base_features.columns:
            momentum_features['away_goals_momentum'] = (
                base_features['away_rolling_avg_goals_for_5'] - base_features['away_rolling_avg_goals_for_20']
            )
        
        # Volatility indicators
        if 'home_rolling_avg_goals_for_10' in base_features.columns and 'home_rolling_avg_goals_against_10' in base_features.columns:
            momentum_features['home_volatility'] = (
                base_features['home_rolling_avg_goals_for_10'] + base_features['home_rolling_avg_goals_against_10']
            )
        
        if 'away_rolling_avg_goals_for_10' in base_features.columns and 'away_rolling_avg_goals_against_10' in base_features.columns:
            momentum_features['away_volatility'] = (
                base_features['away_rolling_avg_goals_for_10'] + base_features['away_rolling_avg_goals_against_10']
            )
        
        return momentum_features
    
    def analyze_feature_importance(self, features_df, target_col='margin'):
        """Analyze feature importance using multiple methods."""
        print("Analyzing feature importance...")
        
        # Add margin column if it doesn't exist
        if 'margin' not in features_df.columns:
            features_df['margin'] = features_df['home_total_goals'] - features_df['away_total_goals']
        
        # Prepare data for analysis
        X = features_df.select_dtypes(include=[np.number]).dropna()
        if target_col in X.columns:
            y = X[target_col]
            X = X.drop(columns=[target_col])
        else:
            print(f"Target column {target_col} not found in features")
            return pd.DataFrame()
        
        # Remove columns with too many missing values
        X = X.dropna(axis=1, thresh=len(X) * 0.5)
        
        if len(X.columns) == 0:
            print("No features available for importance analysis")
            return pd.DataFrame()
        
        # Fill remaining missing values
        X = X.fillna(X.mean())
        
        # Random Forest importance
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X, y)
        rf_importance = dict(zip(X.columns, rf.feature_importances_))
        
        # Mutual Information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_importance = dict(zip(X.columns, mi_scores))
        
        # F-statistic
        f_scores, _ = f_regression(X, y)
        f_importance = dict(zip(X.columns, f_scores))
        
        # Combine importance scores
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'rf_importance': [rf_importance[col] for col in X.columns],
            'mi_importance': [mi_importance[col] for col in X.columns],
            'f_importance': [f_importance[col] for col in X.columns]
        })
        
        # Calculate average importance
        importance_df['avg_importance'] = importance_df[['rf_importance', 'mi_importance', 'f_importance']].mean(axis=1)
        importance_df = importance_df.sort_values('avg_importance', ascending=False)
        
        self.feature_importance = importance_df.to_dict('records')
        
        return importance_df
    
    def create_correlation_analysis(self, features_df):
        """Create correlation analysis for features."""
        print("Creating correlation analysis...")
        
        # Select numerical features
        numerical_features = features_df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        self.correlation_matrix = numerical_features.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(20, 16))
        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))
        sns.heatmap(self.correlation_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16)
        plt.tight_layout()
        plt.savefig('feature_correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Find highly correlated features
        high_corr_features = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                corr_value = self.correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.8:
                    high_corr_features.append({
                        'feature1': self.correlation_matrix.columns[i],
                        'feature2': self.correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return high_corr_features
    
    def run_feature_engineering(self):
        """Run the complete feature engineering pipeline."""
        print("Starting Feature Engineering Pipeline...")
        
        # Load data
        self.load_data()
        
        # Create all feature types
        h2h_features = self.create_head_to_head_features()
        player_features = self.create_player_aggregation_features()
        contextual_features = self.create_contextual_features()
        
        # Create team performance features directly for matches
        print("Creating team performance features...")
        team_performance_features = self.create_match_based_team_features()
        
        # Merge all features
        print("Merging features...")
        self.features_df = self.matches_df.copy()
        self.features_df['match_id'] = self.features_df.index  # Add match_id column
        
        # Merge all feature sets
        self.features_df = self.features_df.merge(team_performance_features, left_index=True, right_index=True, how='left')
        self.features_df = self.features_df.merge(h2h_features, on='match_id', how='left')
        self.features_df = self.features_df.merge(player_features, on='match_id', how='left')
        self.features_df = self.features_df.merge(contextual_features, on='match_id', how='left')
        
        # Create advanced features
        self.features_df = self.create_feature_interactions(self.features_df)
        self.features_df = self.create_momentum_indicators(self.features_df)
        
        # Analyze features
        importance_df = self.analyze_feature_importance(self.features_df)
        high_corr_features = self.create_correlation_analysis(self.features_df)
        
        # Save results
        self.features_df.to_csv('engineered_features.csv', index=False)
        importance_df.to_csv('feature_importance.csv', index=False)
        
        # Create feature importance visualization
        plt.figure(figsize=(12, 8))
        top_features = importance_df.head(20)
        plt.barh(range(len(top_features)), top_features['avg_importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Average Importance Score')
        plt.title('Top 20 Feature Importance Scores')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Feature engineering complete!")
        print(f"Total features created: {len(self.features_df.columns)}")
        print(f"Total samples: {len(self.features_df)}")
        print(f"Top 5 most important features:")
        for i, row in importance_df.head(5).iterrows():
            print(f"  {row['feature']}: {row['avg_importance']:.4f}")
        
        return {
            'features_df': self.features_df,
            'importance_df': importance_df,
            'high_corr_features': high_corr_features,
            'feature_count': len(self.features_df.columns),
            'sample_count': len(self.features_df)
        }
    
    def create_match_based_team_features(self):
        """Create team performance features directly for each match."""
        print("Creating match-based team features...")
        
        match_features = []
        
        for idx, match in self.matches_df.iterrows():
            home_team = match['home_team']
            away_team = match['away_team']
            match_date = match['date']
            
            # Get historical data for home team
            home_history = self.matches_df[
                ((self.matches_df['home_team'] == home_team) | (self.matches_df['away_team'] == home_team)) &
                (self.matches_df['date'] < match_date)
            ].sort_values('date').tail(20)  # Last 20 games
            
            # Get historical data for away team
            away_history = self.matches_df[
                ((self.matches_df['home_team'] == away_team) | (self.matches_df['away_team'] == away_team)) &
                (self.matches_df['date'] < match_date)
            ].sort_values('date').tail(20)  # Last 20 games
            
            # Calculate home team features
            home_features = self._calculate_team_history_features(home_history, home_team, 'home')
            
            # Calculate away team features
            away_features = self._calculate_team_history_features(away_history, away_team, 'away')
            
            # Combine features
            match_features.append({
                **home_features,
                **away_features
            })
        
        return pd.DataFrame(match_features)
    
    def _calculate_team_history_features(self, team_history, team_name, team_type):
        """Calculate features from team's historical performance."""
        features = {}
        
        if len(team_history) > 0:
            # Calculate goals for and against for this team
            team_goals_for = []
            team_goals_against = []
            
            for _, game in team_history.iterrows():
                if game['home_team'] == team_name:
                    team_goals_for.append(game['home_total_goals'])
                    team_goals_against.append(game['away_total_goals'])
                else:
                    team_goals_for.append(game['away_total_goals'])
                    team_goals_against.append(game['home_total_goals'])
            
            # Rolling averages
            for window in [5, 10, 20]:
                if len(team_goals_for) >= window:
                    features[f'{team_type}_rolling_avg_goals_for_{window}'] = np.mean(team_goals_for[-window:])
                    features[f'{team_type}_rolling_avg_goals_against_{window}'] = np.mean(team_goals_against[-window:])
                else:
                    features[f'{team_type}_rolling_avg_goals_for_{window}'] = np.mean(team_goals_for) if team_goals_for else 0
                    features[f'{team_type}_rolling_avg_goals_against_{window}'] = np.mean(team_goals_against) if team_goals_against else 0
            
            # Recent form (last 5 games)
            if len(team_goals_for) >= 5:
                recent_goals_for = np.mean(team_goals_for[-5:])
                recent_goals_against = np.mean(team_goals_against[-5:])
                features[f'{team_type}_recent_form'] = recent_goals_for - recent_goals_against
            else:
                features[f'{team_type}_recent_form'] = 0
            
            # Season averages
            season_games = team_history[team_history['year'] == team_history['year'].iloc[-1]]
            if len(season_games) > 0:
                season_goals_for = []
                season_goals_against = []
                for _, game in season_games.iterrows():
                    if game['home_team'] == team_name:
                        season_goals_for.append(game['home_total_goals'])
                        season_goals_against.append(game['away_total_goals'])
                    else:
                        season_goals_for.append(game['away_total_goals'])
                        season_goals_against.append(game['home_total_goals'])
                
                features[f'{team_type}_season_avg_goals_for'] = np.mean(season_goals_for) if season_goals_for else 0
                features[f'{team_type}_season_avg_goals_against'] = np.mean(season_goals_against) if season_goals_against else 0
            else:
                features[f'{team_type}_season_avg_goals_for'] = 0
                features[f'{team_type}_season_avg_goals_against'] = 0
        else:
            # Default values if no history
            for window in [5, 10, 20]:
                features[f'{team_type}_rolling_avg_goals_for_{window}'] = 0
                features[f'{team_type}_rolling_avg_goals_against_{window}'] = 0
            
            features[f'{team_type}_recent_form'] = 0
            features[f'{team_type}_season_avg_goals_for'] = 0
            features[f'{team_type}_season_avg_goals_against'] = 0
        
        return features

def main():
    """Main function to run the feature engineering pipeline."""
    pipeline = FeatureEngineeringPipeline()
    results = pipeline.run_feature_engineering()
    
    print("\nFeature Engineering Pipeline Complete!")
    print(f"Results saved to:")
    print("  - engineered_features.csv")
    print("  - feature_importance.csv")
    print("  - feature_importance.png")
    print("  - feature_correlation_matrix.png")

if __name__ == "__main__":
    main() 