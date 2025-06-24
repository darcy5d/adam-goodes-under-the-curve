"""
Phase 3A: ML Model Architecture Design
AFL Prediction Model - Machine Learning Architecture Analysis

This script analyzes and compares different ML approaches for AFL prediction,
evaluating predictive accuracy potential, interpretability, computational requirements,
and data efficiency for each approach.
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
import warnings
warnings.filterwarnings('ignore')

class MLArchitectureAnalysis:
    """
    Comprehensive analysis of ML approaches for AFL prediction modeling.
    """
    
    def __init__(self, db_path="afl_data/afl_database.db"):
        self.db_path = db_path
        self.matches_df = None
        self.features_df = None
        self.ml_approaches = {}
        self.approach_evaluation = {}
        
    def load_data(self):
        """Load match data and engineered features."""
        print("Loading data for ML architecture analysis...")
        
        # Load engineered features
        try:
            self.features_df = pd.read_csv('outputs/data/feature_engineering/engineered_features.csv')
            print(f"Loaded {len(self.features_df)} feature samples")
        except FileNotFoundError:
            print("Engineered features not found. Loading from database...")
            self._load_from_database()
        
        # Load match data for target variables
        conn = sqlite3.connect(self.db_path)
        self.matches_df = pd.read_sql_query("""
            SELECT year, date, home_team, away_team, 
                   home_total_goals, away_total_goals
            FROM matches 
            WHERE year >= 1991 
            ORDER BY year, date
        """, conn)
        conn.close()
        
        # Calculate margin and winning team
        self.matches_df['margin'] = self.matches_df['home_total_goals'] - self.matches_df['away_total_goals']
        self.matches_df['winning_team'] = self.matches_df.apply(
            lambda row: row['home_team'] if row['margin'] > 0 else row['away_team'] if row['margin'] < 0 else 'Draw', 
            axis=1
        )
        
        print(f"Loaded {len(self.matches_df)} match records")
        
    def _load_from_database(self):
        """Load and create basic features from database if engineered features not available."""
        conn = sqlite3.connect(self.db_path)
        
        # Create basic features from match data
        matches = pd.read_sql_query("""
            SELECT * FROM matches 
            WHERE year >= 1991 
            ORDER BY year, date
        """, conn)
        
        # Create simple features
        self.features_df = matches.copy()
        self.features_df['total_goals'] = self.features_df['home_total_goals'] + self.features_df['away_total_goals']
        self.features_df['goal_difference'] = self.features_df['home_total_goals'] - self.features_df['away_total_goals']
        
        conn.close()
        
    def analyze_approach_1_traditional_ml(self):
        """
        Approach 1: Traditional ML Methods
        - Linear models, Random Forest, XGBoost
        """
        print("\n=== Approach 1: Traditional ML Methods ===")
        
        approach = {
            'name': 'Traditional ML Methods',
            'description': 'Linear models, tree-based methods, and gradient boosting',
            'methods': ['Linear Regression', 'Ridge/Lasso', 'Random Forest', 'XGBoost', 'Gradient Boosting'],
            'complexity': 'Low to Medium',
            'interpretability': 'High',
            'data_requirements': 'Low',
            'accuracy': 'Medium to High'
        }
        
        # Specific methods for AFL prediction
        afl_methods = {
            'winner_prediction': ['Random Forest', 'XGBoost', 'Logistic Regression'],
            'margin_prediction': ['Linear Regression', 'Ridge Regression', 'Gradient Boosting'],
            'multi_task': ['Random Forest', 'XGBoost with custom loss']
        }
        
        approach['afl_methods'] = afl_methods
        approach['advantages'] = [
            'Highly interpretable with SHAP explanations',
            'Robust to outliers and noise',
            'Handle non-linear relationships well',
            'Feature importance analysis available',
            'Fast training and inference',
            'Good performance with limited data'
        ]
        
        approach['disadvantages'] = [
            'May struggle with complex temporal patterns',
            'Limited ability to capture sequential dependencies',
            'Requires feature engineering for complex relationships',
            'May overfit with too many features',
            'Linear models assume linear relationships'
        ]
        
        approach['computational_requirements'] = {
            'training_time': 'Low to Medium',
            'memory_usage': 'Low',
            'inference_speed': 'Very Fast',
            'scalability': 'High'
        }
        
        self.ml_approaches['traditional_ml'] = approach
        return approach
    
    def analyze_approach_2_deep_learning(self):
        """
        Approach 2: Deep Learning Approaches
        - MLPs, CNNs, RNNs/LSTMs
        """
        print("\n=== Approach 2: Deep Learning Approaches ===")
        
        approach = {
            'name': 'Deep Learning Approaches',
            'description': 'Neural networks, CNNs, RNNs/LSTMs for complex pattern recognition',
            'methods': ['MLP', 'CNN', 'RNN', 'LSTM', 'GRU', '1D-CNN'],
            'complexity': 'High',
            'interpretability': 'Low to Medium',
            'data_requirements': 'High',
            'accuracy': 'High to Very High'
        }
        
        # Specific architectures for AFL prediction
        afl_architectures = {
            'temporal_modeling': ['LSTM', 'GRU', '1D-CNN'],
            'feature_interaction': ['MLP with attention', 'Deep & Wide networks'],
            'multi_task': ['Shared encoder with task-specific heads']
        }
        
        approach['afl_architectures'] = afl_architectures
        approach['advantages'] = [
            'Can capture complex non-linear patterns',
            'Excellent for temporal sequence modeling',
            'Automatic feature learning and interaction',
            'Can handle high-dimensional data',
            'State-of-the-art performance potential',
            'Flexible architecture design'
        ]
        
        approach['disadvantages'] = [
            'Requires large amounts of data',
            'Computationally intensive',
            'Black-box nature limits interpretability',
            'Prone to overfitting with limited data',
            'Complex hyperparameter tuning',
            'Long training times'
        ]
        
        approach['computational_requirements'] = {
            'training_time': 'High',
            'memory_usage': 'High',
            'inference_speed': 'Medium',
            'scalability': 'Medium'
        }
        
        self.ml_approaches['deep_learning'] = approach
        return approach
    
    def analyze_approach_3_specialized_architectures(self):
        """
        Approach 3: Specialized Architectures
        - Attention mechanisms, Graph Neural Networks
        """
        print("\n=== Approach 3: Specialized Architectures ===")
        
        approach = {
            'name': 'Specialized Architectures',
            'description': 'Attention mechanisms, Graph Neural Networks, Transformer-based models',
            'methods': ['Attention Mechanisms', 'Graph Neural Networks', 'Transformers', 'Multi-head Attention'],
            'complexity': 'Very High',
            'interpretability': 'Medium',
            'data_requirements': 'Very High',
            'accuracy': 'Very High'
        }
        
        # Specialized methods for AFL prediction
        afl_specialized = {
            'team_interactions': ['Graph Neural Networks for team relationships'],
            'temporal_attention': ['Attention mechanisms for time series'],
            'multi-modal': ['Combining match stats, player stats, and contextual data']
        }
        
        approach['afl_specialized'] = afl_specialized
        approach['advantages'] = [
            'Can model complex relationships between teams/players',
            'Attention mechanisms provide interpretability',
            'Excellent for capturing long-range dependencies',
            'Can handle multi-modal data effectively',
            'State-of-the-art performance for complex tasks',
            'Flexible architecture for domain-specific needs'
        ]
        
        approach['disadvantages'] = [
            'Very computationally intensive',
            'Requires extensive data for training',
            'Complex implementation and debugging',
            'May be overkill for simpler prediction tasks',
            'Limited interpretability despite attention',
            'Long development and training cycles'
        ]
        
        approach['computational_requirements'] = {
            'training_time': 'Very High',
            'memory_usage': 'Very High',
            'inference_speed': 'Low to Medium',
            'scalability': 'Low to Medium'
        }
        
        self.ml_approaches['specialized_architectures'] = approach
        return approach
    
    def analyze_approach_4_ensemble_meta_learning(self):
        """
        Approach 4: Ensemble and Meta-learning Methods
        - Stacking, blending, meta-learning
        """
        print("\n=== Approach 4: Ensemble and Meta-learning Methods ===")
        
        approach = {
            'name': 'Ensemble and Meta-learning Methods',
            'description': 'Stacking, blending, and meta-learning for improved performance',
            'methods': ['Stacking', 'Blending', 'Voting', 'Meta-learning', 'Neural Network Ensembles'],
            'complexity': 'Medium to High',
            'interpretability': 'Medium',
            'data_requirements': 'Medium',
            'accuracy': 'High to Very High'
        }
        
        # Ensemble methods for AFL prediction
        afl_ensemble = {
            'base_models': ['Random Forest', 'XGBoost', 'Linear Regression', 'Neural Network'],
            'meta_learner': ['Linear Regression', 'Ridge Regression', 'Neural Network'],
            'ensemble_strategy': ['Stacking', 'Blending', 'Weighted Average']
        }
        
        approach['afl_ensemble'] = afl_ensemble
        approach['advantages'] = [
            'Combines strengths of multiple models',
            'Reduces overfitting and improves generalization',
            'Robust performance across different scenarios',
            'Can handle different types of relationships',
            'Good balance of accuracy and interpretability',
            'Flexible combination strategies'
        ]
        
        approach['disadvantages'] = [
            'Increased computational complexity',
            'More complex to implement and maintain',
            'May be harder to interpret than single models',
            'Requires careful model selection and combination',
            'Potential for overfitting if not properly validated',
            'Longer inference times'
        ]
        
        approach['computational_requirements'] = {
            'training_time': 'Medium to High',
            'memory_usage': 'Medium',
            'inference_speed': 'Medium',
            'scalability': 'Medium'
        }
        
        self.ml_approaches['ensemble_meta_learning'] = approach
        return approach
    
    def evaluate_approaches(self):
        """Evaluate all approaches across multiple dimensions."""
        print("\n=== ML Approach Evaluation ===")
        
        evaluation_criteria = {
            'predictive_accuracy': {'weight': 0.35, 'scores': {}},
            'interpretability': {'weight': 0.25, 'scores': {}},
            'computational_efficiency': {'weight': 0.25, 'scores': {}},
            'data_efficiency': {'weight': 0.15, 'scores': {}}
        }
        
        # Score each approach (1-5 scale, 5 being best)
        scores = {
            'traditional_ml': {
                'predictive_accuracy': 4,
                'interpretability': 5,
                'computational_efficiency': 5,
                'data_efficiency': 4
            },
            'deep_learning': {
                'predictive_accuracy': 5,
                'interpretability': 2,
                'computational_efficiency': 2,
                'data_efficiency': 2
            },
            'specialized_architectures': {
                'predictive_accuracy': 5,
                'interpretability': 3,
                'computational_efficiency': 1,
                'data_efficiency': 1
            },
            'ensemble_meta_learning': {
                'predictive_accuracy': 5,
                'interpretability': 3,
                'computational_efficiency': 3,
                'data_efficiency': 3
            }
        }
        
        # Calculate weighted scores
        approach_scores = {}
        for approach_name, approach_scores_dict in scores.items():
            weighted_score = sum(
                approach_scores_dict[criterion] * evaluation_criteria[criterion]['weight']
                for criterion in evaluation_criteria.keys()
            )
            approach_scores[approach_name] = {
                'individual_scores': approach_scores_dict,
                'weighted_score': weighted_score
            }
        
        self.approach_evaluation = {
            'criteria': evaluation_criteria,
            'scores': approach_scores
        }
        
        return approach_scores
    
    def create_approach_comparison_visualization(self):
        """Create visualization comparing all ML approaches."""
        print("\nCreating ML approach comparison visualization...")
        
        approaches = list(self.approach_evaluation['scores'].keys())
        criteria = list(self.approach_evaluation['criteria'].keys())
        
        # Create radar chart data
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(criteria), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, approach in enumerate(approaches):
            scores = [self.approach_evaluation['scores'][approach]['individual_scores'][criterion] 
                     for criterion in criteria]
            scores += scores[:1]  # Complete the circle
            
            ax.plot(angles, scores, 'o-', linewidth=2, label=approach.replace('_', ' ').title(), 
                   color=colors[i])
            ax.fill(angles, scores, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([criterion.replace('_', ' ').title() for criterion in criteria])
        ax.set_ylim(0, 5)
        ax.set_title('ML Approach Comparison for AFL Prediction', size=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('outputs/visualizations/ml_architecture/ml_approach_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("ML approach comparison visualization saved")
    
    def generate_recommendations(self):
        """Generate recommendations based on approach analysis."""
        print("\n=== ML Architecture Recommendations ===")
        
        # Sort approaches by weighted score
        sorted_approaches = sorted(
            self.approach_evaluation['scores'].items(),
            key=lambda x: x[1]['weighted_score'],
            reverse=True
        )
        
        recommendations = {
            'primary_approach': sorted_approaches[0][0],
            'secondary_approaches': [s[0] for s in sorted_approaches[1:]],
            'implementation_priority': [],
            'hybrid_approach': {}
        }
        
        # Implementation priority based on complexity and impact
        implementation_order = [
            ('traditional_ml', 'Start with traditional ML for baseline performance and interpretability'),
            ('ensemble_meta_learning', 'Add ensemble methods for improved performance'),
            ('deep_learning', 'Consider deep learning for complex temporal patterns'),
            ('specialized_architectures', 'Explore specialized architectures for advanced modeling')
        ]
        
        recommendations['implementation_priority'] = implementation_order
        
        # Hybrid approach recommendations
        recommendations['hybrid_approach'] = {
            'phase_1': ['traditional_ml'],
            'phase_2': ['ensemble_meta_learning'],
            'phase_3': ['deep_learning'],
            'rationale': 'Start simple, add complexity incrementally based on performance needs'
        }
        
        return recommendations
    
    def run_analysis(self):
        """Run complete ML architecture analysis."""
        print("Starting ML Architecture Analysis...")
        
        # Load data
        self.load_data()
        
        # Analyze each approach
        self.analyze_approach_1_traditional_ml()
        self.analyze_approach_2_deep_learning()
        self.analyze_approach_3_specialized_architectures()
        self.analyze_approach_4_ensemble_meta_learning()
        
        # Evaluate approaches
        approach_scores = self.evaluate_approaches()
        
        # Create visualization
        self.create_approach_comparison_visualization()
        
        # Generate recommendations
        recommendations = self.generate_recommendations()
        
        # Print summary
        print("\n" + "="*60)
        print("ML ARCHITECTURE ANALYSIS SUMMARY")
        print("="*60)
        
        print(f"\nTop Approach: {recommendations['primary_approach'].replace('_', ' ').title()}")
        print(f"Weighted Score: {approach_scores[recommendations['primary_approach']]['weighted_score']:.2f}")
        
        print("\nImplementation Priority:")
        for approach, rationale in recommendations['implementation_priority']:
            score = approach_scores[approach]['weighted_score']
            print(f"  {approach.replace('_', ' ').title()}: {score:.2f} - {rationale}")
        
        return {
            'approaches': self.ml_approaches,
            'evaluation': self.approach_evaluation,
            'recommendations': recommendations
        }

def main():
    """Main function to run the ML architecture analysis."""
    analyzer = MLArchitectureAnalysis()
    results = analyzer.run_analysis()
    
    # Save results
    import json
    with open('outputs/data/ml_architecture/ml_architecture_analysis_results.json', 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_results = {}
        for key, value in results.items():
            if key == 'approaches':
                json_results[key] = value
            elif key == 'evaluation':
                json_results[key] = {
                    'criteria': value['criteria'],
                    'scores': {
                        k: {
                            'individual_scores': dict(v['individual_scores']),
                            'weighted_score': float(v['weighted_score'])
                        } for k, v in value['scores'].items()
                    }
                }
            elif key == 'recommendations':
                json_results[key] = value
        
        json.dump(json_results, f, indent=2)
    
    print("\nAnalysis complete! Results saved to 'outputs/data/ml_architecture/ml_architecture_analysis_results.json'")

if __name__ == "__main__":
    main() 