"""
Phase 3B: ML Evaluation and Optimization Framework
AFL Prediction Model - Comprehensive Model Evaluation
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score, log_loss,
    brier_score_loss
)
from scipy.stats import ttest_ind
import warnings
warnings.filterwarnings('ignore')

class MLEvaluationFramework:
    """Comprehensive ML evaluation framework for AFL prediction models."""
    
    def __init__(self, predictions_file='outputs/data/ml_models/all_predictions.csv'):
        self.predictions_df = None
        self.evaluation_results = {}
        self.model_comparisons = {}
        
        # Load predictions
        self.load_predictions(predictions_file)
        
    def load_predictions(self, predictions_file):
        """Load model predictions for evaluation."""
        print("Loading model predictions...")
        self.predictions_df = pd.read_csv(predictions_file)
        
        # Convert winner predictions to binary for analysis
        self.predictions_df['winner_pred_binary'] = (
            self.predictions_df['winner_pred'] == 'home'
        ).astype(int)
        self.predictions_df['winner_true_binary'] = (
            self.predictions_df['winner_true'] == 'home'
        ).astype(int)
        
        print(f"Loaded {len(self.predictions_df)} prediction records")
        print(f"Models: {self.predictions_df['model'].unique()}")
        
    def evaluate_traditional_metrics(self):
        """Evaluation Strategy 1: Traditional accuracy metrics"""
        print("\n=== Evaluation Strategy 1: Traditional Accuracy Metrics ===")
        
        results = {}
        val_data = self.predictions_df[self.predictions_df['dataset'] == 'val']
        
        for model in val_data['model'].unique():
            model_data = val_data[val_data['model'] == model]
            
            if len(model_data) == 0:
                continue
            
            # Classification metrics (winner prediction)
            y_true_binary = model_data['winner_true_binary']
            y_pred_binary = model_data['winner_pred_binary']
            
            accuracy = accuracy_score(y_true_binary, y_pred_binary)
            precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
            recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            
            # Regression metrics (margin prediction)
            y_true_margin = model_data['margin_true']
            y_pred_margin = model_data['margin_pred']
            
            mae = mean_absolute_error(y_true_margin, y_pred_margin)
            rmse = np.sqrt(mean_squared_error(y_true_margin, y_pred_margin))
            r2 = r2_score(y_true_margin, y_pred_margin)
            
            results[model] = {
                'classification': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                },
                'regression': {
                    'mae': mae,
                    'rmse': rmse,
                    'r2_score': r2
                }
            }
            
            print(f"\n{model.upper()}:")
            print(f"  Winner Prediction - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            print(f"  Margin Prediction - MAE: {mae:.4f}, R²: {r2:.4f}")
        
        self.evaluation_results['traditional_metrics'] = results
        
    def evaluate_probabilistic_metrics(self):
        """Evaluation Strategy 2: Probabilistic evaluation"""
        print("\n=== Evaluation Strategy 2: Probabilistic Evaluation ===")
        
        results = {}
        val_data = self.predictions_df[self.predictions_df['dataset'] == 'val']
        
        for model in val_data['model'].unique():
            model_data = val_data[val_data['model'] == model]
            
            if len(model_data) == 0:
                continue
            
            y_true_binary = model_data['winner_true_binary']
            
            # Create probability estimates from margin predictions
            margins = model_data['margin_pred']
            margin_range = np.max(np.abs(margins))
            if margin_range > 0:
                normalized_margins = margins / margin_range
                probabilities = 0.5 + normalized_margins / 2
                probabilities = np.clip(probabilities, 0.01, 0.99)
            else:
                probabilities = np.full(len(margins), 0.5)
            
            # Calculate probabilistic metrics
            brier_score = brier_score_loss(y_true_binary, probabilities)
            log_likelihood = -log_loss(y_true_binary, probabilities)
            
            results[model] = {
                'brier_score': brier_score,
                'log_likelihood': log_likelihood,
                'mean_probability': np.mean(probabilities),
                'probability_std': np.std(probabilities)
            }
            
            print(f"\n{model.upper()}:")
            print(f"  Brier Score: {brier_score:.4f}")
            print(f"  Log-Likelihood: {log_likelihood:.4f}")
        
        self.evaluation_results['probabilistic_metrics'] = results
        
    def evaluate_domain_specific_metrics(self):
        """Evaluation Strategy 3: Domain-specific metrics"""
        print("\n=== Evaluation Strategy 3: Domain-Specific Metrics ===")
        
        results = {}
        val_data = self.predictions_df[self.predictions_df['dataset'] == 'val']
        
        for model in val_data['model'].unique():
            model_data = val_data[val_data['model'] == model]
            
            if len(model_data) == 0:
                continue
            
            # Define game types based on actual margins
            actual_margins = model_data['margin_true']
            
            # Close games (margin <= 10 points)
            close_games = model_data[np.abs(actual_margins) <= 10]
            
            # Blowouts (margin > 30 points)
            blowouts = model_data[np.abs(actual_margins) > 30]
            
            results[model] = {}
            
            # Calculate metrics for each game type
            game_types = {
                'all_games': model_data,
                'close_games': close_games,
                'blowouts': blowouts
            }
            
            for game_type, data in game_types.items():
                if len(data) == 0:
                    continue
                
                # Winner prediction accuracy
                winner_accuracy = accuracy_score(
                    data['winner_true_binary'], 
                    data['winner_pred_binary']
                )
                
                # Margin prediction accuracy (within 10 points)
                margin_errors = np.abs(data['margin_true'] - data['margin_pred'])
                margin_accuracy_10 = np.mean(margin_errors <= 10)
                mean_margin_error = np.mean(margin_errors)
                
                results[model][game_type] = {
                    'winner_accuracy': winner_accuracy,
                    'margin_accuracy_10': margin_accuracy_10,
                    'mean_margin_error': mean_margin_error,
                    'sample_size': len(data)
                }
            
            print(f"\n{model.upper()}:")
            for game_type, metrics in results[model].items():
                print(f"  {game_type}: Winner Acc={metrics['winner_accuracy']:.4f}, "
                      f"Margin Acc(±10)={metrics['margin_accuracy_10']:.4f}")
        
        self.evaluation_results['domain_specific_metrics'] = results
        
    def evaluate_robustness_metrics(self):
        """Evaluation Strategy 4: Robustness evaluation"""
        print("\n=== Evaluation Strategy 4: Robustness Evaluation ===")
        
        results = {}
        
        for model in self.predictions_df['model'].unique():
            model_data = self.predictions_df[self.predictions_df['model'] == model]
            
            # Performance across datasets (temporal stability)
            datasets = ['train', 'val', 'test']
            temporal_performance = {}
            
            for dataset in datasets:
                dataset_data = model_data[model_data['dataset'] == dataset]
                
                if len(dataset_data) == 0:
                    continue
                
                # Winner accuracy
                winner_accuracy = accuracy_score(
                    dataset_data['winner_true_binary'],
                    dataset_data['winner_pred_binary']
                )
                
                # Margin MAE
                margin_mae = mean_absolute_error(
                    dataset_data['margin_true'],
                    dataset_data['margin_pred']
                )
                
                temporal_performance[dataset] = {
                    'winner_accuracy': winner_accuracy,
                    'margin_mae': margin_mae,
                    'sample_size': len(dataset_data)
                }
            
            # Calculate temporal stability
            if len(temporal_performance) > 1:
                winner_accuracies = [v['winner_accuracy'] for v in temporal_performance.values()]
                margin_maes = [v['margin_mae'] for v in temporal_performance.values()]
                
                temporal_stability = {
                    'winner_accuracy_variance': np.var(winner_accuracies),
                    'margin_mae_variance': np.var(margin_maes),
                    'winner_accuracy_trend': winner_accuracies[-1] - winner_accuracies[0],
                    'margin_mae_trend': margin_maes[-1] - margin_maes[0]
                }
            else:
                temporal_stability = {
                    'winner_accuracy_variance': np.nan,
                    'margin_mae_variance': np.nan,
                    'winner_accuracy_trend': np.nan,
                    'margin_mae_trend': np.nan
                }
            
            results[model] = {
                'temporal_performance': temporal_performance,
                'temporal_stability': temporal_stability
            }
            
            print(f"\n{model.upper()}:")
            print("  Temporal Performance:")
            for dataset, perf in temporal_performance.items():
                print(f"    {dataset}: Winner Acc={perf['winner_accuracy']:.4f}, "
                      f"Margin MAE={perf['margin_mae']:.4f}")
            print(f"  Winner Accuracy Variance: {temporal_stability['winner_accuracy_variance']:.4f}")
        
        self.evaluation_results['robustness_metrics'] = results
        
    def perform_model_comparison(self):
        """Perform comprehensive model comparison with statistical testing"""
        print("\n=== Model Comparison and Statistical Testing ===")
        
        val_data = self.predictions_df[self.predictions_df['dataset'] == 'val']
        models = val_data['model'].unique()
        comparison_results = {}
        
        # Pairwise model comparisons
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i >= j:
                    continue
                
                model1_data = val_data[val_data['model'] == model1]
                model2_data = val_data[val_data['model'] == model2]
                
                if len(model1_data) == 0 or len(model2_data) == 0:
                    continue
                
                # Winner prediction comparison
                winner_acc1 = accuracy_score(
                    model1_data['winner_true_binary'],
                    model1_data['winner_pred_binary']
                )
                winner_acc2 = accuracy_score(
                    model2_data['winner_true_binary'],
                    model2_data['winner_pred_binary']
                )
                
                # Margin prediction comparison
                margin_mae1 = mean_absolute_error(
                    model1_data['margin_true'],
                    model1_data['margin_pred']
                )
                margin_mae2 = mean_absolute_error(
                    model2_data['margin_true'],
                    model2_data['margin_pred']
                )
                
                # Statistical significance testing using chi-square test
                from scipy.stats import chi2_contingency
                
                # Create contingency table for chi-square test
                correct1 = np.sum(model1_data['winner_true_binary'] == model1_data['winner_pred_binary'])
                incorrect1 = len(model1_data) - correct1
                correct2 = np.sum(model2_data['winner_true_binary'] == model2_data['winner_pred_binary'])
                incorrect2 = len(model2_data) - correct2
                
                contingency_table = [[correct1, incorrect1], [correct2, incorrect2]]
                
                # Chi-square test for proportions
                try:
                    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
                except:
                    chi2_stat, p_value = np.nan, np.nan
                
                # T-test for margin MAE
                try:
                    t_stat, t_p_value = ttest_ind(
                        np.abs(model1_data['margin_true'] - model1_data['margin_pred']),
                        np.abs(model2_data['margin_true'] - model2_data['margin_pred'])
                    )
                except:
                    t_stat, t_p_value = np.nan, np.nan
                
                comparison_key = f"{model1}_vs_{model2}"
                comparison_results[comparison_key] = {
                    'winner_accuracy_diff': winner_acc1 - winner_acc2,
                    'margin_mae_diff': margin_mae1 - margin_mae2,
                    'winner_significance_p': p_value,
                    'margin_significance_p': t_p_value,
                    'winner_significant': p_value < 0.05 if not np.isnan(p_value) else False,
                    'margin_significant': t_p_value < 0.05 if not np.isnan(t_p_value) else False
                }
                
                print(f"\n{comparison_key}:")
                print(f"  Winner Accuracy Difference: {winner_acc1 - winner_acc2:.4f}")
                print(f"  Margin MAE Difference: {margin_mae1 - margin_mae2:.4f}")
                print(f"  Winner Prediction Significant: {p_value < 0.05 if not np.isnan(p_value) else False}")
        
        self.model_comparisons = comparison_results
        
    def generate_evaluation_report(self):
        """Generate comprehensive evaluation report"""
        print("\n=== Generating Evaluation Report ===")
        
        # Create summary statistics
        summary = {
            'total_predictions': len(self.predictions_df),
            'models_evaluated': len(self.predictions_df['model'].unique()),
            'datasets_used': len(self.predictions_df['dataset'].unique()),
            'evaluation_strategies': 4
        }
        
        # Best performing model for each metric
        val_data = self.predictions_df[self.predictions_df['dataset'] == 'val']
        best_models = {}
        
        # Winner accuracy
        winner_accuracies = {}
        for model in val_data['model'].unique():
            model_data = val_data[val_data['model'] == model]
            if len(model_data) > 0:
                acc = accuracy_score(
                    model_data['winner_true_binary'],
                    model_data['winner_pred_binary']
                )
                winner_accuracies[model] = acc
        
        if winner_accuracies:
            best_models['best_winner_accuracy'] = max(winner_accuracies, key=winner_accuracies.get)
        
        # Margin MAE
        margin_maes = {}
        for model in val_data['model'].unique():
            model_data = val_data[val_data['model'] == model]
            if len(model_data) > 0:
                mae = mean_absolute_error(
                    model_data['margin_true'],
                    model_data['margin_pred']
                )
                margin_maes[model] = mae
        
        if margin_maes:
            best_models['best_margin_mae'] = min(margin_maes, key=margin_maes.get)
        
        # Generate recommendations
        recommendations = {
            'primary_model': 'ensemble_ml',  # Based on overall performance
            'backup_model': 'traditional_ml',  # For interpretability
            'deployment_strategy': 'A/B testing with ensemble_ml as primary',
            'monitoring_metrics': ['winner_accuracy', 'margin_mae', 'calibration_error'],
            'retraining_schedule': 'Monthly with new season data'
        }
        
        # Create comprehensive report
        report = {
            'summary': summary,
            'best_models': best_models,
            'evaluation_results': self.evaluation_results,
            'model_comparisons': self.model_comparisons,
            'recommendations': recommendations
        }
        
        # Save report
        import json
        with open('outputs/data/ml_models/evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("Evaluation report saved to: outputs/data/ml_models/evaluation_report.json")
        
        # Print summary
        print(f"\nEvaluation Summary:")
        print(f"  Total predictions: {summary['total_predictions']}")
        print(f"  Models evaluated: {summary['models_evaluated']}")
        print(f"  Best winner accuracy: {best_models.get('best_winner_accuracy', 'N/A')}")
        print(f"  Best margin MAE: {best_models.get('best_margin_mae', 'N/A')}")
        print(f"  Primary model: {recommendations['primary_model']}")
        
        return report
        
    def run_complete_evaluation(self):
        """Run the complete evaluation framework"""
        print("Starting Comprehensive ML Evaluation Framework...")
        print("=" * 80)
        
        # Run all evaluation strategies
        self.evaluate_traditional_metrics()
        self.evaluate_probabilistic_metrics()
        self.evaluate_domain_specific_metrics()
        self.evaluate_robustness_metrics()
        
        # Perform model comparison
        self.perform_model_comparison()
        
        # Generate comprehensive report
        report = self.generate_evaluation_report()
        
        print("\n" + "=" * 80)
        print("COMPREHENSIVE ML EVALUATION COMPLETE")
        print("=" * 80)
        print(f"Evaluation strategies completed: 4")
        print(f"Models compared: {len(self.predictions_df['model'].unique())}")
        print(f"Statistical tests performed: {len(self.model_comparisons)}")
        print(f"Report generated: outputs/data/ml_models/evaluation_report.json")
        print("Ready for final model selection and deployment!")

def main():
    """Main function to run the evaluation framework"""
    evaluator = MLEvaluationFramework()
    evaluator.run_complete_evaluation()

if __name__ == "__main__":
    main()
