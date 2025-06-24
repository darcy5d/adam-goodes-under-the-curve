"""
ML Training Pipeline for AFL Prediction Model
Phase 3A Implementation - Model Training

This script implements the 3 selected models from Phase 3A:
1. Traditional ML (Random Forest/XGBoost)
2. Ensemble/Meta-learning (Stacking)
3. Deep Learning (MLP/LSTM)

Generates predictions for Phase 3B evaluation.
"""

import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
import xgboost as xgb
from sklearn.ensemble import StackingClassifier, StackingRegressor
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

class MLTrainingPipeline:
    """
    Comprehensive ML training pipeline for AFL prediction models.
    """
    
    def __init__(self):
        self.features_df = None
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.models = {}
        self.predictions = {}
        self.scalers = {}
        
    def load_and_prepare_data(self):
        """Load engineered features and prepare data splits."""
        print("Loading and preparing data...")
        
        # Load engineered features
        self.features_df = pd.read_csv('outputs/data/feature_engineering/engineered_features.csv')
        
        # Create time series splits
        self.train_data = self.features_df[self.features_df['year'] <= 2020].copy()
        self.val_data = self.features_df[(self.features_df['year'] >= 2021) & 
                                        (self.features_df['year'] <= 2023)].copy()
        self.test_data = self.features_df[self.features_df['year'] >= 2024].copy()
        
        print(f"Data splits:")
        print(f"  Training: {len(self.train_data)} samples (1991-2020)")
        print(f"  Validation: {len(self.val_data)} samples (2021-2023)")
        print(f"  Test: {len(self.test_data)} samples (2024-2025)")
        
        # Prepare features and targets
        self._prepare_features_and_targets()
        
    def _prepare_features_and_targets(self):
        """Prepare features and targets for all datasets."""
        # Feature columns (exclude metadata, targets, and data leakage features)
        exclude_cols = ['year', 'date', 'home_team', 'away_team', 'margin', 'round', 'venue', 'attendance', 'match_id',
                       'home_total_goals', 'away_total_goals', 'home_total_behinds', 'away_total_behinds']
        feature_cols = [col for col in self.features_df.columns if col not in exclude_cols]
        
        # Prepare each dataset
        for dataset_name, dataset in [('train', self.train_data), 
                                     ('val', self.val_data), 
                                     ('test', self.test_data)]:
            
            # Create winner column from margin for this dataset
            dataset['winner'] = dataset['margin'].apply(
                lambda x: 'home' if x > 0 else 'away' if x < 0 else 'draw'
            )
            
            # Features
            X = dataset[feature_cols].fillna(0)
            
            # Targets
            y_winner = dataset['winner'].fillna('draw')
            y_margin = dataset['margin'].fillna(0)
            
            # Store prepared data
            setattr(self, f'X_{dataset_name}', X)
            setattr(self, f'y_winner_{dataset_name}', y_winner)
            setattr(self, f'y_margin_{dataset_name}', y_margin)
            
            # Fit scaler on training data only
            if dataset_name == 'train':
                self.scalers['winner'] = LabelEncoder()
                self.scalers['winner'].fit(y_winner)
                
                self.scalers['margin'] = StandardScaler()
                self.scalers['margin'].fit(y_margin.values.reshape(-1, 1))
        
        print(f"Prepared {len(feature_cols)} features for modeling")
        
    def implement_model_1_traditional_ml(self):
        """Implement Model 1: Traditional ML (Random Forest/XGBoost)."""
        print("\n=== Model 1: Traditional ML (Random Forest/XGBoost) ===")
        
        # Prepare data
        X_train = self.X_train
        y_winner_train = self.scalers['winner'].transform(self.y_winner_train)
        y_margin_train = self.scalers['margin'].transform(self.y_margin_train.values.reshape(-1, 1)).flatten()
        
        # Winner prediction (classification)
        print("Training winner prediction model...")
        winner_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        winner_model.fit(X_train, y_winner_train)
        
        # Margin prediction (regression)
        print("Training margin prediction model...")
        margin_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        margin_model.fit(X_train, y_margin_train)
        
        # Store models
        self.models['traditional_ml'] = {
            'winner_model': winner_model,
            'margin_model': margin_model,
            'type': 'Random Forest'
        }
        
        # Generate predictions
        self._generate_predictions('traditional_ml', winner_model, margin_model)
        
    def implement_model_2_ensemble_ml(self):
        """Implement Model 2: Ensemble/Meta-learning (Stacking)."""
        print("\n=== Model 2: Ensemble ML (Stacking) ===")
        
        # Prepare data
        X_train = self.X_train
        y_winner_train = self.scalers['winner'].transform(self.y_winner_train)
        y_margin_train = self.scalers['margin'].transform(self.y_margin_train.values.reshape(-1, 1)).flatten()
        
        # Base models for stacking
        base_models_winner = [
            ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
            ('xgb', xgb.XGBClassifier(n_estimators=50, random_state=42)),
            ('lr', LogisticRegression(random_state=42))
        ]
        
        base_models_margin = [
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
            ('xgb', xgb.XGBRegressor(n_estimators=50, random_state=42)),
            ('lr', LinearRegression())
        ]
        
        # Winner prediction (stacking classifier)
        print("Training stacking winner prediction model...")
        winner_model = StackingClassifier(
            estimators=base_models_winner,
            final_estimator=LogisticRegression(),
            cv=5,
            n_jobs=-1
        )
        winner_model.fit(X_train, y_winner_train)
        
        # Margin prediction (stacking regressor)
        print("Training stacking margin prediction model...")
        margin_model = StackingRegressor(
            estimators=base_models_margin,
            final_estimator=LinearRegression(),
            cv=5,
            n_jobs=-1
        )
        margin_model.fit(X_train, y_margin_train)
        
        # Store models
        self.models['ensemble_ml'] = {
            'winner_model': winner_model,
            'margin_model': margin_model,
            'type': 'Stacking Ensemble'
        }
        
        # Generate predictions
        self._generate_predictions('ensemble_ml', winner_model, margin_model)
        
    def implement_model_3_deep_learning(self):
        """Implement Model 3: Deep Learning (MLP/LSTM)."""
        print("\n=== Model 3: Deep Learning (MLP/LSTM) ===")
        
        # For now, use a simple MLP with sklearn
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        
        # Prepare data
        X_train = self.X_train
        y_winner_train = self.scalers['winner'].transform(self.y_winner_train)
        y_margin_train = self.scalers['margin'].transform(self.y_margin_train.values.reshape(-1, 1)).flatten()
        
        # Scale features for neural network
        feature_scaler = StandardScaler()
        X_train_scaled = feature_scaler.fit_transform(X_train)
        
        # Winner prediction (classification)
        print("Training neural network winner prediction model...")
        winner_model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=100,
            random_state=42
        )
        winner_model.fit(X_train_scaled, y_winner_train)
        
        # Margin prediction (regression)
        print("Training neural network margin prediction model...")
        margin_model = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            max_iter=100,
            random_state=42
        )
        margin_model.fit(X_train_scaled, y_margin_train)
        
        # Store models and scaler
        self.models['deep_learning'] = {
            'winner_model': winner_model,
            'margin_model': margin_model,
            'feature_scaler': feature_scaler,
            'type': 'Neural Network'
        }
        
        # Generate predictions
        self._generate_predictions('deep_learning', winner_model, margin_model, feature_scaler)
        
    def _generate_predictions(self, model_name, winner_model, margin_model, feature_scaler=None):
        """Generate predictions for all datasets using the trained model."""
        print(f"Generating predictions for {model_name}...")
        
        predictions = {}
        
        for dataset_name in ['train', 'val', 'test']:
            X = getattr(self, f'X_{dataset_name}')
            
            # Scale features if needed (for neural network)
            if feature_scaler is not None:
                X_scaled = feature_scaler.transform(X)
            else:
                X_scaled = X
            
            # Winner predictions
            winner_pred = winner_model.predict(X_scaled)
            
            # Margin predictions
            margin_pred = margin_model.predict(X_scaled)
            
            # Inverse transform predictions
            winner_pred_original = self.scalers['winner'].inverse_transform(winner_pred)
            margin_pred_original = self.scalers['margin'].inverse_transform(margin_pred.reshape(-1, 1)).flatten()
            
            # Store predictions
            predictions[dataset_name] = {
                'winner_pred': winner_pred_original,
                'margin_pred': margin_pred_original,
                'winner_true': getattr(self, f'y_winner_{dataset_name}'),
                'margin_true': getattr(self, f'y_margin_{dataset_name}')
            }
        
        self.predictions[model_name] = predictions
        
    def save_predictions(self):
        """Save all predictions for evaluation."""
        print("\nSaving predictions for evaluation...")
        
        # Combine all predictions
        all_predictions = []
        
        for model_name, model_predictions in self.predictions.items():
            for dataset_name, predictions in model_predictions.items():
                n_samples = len(predictions['winner_pred'])
                
                for i in range(n_samples):
                    pred_record = {
                        'model': model_name,
                        'dataset': dataset_name,
                        'winner_pred': predictions['winner_pred'][i],
                        'margin_pred': predictions['margin_pred'][i],
                        'winner_true': predictions['winner_true'].iloc[i] if hasattr(predictions['winner_true'], 'iloc') else predictions['winner_true'][i],
                        'margin_true': predictions['margin_true'].iloc[i] if hasattr(predictions['margin_true'], 'iloc') else predictions['margin_true'][i]
                    }
                    all_predictions.append(pred_record)
        
        # Save predictions
        predictions_df = pd.DataFrame(all_predictions)
        predictions_df.to_csv('outputs/data/ml_models/all_predictions.csv', index=False)
        
        print(f"Saved predictions for {len(self.predictions)} models")
        print(f"Total prediction records: {len(predictions_df)}")
        
    def run_training_pipeline(self):
        """Run the complete training pipeline."""
        print("Starting ML Training Pipeline...")
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Implement each model
        self.implement_model_1_traditional_ml()
        self.implement_model_2_ensemble_ml()
        self.implement_model_3_deep_learning()
        
        # Save predictions
        self.save_predictions()
        
        print("\n" + "="*60)
        print("ML TRAINING PIPELINE COMPLETE")
        print("="*60)
        print(f"Models trained: {len(self.models)}")
        print(f"Predictions saved: outputs/data/ml_models/all_predictions.csv")
        print("Ready for Phase 3B evaluation!")

def main():
    """Main function to run the training pipeline."""
    pipeline = MLTrainingPipeline()
    pipeline.run_training_pipeline()

if __name__ == "__main__":
    main() 