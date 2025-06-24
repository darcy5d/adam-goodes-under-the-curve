# AFL Prediction Model

A comprehensive machine learning pipeline for predicting Australian Football League (AFL) match outcomes using historical data from the "AFL-Data-Analysis" GitHub repository.

## Project Overview

This project implements a sophisticated data pipeline, feature engineering, statistical modeling, and machine learning framework to predict AFL match winners and margins. The system uses SQLite with Parquet backup, batch loading with incremental updates, and comprehensive validation.

## Phase Completion Status

### ✅ Phase 1A: Data Pipeline (COMPLETE)
- **Data Loading**: Robust pipeline with SQLite and Parquet storage
- **Validation**: Comprehensive data validation and quality checks
- **Storage**: Efficient batch loading with incremental updates
- **Logging**: Detailed logging and error handling
- **Results**: Successfully loaded 16,649 match records and 670,839 player records

### ✅ Phase 1B: Exploratory Data Analysis (COMPLETE)
- **Comprehensive EDA**: Statistical summaries, visual exploration, time series analysis
- **Data Quality Assessment**: Missing values, outliers, data distribution analysis
- **Temporal Analysis**: Season trends, performance evolution over time
- **Player Analysis**: Individual and team performance metrics
- **Visualizations**: 5 key visualizations generated and saved
- **Documentation**: Detailed EDA report with findings and recommendations

### ✅ Phase 2A: Feature Engineering (COMPLETE)
- **Strategy Analysis**: Evaluated 4 feature engineering approaches
- **Implementation**: Created 123 engineered features from 6,522 matches
- **Feature Categories**:
  - Team performance features (rolling averages, EWM, home/away metrics)
  - Player aggregation features (team composition, experience, star impact)
  - Contextual features (venue, rest days, season effects)
  - Advanced features (interactions, momentum, polynomial terms)
- **Analysis**: Feature importance, correlation studies, and visualizations
- **Documentation**: Comprehensive feature engineering report

### ✅ Phase 2B: Statistical Modeling (COMPLETE)
- **Approach Analysis**: Evaluated 4 statistical modeling approaches
- **Implementation**: Comprehensive statistical modeling framework
- **Components**:
  - Distribution fitting (parametric and non-parametric)
  - Hierarchical Bayesian modeling
  - Team performance distributions
  - Uncertainty quantification with Monte Carlo methods
- **Results**: Detailed statistical analysis and model validation
- **Documentation**: Statistical modeling report with recommendations

### ✅ Phase 3A: ML Model Architecture & Training (COMPLETE)
- **Architecture Analysis**: Evaluated 4 ML approaches, selected top 3
- **Model Implementation**: Successfully trained 3 comprehensive models
- **Models Trained**:
  1. **Traditional ML (Random Forest/XGBoost)**: Multi-task learning for winner and margin prediction
  2. **Ensemble/Meta-learning (Stacking)**: Multiple base models with meta-learner
  3. **Deep Learning (MLP)**: Neural network with multi-task learning
- **Data Preparation**: Time series splits (1991-2020 train, 2021-2023 validation, 2024-2025 test)
- **Features**: 114 engineered features used for modeling
- **Predictions**: Generated predictions for all 3 models across train/validation/test sets
- **Output**: 19,566 prediction records saved for evaluation

### 🔄 Phase 3B: Model Evaluation & Optimization (IN PROGRESS)
- **Evaluation Framework**: Comprehensive metrics and validation strategies
- **Model Comparison**: Performance comparison across all 3 models
- **Optimization**: Hyperparameter tuning and model refinement
- **Documentation**: Evaluation report and final recommendations

## Data Pipeline Architecture

### Storage Strategy
- **Primary**: SQLite database for fast querying and ACID compliance
- **Backup**: Parquet files for efficient storage and data portability
- **Validation**: Comprehensive data quality checks and error handling

### Feature Engineering Strategy
- **Traditional Statistical**: Rolling averages, exponential weighted means, head-to-head records
- **Advanced Time Series**: Momentum indicators, volatility measures, seasonal adjustments
- **Player Interaction**: Team composition analysis, experience metrics, star player impact
- **Contextual Features**: Venue effects, rest days, historical matchups, season progression

## Model Architecture

### Selected Approaches
1. **Traditional ML**: Random Forest and XGBoost with multi-task learning
2. **Ensemble/Meta-learning**: Stacking with multiple base models and meta-learner
3. **Deep Learning**: Multi-layer perceptron with multi-task learning

### Training Strategy
- **Time Series Splits**: Chronological data splitting to prevent data leakage
- **Multi-task Learning**: Simultaneous winner and margin prediction
- **Feature Scaling**: Standardization for neural networks
- **Cross-validation**: Time series cross-validation for robust evaluation

## Output Structure

```
outputs/
├── data/
│   ├── feature_engineering/
│   │   └── engineered_features.csv
│   ├── ml_models/
│   │   └── all_predictions.csv
│   ├── ml_architecture/
│   │   └── ml_architecture_analysis_results.json
│   └── statistical_modeling/
│       ├── statistical_modeling_analysis_results.json
│       └── statistical_modeling_summary.json
├── reports/
│   ├── eda/
│   │   └── EDA_Report.md
│   ├── feature_engineering/
│   │   └── Feature_Engineering_Report.md
│   └── statistical_modeling/
│       └── Statistical_Modeling_Report.md
└── visualizations/
    ├── eda/
    │   ├── data_quality.png
    │   ├── match_analysis.png
    │   ├── player_analysis.png
    │   └── temporal_analysis.png
    ├── feature_engineering/
    │   ├── feature_correlation_matrix.png
    │   ├── feature_importance.png
    │   └── feature_strategy_comparison.png
    ├── ml_architecture/
    │   └── ml_approach_comparison.png
    └── statistical_modeling/
        ├── distribution_fitting_results.png
        ├── hierarchical_model_structure.png
        ├── statistical_modeling_approach_comparison.png
        └── uncertainty_quantification_results.png
```

## Key Findings

### Data Quality
- **Match Data**: 16,649 matches from 1965-2025 with comprehensive statistics
- **Player Data**: 670,839 player records with detailed performance metrics
- **Data Completeness**: High quality with minimal missing values
- **Temporal Coverage**: 60 years of AFL history for robust modeling

### Feature Engineering Results
- **Feature Count**: 123 engineered features from original 20+ base features
- **Feature Categories**: Team performance, player aggregation, contextual, and advanced features
- **Correlation Analysis**: Identified key predictive features and interactions
- **Dimensionality**: Balanced feature richness with computational efficiency

### Model Training Results
- **Training Samples**: 5,649 matches (1991-2020)
- **Validation Samples**: 630 matches (2021-2023)
- **Test Samples**: 243 matches (2024-2025)
- **Features Used**: 114 engineered features
- **Predictions Generated**: 19,566 prediction records across all models and datasets

## Next Steps

### Phase 3B: Model Evaluation & Optimization
1. **Comprehensive Evaluation**: Traditional metrics, probabilistic evaluation, domain-specific metrics
2. **Robustness Testing**: Cross-validation, temporal stability, uncertainty quantification
3. **Model Comparison**: Performance comparison across all 3 approaches
4. **Hyperparameter Optimization**: Fine-tuning for optimal performance
5. **Final Model Selection**: Choose best performing model for deployment

### Future Enhancements
- **Real-time Predictions**: API development for live match predictions
- **Model Monitoring**: Performance tracking and drift detection
- **Feature Updates**: Continuous feature engineering based on new data
- **Ensemble Methods**: Advanced ensemble techniques for improved accuracy

## Technical Requirements

### Dependencies
- Python 3.9+
- pandas, numpy, scikit-learn
- matplotlib, seaborn for visualization
- sqlite3 for database operations
- xgboost for gradient boosting
- tensorflow-macos for deep learning (Apple Silicon optimized)

### Installation
```bash
# Create virtual environment
python -m venv afl2_env
source afl2_env/bin/activate  # On Windows: afl2_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install additional dependencies for Apple Silicon
brew install libomp  # Required for XGBoost
pip install tensorflow-macos tensorflow-metal  # Apple Silicon optimized TensorFlow
```

## Usage

### Running the Pipeline
```bash
# Data pipeline
python data_pipeline.py

# Exploratory data analysis
python eda_comprehensive.py

# Feature engineering
python feature_engineering_pipeline.py

# Statistical modeling
python statistical_modeling_framework.py

# ML training pipeline
python ml_training_pipeline.py
```

### Exploring Outputs
```bash
# Navigate outputs
python navigate_outputs.py
```

## Contributing

This project follows a structured approach to machine learning development with clear phases and deliverables. Each phase builds upon the previous one, ensuring robust and well-documented results.

## License

This project is for educational and research purposes. Please respect the original data sources and licensing requirements. 