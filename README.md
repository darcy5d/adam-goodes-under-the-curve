# 🏈 AFL Match Prediction System

A comprehensive AFL (Australian Football League) match prediction system with an interactive Streamlit dashboard. This system uses machine learning to predict match winners and margins based on historical AFL data.

## 🎯 Features

- **Interactive Dashboard**: Streamlit-based web interface for data exploration and predictions
- **Data Exploration**: AFL-specific insights including current season ladder, rivalries, venue analysis, momentum tracking, and clutch performance
- **Match Predictions**: Real-time prediction of match winners and margins using trained ML models
- **Model Training**: Built-in training interface with feature selection capabilities
- **Clean Architecture**: Organized codebase with clear separation of concerns

## 📁 Project Structure

```
AFL2/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── afl_dashboard.py            # Main Streamlit dashboard application
├── cleanup_repo.py             # Repository maintenance script
├── afl2_env/                   # Python virtual environment
├── afl_data/                   # Data storage
│   ├── afl_database.db         # Main SQLite database with AFL match data
│   └── AFL-Data-Analysis/      # Raw data files
├── scripts/                    # Core processing scripts
│   ├── data_pipeline.py        # Data loading and validation
│   ├── feature_engineering_pipeline.py  # Feature creation and engineering
│   ├── ml_training_pipeline.py # Model training and evaluation
│   └── retrain_clean_model.py  # Clean model training (no data leakage)
├── docs/                       # Documentation and reports
│   ├── reports/               # Analysis reports (EDA, Feature Engineering)
│   └── visualizations/        # Generated charts and plots
└── outputs/                   # Model outputs and results
    └── data/
        ├── feature_engineering/  # Feature engineering results
        └── ml_models/            # Trained models and evaluation data
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone and navigate to the repository
cd AFL2

# Activate the virtual environment
source afl2_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train a Model (First Time)

```bash
# Train a clean model with no data leakage
python scripts/retrain_clean_model.py
```

### 3. Launch the Dashboard

```bash
# Start the Streamlit dashboard
streamlit run afl_dashboard.py
```

The dashboard will be available at `http://localhost:8501`

## 🎮 Dashboard Features

### 🏠 Dashboard Overview
- Latest 10 matches from the dataset
- Model performance metrics and recent predictions
- Quick statistics and data quality insights

### 🔥 Data Explorer
The data exploration section provides AFL-specific insights:

- **🔥 Current Season**: Latest season ladder with wins, losses, and percentage
- **⚔️ Rivalries**: Pre-configured AFL rivalries with head-to-head records
- **📍 Venues**: Venue statistics, home advantages, and team performance by ground
- **💫 Momentum**: Team form tracking with customizable windows
- **🎯 Clutch Performance**: Analysis of team performance in close games

### 🎯 Match Predictions
Real-time match prediction interface:

- **Team Selection**: Choose home and away teams
- **Match Details**: Set venue, date, and other parameters
- **Auto-calculated Features**: 
  - Rest days (automatically calculated from team schedules)
  - Season round estimation from match date
  - Dynamic attendance prediction based on teams, venue, and round
- **Prediction Results**: 
  - Winner probability with confidence metrics
  - Predicted margin with realistic scaling
  - Feature importance breakdown
  - Draw probability for very close matches

### 🏋️ Model Training
Built-in model training interface:

- **Feature Selection**: Toggle between full feature set and pruned features
- **Training Progress**: Real-time training status and metrics
- **Model Comparison**: Performance comparison between different approaches

## 🧠 Machine Learning Architecture

### Data Pipeline
- **Data Loading**: SQLite database with comprehensive AFL match data
- **Data Validation**: Automatic data type conversion and quality checks
- **Feature Storage**: Efficient parquet format for processed features

### Feature Engineering
The system creates 28 clean features (no data leakage):

**Team Performance Features:**
- Rolling averages for goals for/against (5, 10, season)
- Win rates and recent form
- Momentum indicators (trend analysis)

**Head-to-Head Features:**
- Historical matchup records
- Recent performance between teams

**Contextual Features:**
- Venue experience and home advantage
- Rest days between matches
- Season progression indicators

**Advanced Features:**
- Feature interactions and squared terms
- Normalized venue statistics
- Dynamic attendance modeling

### Model Architecture
- **Winner Prediction**: RandomForestClassifier for match outcome
- **Margin Prediction**: RandomForestRegressor with confidence-based scaling
- **Feature Selection**: 28 carefully selected features avoiding data leakage
- **Confidence Scaling**: Post-prediction margin adjustment based on winner confidence

## 📊 Model Performance

The clean model (without data leakage) typically achieves:
- **Winner Accuracy**: ~65-70% (realistic for AFL)
- **Margin MAE**: ~20-25 points (industry competitive)
- **Feature Importance**: Balanced across team form, venue, and context

## 🔧 Technical Details

### Dependencies
- **Streamlit**: Interactive web dashboard
- **Pandas/NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning models
- **Plotly**: Interactive visualizations
- **SQLite**: Database storage

### Data Sources
- Historical AFL match data including scores, venues, dates
- Team performance statistics
- Venue information and crowd data

### Feature Engineering Philosophy
- **No Data Leakage**: Features only use information available before match start
- **AFL-Specific**: Features designed for Australian Rules Football dynamics
- **Robust**: Handles missing data and edge cases gracefully

## 🛠️ Development & Maintenance

### Adding New Features
1. Modify `scripts/feature_engineering_pipeline.py`
2. Update feature blacklist in training pipeline if needed
3. Retrain model with `python scripts/retrain_clean_model.py`

### Model Retraining
```bash
# Clean model training (recommended)
python scripts/retrain_clean_model.py

# Or use the dashboard training interface
# Navigate to "Train Models" in the sidebar
```

### Data Updates
```bash
# Update database with new match data
python scripts/data_pipeline.py

# Regenerate features
python scripts/feature_engineering_pipeline.py
```

### Repository Cleanup
```bash
# Clean up temporary files and reorganize
python cleanup_repo.py
```

## 🎯 Usage Tips

### For Best Predictions:
- Ensure model is trained on recent data
- Use realistic team names (check dropdown options)
- Verify venue names match database entries
- Consider rest days and season context

### Dashboard Navigation:
- Start with "Data Explorer" to understand the data
- Use "Match Predictions" for forecasting
- Check "Dashboard Overview" for model health
- Use "Train Models" to retrain with new data

### Troubleshooting:
- If predictions fail, check team/venue name spelling
- If dashboard won't load, ensure virtual environment is activated
- If model not found, run `python scripts/retrain_clean_model.py`

## 📈 Future Enhancements

- Real-time data integration from AFL APIs
- Player-level statistics and injury data
- Weather and ground condition factors
- Advanced ensemble methods
- Mobile-responsive dashboard design

## 📝 License & Credits

This project is designed for AFL match prediction research and education. AFL data used under fair use for analytical purposes.

---

**🏈 Ready to predict some AFL matches? Fire up the dashboard and explore!**

```bash
source afl2_env/bin/activate
streamlit run afl_dashboard.py
```