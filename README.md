# 🏈 AFL Match Prediction System

A comprehensive AFL (Australian Football League) match prediction system with an interactive Streamlit dashboard. This system uses machine learning to predict match winners and margins based on historical AFL data.

## 📊 Data Sources

- **Primary Source**: [AFL-Data-Analysis Repository](https://github.com/akareen/AFL-Data-Analysis)
- **Data Coverage**: 1897-2025 (historical to current)
- **Data Types**: Match results, player statistics, team performance
- **Attribution**: This project builds upon the excellent work by akareen in collecting and organizing AFL historical data

### Current Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Matches** | 16,649 |
| **Total Player Records** | 670,839 |
| **Year Range** | 1897-2025 (129 years) |
| **Unique Teams** | 25 (matches), 24 (players) |
| **Unique Venues** | 50 |
| **Database Size** | ~43.5 MB |

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

### Prerequisites

- Python 3.8+
- Git
- 2GB+ free disk space

### 1. Repository Setup

```bash
# Clone this repository
git clone https://github.com/darcy5d/adam-goodes-under-the-curve.git
cd adam-goodes-under-the-curve

# Create and activate virtual environment
python -m venv afl2_env
source afl2_env/bin/activate  # On Windows: afl2_env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

**Option A: Use Existing Database (Recommended)**
```bash
# The repository includes a pre-built database (afl_data/afl_database.db)
# Skip to step 3 if this file exists and is recent
```

**Option B: Rebuild Database from Source**
```bash
# Method 1: Original repository data only
python scripts/data_pipeline.py

# Method 2: Enhanced scraping pipeline (full historical scraping)
python scripts/enhanced_data_pipeline.py --start_year 2020 --end_year 2025 --scrape_matches

# Method 3: Hybrid evergreen pipeline (RECOMMENDED!)
python scripts/hybrid_data_pipeline.py

# Method 4: Validate scraping vs repository data
python scripts/test_scraping_pipeline.py

# This will:
# 1. Intelligently combine repository data with real-time scraping
# 2. Automatically detect data cutoffs and switch to scraping for newer seasons
# 3. Create afl_data/afl_database.db with both historical and current data
# 4. Generate comprehensive quality reports
```

### 3. Train a Model (First Time)

```bash
# Train a clean model with no data leakage
python scripts/retrain_clean_model.py
```

### 4. Launch the Dashboard

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

## 📋 Data Schema

### Match Data Structure
```sql
matches (
    id INTEGER PRIMARY KEY,
    year INTEGER NOT NULL,
    ground TEXT,
    venue TEXT,
    date TEXT NOT NULL,
    home_team TEXT NOT NULL,
    away_team TEXT NOT NULL,
    home_team_goals_by_quarter TEXT,
    home_team_behinds_by_quarter TEXT,
    away_team_goals_by_quarter TEXT,
    away_team_behinds_by_quarter TEXT,
    home_total_goals INTEGER,
    home_total_behinds INTEGER,
    away_total_goals INTEGER,
    away_total_behinds INTEGER,
    winning_team TEXT,
    margin INTEGER,
    created_at TIMESTAMP
)
```

### Player Data Structure
```sql
players (
    id INTEGER PRIMARY KEY,
    team TEXT,
    year INTEGER,
    games_played INTEGER,
    opponent TEXT,
    round TEXT,
    result TEXT,
    jersey_number INTEGER,
    kicks INTEGER,
    marks INTEGER,
    handballs INTEGER,
    disposals INTEGER,
    goals INTEGER,
    behinds INTEGER,
    hit_outs INTEGER,
    tackles INTEGER,
    rebound_50s INTEGER,
    inside_50s INTEGER,
    clearances INTEGER,
    clangers INTEGER,
    free_kicks_for INTEGER,
    free_kicks_against INTEGER,
    brownlow_votes INTEGER,
    contested_possessions INTEGER,
    uncontested_possessions INTEGER,
    contested_marks INTEGER,
    marks_inside_50 INTEGER,
    one_percenters INTEGER,
    bounces INTEGER,
    goal_assist INTEGER,
    percentage_of_game_played REAL,
    first_name TEXT,
    last_name TEXT,
    born_date TEXT,
    debut_date TEXT,
    height INTEGER,
    weight INTEGER,
    created_at TIMESTAMP
)
```

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

## 🙏 Credits & Attribution

### Data Sources
- **Primary Data**: [AFL-Data-Analysis Repository](https://github.com/akareen/AFL-Data-Analysis) by akareen
- **Original AFL Data**: Australian Football League historical match and player statistics
- **Data Period**: 1897-2025 (129 years of AFL history)

### Data Processing
This project builds upon the excellent data collection and organization work by:
- **akareen**: Original AFL data collection, cleaning, and CSV organization
- **AFL**: Official match records and statistics

### Our Contributions
- Machine learning prediction models with clean feature engineering
- Interactive Streamlit dashboard for data exploration and predictions
- Comprehensive data pipeline with validation and quality checks
- Repository organization and production-ready structure

## 📝 License & Usage

- **Research & Education**: This project is designed for AFL match prediction research and educational purposes
- **Data Attribution**: AFL data used under fair use for analytical purposes, original collection by akareen
- **Non-Commercial**: Please respect the original data sources and use responsibly
- **Code License**: Open source - feel free to learn from and improve upon this work

### Data Update Process

To keep data current:

1. **Enhanced Scraping (NEW!)**: Use `python scripts/enhanced_data_pipeline.py --start_year 2024 --scrape_matches` to get latest data directly from AFL Tables
2. **Repository Method**: Monitor [AFL-Data-Analysis](https://github.com/akareen/AFL-Data-Analysis) for updates, then run `python scripts/data_pipeline.py`
3. **Validation**: Run `python scripts/test_scraping_pipeline.py` to compare data sources and ensure quality
4. **Retrain Models**: Use `python scripts/retrain_clean_model.py` with fresh data
5. **Update Features**: Modify feature engineering if new data columns are available

### Enhanced Scraping Features 🆕

Our enhanced data pipeline now includes three powerful approaches:

#### 🤖 Hybrid Evergreen Pipeline (RECOMMENDED)
- **Automatic Data Detection**: Intelligently detects repository data cutoff year
- **Seamless Transition**: Uses repository for historical data, scraping for current seasons
- **Evergreen Operation**: Automatically handles 2026, 2027, and beyond without manual updates
- **Smart Strategy**: Determines optimal data update strategy based on current date and existing data
- **Zero Configuration**: Works out of the box for any future AFL season

#### 🕷️ Enhanced Scraping Pipeline  
- **Direct AFL Tables Scraping**: Get data directly from the primary source
- **Data Quality Validation**: Compare scraped data with repository data for accuracy
- **Intelligent Fallback**: Automatically falls back to repository data if scraping fails
- **Completeness Verification**: Ensures no matches are missed
- **Rate Limiting**: Respectful scraping with appropriate delays
- **Comprehensive Reporting**: Detailed quality and completeness reports

#### 📊 Validation & Testing Tools
- **Scraping Feasibility Testing**: Validates AFL Tables accessibility and data structure
- **Data Comparison**: Compares scraped vs repository data for accuracy
- **Quality Metrics**: Comprehensive data quality scoring and reporting

**Usage Examples:**
```bash
# Evergreen hybrid approach (recommended for production)
python scripts/hybrid_data_pipeline.py

# Force scraping even if repository is current
python scripts/hybrid_data_pipeline.py --force_scraping

# Full historical scraping (for research/backup)
python scripts/enhanced_data_pipeline.py --start_year 2023 --end_year 2025 --scrape_matches

# Test scraping feasibility and data quality
python scripts/simple_scraping_test.py
```

---

**🏈 Ready to predict some AFL matches? Fire up the dashboard and explore!**

```bash
source afl2_env/bin/activate
streamlit run afl_dashboard.py
```

## 🔗 Links

- **This Repository**: [AFL Match Prediction System](https://github.com/darcy5d/adam-goodes-under-the-curve)
- **Original Data Source**: [AFL-Data-Analysis](https://github.com/akareen/AFL-Data-Analysis)
- **AFL Official**: [www.afl.com.au](https://www.afl.com.au)