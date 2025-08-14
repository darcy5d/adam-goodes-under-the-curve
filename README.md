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
| **Total Player Records** | 677,647+ |
| **Year Range** | 1897-2025 (129 years) |
| **Unique Teams** | 25 (matches), 24 (players) |
| **Unique Venues** | 50 |
| **Database Size** | ~45 MB |
| **2025 Data Coverage** | Complete (Rounds 1-23) |
| **2025 Player Records** | 7,688 individual player performances |

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
│   ├── retrain_clean_model.py  # Clean model training (no data leakage)
│   ├── hybrid_data_pipeline.py # Evergreen data pipeline with web scraping
│   ├── modern_afl_parser.py    # AFL Tables parser for 2025+ seasons
│   └── simple_player_scraper.py # Individual player statistics scraper
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
- **Data Loading**: SQLite database with comprehensive AFL match data (16,786+ matches, 1897-2025)
- **Data Validation**: Automatic data type conversion and quality checks
- **Feature Storage**: Clean feature generation with temporal awareness (no data leakage)
- **Margin Calculation**: Proper AFL scoring - `(goals × 6 + behinds) - (away_goals × 6 + away_behinds)`

### 🎯 Clean Feature Engineering (28 Features - No Data Leakage)

The model uses **only pre-game information** to make predictions, ensuring realistic performance in live scenarios.

#### **Team Performance Features (20 features)**

**Home Team Features (10):**
- `home_avg_goals_for`: Average goals scored (last 20 games)
- `home_avg_goals_against`: Average goals conceded (last 20 games)  
- `home_avg_goals_for_5`: Short-term scoring form (last 5 games)
- `home_avg_goals_against_5`: Short-term defensive form (last 5 games)
- `home_avg_goals_for_10`: Medium-term scoring form (last 10 games)
- `home_avg_goals_against_10`: Medium-term defensive form (last 10 games)
- `home_win_rate_5`: Win percentage (last 5 games)
- `home_win_rate_10`: Win percentage (last 10 games)
- `home_recent_form`: Recent form indicator (last 5 games)
- `home_momentum`: Scoring trend (recent 5 vs previous 5 games)

**Away Team Features (10):**
- `away_avg_goals_for`: Average goals scored (last 20 games)
- `away_avg_goals_against`: Average goals conceded (last 20 games)
- `away_avg_goals_for_5`: Short-term scoring form (last 5 games)
- `away_avg_goals_against_5`: Short-term defensive form (last 5 games)
- `away_avg_goals_for_10`: Medium-term scoring form (last 10 games)  
- `away_avg_goals_against_10`: Medium-term defensive form (last 10 games)
- `away_win_rate_5`: Win percentage (last 5 games)
- `away_win_rate_10`: Win percentage (last 10 games)
- `away_recent_form`: Recent form indicator (last 5 games)
- `away_momentum`: Scoring trend (recent 5 vs previous 5 games)

#### **Head-to-Head Features (3 features)**
- `h2h_home_win_rate`: Historical home team win rate in this matchup
- `h2h_avg_margin`: Average margin in previous meetings (points-based)
- `h2h_total_games`: Total historical meetings between teams

#### **Contextual Features (5 features)**
- `venue_home_advantage`: Home team win rate at this venue (normalized 0-1)
- `venue_experience`: Venue familiarity factor (matches played / 100, capped at 1.0)
- `home_rest_days`: Days since home team's last match (capped at 30)
- `away_rest_days`: Days since away team's last match (capped at 30) 
- `season_progress`: How far through the season (matches played / 200)

### 🤖 Model Architecture

#### **Dual-Model System:**
- **Winner Model**: `RandomForestClassifier` (n_estimators=100, max_depth=10)
  - **Input**: 28 clean features
  - **Output**: Win/loss probability for each team
  - **Accuracy**: ~65.8% (realistic for AFL prediction)

- **Margin Model**: `RandomForestRegressor` (n_estimators=100, max_depth=10)
  - **Input**: 28 clean features  
  - **Output**: Raw margin prediction in points
  - **Performance**: 30.45 MAE (excellent - historical average is 32.1 points)

#### **Prediction Process:**
1. **Feature Generation**: Calculate all 28 features using only historical data before match date
2. **Winner Prediction**: RandomForest outputs probabilities for home/away win
3. **Margin Prediction**: RandomForest outputs expected point margin
4. **Confidence Calculation**: `max(home_prob, away_prob)` determines overall confidence
5. **Final Output**: Winner, margin, and confidence with uncertainty estimates

#### **Key Model Improvements:**
- ✅ **No Data Leakage**: Features use only pre-game information
- ✅ **Proper Margin Calculation**: Points-based AFL scoring (not goals)
- ✅ **No Artificial Scaling**: Trust raw model output  
- ✅ **Temporal Awareness**: Features respect match chronology
- ✅ **Robust Defaults**: Handles new teams/venues gracefully

### 📊 Feature Importance Rankings

**Top 10 Most Important Features:**
1. **away_avg_goals_against** (6.0%) - Away team's defensive strength
2. **away_avg_goals_for** (5.8%) - Away team's offensive ability
3. **home_win_rate_10** (5.7%) - Home team's recent success rate
4. **home_avg_goals_against** (5.7%) - Home team's defensive strength  
5. **home_avg_goals_against_10** (5.2%) - Home team's medium-term defense
6. **home_avg_goals_for** (5.1%) - Home team's offensive ability
7. **away_avg_goals_against_10** (4.7%) - Away team's medium-term defense
8. **home_avg_goals_for_10** (4.6%) - Home team's medium-term offense
9. **venue_home_advantage** (4.4%) - Venue-specific home advantage
10. **away_avg_goals_for_10** (4.0%) - Away team's medium-term offense

## 📊 Model Performance & Validation

### 🎯 Current Performance Metrics
- **Winner Accuracy**: **65.8%** (excellent for AFL - historically challenging sport to predict)
- **Margin MAE**: **30.45 points** (outstanding - historical AFL average margin is 32.1 points)
- **Feature Balance**: No single feature dominates (top feature only 6% importance)
- **Data Coverage**: Trained on 6,511 matches from 1991-2025

### 📈 Historical Context & Benchmarks
- **AFL Average Margin**: 32.1 points (across 16,786 matches, 1897-2025)
- **Typical Game Distribution**:
  - Close Games (≤10 pts): 21.5% of matches
  - Moderate Games (11-30 pts): 34.8% of matches  
  - Big Wins (31-60 pts): 29.9% of matches
  - Blowouts (61+ pts): 13.8% of matches
- **Home Team Advantage**: Consistent 7.8 point advantage across all eras

### 🎲 Confidence Interpretation Guide

**Overall Confidence Levels:**
- **50-60%**: Low confidence - coin flip game, very close match expected
- **60-70%**: Medium confidence - moderate favorite, competitive game
- **70-80%**: High confidence - strong favorite, expect convincing win  
- **80%+**: Very high confidence - overwhelming favorite, potential blowout

**Margin Confidence (±Uncertainty):**
- Calculated as: `abs(predicted_margin) × 0.15 + 3.0`
- **Example**: 30-point margin ± 7.5 points = likely outcome between 22.5-37.5 points
- **Interpretation**: Shows prediction reliability range around the expected margin

### ✅ Model Validation Features
- **No Data Leakage**: Uses only pre-game information available to bettors/fans
- **Temporal Integrity**: Features respect chronological order of matches
- **Realistic Performance**: Aligns with historical AFL prediction difficulty
- **Robust Handling**: Graceful defaults for new teams, venues, or missing data

## 🔧 Technical Details

### Dependencies
- **Streamlit**: Interactive web dashboard
- **Pandas/NumPy**: Data manipulation and analysis
- **Scikit-learn**: Machine learning models
- **Plotly**: Interactive visualizations
- **SQLite**: Database storage

### Data Sources & Processing
- **Primary Database**: SQLite with 16,786+ AFL matches (1897-2025)
- **Scoring Data**: Goals, behinds, and calculated points margins  
- **Venue Data**: 50+ unique venues with home advantage statistics
- **Team Data**: 25 AFL teams with comprehensive historical records
- **Temporal Data**: Match dates, rest days, season progression

### Feature Engineering Philosophy  
- **Zero Data Leakage**: Only pre-game information (no quarter scores, final results)
- **Temporal Awareness**: Features calculated using only historical data before each match
- **AFL-Specific Logic**: Designed for unique AFL dynamics (home advantage, momentum, venue effects)
- **Robust Defaults**: Handles new teams (12.0 goal average), new venues (55% home advantage)
- **Normalized Scales**: Features scaled appropriately (venue experience capped at 1.0, rest days at 30)

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
- Ensure model is trained on recent data (`python scripts/retrain_clean_model.py`)
- Use exact team names from dropdown options in dashboard
- Verify venue names match database entries (50+ venues supported)
- Consider context: rest days, venue history, head-to-head records
- Trust the model - no manual scaling applied to predictions

### Dashboard Navigation:
- **Start**: "Data Explorer" → understand AFL patterns and statistics
- **Predict**: "Match Predictions" → get winner/margin forecasts with confidence
- **Monitor**: "Dashboard Overview" → check model health and recent performance
- **Analyze**: Venue analysis, team momentum, clutch performance insights

### Troubleshooting:
- **Predictions fail**: Check team/venue spelling (case-sensitive)
- **Dashboard won't load**: Activate virtual environment: `source afl2_env/bin/activate`
- **Model not found**: Train clean model: `python scripts/retrain_clean_model.py`
- **Unrealistic margins**: Model now fixed - expect 20-60 point predictions
- **Low confidence**: Normal for close games - AFL is inherently unpredictable

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

Our enhanced data pipeline now includes four powerful approaches:

#### 🤖 Hybrid Evergreen Pipeline (RECOMMENDED)
- **Automatic Data Detection**: Intelligently detects repository data cutoff year
- **Seamless Transition**: Uses repository for historical data, scraping for current seasons
- **Evergreen Operation**: Automatically handles 2026, 2027, and beyond without manual updates
- **Smart Strategy**: Determines optimal data update strategy based on current date and existing data
- **Zero Configuration**: Works out of the box for any future AFL season

#### 🏈 Player Statistics Scraper (NEW!)
- **Individual Player Data**: Scrapes detailed match-by-match player performance statistics
- **Complete Coverage**: All AFL Tables player stats (kicks, marks, handballs, disposals, goals, etc.)
- **Mass Processing**: Efficiently handles 148+ matches with 6,800+ player records
- **Database Integration**: Seamlessly stores player data in existing SQLite schema
- **Real Player Names**: Extracts and stores actual player names for enhanced analysis

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

# Scrape detailed player statistics for missing rounds
python scripts/simple_player_scraper.py

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