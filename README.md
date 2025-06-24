# AFL Prediction Model - Data Pipeline & Analysis

## Project Overview

This project implements a comprehensive data pipeline and exploratory analysis for building an AFL (Australian Football League) prediction model. The pipeline extracts, validates, and stores historical AFL data from 1897 to 2025, providing a robust foundation for machine learning models.

### Objectives

- **Data Foundation**: Establish a reliable, validated dataset of AFL match and player statistics
- **Data Quality**: Implement comprehensive validation and quality checks
- **Exploratory Analysis**: Conduct thorough EDA to understand patterns and inform model development
- **Performance**: Optimize data storage and query performance
- **Extensibility**: Design for future model development and data updates
- **Reproducibility**: Ensure consistent data processing across environments

## Data Sources

- **Primary Source**: [AFL-Data-Analysis Repository](https://github.com/akareen/AFL-Data-Analysis)
- **Data Coverage**: 1897-2025 (historical to current)
- **Data Types**: Match results, player statistics, team performance

### Current Dataset Statistics

| Metric | Value |
|--------|-------|
| **Total Matches** | 16,649 |
| **Total Player Records** | 670,839 |
| **Year Range** | 1897-2025 (129 years) |
| **Unique Teams** | 25 (matches), 24 (players) |
| **Unique Venues** | 50 |
| **Database Size** | ~43.5 MB |

## Phase 1A: Data Pipeline Implementation ✅ COMPLETED

### Data Structure

#### Match Data Schema
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

#### Player Data Schema
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

## Phase 1B: Exploratory Data Analysis ✅ COMPLETED

### EDA Methodology

The comprehensive EDA implemented four distinct analytical approaches:

1. **Statistical Summary Approach**: Descriptive statistics, distributions, and central tendency measures
2. **Visual Exploration Approach**: Trend analysis, relationship mapping, and pattern identification
3. **Time Series Analysis**: Temporal patterns, seasonality, and era-based comparisons
4. **Data Quality Assessment**: Missing data analysis, outlier detection, and consistency checks

### Key Findings

#### Temporal Analysis
- **Data Growth**: Modern era (1991-2025) shows 2x more matches per year compared to early era
- **Scoring Evolution**: Gradual increase from 20.0 (early era) to 26.4 (modern era) total goals per match
- **Home Advantage Stability**: Remarkably consistent ~1.1 goal advantage across all eras
- **Seasonal Patterns**: Round 22 shows highest scoring, Grand Final shows lowest

#### Match Analysis
- **Home Advantage**: Consistent 1.1 goals across all eras (Early: 1.1, Mid: 1.1, Modern: 1.2)
- **Scoring Distribution**: Normal distribution centered around 24 total goals
- **Margin Distribution**: Slightly right-skewed, indicating some high-scoring games
- **Round Patterns**: Finals games show lower scoring (defensive play)

#### Player Analysis
- **Core Statistics Completeness**: Kicks (62%), Marks (58%), Handballs (60%), Disposals (63%)
- **Performance Distributions**: Normal distributions with expected outliers
- **Career Longevity**: Average 3.2 years, with modern era showing longer careers
- **Outlier Analysis**: Goals show 9.1% outliers, disposals show 0.6% outliers
- **Temporal Trends**: Average disposals range from 13-17 per game (1965-2025), showing realistic historical patterns

#### Data Quality Assessment
- **Match Data Quality**: 3.9% missing rate overall, core data highly complete
- **Player Data Quality**: 54.1% missing rate, but core statistics >50% complete
- **Data Consistency**: 100% year overlap, 96% team consistency between datasets
- **Problematic Columns**: Attendance (89% missing), Brownlow votes (97% missing)

### Era Comparison

| Era | Period | Matches | Avg Matches/Year | Home Goals | Away Goals | Home Advantage |
|-----|--------|---------|------------------|------------|------------|----------------|
| **Early Era** | 1897-1950 | 4,983 | 92.3 | 10.6 | 9.4 | 1.1 |
| **Mid Era** | 1951-1990 | 5,144 | 128.6 | 13.3 | 12.2 | 1.1 |
| **Modern Era** | 1991-2025 | 6,522 | 186.3 | 13.8 | 12.6 | 1.2 |

### Visualizations Generated

The EDA produced four comprehensive visualization sets:

1. **Temporal Analysis** (`eda_output/temporal_analysis.png`): Data volume and scoring trends over time
2. **Match Analysis** (`eda_output/match_analysis.png`): Home advantage and scoring distributions
3. **Player Analysis** (`eda_output/player_analysis.png`): Performance distributions and career patterns
4. **Data Quality** (`eda_output/data_quality.png`): Missing data patterns and outlier analysis

## Technology Decisions & Rationale

### 1. Data Storage Analysis

| Approach | Performance | Scalability | Ease of Use | Extensibility | Decision |
|----------|-------------|-------------|-------------|---------------|----------|
| **SQLite** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ **Chosen** |
| PostgreSQL | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Considered |
| Parquet Files | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ✅ **Backup** |
| CSV with Indexing | ⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | Rejected |

**Rationale**: SQLite provides excellent performance for single-user applications, ACID compliance, and simple setup. Parquet files serve as backup and analytical format.

### 2. Data Loading Strategies

| Approach | Memory Efficiency | Complexity | Update Frequency | Decision |
|----------|-------------------|------------|------------------|----------|
| **Batch Loading** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Low | ✅ **Chosen** |
| Streaming | ⭐⭐⭐⭐⭐ | ⭐⭐ | High | Future consideration |
| Incremental | ⭐⭐⭐⭐ | ⭐⭐⭐ | Medium | ✅ **Future** |

**Rationale**: Batch loading is optimal for historical data processing. Incremental updates will be implemented for ongoing data collection.

### 3. Data Validation Approaches

| Approach | Coverage | Performance | Domain Knowledge | Decision |
|----------|----------|-------------|------------------|----------|
| **Schema Validation** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Low | ✅ **Implemented** |
| **Statistical Validation** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Medium | ✅ **Implemented** |
| **Manual Inspection** | ⭐⭐⭐⭐⭐ | ⭐⭐ | High | ✅ **Targeted** |

**Rationale**: Combined approach ensures comprehensive data quality while maintaining performance.

## Setup Instructions

### Prerequisites

- Python 3.8+
- Git
- 2GB+ free disk space

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AFL2
   ```

2. **Create virtual environment**
   ```bash
   python -m venv afl2_env
   source afl2_env/bin/activate  # On Windows: afl2_env\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the data pipeline**
   ```bash
   python data_pipeline.py
   ```

5. **Run comprehensive EDA**
   ```bash
   python eda_comprehensive.py
   ```

6. **Navigate outputs** (optional)
   ```bash
   python navigate_outputs.py
   ```

### Directory Structure

```
AFL2/
├── data_pipeline.py                    # Main data pipeline
├── eda_comprehensive.py                # Comprehensive EDA analysis
├── eda_analysis.py                     # Basic EDA analysis
├── feature_engineering_analysis.py     # Feature engineering strategy analysis
├── feature_engineering_pipeline.py     # Feature engineering implementation
├── statistical_modeling_analysis.py    # Statistical modeling approach analysis
├── statistical_modeling_framework.py   # Statistical modeling implementation
├── requirements.txt                    # Python dependencies
├── README.md                          # This file
├── afl_data/                          # Data directory (created by pipeline)
│   ├── AFL-Data-Analysis/             # Cloned repository
│   ├── afl_database.db                # SQLite database
│   └── parquet/                       # Parquet backup files
│       ├── matches.parquet
│       └── players.parquet
├── outputs/                           # All analysis outputs
│   ├── visualizations/                # All generated visualizations
│   │   ├── eda/                      # EDA visualizations
│   │   │   ├── data_quality.png
│   │   │   ├── match_analysis.png
│   │   │   ├── player_analysis.png
│   │   │   ├── temporal_analysis.png
│   │   │   └── eda_analysis_results.json
│   │   ├── feature_engineering/       # Feature engineering visualizations
│   │   │   ├── feature_importance.png
│   │   │   ├── feature_correlation_matrix.png
│   │   │   └── feature_strategy_comparison.png
│   │   └── statistical_modeling/      # Statistical modeling visualizations
│   │       ├── distribution_fitting_results.png
│   │       ├── hierarchical_model_structure.png
│   │       ├── uncertainty_quantification_results.png
│   │       └── statistical_modeling_approach_comparison.png
│   ├── reports/                       # All generated reports
│   │   ├── eda/                      # EDA reports
│   │   │   └── EDA_Report.md
│   │   ├── feature_engineering/       # Feature engineering reports
│   │   │   └── Feature_Engineering_Report.md
│   │   └── statistical_modeling/      # Statistical modeling reports
│   │       └── Statistical_Modeling_Report.md
│   └── data/                          # All generated data files
│       ├── pipeline/                  # Pipeline outputs
│       │   └── afl_pipeline.log
│       ├── feature_engineering/       # Feature engineering data
│       │   ├── engineered_features.csv
│       │   ├── feature_importance.csv
│       │   └── feature_engineering_analysis_results.json
│       └── statistical_modeling/      # Statistical modeling data
│           ├── statistical_modeling_analysis_results.json
│           └── statistical_modeling_summary.json
└── afl2_env/                          # Virtual environment
```

### Organized Outputs

All analysis outputs are now organized in the `outputs/` directory for better project structure:

**📊 Visualizations** (`outputs/visualizations/`)
- **EDA**: Data quality, match analysis, player analysis, temporal analysis
- **Feature Engineering**: Feature importance, correlation matrix, strategy comparison
- **Statistical Modeling**: Distribution fitting, hierarchical structure, uncertainty quantification

**📄 Reports** (`outputs/reports/`)
- **EDA**: Comprehensive EDA report with findings and insights
- **Feature Engineering**: Feature engineering strategy and implementation report
- **Statistical Modeling**: Statistical distribution modeling report

**💾 Data Files** (`outputs/data/`)
- **Pipeline**: Execution logs and pipeline outputs
- **Feature Engineering**: Engineered features dataset, importance rankings, analysis results
- **Statistical Modeling**: Distribution fitting results, hierarchical model data

**🔍 Navigation**: Use `python navigate_outputs.py` to explore the organized outputs interactively.

## Project Status

### ✅ Phase 1: Data Foundation & Analysis (COMPLETED)

**Phase 1A: Data Pipeline**
- [x] **Data Pipeline Architecture**: Complete pipeline with modular design
- [x] **Repository Integration**: Automated cloning and updating of AFL data repository
- [x] **Data Loading**: Robust CSV file discovery and loading with column mapping
- [x] **Data Validation**: Comprehensive schema and statistical validation
- [x] **Database Setup**: SQLite database with proper indexing
- [x] **Data Storage**: Dual storage (SQLite + Parquet) for reliability
- [x] **Quality Reporting**: Automated data quality metrics and reporting
- [x] **Performance Benchmarks**: Query performance and storage efficiency metrics
- [x] **Logging**: Comprehensive logging throughout the pipeline

**Phase 1B: Exploratory Data Analysis**
- [x] **Statistical Summary Analysis**: Descriptive statistics and distributions
- [x] **Visual Exploration**: Trend analysis and pattern identification
- [x] **Time Series Analysis**: Temporal patterns and era comparisons
- [x] **Data Quality Assessment**: Missing data and outlier analysis
- [x] **Comprehensive Visualizations**: Four detailed visualization sets
- [x] **Recommendations**: Clear guidance for model development
- [x] **Detailed Report**: Complete EDA report with findings and insights
- [x] **Data Quality Verification**: Confirmed temporal data integrity and realistic patterns

### ✅ Phase 2: Advanced Analytics & Modeling Foundation (COMPLETED)

**Phase 2A: Advanced Feature Engineering**
- [x] **Strategy Analysis**: Evaluated 4 feature engineering approaches
- [x] **Feature Creation**: 123 engineered features from 6,522 matches
- [x] **Feature Analysis**: Importance ranking and correlation analysis
- [x] **Comprehensive Documentation**: Complete feature engineering report

**Phase 2B: Statistical Distribution Modeling**
- [x] **Approach Analysis**: Evaluated 4 statistical modeling approaches
- [x] **Distribution Fitting**: 6 player statistics with appropriate distributions
- [x] **Hierarchical Modeling**: 3-level hierarchy for round/team/era analysis
- [x] **Team Performance**: Team-level distribution aggregation
- [x] **Uncertainty Quantification**: Monte Carlo methods with bootstrap confidence intervals
- [x] **Model Validation**: All distributions passed goodness-of-fit tests
- [x] **Comprehensive Documentation**: Complete statistical modeling report

### 🚀 Phase 3: Model Development & Training (READY TO START)

**Model Development Framework**
1. **Model Architecture Selection**: Implement multiple model types (tree-based, linear, neural networks)
2. **Feature Selection**: Use top 50 features for primary models, all 123 for ensemble
3. **Statistical Integration**: Incorporate fitted distributions and hierarchical structure
4. **Uncertainty Propagation**: Use Monte Carlo methods for prediction uncertainty
5. **Hyperparameter Tuning**: Optimize model parameters using cross-validation
6. **Ensemble Methods**: Combine multiple models for improved performance
7. **Validation Strategy**: Use recommended timeline (1991-2020 train, 2021-2023 validate, 2024-2025 test)

## Key Recommendations for Phase 2

### Reliable Data Subsets

**Primary Training Data (1991-2025):**
- Most complete and consistent data
- Relevant to current game rules
- Core statistics >50% complete

**Secondary Training Data (1951-1990):**
- Good for historical patterns
- Balanced data quality
- Useful for trend analysis

**Avoid for Training (1897-1950):**
- Inconsistent data quality
- Different game rules
- Limited statistical recording

### Most Predictive Features

**Recommended (Completeness >50%):**
1. **Kicks** (62% complete) - Core possession metric
2. **Marks** (58% complete) - Key defensive/offensive stat
3. **Handballs** (60% complete) - Possession chain metric
4. **Disposals** (63% complete) - Total possession metric

**Secondary (Completeness 30-50%):**
- Goals (32% complete) - Scoring metric
- Tackles (36% complete) - Defensive metric

### Preprocessing Strategies

1. **Missing Value Handling**: Era-specific imputation for statistical columns
2. **Outlier Management**: Remove statistical outliers using IQR method
3. **Feature Engineering**: Efficiency ratios, era-normalized statistics
4. **Data Normalization**: Standardize statistics by era to account for rule changes

### Model Training Timeline

- **Training**: 1991-2020 (30 years of modern data)
- **Validation**: 2021-2023 (3 years)
- **Testing**: 2024-2025 (2 years)

## Issues Encountered & Solutions

### Data Loading Challenges

1. **Column Mapping Issues**: Fixed column name mapping for match data (`team_1_team_name` → `home_team`)
2. **File Discovery**: Updated file pattern matching for correct data file identification
3. **Data Type Conversion**: Added proper numeric conversion for all statistical columns

### Data Quality Issues

1. **Missing Data**: Identified and documented missing data patterns by era
2. **Outliers**: Implemented statistical outlier detection and analysis
3. **Consistency**: Verified data consistency between match and player datasets
4. **Temporal Data**: Confirmed year ranges and disposal patterns are realistic and accurate

## Next Steps

1. **Model Development**: Build prediction models using 123 engineered features and statistical distributions
2. **Statistical Integration**: Incorporate fitted distributions and hierarchical structure into models
3. **Uncertainty Quantification**: Use Monte Carlo methods for prediction uncertainty
4. **Validation**: Use recommended timeline for realistic performance assessment
5. **Deployment**: Prepare model for production use

---

*Last updated: December 2024*  
*Phase 1 & Phase 2 completed successfully - Ready for Phase 3 Model Development* 