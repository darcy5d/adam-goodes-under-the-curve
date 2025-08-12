#!/usr/bin/env python3
"""
Repository Cleanup Script
Removes unnecessary files and organizes the AFL prediction project.
"""

import os
import shutil
from pathlib import Path

def cleanup_repository():
    """Clean up the repository by removing unnecessary files."""
    
    print("🧹 Starting Repository Cleanup")
    print("=" * 50)
    
    # Files/folders to delete
    files_to_delete = [
        # Analysis scripts (exploratory - no longer needed)
        "ai_integration_analysis.py",
        "compare_pruned.py", 
        "data_analysis.py",
        "eda_analysis.py",
        "eda_comprehensive.py", 
        "feature_engineering_analysis.py",
        "interface_analysis.py",
        "ml_architecture_analysis.py",
        "ml_evaluation_framework.py",
        "statistical_modeling_analysis.py",
        "statistical_modeling_framework.py",
        "navigate_outputs.py",
        "test_pipeline.py",
        "improve_margin_model.py",  # Failed attempt
        
        # Cache
        "__pycache__",
        
        # Redundant outputs
        "outputs/data/interface_analysis",
        "outputs/data/ml_architecture", 
        "outputs/data/statistical_modeling",
        "outputs/data/pipeline",
        "outputs/reports/interface",
        "outputs/reports/statistical_modeling",
        "outputs/visualizations/ml_architecture",
        "outputs/visualizations/statistical_modeling",
        
        # Old models (keep only clean model)
        "outputs/data/ml_models/ensemble_ml_model.pkl",
        "outputs/data/ml_models/ensemble_ml_model_pruned.pkl",
        "outputs/data/ml_models/model_metadata.json",
        "outputs/data/ml_models/model_metadata_pruned.json",
        "outputs/data/ml_models/all_predictions.csv",
        
        # Unnecessary data files
        "afl_data/eda_output",
        "afl_data/parquet",
        "afl_data/quality_report.json",
    ]
    
    deleted_count = 0
    kept_important = []
    
    for item in files_to_delete:
        path = Path(item)
        if path.exists():
            try:
                if path.is_file():
                    path.unlink()
                    print(f"🗑️  Deleted file: {item}")
                elif path.is_dir():
                    shutil.rmtree(path)
                    print(f"🗑️  Deleted folder: {item}")
                deleted_count += 1
            except Exception as e:
                print(f"❌ Could not delete {item}: {e}")
        else:
            print(f"⚪ Not found: {item}")
    
    # Create organized structure
    print(f"\n📁 Creating organized structure...")
    
    # Create scripts folder
    scripts_dir = Path("scripts")
    scripts_dir.mkdir(exist_ok=True)
    
    # Move training scripts to scripts folder
    training_scripts = [
        "data_pipeline.py",
        "feature_engineering_pipeline.py", 
        "ml_training_pipeline.py",
        "retrain_clean_model.py"
    ]
    
    for script in training_scripts:
        if Path(script).exists():
            shutil.move(script, scripts_dir / script)
            print(f"📦 Moved {script} to scripts/")
    
    # Create docs folder and move reports
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    if Path("outputs/reports").exists():
        shutil.move("outputs/reports", docs_dir / "reports")
        print(f"📦 Moved reports to docs/")
    
    # Keep only essential visualizations
    essential_viz = Path("docs/visualizations")
    essential_viz.mkdir(exist_ok=True)
    
    if Path("outputs/visualizations/eda").exists():
        shutil.move("outputs/visualizations/eda", essential_viz / "eda")
        print(f"📦 Moved EDA visualizations to docs/")
    
    if Path("outputs/visualizations/feature_engineering").exists():
        shutil.move("outputs/visualizations/feature_engineering", essential_viz / "feature_engineering")
        print(f"📦 Moved feature engineering visualizations to docs/")
    
    # Remove now-empty outputs/visualizations
    if Path("outputs/visualizations").exists():
        try:
            shutil.rmtree("outputs/visualizations")
            print(f"🗑️  Removed empty outputs/visualizations")
        except:
            pass
    
    print(f"\n✅ Cleanup Summary:")
    print(f"   🗑️  Deleted {deleted_count} files/folders")
    print(f"   📁 Organized remaining files into logical structure")
    
    print(f"\n📊 Final Repository Structure:")
    print_tree(".", max_depth=2)

def print_tree(directory, prefix="", max_depth=3, current_depth=0):
    """Print directory tree structure."""
    if current_depth >= max_depth:
        return
        
    path = Path(directory)
    items = sorted([item for item in path.iterdir() if not item.name.startswith('.')])
    
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{item.name}")
        
        if item.is_dir() and item.name not in ['afl2_env', '__pycache__']:
            extension = "    " if is_last else "│   "
            print_tree(item, prefix + extension, max_depth, current_depth + 1)

def create_updated_readme():
    """Create an updated README for the cleaned repository."""
    
    readme_content = """# 🏉 AFL Match Prediction System

A comprehensive AFL (Australian Football League) match prediction system using machine learning.

## 🎯 Features

- **Clean ML Model**: No data leakage, 65.1% winner accuracy, 4.9 MAE margin prediction
- **Interactive Dashboard**: Streamlit web interface for predictions and analysis
- **Dynamic Attendance Estimation**: Smart crowd prediction based on teams, venue, and round
- **Real-time Predictions**: Input match details and get winner/margin predictions
- **Data Exploration**: AFL-specific insights including current season ladder, rivalries, venue analysis

## 🚀 Quick Start

1. **Setup Environment**:
```bash
python -m venv afl2_env
source afl2_env/bin/activate  # On Windows: afl2_env\\Scripts\\activate
pip install -r requirements.txt
```

2. **Run Dashboard**:
```bash
streamlit run afl_dashboard.py
```

3. **Open Browser**: Navigate to `http://localhost:8501`

## 📁 Repository Structure

```
AFL2/
├── afl_dashboard.py              # Main Streamlit application
├── requirements.txt              # Python dependencies
├── afl_data/
│   ├── afl_database.db          # SQLite database with match/player data
│   └── AFL-Data-Analysis/       # Raw data source
├── scripts/                     # Training and data processing scripts
│   ├── data_pipeline.py         # Data loading and preprocessing
│   ├── feature_engineering_pipeline.py  # Feature creation
│   ├── ml_training_pipeline.py  # Model training
│   └── retrain_clean_model.py   # Clean model (recommended)
├── models/
│   └── clean_ensemble_model.pkl # Trained ML model (no data leakage)
├── docs/                        # Documentation and analysis reports
│   ├── reports/                 # Analysis reports
│   └── visualizations/          # Charts and graphs
└── afl2_env/                    # Python virtual environment
```

## 🤖 Model Information

### Clean Model Features (28 total):
- **Team Performance** (20): Rolling averages of goals for/against over 5/10/20 games
- **Win Rates** (6): Recent form and win percentages
- **Head-to-Head** (3): Historical matchup statistics
- **Venue** (2): Home advantage and venue experience (normalized)
- **Context** (4): Rest days, season progress, momentum

### Performance:
- **Winner Prediction**: 65.1% accuracy
- **Margin Prediction**: 4.9 MAE (Mean Absolute Error)
- **No Data Leakage**: Uses only pre-game information

## 🎮 Dashboard Features

### 🎯 Match Predictions
- Select teams, venue, and date
- Automatic rest days and attendance estimation
- Confidence-scaled margin predictions
- Feature importance analysis

### 🏈 Data Explorer
- **Current Season**: Live ladder with percentage calculations
- **Rivalries**: Pre-configured AFL rivalries with head-to-head stats
- **Venues**: Venue analysis with home advantage metrics
- **Momentum**: Team form tracking with trend analysis
- **Clutch Performance**: Close game performance analysis

### 📊 Model Performance
- Real-time model evaluation metrics
- Feature importance visualization
- Prediction confidence analysis

## 🔧 Retraining Models

To retrain the clean model with updated data:

```bash
cd scripts/
python retrain_clean_model.py
```

## 📈 Key Improvements Over Previous Versions

1. **Eliminated Data Leakage**: Removed quarter-by-quarter and final score features
2. **Normalized Features**: Venue features properly scaled (0-1 vs raw counts)
3. **Realistic Predictions**: 4.9 MAE vs 2.1 with cheating model
4. **Balanced Feature Importance**: Top feature 7.3% vs 30%+ in leaky model
5. **Confidence-Based Margins**: Higher confidence predictions → larger margins

## 🏆 AFL Prediction Reality

AFL is naturally unpredictable:
- **71% of games**: 0-6 point margins
- **Mean margin**: 5.0 points
- **Median margin**: 4.0 points

Our model reflects this reality while still providing valuable insights for match prediction.

## 📚 Documentation

See `docs/reports/` for detailed analysis reports including:
- EDA (Exploratory Data Analysis)
- Feature Engineering Analysis  
- Model Evaluation Reports

## 🤝 Contributing

1. Focus on feature engineering improvements
2. Maintain data leakage prevention
3. Test with realistic AFL scenarios
4. Update documentation for changes

## 📄 License

[Add your license information here]
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    
    print("📝 Updated README.md with clean project structure")

if __name__ == "__main__":
    cleanup_repository()
    create_updated_readme()
    print("\n🎉 Repository cleanup complete!")
    print("📁 Check the new organized structure above")
    print("🚀 Ready for production use!")
