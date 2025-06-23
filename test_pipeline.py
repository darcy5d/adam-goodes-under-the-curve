#!/usr/bin/env python3
"""
Test script for AFL Data Pipeline
Verifies pipeline components work correctly.
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_pipeline import AFLDataPipeline
from data_analysis import AFLDataAnalyzer

def test_pipeline_components():
    """Test individual pipeline components."""
    print("Testing AFL Data Pipeline Components...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Using temporary directory: {temp_dir}")
        
        # Test pipeline initialization
        try:
            pipeline = AFLDataPipeline(data_dir=temp_dir)
            print("✅ Pipeline initialization successful")
        except Exception as e:
            print(f"❌ Pipeline initialization failed: {e}")
            return False
        
        # Test database initialization
        try:
            db_path = Path(temp_dir) / "afl_database.db"
            if db_path.exists():
                print("✅ Database initialization successful")
            else:
                print("❌ Database initialization failed")
                return False
        except Exception as e:
            print(f"❌ Database initialization failed: {e}")
            return False
        
        # Test analyzer initialization
        try:
            analyzer = AFLDataAnalyzer(data_dir=temp_dir)
            print("✅ Analyzer initialization successful")
        except Exception as e:
            print(f"❌ Analyzer initialization failed: {e}")
            return False
    
    print("✅ All pipeline components working correctly")
    return True

def test_data_structures():
    """Test data structure definitions."""
    print("\nTesting Data Structures...")
    
    try:
        from data_pipeline import DataQualityMetrics
        
        # Test DataQualityMetrics
        metrics = DataQualityMetrics(
            total_records=100,
            missing_values={'col1': 5},
            invalid_values={'col2': 2},
            outliers={'col3': 1},
            duplicates=3,
            date_range=('2020-01-01', '2020-12-31'),
            team_count=18,
            venue_count=25
        )
        
        print("✅ DataQualityMetrics structure working")
        
        # Test that all fields are accessible
        assert metrics.total_records == 100
        assert len(metrics.missing_values) == 1
        assert metrics.team_count == 18
        
        print("✅ DataQualityMetrics field access working")
        
    except Exception as e:
        print(f"❌ Data structure test failed: {e}")
        return False
    
    return True

def test_imports():
    """Test that all required imports work."""
    print("\nTesting Imports...")
    
    required_modules = [
        'pandas',
        'numpy',
        'sqlite3',
        'git',
        'requests',
        'tqdm',
        'json',
        'pathlib',
        'matplotlib',
        'seaborn'
    ]
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module} imported successfully")
        except ImportError as e:
            print(f"❌ {module} import failed: {e}")
            return False
    
    return True

def main():
    """Run all tests."""
    print("="*50)
    print("AFL DATA PIPELINE TEST SUITE")
    print("="*50)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Structure Test", test_data_structures),
        ("Pipeline Component Test", test_pipeline_components)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print("\n" + "="*50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("="*50)
    
    if passed == total:
        print("🎉 All tests passed! Pipeline is ready to run.")
        print("\nNext steps:")
        print("1. Run: python data_pipeline.py")
        print("2. Run: python data_analysis.py")
        print("3. Check generated reports and visualizations")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 