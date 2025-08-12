#!/usr/bin/env python3
"""
Test script for the enhanced scraping pipeline.
Validates data completeness and quality compared to existing data.
"""

import pandas as pd
import sqlite3
import numpy as np
from pathlib import Path
import logging
from enhanced_data_pipeline import AFLScrapingPipeline
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScrapingPipelineValidator:
    """Validates the enhanced scraping pipeline against existing data."""
    
    def __init__(self, data_dir: str = "afl_data"):
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / "afl_database.db"
        self.test_results = {}
        
    def backup_existing_data(self):
        """Backup existing database before testing."""
        backup_path = self.db_path.with_suffix('.backup.db')
        
        if self.db_path.exists():
            import shutil
            shutil.copy2(self.db_path, backup_path)
            logger.info(f"Backed up existing database to {backup_path}")
            return backup_path
        return None
    
    def get_existing_data_stats(self):
        """Get statistics from existing data."""
        if not self.db_path.exists():
            return None
        
        conn = sqlite3.connect(self.db_path)
        
        stats = {
            'matches': {
                'total_count': pd.read_sql_query("SELECT COUNT(*) as count FROM matches", conn).iloc[0]['count'],
                'year_range': pd.read_sql_query("SELECT MIN(year) as min_year, MAX(year) as max_year FROM matches", conn).to_dict('records')[0],
                'teams_count': pd.read_sql_query("SELECT COUNT(DISTINCT home_team) as count FROM matches", conn).iloc[0]['count'],
                'venues_count': pd.read_sql_query("SELECT COUNT(DISTINCT venue) as count FROM matches WHERE venue IS NOT NULL", conn).iloc[0]['count']
            },
            'players': {
                'total_count': pd.read_sql_query("SELECT COUNT(*) as count FROM players", conn).iloc[0]['count'],
                'unique_teams': pd.read_sql_query("SELECT COUNT(DISTINCT team) as count FROM players WHERE team IS NOT NULL", conn).iloc[0]['count'],
                'year_range': pd.read_sql_query("SELECT MIN(year) as min_year, MAX(year) as max_year FROM players WHERE year IS NOT NULL", conn).to_dict('records')[0]
            }
        }
        
        conn.close()
        return stats
    
    def test_small_scrape(self, test_year: int = 2023):
        """Test scraping a small dataset (single year) for validation."""
        logger.info(f"Testing scraping for year {test_year}...")
        
        # Create test pipeline
        pipeline = AFLScrapingPipeline(self.data_dir)
        
        # Create temporary test database
        test_db_path = self.data_dir / f"test_scraping_{test_year}.db"
        pipeline.db_path = test_db_path
        pipeline.init_database()
        
        try:
            # Run pipeline for single year
            report = pipeline.run_enhanced_pipeline(
                start_year=test_year,
                end_year=test_year,
                scrape_matches=True,
                scrape_players=False,  # Start with matches only
                verify_with_existing=False
            )
            
            # Analyze results
            test_results = self.analyze_test_results(test_db_path, test_year)
            test_results['scraping_report'] = report
            
            return test_results
            
        except Exception as e:
            logger.error(f"Test scraping failed: {e}")
            return {'success': False, 'error': str(e)}
        
        finally:
            # Cleanup test database
            if test_db_path.exists():
                test_db_path.unlink()
    
    def analyze_test_results(self, test_db_path: Path, test_year: int):
        """Analyze the results of test scraping."""
        
        if not test_db_path.exists():
            return {'success': False, 'error': 'Test database not found'}
        
        conn = sqlite3.connect(test_db_path)
        
        try:
            # Get scraped data
            matches_df = pd.read_sql_query(f"SELECT * FROM matches WHERE year = {test_year}", conn)
            
            analysis = {
                'success': True,
                'year': test_year,
                'scraped_matches': len(matches_df),
                'data_quality': self.assess_data_quality(matches_df),
                'completeness_check': self.check_data_completeness(matches_df, test_year),
                'sample_matches': matches_df.head(5).to_dict('records') if not matches_df.empty else []
            }
            
            return analysis
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
        
        finally:
            conn.close()
    
    def assess_data_quality(self, matches_df: pd.DataFrame):
        """Assess the quality of scraped data."""
        if matches_df.empty:
            return {'overall_score': 0.0, 'issues': ['No data scraped']}
        
        quality_checks = {
            'has_teams': (matches_df['home_team'].notna() & matches_df['away_team'].notna()).mean(),
            'has_dates': matches_df['date'].notna().mean(),
            'has_venues': matches_df['venue'].notna().mean(),
            'has_scores': (matches_df['home_total_goals'].notna() & matches_df['away_total_goals'].notna()).mean(),
            'valid_scores': ((matches_df['home_total_goals'] >= 0) & (matches_df['away_total_goals'] >= 0)).mean() if 'home_total_goals' in matches_df.columns else 0.0
        }
        
        # Calculate overall quality score
        overall_score = np.mean(list(quality_checks.values()))
        
        # Identify issues
        issues = []
        if quality_checks['has_teams'] < 1.0:
            issues.append('Missing team information')
        if quality_checks['has_dates'] < 0.9:
            issues.append('Missing date information')
        if quality_checks['has_scores'] < 0.8:
            issues.append('Missing score information')
        
        return {
            'overall_score': overall_score,
            'individual_checks': quality_checks,
            'issues': issues
        }
    
    def check_data_completeness(self, matches_df: pd.DataFrame, year: int):
        """Check if scraped data is complete for the given year."""
        
        # Expected number of matches per year (roughly)
        expected_ranges = {
            'modern': (185, 210),  # 2000+
            'mid': (120, 180),     # 1970-1999
            'early': (60, 120)     # Before 1970
        }
        
        if year >= 2000:
            expected_min, expected_max = expected_ranges['modern']
        elif year >= 1970:
            expected_min, expected_max = expected_ranges['mid']
        else:
            expected_min, expected_max = expected_ranges['early']
        
        actual_count = len(matches_df)
        
        completeness = {
            'expected_range': (expected_min, expected_max),
            'actual_count': actual_count,
            'completeness_score': min(1.0, actual_count / expected_min) if expected_min > 0 else 0.0,
            'within_expected_range': expected_min <= actual_count <= expected_max
        }
        
        return completeness
    
    def compare_scraping_vs_repository(self, test_year: int = 2023):
        """Compare scraping results with repository data for the same year."""
        logger.info(f"Comparing scraping vs repository data for {test_year}...")
        
        # Get data from current database (repository source)
        if not self.db_path.exists():
            return {'error': 'No existing database to compare against'}
        
        conn = sqlite3.connect(self.db_path)
        repo_data = pd.read_sql_query(f"SELECT * FROM matches WHERE year = {test_year}", conn)
        conn.close()
        
        # Get scraped data
        scraping_results = self.test_small_scrape(test_year)
        
        if not scraping_results.get('success', False):
            return {'error': 'Scraping test failed', 'scraping_results': scraping_results}
        
        comparison = {
            'repository_matches': len(repo_data),
            'scraped_matches': scraping_results['scraped_matches'],
            'scraping_quality': scraping_results['data_quality']['overall_score'],
            'recommendation': self.get_recommendation(repo_data, scraping_results)
        }
        
        return comparison
    
    def get_recommendation(self, repo_data: pd.DataFrame, scraping_results: dict):
        """Provide recommendation based on comparison."""
        
        repo_count = len(repo_data)
        scraped_count = scraping_results['scraped_matches']
        quality_score = scraping_results['data_quality']['overall_score']
        
        if quality_score < 0.7:
            return "Use repository data - scraping quality too low"
        elif scraped_count > repo_count * 1.1:
            return "Use scraping - more complete data"
        elif scraped_count < repo_count * 0.8:
            return "Use repository data - scraping incomplete"
        else:
            return "Either source acceptable - consider using scraping for latest data"
    
    def run_comprehensive_validation(self):
        """Run comprehensive validation of scraping pipeline."""
        logger.info("Running comprehensive scraping pipeline validation...")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'existing_data_stats': self.get_existing_data_stats(),
            'test_scraping_2023': self.test_small_scrape(2023),
            'test_scraping_2022': self.test_small_scrape(2022),
            'comparison_2023': self.compare_scraping_vs_repository(2023),
            'overall_recommendation': None
        }
        
        # Generate overall recommendation
        if (validation_results['test_scraping_2023'].get('success', False) and
            validation_results['test_scraping_2023']['data_quality']['overall_score'] > 0.8):
            validation_results['overall_recommendation'] = "Scraping pipeline ready for production use"
        else:
            validation_results['overall_recommendation'] = "Continue using repository data - scraping needs improvement"
        
        # Save validation report
        report_path = self.data_dir / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(validation_results, f, indent=2, default=str)
        
        logger.info(f"Validation report saved to {report_path}")
        return validation_results

def main():
    """Main function to run validation tests."""
    
    print("\n" + "="*60)
    print("AFL SCRAPING PIPELINE VALIDATION")
    print("="*60)
    
    validator = ScrapingPipelineValidator()
    
    # Backup existing data
    backup_path = validator.backup_existing_data()
    
    try:
        # Run comprehensive validation
        results = validator.run_comprehensive_validation()
        
        # Print summary
        print(f"\nValidation Results Summary:")
        print(f"- Existing matches in DB: {results.get('existing_data_stats', {}).get('matches', {}).get('total_count', 'N/A')}")
        print(f"- Test scraping 2023: {'SUCCESS' if results['test_scraping_2023'].get('success') else 'FAILED'}")
        print(f"- Data quality score: {results['test_scraping_2023'].get('data_quality', {}).get('overall_score', 0):.2f}")
        print(f"- Overall recommendation: {results['overall_recommendation']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {'error': str(e)}

if __name__ == "__main__":
    main()
