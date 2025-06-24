"""
Phase 2B: Statistical Distribution Modeling Framework
AFL Prediction Model - Statistical Modeling Implementation

This script implements a comprehensive statistical modeling framework including:
1. Distribution fitting with goodness-of-fit testing
2. Hierarchical modeling for player/team/era levels
3. Team performance distribution aggregation
4. Uncertainty quantification with Monte Carlo methods
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import norm, gamma, beta, expon, weibull_min, lognorm, poisson, nbinom
from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class StatisticalModelingFramework:
    """
    Comprehensive statistical modeling framework for AFL data.
    """
    
    def __init__(self, db_path="afl_data/afl_database.db"):
        self.db_path = db_path
        self.players_df = None
        self.matches_df = None
        self.distribution_fits = {}
        self.hierarchical_models = {}
        self.team_distributions = {}
        self.uncertainty_quantification = {}
        
    def load_data(self):
        """Load player and match data from database."""
        print("Loading data for statistical modeling...")
        
        conn = sqlite3.connect(self.db_path)
        
        # Load player data with key statistics
        self.players_df = pd.read_sql_query("""
            SELECT * FROM players 
            WHERE year >= 1991 
            AND disposals IS NOT NULL
            AND kicks IS NOT NULL
            AND marks IS NOT NULL
            AND handballs IS NOT NULL
            ORDER BY year, team
        """, conn)
        
        # Load match data
        self.matches_df = pd.read_sql_query("""
            SELECT * FROM matches 
            WHERE year >= 1991 
            ORDER BY year, date
        """, conn)
        
        conn.close()
        
        print(f"Loaded {len(self.players_df)} player records and {len(self.matches_df)} matches")
        
    def fit_distributions(self):
        """Fit appropriate distributions to each player statistic."""
        print("Fitting distributions to player statistics...")
        
        # Key statistics to model
        key_stats = ['disposals', 'kicks', 'marks', 'handballs', 'goals', 'tackles']
        
        for stat in key_stats:
            if stat in self.players_df.columns:
                data = self.players_df[stat].dropna()
                
                if len(data) > 0:
                    print(f"\nFitting distributions for {stat}...")
                    
                    # Fit multiple distributions
                    fits = self._fit_multiple_distributions(data, stat)
                    
                    # Test goodness of fit
                    best_fit = self._test_goodness_of_fit(data, fits, stat)
                    
                    self.distribution_fits[stat] = {
                        'data': data,
                        'fits': fits,
                        'best_fit': best_fit,
                        'summary_stats': {
                            'mean': data.mean(),
                            'std': data.std(),
                            'skew': data.skew(),
                            'kurtosis': data.kurtosis(),
                            'min': data.min(),
                            'max': data.max()
                        }
                    }
    
    def _fit_multiple_distributions(self, data, stat_name):
        """Fit multiple distributions to the data."""
        fits = {}
        
        # Handle different data characteristics
        if stat_name in ['goals']:
            # Discrete data - try Poisson and Negative Binomial
            try:
                lambda_pois = data.mean()
                fits['poisson'] = {'dist': poisson, 'params': {'mu': lambda_pois}}
            except:
                pass
            
            try:
                # Negative binomial parameters
                mean_val = data.mean()
                var_val = data.var()
                if var_val > mean_val:
                    p = mean_val / var_val
                    n = mean_val * p / (1 - p)
                    fits['negative_binomial'] = {'dist': nbinom, 'params': {'n': n, 'p': p}}
            except:
                pass
                
        else:
            # Continuous data - try various distributions
            try:
                # Normal distribution
                mu, sigma = norm.fit(data)
                fits['normal'] = {'dist': norm, 'params': {'loc': mu, 'scale': sigma}}
            except:
                pass
            
            try:
                # Gamma distribution
                a, loc, scale = gamma.fit(data)
                fits['gamma'] = {'dist': gamma, 'params': {'a': a, 'loc': loc, 'scale': scale}}
            except:
                pass
            
            try:
                # Log-normal distribution
                s, loc, scale = lognorm.fit(data)
                fits['lognormal'] = {'dist': lognorm, 'params': {'s': s, 'loc': loc, 'scale': scale}}
            except:
                pass
            
            try:
                # Weibull distribution
                c, loc, scale = weibull_min.fit(data)
                fits['weibull'] = {'dist': weibull_min, 'params': {'c': c, 'loc': loc, 'scale': scale}}
            except:
                pass
        
        return fits
    
    def _test_goodness_of_fit(self, data, fits, stat_name):
        """Test goodness of fit for each distribution."""
        best_fit = None
        best_score = float('inf')
        
        for dist_name, fit_info in fits.items():
            try:
                dist = fit_info['dist']
                params = fit_info['params']
                
                # Perform Kolmogorov-Smirnov test
                if hasattr(dist, 'cdf'):
                    ks_stat, p_value = stats.kstest(data, dist.cdf, args=tuple(params.values()))
                    
                    # For discrete distributions, use chi-square test
                    if dist_name in ['poisson', 'negative_binomial']:
                        # Create histogram
                        hist, bins = np.histogram(data, bins='auto', density=True)
                        bin_centers = (bins[:-1] + bins[1:]) / 2
                        
                        # Calculate expected frequencies
                        expected = dist.pmf(bin_centers, **params)
                        observed = hist
                        
                        # Chi-square test
                        chi2_stat, p_value = stats.chisquare(observed, expected)
                        test_stat = chi2_stat
                    else:
                        test_stat = ks_stat
                    
                    if test_stat < best_score:
                        best_score = test_stat
                        best_fit = {
                            'distribution': dist_name,
                            'params': params,
                            'test_statistic': test_stat,
                            'p_value': p_value
                        }
                        
            except Exception as e:
                print(f"Error testing {dist_name} for {stat_name}: {e}")
                continue
        
        return best_fit
    
    def create_hierarchical_models(self):
        """Create hierarchical models for player/team/era levels."""
        print("Creating hierarchical models...")
        
        key_stats = ['disposals', 'kicks', 'marks', 'handballs']
        
        for stat in key_stats:
            if stat in self.players_df.columns:
                print(f"Creating hierarchical model for {stat}...")
                
                # Team-level analysis (since we don't have individual player names)
                team_stats = self.players_df.groupby('team')[stat].agg(['mean', 'std', 'count']).reset_index()
                
                # Era-level analysis
                self.players_df['decade'] = (self.players_df['year'] // 10) * 10
                era_stats = self.players_df.groupby('decade')[stat].agg(['mean', 'std', 'count']).reset_index()
                
                # Round-level analysis (as proxy for player-level)
                round_stats = self.players_df.groupby('round')[stat].agg(['mean', 'std', 'count']).reset_index()
                
                hierarchical_model = {
                    'statistic': stat,
                    'round_level': {
                        'data': round_stats,
                        'mean': round_stats['mean'].mean(),
                        'std': round_stats['mean'].std(),
                        'n_rounds': len(round_stats)
                    },
                    'team_level': {
                        'data': team_stats,
                        'mean': team_stats['mean'].mean(),
                        'std': team_stats['mean'].std(),
                        'n_teams': len(team_stats)
                    },
                    'era_level': {
                        'data': era_stats,
                        'mean': era_stats['mean'].mean(),
                        'std': era_stats['mean'].std(),
                        'n_eras': len(era_stats)
                    }
                }
                
                self.hierarchical_models[stat] = hierarchical_model
    
    def create_team_performance_distributions(self):
        """Create team performance distributions from player aggregations."""
        print("Creating team performance distributions...")
        
        # Aggregate player statistics to team level
        team_aggregations = {}
        
        key_stats = ['disposals', 'kicks', 'marks', 'handballs', 'goals', 'tackles']
        
        for stat in key_stats:
            if stat in self.players_df.columns:
                # Team-level aggregations
                team_stats = self.players_df.groupby(['team', 'year'])[stat].agg([
                    'sum', 'mean', 'std', 'count'
                ]).reset_index()
                
                team_aggregations[stat] = team_stats
        
        # Create team performance models
        for stat, team_data in team_aggregations.items():
            print(f"\nModeling team {stat} distributions...")
            
            # Fit distributions to team-level data
            team_sums = team_data['sum'].dropna()
            team_means = team_data['mean'].dropna()
            
            if len(team_sums) > 0:
                # Fit distributions to team totals
                sum_fits = self._fit_multiple_distributions(team_sums, f'team_{stat}_total')
                best_sum_fit = self._test_goodness_of_fit(team_sums, sum_fits, f'team_{stat}_total')
                
                # Fit distributions to team averages
                mean_fits = self._fit_multiple_distributions(team_means, f'team_{stat}_average')
                best_mean_fit = self._test_goodness_of_fit(team_means, mean_fits, f'team_{stat}_average')
                
                self.team_distributions[stat] = {
                    'team_totals': {
                        'data': team_sums,
                        'fits': sum_fits,
                        'best_fit': best_sum_fit
                    },
                    'team_averages': {
                        'data': team_means,
                        'fits': mean_fits,
                        'best_fit': best_mean_fit
                    },
                    'summary': {
                        'n_teams': len(team_data['team'].unique()),
                        'n_years': len(team_data['year'].unique()),
                        'total_observations': len(team_data)
                    }
                }
    
    def implement_uncertainty_quantification(self):
        """Implement uncertainty quantification with Monte Carlo methods."""
        print("Implementing uncertainty quantification...")
        
        mc_results = {}
        
        for stat, fit_info in self.distribution_fits.items():
            print(f"Quantifying uncertainty for {stat}...")
            
            data = fit_info['data']
            best_fit = fit_info['best_fit']
            
            if best_fit:
                # Bootstrap confidence intervals for parameters
                n_bootstrap = 1000
                bootstrap_params = []
                
                for _ in range(n_bootstrap):
                    bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
                    
                    try:
                        dist = best_fit['distribution']
                        if dist == 'normal':
                            mu, sigma = norm.fit(bootstrap_sample)
                            bootstrap_params.append({'loc': mu, 'scale': sigma})
                        elif dist == 'gamma':
                            a, loc, scale = gamma.fit(bootstrap_sample)
                            bootstrap_params.append({'a': a, 'loc': loc, 'scale': scale})
                        elif dist == 'poisson':
                            lambda_val = bootstrap_sample.mean()
                            bootstrap_params.append({'mu': lambda_val})
                    except:
                        continue
                
                if bootstrap_params:
                    # Calculate confidence intervals
                    param_ci = {}
                    for param_name in bootstrap_params[0].keys():
                        param_values = [params[param_name] for params in bootstrap_params]
                        param_ci[param_name] = {
                            'mean': np.mean(param_values),
                            'std': np.std(param_values),
                            'ci_95_lower': np.percentile(param_values, 2.5),
                            'ci_95_upper': np.percentile(param_values, 97.5)
                        }
                    
                    mc_results[stat] = {
                        'bootstrap_params': bootstrap_params,
                        'confidence_intervals': param_ci,
                        'n_bootstrap': n_bootstrap
                    }
        
        self.uncertainty_quantification = {
            'monte_carlo_results': mc_results
        }
    
    def create_distribution_visualizations(self):
        """Create visualizations for distribution fits and hierarchical models."""
        print("Creating distribution visualizations...")
        
        # Create subplots for each statistic
        key_stats = ['disposals', 'kicks', 'marks', 'handballs', 'goals', 'tackles']
        n_stats = len([stat for stat in key_stats if stat in self.distribution_fits])
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Distribution Fitting Results for AFL Player Statistics', fontsize=16)
        
        axes = axes.flatten()
        plot_idx = 0
        
        for stat in key_stats:
            if stat in self.distribution_fits:
                ax = axes[plot_idx]
                
                fit_info = self.distribution_fits[stat]
                data = fit_info['data']
                best_fit = fit_info['best_fit']
                
                # Plot histogram of data
                ax.hist(data, bins=30, density=True, alpha=0.7, label='Data', color='lightblue')
                
                # Plot best fit distribution
                if best_fit:
                    x = np.linspace(data.min(), data.max(), 100)
                    dist_name = best_fit['distribution']
                    params = best_fit['params']
                    
                    if dist_name == 'normal':
                        y = norm.pdf(x, **params)
                    elif dist_name == 'gamma':
                        y = gamma.pdf(x, **params)
                    elif dist_name == 'poisson':
                        y = poisson.pmf(x.astype(int), **params)
                    elif dist_name == 'lognormal':
                        y = lognorm.pdf(x, **params)
                    else:
                        y = np.zeros_like(x)
                    
                    ax.plot(x, y, 'r-', linewidth=2, label=f'Best Fit: {dist_name}')
                
                ax.set_title(f'{stat.title()} Distribution')
                ax.set_xlabel(stat.title())
                ax.set_ylabel('Density')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                plot_idx += 1
        
        # Remove empty subplots
        for i in range(plot_idx, len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('distribution_fitting_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create hierarchical model visualization
        self._create_hierarchical_visualization()
        
        # Create uncertainty quantification visualization
        self._create_uncertainty_visualization()
    
    def _create_hierarchical_visualization(self):
        """Create visualization for hierarchical models."""
        print("Creating hierarchical model visualization...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Hierarchical Model Structure', fontsize=16)
        
        key_stats = ['disposals', 'kicks', 'marks', 'handballs']
        
        for i, stat in enumerate(key_stats):
            if stat in self.hierarchical_models:
                ax = axes[i // 2, i % 2]
                
                model = self.hierarchical_models[stat]
                
                # Plot means at different levels
                levels = ['round_level', 'team_level', 'era_level']
                means = [model[level]['mean'] for level in levels]
                stds = [model[level]['std'] for level in levels]
                
                x_pos = np.arange(len(levels))
                ax.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
                ax.set_xticks(x_pos)
                ax.set_xticklabels(['Round', 'Team', 'Era'])
                ax.set_title(f'{stat.title()} - Hierarchical Means')
                ax.set_ylabel('Mean Value')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('hierarchical_model_structure.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_uncertainty_visualization(self):
        """Create visualization for uncertainty quantification."""
        print("Creating uncertainty quantification visualization...")
        
        if not self.uncertainty_quantification:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Uncertainty Quantification Results', fontsize=16)
        
        # Plot 1: Parameter confidence intervals
        ax1 = axes[0, 0]
        stats_with_ci = [stat for stat in ['disposals', 'kicks', 'marks', 'handballs'] 
                        if stat in self.uncertainty_quantification['monte_carlo_results']]
        
        if stats_with_ci:
            stat_names = []
            ci_widths = []
            
            for stat in stats_with_ci:
                mc_result = self.uncertainty_quantification['monte_carlo_results'][stat]
                if 'confidence_intervals' in mc_result:
                    for param, ci in mc_result['confidence_intervals'].items():
                        stat_names.append(f'{stat}_{param}')
                        ci_widths.append(ci['ci_95_upper'] - ci['ci_95_lower'])
            
            if ci_widths:
                ax1.bar(range(len(ci_widths)), ci_widths)
                ax1.set_xticks(range(len(stat_names)))
                ax1.set_xticklabels(stat_names, rotation=45)
                ax1.set_title('95% Confidence Interval Widths')
                ax1.set_ylabel('CI Width')
                ax1.grid(True, alpha=0.3)
        
        # Plot 2: Prediction uncertainty
        ax2 = axes[0, 1]
        if 'uncertainty_propagation' in self.uncertainty_quantification:
            up = self.uncertainty_quantification['uncertainty_propagation']
            stats_with_pred = list(up.keys())
            
            if stats_with_pred:
                means = [up[stat]['prediction_mean'] for stat in stats_with_pred]
                stds = [up[stat]['prediction_std'] for stat in stats_with_pred]
                
                ax2.bar(stats_with_pred, means, yerr=stds, capsize=5, alpha=0.7)
                ax2.set_title('Prediction Uncertainty')
                ax2.set_ylabel('Predicted Mean')
                ax2.grid(True, alpha=0.3)
        
        # Plot 3: Bootstrap parameter distributions
        ax3 = axes[1, 0]
        if stats_with_ci and 'disposals' in self.uncertainty_quantification['monte_carlo_results']:
            mc_result = self.uncertainty_quantification['monte_carlo_results']['disposals']
            if 'bootstrap_params' in mc_result:
                loc_values = [params.get('loc', 0) for params in mc_result['bootstrap_params']]
                ax3.hist(loc_values, bins=30, alpha=0.7)
                ax3.set_title('Bootstrap Distribution - Disposals Location')
                ax3.set_xlabel('Location Parameter')
                ax3.set_ylabel('Frequency')
                ax3.grid(True, alpha=0.3)
        
        # Plot 4: Monte Carlo convergence
        ax4 = axes[1, 1]
        if 'uncertainty_propagation' in self.uncertainty_quantification and 'disposals' in up:
            predictions = up['disposals']['predictions']
            cumulative_mean = np.cumsum(predictions) / np.arange(1, len(predictions) + 1)
            ax4.plot(cumulative_mean)
            ax4.axhline(y=up['disposals']['prediction_mean'], color='r', linestyle='--', label='Final Mean')
            ax4.set_title('Monte Carlo Convergence - Disposals')
            ax4.set_xlabel('Simulation Number')
            ax4.set_ylabel('Cumulative Mean')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('uncertainty_quantification_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def run_statistical_modeling(self):
        """Run the complete statistical modeling framework."""
        print("Starting Statistical Distribution Modeling Framework...")
        
        # Load data
        self.load_data()
        
        # Implement all components
        self.fit_distributions()
        self.create_hierarchical_models()
        self.create_team_performance_distributions()
        self.implement_uncertainty_quantification()
        
        # Create visualizations
        self.create_distribution_visualizations()
        
        # Generate summary report
        self._generate_summary_report()
        
        print(f"\nStatistical modeling complete!")
        print(f"Distributions fitted: {len(self.distribution_fits)}")
        print(f"Hierarchical models: {len(self.hierarchical_models)}")
        print(f"Team distributions: {len(self.team_distributions)}")
        print(f"Uncertainty quantification: {len(self.uncertainty_quantification.get('monte_carlo_results', {}))}")
        
        return {
            'distribution_fits': self.distribution_fits,
            'hierarchical_models': self.hierarchical_models,
            'team_distributions': self.team_distributions,
            'uncertainty_quantification': self.uncertainty_quantification
        }
    
    def _generate_summary_report(self):
        """Generate a summary report of the statistical modeling results."""
        print("\nGenerating summary report...")
        
        summary = {
            'distribution_fitting': {},
            'hierarchical_modeling': {},
            'team_performance': {},
            'uncertainty_quantification': {}
        }
        
        # Distribution fitting summary
        for stat, fit_info in self.distribution_fits.items():
            summary['distribution_fitting'][stat] = {
                'best_distribution': fit_info['best_fit']['distribution'] if fit_info['best_fit'] else 'None',
                'test_statistic': fit_info['best_fit']['test_statistic'] if fit_info['best_fit'] else None,
                'p_value': fit_info['best_fit']['p_value'] if fit_info['best_fit'] else None,
                'data_points': len(fit_info['data']),
                'mean': fit_info['summary_stats']['mean'],
                'std': fit_info['summary_stats']['std']
            }
        
        # Hierarchical modeling summary
        for stat, model in self.hierarchical_models.items():
            summary['hierarchical_modeling'][stat] = {
                'round_level_n': model['round_level']['n_rounds'],
                'team_level_n': model['team_level']['n_teams'],
                'era_level_n': model['era_level']['n_eras'],
                'round_mean': model['round_level']['mean'],
                'team_mean': model['team_level']['mean'],
                'era_mean': model['era_level']['mean']
            }
        
        # Team performance summary
        for stat, team_info in self.team_distributions.items():
            summary['team_performance'][stat] = {
                'n_teams': team_info['summary']['n_teams'],
                'n_years': team_info['summary']['n_years'],
                'total_observations': team_info['summary']['total_observations'],
                'best_fit_totals': team_info['team_totals']['best_fit']['distribution'] if team_info['team_totals']['best_fit'] else 'None',
                'best_fit_averages': team_info['team_averages']['best_fit']['distribution'] if team_info['team_averages']['best_fit'] else 'None'
            }
        
        # Uncertainty quantification summary
        if self.uncertainty_quantification:
            summary['uncertainty_quantification'] = {
                'monte_carlo_stats': len(self.uncertainty_quantification.get('monte_carlo_results', {})),
                'uncertainty_propagation_stats': len(self.uncertainty_quantification.get('uncertainty_propagation', {}))
            }
        
        # Save summary to file
        import json
        with open('statistical_modeling_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print("Summary report saved to 'statistical_modeling_summary.json'")

def main():
    """Main function to run the statistical modeling framework."""
    framework = StatisticalModelingFramework()
    results = framework.run_statistical_modeling()
    
    print("\nStatistical Modeling Framework Complete!")
    print(f"Results saved to:")
    print("  - distribution_fitting_results.png")
    print("  - hierarchical_model_structure.png")
    print("  - uncertainty_quantification_results.png")
    print("  - statistical_modeling_summary.json")

if __name__ == "__main__":
    main() 