#!/usr/bin/env python3
"""
run_validation_experiments.py

Run forecasting experiments across multiple synthetic datasets to validate:
1. Model robustness across different configurations
2. Forecast accuracy metrics
3. Low-data fallback behavior
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import sys

PYTHON_BIN = "/Users/P36276@intertech.com.tr/Projects/agile_release_forecast/.venv/bin/python"

CONFIGURATIONS = [
    {
        "name": "small_low_cv",
        "csv": "validation_datasets/small_low_cv.csv",
        "description": "12 sprints, 7-day, low volatility",
        "remaining_effort": 489.6
    },
    {
        "name": "small_high_cv",
        "csv": "validation_datasets/small_high_cv.csv",
        "description": "12 sprints, 7-day, high volatility",
        "remaining_effort": 417.0
    },
    {
        "name": "medium_mixed",
        "csv": "validation_datasets/medium_mixed.csv",
        "description": "24 sprints, mixed length",
        "remaining_effort": 1923.4
    },
    {
        "name": "large_stable",
        "csv": "validation_datasets/large_stable.csv",
        "description": "48 sprints, 14-day, stable",
        "remaining_effort": 4145.6
    },
    {
        "name": "large_volatile",
        "csv": "validation_datasets/large_volatile.csv",
        "description": "48 sprints, mixed, volatile",
        "remaining_effort": 4282.6
    },
    {
        "name": "xlarge_realistic",
        "csv": "validation_datasets/xlarge_realistic.csv",
        "description": "96 sprints, realistic pattern",
        "remaining_effort": 7678.6
    }
]

def compute_metrics_for_dataset(csv_path):
    """Compute basic metrics from dataset"""
    df = pd.read_csv(csv_path)
    
    # Calculate throughput
    df['calendar_days'] = (pd.to_datetime(df['end_date']) - pd.to_datetime(df['start_date'])).dt.days + 1
    df['weekend_days'] = df.apply(lambda row: sum(1 for i in range(row['calendar_days']) 
                                   if (pd.to_datetime(row['start_date']) + pd.Timedelta(days=i)).weekday() >= 5), axis=1)
    df['effective_days'] = df['calendar_days'] - df['weekend_days']
    df['net_done'] = df['velocity'] - df['scope_added']
    df['throughput'] = df['net_done'] / df['effective_days']
    
    return {
        'n_sprints': len(df),
        'total_velocity': df['velocity'].sum(),
        'mean_throughput': df['throughput'].mean(),
        'std_throughput': df['throughput'].std(),
        'cv_throughput': df['throughput'].std() / df['throughput'].mean(),
        'mean_unplanned': (df['scope_added'] / df['velocity']).mean(),
        'mean_bug_pct': df['percent_bug'].mean()
    }

def run_forecast(config):
    """Run forecast for a configuration"""
    print(f"\n{'='*80}")
    print(f"Running: {config['name']} - {config['description']}")
    print(f"{'='*80}")
    
    # Create config file
    hist_dir = f"validation_datasets/{config['name']}_history"
    forecast_config = {
        "csv_path": config['csv'],
        "total_release_effort": config['remaining_effort'],
        "effort_unit": "person_days",
        "use_features": ["prev_daily_rate", "unplanned_fraction", "percent_bug"],
        "n_sims": 5000,
        "recent_sprint_window": 6,
        "save_model_path": f"validation_datasets/{config['name']}_model.json",
        "history_path": f"{hist_dir}/model_history.jsonl",
        "output_paths": {
            "sims_csv": f"{hist_dir}/sims_output_regression.csv"
        },
        "compute_effort_metrics": {"enabled": True, "window": 6},
        "generate_plots": False
    }
    
    config_path = f"validation_datasets/{config['name']}_config.json"
    with open(config_path, 'w') as f:
        json.dump(forecast_config, f, indent=2)
    
    # Run forecast
    print(f"Running forecast_release.py...")
    cmd = [PYTHON_BIN, "forecast_release.py", "--config", config_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR running forecast: {result.stderr[:500]}")
        return None
    
    print(f"Forecast completed successfully")
    
    # Check if files exist
    model_path = forecast_config['save_model_path']
    hist_dir = f"validation_datasets/{config['name']}_history"
    sims_path = f"{hist_dir}/sims_output_regression.csv"
    
    if not Path(model_path).exists():
        print(f"ERROR: Model file not found: {model_path}")
        return None
    
    if not Path(sims_path).exists():
        print(f"ERROR: Sims file not found: {sims_path}")
        return None
    
    # Load results
    with open(model_path, 'r') as f:
        model_artifact = json.load(f)
    
    # Load simulation results
    sims_df = pd.read_csv(sims_path)
    
    # Compute dataset metrics
    dataset_metrics = compute_metrics_for_dataset(config['csv'])
    
    # Compute distribution from simulations
    percentiles = [10, 25, 50, 75, 90, 95]
    dist_days = np.percentile(sims_df['days_needed'], percentiles)
    dist_sprints = np.percentile(sims_df['sprints_needed'], percentiles)
    
    # Extract key results
    results = {
        'config_name': config['name'],
        'description': config['description'],
        'n_sprints': dataset_metrics['n_sprints'],
        'cv_throughput': dataset_metrics['cv_throughput'],
        'mean_unplanned': dataset_metrics['mean_unplanned'],
        'low_data_mode': model_artifact.get('low_data_mode', False),
        'p10_days': dist_days[0],
        'p25_days': dist_days[1],
        'p50_days': dist_days[2],
        'p75_days': dist_days[3],
        'p90_days': dist_days[4],
        'p50_sprints': dist_sprints[2],
        'p90_sprints': dist_sprints[4],
        'iqr_days': dist_days[3] - dist_days[1],
        'forecast_spread': (dist_days[4] - dist_days[0]) / dist_days[2]
    }
    
    print(f"\nResults for {config['name']}:")
    print(f"  Low Data Mode: {results['low_data_mode']}")
    print(f"  Throughput CV: {results['cv_throughput']:.3f}")
    print(f"  P50 forecast: {results['p50_days']:.1f} days ({results['p50_sprints']:.1f} sprints)")
    print(f"  P90 forecast: {results['p90_days']:.1f} days ({results['p90_sprints']:.1f} sprints)")
    print(f"  Forecast spread (P90-P10)/P50: {results['forecast_spread']:.2f}")
    
    return results

def main():
    print("Multi-Configuration Validation Experiments")
    print("=" * 80)
    
    results = []
    for config in CONFIGURATIONS:
        result = run_forecast(config)
        if result:
            results.append(result)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv("validation_datasets/validation_results.csv", index=False)
    
    print("\n" + "=" * 80)
    print("SUMMARY: All Configurations")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    # Generate summary statistics
    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)
    
    # Low data mode analysis
    low_data_configs = results_df[results_df['low_data_mode'] == True]
    print(f"\n1. Low-Data Fallback triggered in {len(low_data_configs)} configurations:")
    for _, row in low_data_configs.iterrows():
        print(f"   - {row['config_name']}: n={row['n_sprints']} sprints")
    
    # Stability analysis
    stable_configs = results_df[results_df['n_sprints'] >= 24]
    print(f"\n2. Model stability (n≥24 sprints):")
    print(f"   - Mean forecast spread: {stable_configs['forecast_spread'].mean():.2f}")
    print(f"   - Std forecast spread: {stable_configs['forecast_spread'].std():.2f}")
    
    # Volatility impact
    print(f"\n3. Volatility impact:")
    print(f"   - Low CV (small_low_cv): spread={results_df[results_df['config_name']=='small_low_cv']['forecast_spread'].values[0]:.2f}")
    print(f"   - High CV (small_high_cv): spread={results_df[results_df['config_name']=='small_high_cv']['forecast_spread'].values[0]:.2f}")
    
    print("\n✓ Validation complete. Results saved to validation_datasets/validation_results.csv")

if __name__ == "__main__":
    main()
