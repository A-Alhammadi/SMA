# debug_regime_optout.py

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# Try to import our modules
try:
    from enhanced_config import STRATEGY_CONFIG, TRAINING_START, TRAINING_END, TESTING_START, TESTING_END, CURRENCY
    from database import DatabaseHandler
    from enhanced_sma import calculate_volatility, detect_volatility_regimes
except ImportError:
    print("Error: Could not import required modules.")
    sys.exit(1)

def main():
    """Analyze regime distribution and test the opt-out functionality"""
    
    print("=== REGIME OPT-OUT DEBUGGING ===")
    print(f"Currency: {CURRENCY}")
    print(f"Training period: {TRAINING_START} to {TRAINING_END}")
    print(f"Testing period: {TESTING_START} to {TESTING_END}")
    
    # Print current opt-out settings
    regime_opt_out = STRATEGY_CONFIG['regime_detection'].get('regime_opt_out', {})
    print("\nCurrent regime opt-out settings:")
    for regime, opt_out in regime_opt_out.items():
        print(f"Regime {regime}: {'Opt-out enabled' if opt_out else 'Trading enabled'}")
    
    try:
        # Initialize database connection
        db = DatabaseHandler()
        
        # Fetch training data
        print(f"\nFetching training data from {TRAINING_START} to {TRAINING_END}...")
        train_df = db.get_historical_data(CURRENCY, TRAINING_START, TRAINING_END)
        
        # Fetch test data
        print(f"Fetching testing data from {TESTING_START} to {TESTING_END}...")
        test_df = db.get_historical_data(CURRENCY, TESTING_START, TESTING_END)
        
        # Check data availability
        if len(train_df) < 100 or len(test_df) < 20:
            print("Not enough data for analysis")
            db.close()
            return
        
        # Calculate volatility for training data
        volatility_method = STRATEGY_CONFIG['volatility']['methods'][0]  # Use first method
        vol_lookback = STRATEGY_CONFIG['volatility']['lookback_periods'][0]  # Use first lookback
        
        print(f"\nCalculating volatility using {volatility_method} method...")
        train_vol = calculate_volatility(train_df, method=volatility_method, window=vol_lookback)
        test_vol = calculate_volatility(test_df, method=volatility_method, window=vol_lookback)
        
        # Detect regimes for training data
        print("Detecting volatility regimes...")
        train_regimes = detect_volatility_regimes(
            train_df, 
            train_vol, 
            method=STRATEGY_CONFIG['regime_detection']['method'],
            n_regimes=STRATEGY_CONFIG['regime_detection']['n_regimes'],
            stability_period=STRATEGY_CONFIG['regime_detection']['regime_stability_period']
        )
        
        # Detect regimes for test data (using the same process for consistency)
        test_regimes = detect_volatility_regimes(
            test_df, 
            test_vol, 
            method=STRATEGY_CONFIG['regime_detection']['method'],
            n_regimes=STRATEGY_CONFIG['regime_detection']['n_regimes'],
            stability_period=STRATEGY_CONFIG['regime_detection']['regime_stability_period']
        )
        
        # Analyze regime distributions
        analyze_regimes("Training", train_regimes)
        analyze_regimes("Testing", test_regimes)
        
        # Calculate percentage of periods affected by opt-out
        total_periods = len(test_regimes)
        affected_periods = 0
        
        for regime, opt_out in regime_opt_out.items():
            if opt_out:
                regime_count = (test_regimes == regime).sum()
                affected_periods += regime_count
                percentage = (regime_count / total_periods) * 100
                print(f"\nRegime {regime} is set to opt-out and appears in {regime_count} periods ({percentage:.2f}% of test data)")
        
        overall_affected = (affected_periods / total_periods) * 100
        print(f"\nOpt-out affects {affected_periods} out of {total_periods} periods ({overall_affected:.2f}% of test data)")
        
        if overall_affected > 95:
            print("\nWARNING: Your opt-out settings affect more than 95% of the test period!")
            print("This is likely why you're seeing 0 trades.")
        
        # Plot regime distributions
        plot_regime_distributions(train_regimes, test_regimes, regime_opt_out)
        
        # Close database connection
        db.close()
        
    except Exception as e:
        print(f"Error in analysis: {e}")
        import traceback
        traceback.print_exc()

def analyze_regimes(period_name, regimes):
    """Analyze and print regime distribution"""
    
    total = len(regimes)
    counts = regimes.value_counts().sort_index()
    
    print(f"\n{period_name} Period Regime Distribution:")
    print(f"Total periods: {total}")
    
    for regime, count in counts.items():
        percentage = (count / total) * 100
        print(f"Regime {regime}: {count} periods ({percentage:.2f}%)")

def plot_regime_distributions(train_regimes, test_regimes, regime_opt_out):
    """Plot training vs testing regime distributions"""
    
    plt.figure(figsize=(12, 8))
    
    # Create distribution data
    train_counts = train_regimes.value_counts().sort_index()
    test_counts = test_regimes.value_counts().sort_index()
    
    train_pct = (train_counts / len(train_regimes)) * 100
    test_pct = (test_counts / len(test_regimes)) * 100
    
    # Plot regime distributions
    regimes = np.arange(STRATEGY_CONFIG['regime_detection']['n_regimes'])
    width = 0.35
    
    # Handle missing regimes
    train_values = [train_pct.get(r, 0) for r in regimes]
    test_values = [test_pct.get(r, 0) for r in regimes]
    
    plt.bar(regimes - width/2, train_values, width, label='Training', color='blue', alpha=0.7)
    plt.bar(regimes + width/2, test_values, width, label='Testing', color='green', alpha=0.7)
    
    # Highlight opted-out regimes
    for regime, opt_out in regime_opt_out.items():
        if opt_out:
            plt.axvspan(regime-0.5, regime+0.5, color='red', alpha=0.2)
            plt.text(regime, max(train_values + test_values) + 5, "OPT-OUT", 
                    ha='center', color='red', fontweight='bold')
    
    plt.xlabel('Regime')
    plt.ylabel('Percentage of Periods (%)')
    plt.title('Regime Distribution: Training vs Testing')
    plt.xticks(regimes)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(train_values):
        plt.text(i - width/2, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    for i, v in enumerate(test_values):
        plt.text(i + width/2, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # Save the plot
    plt.tight_layout()
    plt.savefig('regime_distribution_debug.png', dpi=300)
    plt.close()
    
    print("\nRegime distribution plot saved as 'regime_distribution_debug.png'")

if __name__ == "__main__":
    main()