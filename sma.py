# sma.py
# SMA Strategy with Separate Parameter Optimization for Each Volatility Regime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from itertools import product
from database import DatabaseHandler
from backtest import (
    backtest_strategy, 
    buy_and_hold,
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_volatility_adjusted_return,
    analyze_performance
)
from config import (
    # General backtest settings
    TRADING_FREQUENCY,
    TRAINING_START,
    TRAINING_END,
    TESTING_START,
    TESTING_END,
    CURRENCY,
    INITIAL_CAPITAL,
    TRADING_FEE_PCT,
    
    # SMA-specific settings
    USE_VOLATILITY_REGIMES,
    SMA_SHORT_WINDOWS,
    SMA_LONG_WINDOWS,
    SMA_VOLATILITY_THRESHOLDS,
    SMA_VOLATILITY_WINDOWS,
    SAVE_RESULTS,
    PLOT_RESULTS,
    RESULTS_DIR,
    
    # Position sizing settings
    USE_POSITION_SIZING,
    POSITION_SIZING_METHOD,
    MAX_POSITION_SIZE,
    MIN_POSITION_SIZE,
    VOLATILITY_LOOKBACK,
    
    # Volatility calculation settings
    VOLATILITY_METHOD,
    TARGET_VOLATILITY,
    
    # Expanding window settings
    USE_EXPANDING_WINDOW,
    INITIAL_WINDOW,
    STEP_SIZE,
    MIN_TRAINING_SIZE,
    TEST_PERIOD_SIZE,
    
    # Sectional testing settings (if defined)
    USE_SECTIONAL_TESTING,
    SECTION_SIZE,
    AGGREGATE_RESULTS
)
# ==================== HELPER FUNCTIONS ====================
def calculate_volatility_zscore(df, window):
    """Calculate price volatility using z-score of rolling standard deviation of returns"""
    returns = df["close_price"].pct_change()
    
    # Calculate rolling standard deviation
    vol = returns.rolling(window=window).std()
    
    # Calculate z-score of volatility (how many standard deviations from the mean)
    # Use a longer lookback to get a more stable baseline
    lookback = window * 3  # 3x the volatility window for the baseline
    vol_mean = vol.rolling(window=lookback).mean()
    vol_std = vol.rolling(window=lookback).std()
    
    # Calculate z-score
    vol_zscore = (vol - vol_mean) / vol_std
    
    return vol_zscore

def identify_volatility_regimes(df, vol_window, vol_threshold):
    """
    Split the data into high, normal, and low volatility regimes using z-scores
    Returns a regime classifier series
    """
    vol_zscore = calculate_volatility_zscore(df, vol_window)
    
    # Classify regimes based on z-score thresholds
    regime = pd.Series(0, index=df.index)  # Default to normal regime
    regime[vol_zscore > vol_threshold] = 1  # High volatility
    regime[vol_zscore < -vol_threshold] = -1  # Low volatility
    
    # Forward fill NaN values that occur at the start due to rolling calculations
    regime = regime.fillna(0)
    
    return regime, vol_zscore

# Add these functions to your sma.py file

def calculate_parkinson_volatility(df, window=VOLATILITY_LOOKBACK):
    """
    Calculates volatility using Parkinson's Range formula.
    This method uses high-low ranges and is more efficient than standard deviation.
    
    Formula: sqrt((1 / (4 * ln(2) * n)) * sum((ln(high/low))^2))
    
    Parameters:
    -----------
    df : DataFrame with price data containing high_price and low_price columns
    window : int - Lookback period for volatility calculation
    
    Returns:
    --------
    pd.Series : Volatility series
    """
    # Constant factor in Parkinson's formula
    const = 1.0 / (4.0 * np.log(2.0))
    
    # Calculate log of high/low ratio squared
    log_hl_ratio = np.log(df['high_price'] / df['low_price']) ** 2
    
    # Calculate Parkinson's volatility
    parkinsons_vol = np.sqrt(const * log_hl_ratio.rolling(window=window).sum() / window)
    
    # Handle leading NaN values
    parkinsons_vol = parkinsons_vol.fillna(method='bfill')
    
    # Annualize volatility based on trading frequency
    if TRADING_FREQUENCY == "1H":
        # For hourly data: multiply by sqrt(24*365) for annual
        annualized_vol = parkinsons_vol * np.sqrt(24 * 365)
    elif TRADING_FREQUENCY == "1D":
        # For daily data: multiply by sqrt(365) for annual
        annualized_vol = parkinsons_vol * np.sqrt(365)
    else:
        annualized_vol = parkinsons_vol
    
    return annualized_vol

def calculate_standard_volatility(df, window=VOLATILITY_LOOKBACK):
    """
    Calculates volatility using standard deviation of returns.
    
    Parameters:
    -----------
    df : DataFrame with price data
    window : int - Lookback period for volatility calculation
    
    Returns:
    --------
    pd.Series : Volatility series
    """
    # Calculate returns
    returns = df['close_price'].pct_change().fillna(0)
    
    # Calculate standard deviation
    volatility = returns.rolling(window=window).std()
    
    # Annualize volatility based on trading frequency
    if TRADING_FREQUENCY == "1H":
        # For hourly data: multiply by sqrt(24*365) for annual
        annualized_vol = volatility * np.sqrt(24 * 365)
    elif TRADING_FREQUENCY == "1D":
        # For daily data: multiply by sqrt(365) for annual
        annualized_vol = volatility * np.sqrt(365)
    else:
        annualized_vol = volatility
    
    return annualized_vol

def calculate_volatility(df, method=VOLATILITY_METHOD, window=VOLATILITY_LOOKBACK):
    """
    Calculate volatility using the specified method.
    
    Parameters:
    -----------
    df : DataFrame with price data
    method : str - Method to use for volatility calculation
    window : int - Lookback period for volatility calculation
    
    Returns:
    --------
    pd.Series : Volatility series
    """
    if method == "parkinson":
        return calculate_parkinson_volatility(df, window)
    else:  # Use standard volatility as default or fallback
        return calculate_standard_volatility(df, window)

def size_position(signal, volatility, method=POSITION_SIZING_METHOD):
    """
    Scale position size based on volatility.
    
    Parameters:
    -----------
    signal : pd.Series - Signal series (-1, 0, or 1)
    volatility : pd.Series - Volatility series
    method : str - Method to use for position sizing
    
    Returns:
    --------
    pd.Series : Scaled position size
    """
    if method == "volatility":
        # Invert volatility to scale position sizes
        # When volatility is high, position size is reduced
        if volatility.min() > 0:
            # Use target volatility / current volatility
            scale_factor = TARGET_VOLATILITY / volatility
            
            # Clip to ensure position sizes are within bounds
            scale_factor = scale_factor.clip(MIN_POSITION_SIZE, MAX_POSITION_SIZE)
            
            # Apply scaling to signal
            scaled_position = signal * scale_factor
            
            return scaled_position
    
    # Default: return original signal (binary position sizing)
    return signal

# Add these functions to your sma.py file

def generate_expanding_windows(start_date, end_date, initial_window_size, step_size, test_size):
    """
    Generate expanding windows for time series cross-validation.
    
    Parameters:
    -----------
    start_date : str - Start date for first window
    end_date : str - End date for entire period
    initial_window_size : int - Size of initial window in days
    step_size : int - Step size in days
    test_size : int - Size of test period in days
    
    Returns:
    --------
    list : List of (train_start, train_end, test_start, test_end) date tuples
    """
    # Convert dates to pandas datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Create list to store windows
    windows = []
    
    # Set initial training window
    train_start = start
    train_end = train_start + pd.Timedelta(days=initial_window_size)
    
    # Generate windows until we reach the end date
    while train_end < end:
        # Ensure train_end doesn't go beyond the end date
        train_end = min(train_end, end - pd.Timedelta(days=test_size))
        
        # Define test window
        test_start = train_end
        test_end = test_start + pd.Timedelta(days=test_size)
        
        # Ensure test_end doesn't go beyond the end date
        test_end = min(test_end, end)
        
        # Store window tuple
        windows.append((train_start, train_end, test_start, test_end))
        
        # Create next window
        train_end = train_end + pd.Timedelta(days=step_size)
    
    # Convert dates back to strings
    windows = [(ts.strftime('%Y-%m-%d'), te.strftime('%Y-%m-%d'), 
                vs.strftime('%Y-%m-%d'), ve.strftime('%Y-%m-%d')) 
               for ts, te, vs, ve in windows]
    
    return windows

def run_expanding_window_test():
    """
    Run expanding window tests for the SMA strategy.
    """
    # Initialize database connection
    db = DatabaseHandler()
    
    # Generate expanding windows
    windows = generate_expanding_windows(
        INITIAL_WINDOW, 
        TESTING_END, 
        MIN_TRAINING_SIZE, 
        STEP_SIZE, 
        TEST_PERIOD_SIZE
    )
    
    print(f"Generated {len(windows)} expanding windows for testing.")
    
    # Store results for each window
    window_results = []
    
    # Test on each window
    for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
        print(f"\nWindow {i+1}/{len(windows)}:")
        print(f"Training: {train_start} to {train_end}")
        print(f"Testing: {test_start} to {test_end}")
        
        # Fetch training data
        train_df = db.get_historical_data(CURRENCY, train_start, train_end)
        
        if len(train_df) < 100:
            print(f"Insufficient training data. Skipping window.")
            continue
        
        # Find optimal parameters
        if USE_VOLATILITY_REGIMES:
            best_params, train_return, _ = find_optimal_volatility_settings(train_df)
        else:
            best_params, train_return, _ = find_best_simple_sma_parameters(train_df)
        
        # Fetch test data
        test_df = db.get_historical_data(CURRENCY, test_start, test_end)
        
        if len(test_df) < 100:
            print(f"Insufficient test data. Skipping window.")
            continue
        
        # Calculate buy & hold return for this test period
        buy_hold_return, buy_hold_value = buy_and_hold(test_df, INITIAL_CAPITAL)
        
        # Apply strategy to test data
        if USE_VOLATILITY_REGIMES:
            result_df = apply_regime_specific_strategy(test_df, best_params)
            regime_data = True
        else:
            result_df = apply_simple_sma_strategy(test_df, best_params)
            regime_data = False
        
        # Analyze results
        test_results = analyze_performance(result_df, INITIAL_CAPITAL, regime_data)
        
        # Store window parameters and results
        window_result = {
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'train_return': train_return,
            'test_return': test_results['Total Return'],
            'test_sharpe': test_results['Sharpe Ratio'],
            'test_max_dd': test_results['Max Drawdown'],
            'test_annual_return': test_results['Annual Return'],
            'test_trades': test_results['Number of Trades'],
            'buy_hold_return': buy_hold_return,
            'outperformance': test_results['Total Return'] - buy_hold_return,
            'params': best_params
        }
        window_results.append(window_result)
        
        # Print test results
        print(f"Training Return: {train_return:.2%}")
        print(f"Test Return: {test_results['Total Return']:.2%}")
        print(f"Buy & Hold Return: {buy_hold_return:.2%}")
        print(f"Outperformance: {test_results['Total Return'] - buy_hold_return:.2%}")
        print(f"Test Sharpe: {test_results['Sharpe Ratio']:.2f}")
        print(f"Test Max Drawdown: {test_results['Max Drawdown']:.2%}")
    
    # Close database connection
    db.close()
    
    # Calculate aggregate statistics
    if window_results:
        avg_test_return = np.mean([wr['test_return'] for wr in window_results])
        avg_test_sharpe = np.mean([wr['test_sharpe'] for wr in window_results])
        avg_test_max_dd = np.mean([wr['test_max_dd'] for wr in window_results])
        avg_bh_return = np.mean([wr['buy_hold_return'] for wr in window_results])
        avg_outperformance = np.mean([wr['outperformance'] for wr in window_results])
        
        print("\n===== Aggregate Results =====")
        print(f"Number of Windows: {len(window_results)}")
        print(f"Average Test Return: {avg_test_return:.2%}")
        print(f"Average Buy & Hold Return: {avg_bh_return:.2%}")
        print(f"Average Outperformance: {avg_outperformance:.2%}")
        print(f"Average Test Sharpe: {avg_test_sharpe:.2f}")
        print(f"Average Test Max Drawdown: {avg_test_max_dd:.2%}")
        
        # Calculate success rate (% of periods outperforming buy & hold)
        outperformed_periods = sum(1 for wr in window_results if wr['outperformance'] > 0)
        success_rate = outperformed_periods / len(window_results) if len(window_results) > 0 else 0
        print(f"Success Rate (outperformed B&H): {success_rate:.2%}")
        
        # Save detailed results
        if SAVE_RESULTS:
            if not os.path.exists(RESULTS_DIR):
                os.makedirs(RESULTS_DIR)
                
            # Create summary text file
            create_expanding_window_summary(window_results)
                
            # Convert to DataFrame for easy saving
            results_df = pd.DataFrame(window_results)
            
            # Save to CSV
            results_df.to_csv(os.path.join(RESULTS_DIR, f'expanding_window_results_{CURRENCY.replace("/", "_")}.csv'), index=False)
            print(f"Detailed results saved to {os.path.join(RESULTS_DIR, f'expanding_window_results_{CURRENCY.replace('/', '_')}.csv')}")
    
    return window_results

def create_expanding_window_summary(window_results):
    """Create a detailed summary text file for expanding window results"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Define filename for the summary
    filename = os.path.join(RESULTS_DIR, f'expanding_window_summary_{CURRENCY.replace("/", "_")}.txt')
    
    with open(filename, 'w') as f:
        # Write header
        f.write(f"===== EXPANDING WINDOW TESTING SUMMARY FOR {CURRENCY} =====\n\n")
        
        # Strategy information
        f.write("Strategy: ")
        if USE_VOLATILITY_REGIMES:
            f.write("Volatility Regime-based SMA\n")
        else:
            f.write("Simple SMA\n")
        
        if USE_POSITION_SIZING:
            f.write(f"Position Sizing: {POSITION_SIZING_METHOD}\n")
            f.write(f"Volatility Method: {VOLATILITY_METHOD}\n")
            f.write(f"Target Volatility: {TARGET_VOLATILITY:.2%}\n")
        else:
            f.write("Position Sizing: Disabled (using binary signals)\n")
        
        f.write(f"Trading Fee: {TRADING_FEE_PCT:.4%} per trade\n")
        f.write(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}\n\n")
        
        # Window settings
        f.write(f"Initial Window: {INITIAL_WINDOW}\n")
        f.write(f"Minimum Training Size: {MIN_TRAINING_SIZE} days\n")
        f.write(f"Step Size: {STEP_SIZE} days\n")
        f.write(f"Test Period Size: {TEST_PERIOD_SIZE} days\n")
        f.write(f"Total Windows: {len(window_results)}\n\n")
        
        # Results for each window
        f.write("===== WINDOW RESULTS =====\n\n")
        for i, result in enumerate(window_results):
            f.write(f"Window {i+1}:\n")
            f.write(f"  Training: {result['train_start']} to {result['train_end']}\n")
            f.write(f"  Testing: {result['test_start']} to {result['test_end']}\n")
            f.write(f"  Training Return: {result['train_return']:.4%}\n")
            f.write(f"  Test Return: {result['test_return']:.4%}\n")
            f.write(f"  Buy & Hold Return: {result['buy_hold_return']:.4%}\n")
            f.write(f"  Outperformance: {result['outperformance']:.4%}\n")
            f.write(f"  Sharpe Ratio: {result['test_sharpe']:.4f}\n")
            f.write(f"  Max Drawdown: {result['test_max_dd']:.4%}\n")
            f.write(f"  Number of Trades: {result['test_trades']}\n")
            
            # Write parameters
            f.write("  Parameters:\n")
            if USE_VOLATILITY_REGIMES:
                params = result['params']
                f.write(f"    Volatility Window: {params['vol_window']} hours\n")
                f.write(f"    Volatility Z-score Threshold: {params['vol_threshold']}\n")
                f.write(f"    High Vol SMA: {params['high_vol_params']}\n")
                f.write(f"    Normal Vol SMA: {params['normal_vol_params']}\n")
                f.write(f"    Low Vol SMA: {params['low_vol_params']}\n")
            else:
                params = result['params']
                f.write(f"    Short MA: {params['short_window']}\n")
                f.write(f"    Long MA: {params['long_window']}\n")
            
            f.write("\n")
        
        # Calculate aggregate statistics
        avg_test_return = np.mean([wr['test_return'] for wr in window_results])
        avg_test_sharpe = np.mean([wr['test_sharpe'] for wr in window_results])
        avg_test_max_dd = np.mean([wr['test_max_dd'] for wr in window_results])
        avg_bh_return = np.mean([wr['buy_hold_return'] for wr in window_results])
        avg_outperformance = np.mean([wr['outperformance'] for wr in window_results])
        
        # Calculate success rate (% of periods outperforming buy & hold)
        outperformed_periods = sum(1 for wr in window_results if wr['outperformance'] > 0)
        success_rate = outperformed_periods / len(window_results) if len(window_results) > 0 else 0
        
        # Write aggregate results
        f.write("===== AGGREGATE RESULTS =====\n\n")
        f.write(f"Average Test Return: {avg_test_return:.4%}\n")
        f.write(f"Average Buy & Hold Return: {avg_bh_return:.4%}\n")
        f.write(f"Average Outperformance: {avg_outperformance:.4%}\n")
        f.write(f"Average Sharpe Ratio: {avg_test_sharpe:.4f}\n")
        f.write(f"Average Max Drawdown: {avg_test_max_dd:.4%}\n")
        f.write(f"Success Rate (outperformed B&H): {success_rate:.2%} ({outperformed_periods}/{len(window_results)})\n")
    
    print(f"Expanding window summary saved to {filename}")

def test_sma_parameters(df, regime, regime_type, short_window, long_window):
    """
    Test a specific SMA parameter combination on a specific volatility regime
    
    Parameters:
    -----------
    df : DataFrame with price data
    regime : Series with regime classification (1=high, 0=normal, -1=low)
    regime_type : int - The regime to test (1=high, 0=normal, -1=low)
    short_window : int - Short SMA window
    long_window : int - Long SMA window
    
    Returns:
    --------
    return_pct : float - Return percentage for this parameter combination within this regime
    """
    # Skip invalid combinations
    if short_window >= long_window:
        return -float('inf')
    
    # Calculate SMA signals
    short_ma = df["close_price"].rolling(window=short_window).mean()
    long_ma = df["close_price"].rolling(window=long_window).mean()
    
    # Generate signal (+1 for buy, -1 for sell)
    signal = pd.Series(0, index=df.index)
    signal[short_ma > long_ma] = 1  # Buy when short MA crosses above long MA
    signal[short_ma <= long_ma] = -1  # Sell when short MA crosses below long MA
    
    # Get returns for the entire period
    returns = df["close_price"].pct_change().fillna(0)
    
    # Calculate position changes (when a trade occurs)
    position_changes = signal.diff().fillna(0).abs()
    
    # Apply trading costs when position changes
    trading_costs = position_changes * TRADING_FEE_PCT
    
    # Adjust returns for trading costs
    adjusted_returns = returns - trading_costs
    
    # Create mask for the specific regime
    regime_mask = (regime == regime_type)
    
    # Only include returns for the specific regime
    regime_returns = adjusted_returns[regime_mask]
    regime_signal = signal.shift(1).fillna(0)[regime_mask]  # Shift by 1 to avoid lookahead bias
    
    # Calculate strategy returns within this regime
    strategy_returns = regime_signal * regime_returns
    
    # If the regime has no data, return -inf
    if len(strategy_returns) == 0:
        return -float('inf')
    
    # Calculate the total return percentage within this regime
    # Use compound returns instead of summing
    return_pct = (1 + strategy_returns).prod() - 1
    
    return return_pct

def test_simple_sma_parameters(df, short_window, long_window):
    """
    Test a single SMA parameter combination on the entire dataset without volatility regimes
    
    Parameters:
    -----------
    df : DataFrame with price data
    short_window : int - Short SMA window
    long_window : int - Long SMA window
    
    Returns:
    --------
    return_pct : float - Return percentage for this parameter combination
    """
    # Skip invalid combinations
    if short_window >= long_window:
        return -float('inf')
    
    # Calculate SMA signals
    short_ma = df["close_price"].rolling(window=short_window).mean()
    long_ma = df["close_price"].rolling(window=long_window).mean()
    
    # Generate signal (+1 for buy, -1 for sell)
    signal = pd.Series(0, index=df.index)
    signal[short_ma > long_ma] = 1  # Buy when short MA crosses above long MA
    signal[short_ma <= long_ma] = -1  # Sell when short MA crosses below long MA
    
    # Get returns for the entire period
    returns = df["close_price"].pct_change().fillna(0)
    
    # Calculate position changes (when a trade occurs)
    position_changes = signal.diff().fillna(0).abs()
    
    # Apply trading costs when position changes
    trading_costs = position_changes * TRADING_FEE_PCT
    
    # Adjust returns for trading costs
    adjusted_returns = returns - trading_costs
    
    # Calculate strategy returns
    strategy_returns = signal.shift(1).fillna(0) * adjusted_returns  # Shift by 1 to avoid lookahead bias
    
    # Calculate the total return percentage (compound)
    return_pct = (1 + strategy_returns).prod() - 1
    
    return return_pct

def find_best_parameters_by_regime(df, vol_window, vol_threshold):
    """
    Find the best SMA parameters for each volatility regime
    """
    # Identify volatility regimes
    regime, _ = identify_volatility_regimes(df, vol_window, vol_threshold)
    
    # Calculate the percentage of time spent in each regime
    high_vol_pct = (regime == 1).mean()  # Percentage as decimal
    normal_vol_pct = (regime == 0).mean()
    low_vol_pct = (regime == -1).mean()
    
    # Check if we have enough data for each regime
    min_data_points = 24 * 7  # At least 1 week of data
    
    regime_counts = {
        "high": (regime == 1).sum(),
        "normal": (regime == 0).sum(),
        "low": (regime == -1).sum()
    }
    
    for regime_name, count in regime_counts.items():
        if count < min_data_points:
            print(f"Warning: Not enough {regime_name} volatility data points ({count}) for reliable optimization.")
    
    # Initialize results for each regime
    best_params = {}
    best_returns = {}
    
    # Test every parameter combination for each regime type
    for regime_type, regime_name in [(1, "high"), (0, "normal"), (-1, "low")]:
        best_return = -float('inf')
        best_short = None
        best_long = None
        
        # Skip regimes with insufficient data
        if regime_counts[regime_name] < min_data_points:
            print(f"Skipping parameter optimization for {regime_name} volatility due to insufficient data.")
            # Use reasonable defaults
            best_params[regime_name] = (SMA_SHORT_WINDOWS[0], SMA_LONG_WINDOWS[0])
            best_returns[regime_name] = -float('inf')
            continue
        
        # Test all combinations of short and long windows
        for short_window, long_window in product(SMA_SHORT_WINDOWS, SMA_LONG_WINDOWS):
            if short_window >= long_window:
                continue
                
            # Test this parameter combination in this specific regime
            return_pct = test_sma_parameters(df, regime, regime_type, short_window, long_window)
            
            # Update best parameters if better
            if return_pct > best_return:
                best_return = return_pct
                best_short = short_window
                best_long = long_window
        
        # Store best parameters and return for this regime
        if best_short is not None and best_long is not None:
            best_params[regime_name] = (best_short, best_long)
            best_returns[regime_name] = best_return
        else:
            # Use some reasonable defaults if we couldn't find good parameters
            best_params[regime_name] = (SMA_SHORT_WINDOWS[0], SMA_LONG_WINDOWS[0])
            best_returns[regime_name] = -float('inf')
    
    # Return the best parameters and additional information
    results = {
        'best_params': best_params,
        'best_returns': best_returns,
        'regime_distribution': {
            'high_vol_pct': high_vol_pct,
            'normal_vol_pct': normal_vol_pct,
            'low_vol_pct': low_vol_pct
        }
    }
    
    return results

def find_best_simple_sma_parameters(df):
    """
    Find the best single set of SMA parameters for the entire dataset
    without considering volatility regimes
    """
    best_return = -float('inf')
    best_short = None
    best_long = None
    
    # Store all results for analysis
    all_results = []
    
    # Test all combinations of short and long windows
    total_combos = len(SMA_SHORT_WINDOWS) * len(SMA_LONG_WINDOWS)
    print(f"Testing {total_combos} SMA parameter combinations...")
    
    start_time = time.time()
    processed = 0
    
    for short_window, long_window in product(SMA_SHORT_WINDOWS, SMA_LONG_WINDOWS):
        processed += 1
        
        if short_window >= long_window:
            continue
            
        # Test this parameter combination on the entire dataset
        return_pct = test_simple_sma_parameters(df, short_window, long_window)
        
        # Store result
        result = {
            'short_window': short_window,
            'long_window': long_window,
            'return': return_pct
        }
        all_results.append(result)
        
        # Update best parameters if better
        if return_pct > best_return:
            best_return = return_pct
            best_short = short_window
            best_long = long_window
        
        # Print progress
        if processed % 10 == 0 or processed == total_combos:
            elapsed = time.time() - start_time
            remaining = (elapsed / processed) * (total_combos - processed)
            print(f"Progress: {processed}/{total_combos} ({processed/total_combos*100:.1f}%) - "
                  f"Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s")
    
    # Save all results to CSV
    if SAVE_RESULTS and all_results:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
            
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values(by='return', ascending=False)
        results_df.to_csv(os.path.join(RESULTS_DIR, f'simple_sma_optimization_results_{CURRENCY.replace("/", "_")}.csv'), index=False)
    
    # Analyze top results
    top_results = sorted(all_results, key=lambda x: x['return'], reverse=True)[:5]
    print("\n===== Top 5 SMA Parameter Combinations =====")
    for i, result in enumerate(top_results, 1):
        print(f"{i}. Short MA: {result['short_window']}, Long MA: {result['long_window']} - Return: {result['return']:.2%}")
    
    if best_short is None or best_long is None:
        print("Could not find valid parameters. Using defaults.")
        best_short = SMA_SHORT_WINDOWS[0]
        best_long = SMA_LONG_WINDOWS[0]
        best_return = -float('inf')
    
    best_params = {
        'short_window': best_short,
        'long_window': best_long
    }
    
    return best_params, best_return, all_results

def find_optimal_volatility_settings(df):
    """
    Find the optimal volatility window and threshold
    by testing which combination leads to the best overall performance
    """
    best_vol_window = None
    best_vol_threshold = None
    best_params = None
    best_overall_return = -float('inf')
    
    # Store all results for analysis
    all_results = []
    
    # Test all combinations of volatility window and threshold
    total_combos = len(SMA_VOLATILITY_WINDOWS) * len(SMA_VOLATILITY_THRESHOLDS)
    print(f"Testing {total_combos} volatility settings...")
    
    start_time = time.time()
    processed = 0
    
    for vol_window, vol_threshold in product(SMA_VOLATILITY_WINDOWS, SMA_VOLATILITY_THRESHOLDS):
        processed += 1
        
        # Find the best parameters for each regime with these volatility settings
        regime_results = find_best_parameters_by_regime(df, vol_window, vol_threshold)
        
        # Test the combined strategy with these best parameters
        combined_return = test_combined_strategy(
            df,
            vol_window,
            vol_threshold,
            regime_results['best_params']['high'],
            regime_results['best_params']['normal'],
            regime_results['best_params']['low']
        )
        
        # Store result
        result = {
            'vol_window': vol_window,
            'vol_threshold': vol_threshold,
            'high_params': regime_results['best_params']['high'],
            'normal_params': regime_results['best_params']['normal'],
            'low_params': regime_results['best_params']['low'],
            'high_return': regime_results['best_returns']['high'],
            'normal_return': regime_results['best_returns']['normal'],
            'low_return': regime_results['best_returns']['low'],
            'combined_return': combined_return,
            'high_vol_pct': regime_results['regime_distribution']['high_vol_pct'],
            'normal_vol_pct': regime_results['regime_distribution']['normal_vol_pct'],
            'low_vol_pct': regime_results['regime_distribution']['low_vol_pct']
        }
        all_results.append(result)
        
        # Update best parameters if better overall
        if combined_return > best_overall_return:
            best_overall_return = combined_return
            best_vol_window = vol_window
            best_vol_threshold = vol_threshold
            best_params = {
                'vol_window': vol_window,
                'vol_threshold': vol_threshold,
                'high_vol_params': regime_results['best_params']['high'],
                'normal_vol_params': regime_results['best_params']['normal'],
                'low_vol_params': regime_results['best_params']['low']
            }
        
        # Print progress
        if processed % 5 == 0 or processed == total_combos:
            elapsed = time.time() - start_time
            remaining = (elapsed / processed) * (total_combos - processed)
            print(f"Progress: {processed}/{total_combos} ({processed/total_combos*100:.1f}%) - "
                  f"Elapsed: {elapsed:.1f}s, Remaining: {remaining:.1f}s")
    
    # Save all results to CSV
    if SAVE_RESULTS and all_results:
        if not os.path.exists(RESULTS_DIR):
            os.makedirs(RESULTS_DIR)
            
        results_df = pd.DataFrame(all_results)
        # Convert tuple columns to string for easier CSV export
        for col in ['high_params', 'normal_params', 'low_params']:
            results_df[col] = results_df[col].apply(str)
            
        results_df = results_df.sort_values(by='combined_return', ascending=False)
        results_df.to_csv(os.path.join(RESULTS_DIR, f'volatility_optimization_results_{CURRENCY.replace("/", "_")}.csv'), index=False)
    
    # Analyze top results
    top_results = sorted(all_results, key=lambda x: x['combined_return'], reverse=True)[:5]
    print("\n===== Top 5 Volatility Settings =====")
    for i, result in enumerate(top_results, 1):
        print(f"{i}. Window: {result['vol_window']}, Z-score Threshold: {result['vol_threshold']} - Return: {result['combined_return']:.2%}")
        print(f"   High Vol: {result['high_params']} - Return: {result['high_return']:.2%} ({result['high_vol_pct']:.1%} of time)")
        print(f"   Normal Vol: {result['normal_params']} - Return: {result['normal_return']:.2%} ({result['normal_vol_pct']:.1%} of time)")
        print(f"   Low Vol: {result['low_params']} - Return: {result['low_return']:.2%} ({result['low_vol_pct']:.1%} of time)")
    
    return best_params, best_overall_return, all_results

def test_combined_strategy(df, vol_window, vol_threshold, high_vol_params, normal_vol_params, low_vol_params):
    """
    Test the combined strategy that uses different SMA parameters for each volatility regime
    """
    # Identify volatility regimes
    regime, _ = identify_volatility_regimes(df, vol_window, vol_threshold)
    
    # Calculate SMA signals for each regime
    high_short, high_long = high_vol_params
    normal_short, normal_long = normal_vol_params
    low_short, low_long = low_vol_params
    
    # High volatility SMA
    high_short_ma = df["close_price"].rolling(window=high_short).mean()
    high_long_ma = df["close_price"].rolling(window=high_long).mean()
    high_signal = pd.Series(0, index=df.index)
    high_signal[high_short_ma > high_long_ma] = 1
    high_signal[high_short_ma <= high_long_ma] = -1
    
    # Normal volatility SMA
    normal_short_ma = df["close_price"].rolling(window=normal_short).mean()
    normal_long_ma = df["close_price"].rolling(window=normal_long).mean()
    normal_signal = pd.Series(0, index=df.index)
    normal_signal[normal_short_ma > normal_long_ma] = 1
    normal_signal[normal_short_ma <= normal_long_ma] = -1
    
    # Low volatility SMA
    low_short_ma = df["close_price"].rolling(window=low_short).mean()
    low_long_ma = df["close_price"].rolling(window=low_long).mean()
    low_signal = pd.Series(0, index=df.index)
    low_signal[low_short_ma > low_long_ma] = 1
    low_signal[low_short_ma <= low_long_ma] = -1
    
    # Combine signals based on volatility regime
    combined_signal = pd.Series(0, index=df.index)
    combined_signal[regime == 1] = high_signal[regime == 1]
    combined_signal[regime == 0] = normal_signal[regime == 0]
    combined_signal[regime == -1] = low_signal[regime == -1]
    
    # Calculate position changes (when a trade occurs)
    position_changes = combined_signal.diff().fillna(0).abs()
    
    # Apply trading costs when position changes
    trading_costs = position_changes * TRADING_FEE_PCT
    
    # Prepare data for backtest
    returns = df["close_price"].pct_change().fillna(0)
    
    # Adjust returns for trading costs
    adjusted_returns = returns - trading_costs
    
    # Calculate strategy returns
    strategy_returns = combined_signal.shift(1).fillna(0) * adjusted_returns
    
    # Calculate total return (compound)
    return (1 + strategy_returns).prod() - 1

def apply_regime_specific_strategy(df, params):
    """
    Apply the regime-specific strategy to testing data
    """
    # Extract parameters
    vol_window = params['vol_window']
    vol_threshold = params['vol_threshold']
    high_vol_params = params['high_vol_params']
    normal_vol_params = params['normal_vol_params']
    low_vol_params = params['low_vol_params']
    
    # Identify volatility regimes
    regime, volatility_zscore = identify_volatility_regimes(df, vol_window, vol_threshold)
    
    # Calculate actual volatility for position sizing
    if USE_POSITION_SIZING:
        actual_volatility = calculate_volatility(df, VOLATILITY_METHOD, VOLATILITY_LOOKBACK)
    
    # Calculate SMA signals for each regime
    high_short, high_long = high_vol_params
    normal_short, normal_long = normal_vol_params
    low_short, low_long = low_vol_params
    
    # High volatility SMA
    high_short_ma = df["close_price"].rolling(window=high_short).mean()
    high_long_ma = df["close_price"].rolling(window=high_long).mean()
    high_signal = pd.Series(0, index=df.index)
    high_signal[high_short_ma > high_long_ma] = 1
    high_signal[high_short_ma <= high_long_ma] = -1
    
    # Normal volatility SMA
    normal_short_ma = df["close_price"].rolling(window=normal_short).mean()
    normal_long_ma = df["close_price"].rolling(window=normal_long).mean()
    normal_signal = pd.Series(0, index=df.index)
    normal_signal[normal_short_ma > normal_long_ma] = 1
    normal_signal[normal_short_ma <= normal_long_ma] = -1
    
    # Low volatility SMA
    low_short_ma = df["close_price"].rolling(window=low_short).mean()
    low_long_ma = df["close_price"].rolling(window=low_long).mean()
    low_signal = pd.Series(0, index=df.index)
    low_signal[low_short_ma > low_long_ma] = 1
    low_signal[low_short_ma <= low_long_ma] = -1
    
    # Combine signals based on volatility regime
    combined_signal = pd.Series(0, index=df.index)
    combined_signal[regime == 1] = high_signal[regime == 1]
    combined_signal[regime == 0] = normal_signal[regime == 0]
    combined_signal[regime == -1] = low_signal[regime == -1]
    
    # Apply position sizing if enabled
    if USE_POSITION_SIZING:
        sized_position = size_position(combined_signal, actual_volatility, POSITION_SIZING_METHOD)
    else:
        sized_position = combined_signal
    
    # Get returns
    returns = df["close_price"].pct_change().fillna(0)
    
    # Calculate position changes
    position_changes = sized_position.diff().fillna(0).abs()
    
    # Apply trading costs
    trading_costs = position_changes * TRADING_FEE_PCT
    
    # Adjust returns for trading costs
    adjusted_returns = returns - trading_costs
    
    # Calculate strategy returns
    strategy_returns = sized_position.shift(1).fillna(0) * adjusted_returns
    
    # Calculate cumulative returns
    strategy_cumulative = (1 + strategy_returns).cumprod()
    
    # Create a result DataFrame with everything needed for analysis
    result_df = pd.DataFrame({
        "close_price": df["close_price"],
        "volatility_zscore": volatility_zscore,
        "regime": regime,
        "high_signal": high_signal,
        "normal_signal": normal_signal,
        "low_signal": low_signal,
        "signal": combined_signal,
        "position": sized_position,
        "returns": returns,
        "trading_costs": trading_costs,
        "strategy_returns": strategy_returns,
        "strategy_cumulative": strategy_cumulative
    })
    
    # Add actual volatility if position sizing is enabled
    if USE_POSITION_SIZING:
        result_df["actual_volatility"] = actual_volatility
    
    return result_df

def apply_simple_sma_strategy(df, params):
    """
    Apply a simple SMA strategy without volatility regimes
    """
    # Extract parameters
    short_window = params['short_window']
    long_window = params['long_window']
    
    # Calculate actual volatility for position sizing
    if USE_POSITION_SIZING:
        actual_volatility = calculate_volatility(df, VOLATILITY_METHOD, VOLATILITY_LOOKBACK)
    
    # Calculate SMA signals
    short_ma = df["close_price"].rolling(window=short_window).mean()
    long_ma = df["close_price"].rolling(window=long_window).mean()
    
    # Generate signal (+1 for buy, -1 for sell)
    signal = pd.Series(0, index=df.index)
    signal[short_ma > long_ma] = 1  # Buy when short MA crosses above long MA
    signal[short_ma <= long_ma] = -1  # Sell when short MA crosses below long MA
    
    # Apply position sizing if enabled
    if USE_POSITION_SIZING:
        sized_position = size_position(signal, actual_volatility, POSITION_SIZING_METHOD)
    else:
        sized_position = signal
    
    # Get returns
    returns = df["close_price"].pct_change().fillna(0)
    
    # Calculate position changes
    position_changes = sized_position.diff().fillna(0).abs()
    
    # Apply trading costs
    trading_costs = position_changes * TRADING_FEE_PCT
    
    # Adjust returns for trading costs
    adjusted_returns = returns - trading_costs
    
    # Calculate strategy returns
    strategy_returns = sized_position.shift(1).fillna(0) * adjusted_returns
    
    # Calculate cumulative returns
    strategy_cumulative = (1 + strategy_returns).cumprod()
    
    # Create a result DataFrame with everything needed for analysis
    result_df = pd.DataFrame({
        "close_price": df["close_price"],
        "short_ma": short_ma,
        "long_ma": long_ma,
        "signal": signal,
        "position": sized_position,
        "returns": returns,
        "trading_costs": trading_costs,
        "strategy_returns": strategy_returns,
        "strategy_cumulative": strategy_cumulative
    })
    
    # Add actual volatility if position sizing is enabled
    if USE_POSITION_SIZING:
        result_df["actual_volatility"] = actual_volatility
    
    return result_df

def run_sectional_testing():
    """
    Run tests by dividing the test period into multiple sections.
    Uses one training period but tests on multiple consecutive time periods.
    """
    # Initialize database connection
    db = DatabaseHandler()
    
    # Fetch training data once
    print(f"Fetching training data: {TRAINING_START} to {TRAINING_END}")
    train_df = db.get_historical_data(CURRENCY, TRAINING_START, TRAINING_END)
    
    if len(train_df) < 100:
        print(f"Insufficient training data for {CURRENCY}. Exiting.")
        db.close()
        return
    
    # Find optimal parameters (once for all test sections)
    if USE_VOLATILITY_REGIMES:
        print("Using volatility-based regime switching SMA strategy")
        best_params, train_return, _ = find_optimal_volatility_settings(train_df)
    else:
        print("Using simple SMA strategy without volatility regimes")
        best_params, train_return, _ = find_best_simple_sma_parameters(train_df)
    
    print(f"\nTraining Return: {train_return:.2%}")
    print(f"Training Parameters: {best_params}")
    
    # Save optimal parameters (add this line)
    save_results(best_params, train_return, {}, using_regimes=USE_VOLATILITY_REGIMES)
    
    # Generate test sections
    test_start = pd.to_datetime(TESTING_START)
    test_end = pd.to_datetime(TESTING_END)
    
    section_starts = []
    section_ends = []
    
    current_start = test_start
    while current_start < test_end:
        current_end = current_start + pd.Timedelta(days=SECTION_SIZE)
        
        # Ensure we don't go beyond the overall test end date
        current_end = min(current_end, test_end)
        
        section_starts.append(current_start)
        section_ends.append(current_end)
        
        # Set next section start
        current_start = current_end
    
    print(f"\nDivided testing period into {len(section_starts)} sections of {SECTION_SIZE} days each.")
    
    # Store results for each section
    section_results = []
    
    # Test on each section
    for i, (section_start, section_end) in enumerate(zip(section_starts, section_ends)):
        section_start_str = section_start.strftime('%Y-%m-%d')
        section_end_str = section_end.strftime('%Y-%m-%d')
        
        print(f"\nSection {i+1}/{len(section_starts)}: {section_start_str} to {section_end_str}")
        
        # Fetch test data for this section
        section_df = db.get_historical_data(CURRENCY, section_start_str, section_end_str)
        
        if len(section_df) < 50:  # Minimum data points for testing
            print(f"Insufficient test data for section. Skipping.")
            continue
        
        # Apply strategy to test data
        if USE_VOLATILITY_REGIMES:
            result_df = apply_regime_specific_strategy(section_df, best_params)
            regime_data = True
        else:
            result_df = apply_simple_sma_strategy(section_df, best_params)
            regime_data = False
        
        # Analyze results for this section
        test_results = analyze_performance(result_df, INITIAL_CAPITAL, regime_data)
        
        # Add section dates to results
        test_results['Section Start'] = section_start_str
        test_results['Section End'] = section_end_str
        test_results['Section'] = i + 1
        
        # Store section results
        section_results.append(test_results)
        
        # Print key section results
        print(f"  Section Return: {test_results['Total Return']:.2%}")
        print(f"  Section Sharpe: {test_results['Sharpe Ratio']:.2f}")
        print(f"  Section Max Drawdown: {test_results['Max Drawdown']:.2%}")
    
    # Close database connection
    db.close()
    
    # Create summary text file with results for each section
    if SAVE_RESULTS and section_results:
        create_sectional_summary(best_params, train_return, section_results)
    
    # Calculate aggregate statistics if we have results
    if section_results and AGGREGATE_RESULTS:
        avg_return = np.mean([sr['Total Return'] for sr in section_results])
        avg_sharpe = np.mean([sr['Sharpe Ratio'] for sr in section_results])
        avg_max_dd = np.mean([sr['Max Drawdown'] for sr in section_results])
        avg_num_trades = np.mean([sr['Number of Trades'] for sr in section_results])
        
        # Calculate compounded return across all sections
        compounded_return = 1.0
        for sr in section_results:
            compounded_return *= (1 + sr['Total Return'])
        compounded_return -= 1.0
        
        print("\n===== Aggregate Results Across All Sections =====")
        print(f"Number of Sections: {len(section_results)}")
        print(f"Average Section Return: {avg_return:.2%}")
        print(f"Compounded Return: {compounded_return:.2%}")
        print(f"Average Sharpe Ratio: {avg_sharpe:.2f}")
        print(f"Average Max Drawdown: {avg_max_dd:.2%}")
        print(f"Average Trades per Section: {avg_num_trades:.1f}")
        
        # Save detailed results
        if SAVE_RESULTS:
            if not os.path.exists(RESULTS_DIR):
                os.makedirs(RESULTS_DIR)
                
            # Convert to DataFrame for easy saving
            results_df = pd.DataFrame(section_results)
            
            # Save to CSV
            results_df.to_csv(os.path.join(RESULTS_DIR, f'sectional_test_results_{CURRENCY.replace("/", "_")}.csv'), index=False)
            print(f"Detailed section results saved to {os.path.join(RESULTS_DIR, f'sectional_test_results_{CURRENCY.replace('/', '_')}.csv')}")
    
    return section_results

# Add this new function to create a summary text file
def create_sectional_summary(params, train_return, section_results):
    """Create a clear summary text file with results for each section"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Define filename for the summary
    filename = os.path.join(RESULTS_DIR, f'sectional_summary_{CURRENCY.replace("/", "_")}.txt')
    
    with open(filename, 'w') as f:
        # Write header
        f.write(f"===== SECTIONAL TESTING SUMMARY FOR {CURRENCY} =====\n\n")
        
        # Training period
        f.write(f"Training Period: {TRAINING_START} to {TRAINING_END}\n")
        f.write(f"Training Return: {train_return:.4%}\n\n")
        
        # Strategy information
        f.write("Strategy: ")
        if USE_VOLATILITY_REGIMES:
            f.write("Volatility Regime-based SMA\n")
            f.write(f"Volatility Window: {params['vol_window']} hours\n")
            f.write(f"Volatility Z-score Threshold: {params['vol_threshold']}\n\n")
            
            f.write("High Volatility Parameters:\n")
            f.write(f"  Short MA: {params['high_vol_params'][0]}\n")
            f.write(f"  Long MA: {params['high_vol_params'][1]}\n\n")
            
            f.write("Normal Volatility Parameters:\n")
            f.write(f"  Short MA: {params['normal_vol_params'][0]}\n")
            f.write(f"  Long MA: {params['normal_vol_params'][1]}\n\n")
            
            f.write("Low Volatility Parameters:\n")
            f.write(f"  Short MA: {params['low_vol_params'][0]}\n")
            f.write(f"  Long MA: {params['low_vol_params'][1]}\n\n")
        else:
            f.write("Simple SMA\n")
            f.write(f"Short MA: {params['short_window']}\n")
            f.write(f"Long MA: {params['long_window']}\n\n")
        
        # Position sizing information
        if USE_POSITION_SIZING:
            f.write(f"Position Sizing: {POSITION_SIZING_METHOD}\n")
            f.write(f"Volatility Method: {VOLATILITY_METHOD}\n")
            f.write(f"Target Volatility: {TARGET_VOLATILITY:.2%}\n\n")
        else:
            f.write("Position Sizing: Disabled (using binary signals)\n\n")
        
        f.write(f"Trading Fee: {TRADING_FEE_PCT:.4%} per trade\n")
        f.write(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}\n\n")
        
        # Section details
        f.write(f"Testing Period: {TESTING_START} to {TESTING_END}\n")
        f.write(f"Section Size: {SECTION_SIZE} days\n")
        f.write(f"Number of Sections: {len(section_results)}\n\n")
        
        # Results for each section
        f.write("===== SECTION RESULTS =====\n\n")
        for i, result in enumerate(section_results):
            f.write(f"Section {i+1}: {result['Section Start']} to {result['Section End']}\n")
            f.write(f"  Total Return: {result['Total Return']:.4%}\n")
            f.write(f"  Annual Return: {result['Annual Return']:.4%}\n")
            f.write(f"  Sharpe Ratio: {result['Sharpe Ratio']:.4f}\n")
            f.write(f"  Max Drawdown: {result['Max Drawdown']:.4%}\n")
            f.write(f"  Number of Trades: {result['Number of Trades']}\n")
            f.write(f"  Buy & Hold Return: {result['Buy & Hold Return']:.4%}\n")
            f.write(f"  Outperformance: {result['Outperformance']:.4%}\n")
            
            # Add regime-specific metrics if using volatility regimes
            if USE_VOLATILITY_REGIMES and "High Volatility % of Time" in result:
                f.write(f"  High Vol: {result['High Volatility % of Time']:.2%} of time | Return: {result['High Volatility Return']:.4%}\n")
                f.write(f"  Normal Vol: {result['Normal Volatility % of Time']:.2%} of time | Return: {result['Normal Volatility Return']:.4%}\n")
                f.write(f"  Low Vol: {result['Low Volatility % of Time']:.2%} of time | Return: {result['Low Volatility Return']:.4%}\n")
            
            f.write("\n")
        
        # Calculate aggregate statistics
        avg_return = np.mean([r['Total Return'] for r in section_results])
        avg_sharpe = np.mean([r['Sharpe Ratio'] for r in section_results])
        avg_max_dd = np.mean([r['Max Drawdown'] for r in section_results])
        avg_num_trades = np.mean([r['Number of Trades'] for r in section_results])
        avg_outperformance = np.mean([r['Outperformance'] for r in section_results])
        
        # Calculate compounded return across all sections
        compounded_return = 1.0
        for r in section_results:
            compounded_return *= (1 + r['Total Return'])
        compounded_return -= 1.0
        
        # Write aggregate results
        f.write("===== AGGREGATE RESULTS =====\n\n")
        f.write(f"Average Section Return: {avg_return:.4%}\n")
        f.write(f"Compounded Return Across All Sections: {compounded_return:.4%}\n")
        f.write(f"Average Sharpe Ratio: {avg_sharpe:.4f}\n")
        f.write(f"Average Max Drawdown: {avg_max_dd:.4%}\n")
        f.write(f"Average Trades per Section: {avg_num_trades:.1f}\n")
        f.write(f"Average Outperformance vs Buy & Hold: {avg_outperformance:.4%}\n")
    
    print(f"Sectional summary saved to {filename}")

def plot_test_results(df, params, results):
    """Plot the test results with clear visualization of regime changes and performance"""
    if not PLOT_RESULTS:
        return
        
    # Create output directory if needed
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Check if we're using volatility regimes or simple SMA
    using_regimes = 'regime' in df.columns
    
    # Check if position sizing is being used
    position_sizing_used = 'position' in df.columns and USE_POSITION_SIZING
    
    # Determine number of subplots based on features used
    if using_regimes:
        num_plots = 6 if position_sizing_used else 5
    else:
        num_plots = 5 if position_sizing_used else 4
    
    if using_regimes:
        # Extract parameters for better labeling
        high_short, high_long = params['high_vol_params']
        normal_short, normal_long = params['normal_vol_params']
        low_short, low_long = params['low_vol_params']
        
        # Prepare data for plotting
        price = df["close_price"]
        regime = df["regime"]
        signal = df["signal"]
        position = df["position"] if position_sizing_used else signal
        strategy_cumulative = df["strategy_cumulative"] * INITIAL_CAPITAL
        
        # Calculate buy & hold equity curve
        buy_hold_cumulative = (1 + df["returns"]).cumprod() * INITIAL_CAPITAL
        
        # Drawdown calculation
        running_max = strategy_cumulative.cummax()
        drawdown = (strategy_cumulative / running_max - 1) * 100  # as percentage
        
        # Regime-specific returns
        high_vol_returns = df["strategy_returns"][regime == 1].cumsum() * 100  # as percentage
        normal_vol_returns = df["strategy_returns"][regime == 0].cumsum() * 100
        low_vol_returns = df["strategy_returns"][regime == -1].cumsum() * 100
        
        # Create figure with subplots
        fig, axs = plt.subplots(num_plots, 1, figsize=(14, 5 * num_plots), 
                                 gridspec_kw={'height_ratios': [2] + [1] * (num_plots-1)})
        
        # Plot 1: Price and Performance
        ax1 = axs[0]
        ax1.set_title(f'SMA Strategy with Regime-Specific Parameters for {CURRENCY}', fontsize=16)
        ax1.plot(price.index, price, color='gray', alpha=0.6, label='Price')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(strategy_cumulative.index, strategy_cumulative, 'b-', label='Strategy')
        ax1_twin.plot(buy_hold_cumulative.index, buy_hold_cumulative, 'r--', label='Buy & Hold')
        ax1.set_ylabel('Price')
        ax1_twin.set_ylabel('Portfolio Value ($)')
        ax1_twin.legend(loc='upper left')
        
        # Plot 2: Volatility Regimes
        ax2 = axs[1]
        ax2.set_title('Volatility Regimes and Signals', fontsize=14)
        # Plot regimes as background colors
        high_vol_periods = regime == 1
        normal_vol_periods = regime == 0
        low_vol_periods = regime == -1
        
        # Plot signals on top of regimes
        for i in range(len(df)):
            if i == 0:
                continue
                
            start = df.index[i-1]
            end = df.index[i]
            
            if high_vol_periods.iloc[i]:
                ax2.axvspan(start, end, color='red', alpha=0.2)
            elif normal_vol_periods.iloc[i]:
                ax2.axvspan(start, end, color='gray', alpha=0.2)
            elif low_vol_periods.iloc[i]:
                ax2.axvspan(start, end, color='green', alpha=0.2)
        
        # Plot position as line
        ax2.plot(position.index, position, 'k-', label='Position Size')
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Add horizontal lines at +/-1 if using position sizing
        if position_sizing_used:
            ax2.axhline(y=1, color='k', linestyle='--', alpha=0.3)
            ax2.axhline(y=-1, color='k', linestyle='--', alpha=0.3)
            ax2.set_ylabel('Position Size')
        else:
            ax2.set_ylim(-1.5, 1.5)
            ax2.set_yticks([-1, 0, 1])
            ax2.set_yticklabels(['Short', 'Neutral', 'Long'])
        
        # Create a custom legend for the regimes
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.2, label=f'High Vol ({high_short}/{high_long})'),
            Patch(facecolor='gray', alpha=0.2, label=f'Normal Vol ({normal_short}/{normal_long})'),
            Patch(facecolor='green', alpha=0.2, label=f'Low Vol ({low_short}/{low_long})')
        ]
        ax2.legend(handles=legend_elements, loc='upper right')
        
        # Plot additional position sizing information if enabled
        current_plot = 2
        if position_sizing_used:
            ax_vol = axs[current_plot]
            current_plot += 1
            ax_vol.set_title('Volatility and Position Sizing', fontsize=14)
            ax_vol.plot(df.index, df['actual_volatility'], 'b-', label='Volatility')
            ax_vol_twin = ax_vol.twinx()
            ax_vol_twin.plot(df.index, position.abs(), 'r-', label='Position Size')
            ax_vol.set_ylabel('Volatility')
            ax_vol_twin.set_ylabel('Abs Position Size')
            ax_vol.legend(loc='upper left')
            ax_vol_twin.legend(loc='upper right')
        
        # Plot Drawdown
        ax_dd = axs[current_plot]
        current_plot += 1
        ax_dd.set_title(f'Drawdown (Max: {results["Max Drawdown"]:.2%})', fontsize=14)
        ax_dd.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax_dd.set_ylabel('Drawdown (%)')
        ax_dd.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Plot Cumulative Returns by Regime
        ax_regime = axs[current_plot]
        current_plot += 1
        ax_regime.set_title('Cumulative Returns by Volatility Regime', fontsize=14)
        if len(high_vol_returns) > 0:
            ax_regime.plot(high_vol_returns.index, high_vol_returns, 'r-', 
                     label=f'High Vol Return: {results["High Volatility Return"]:.2%}, Sharpe: {results["High Volatility Sharpe"]:.2f}')
        if len(normal_vol_returns) > 0:
            ax_regime.plot(normal_vol_returns.index, normal_vol_returns, 'k-', 
                     label=f'Normal Vol Return: {results["Normal Volatility Return"]:.2%}, Sharpe: {results["Normal Volatility Sharpe"]:.2f}')
        if len(low_vol_returns) > 0:
            ax_regime.plot(low_vol_returns.index, low_vol_returns, 'g-', 
                     label=f'Low Vol Return: {results["Low Volatility Return"]:.2%}, Sharpe: {results["Low Volatility Sharpe"]:.2f}')
        ax_regime.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax_regime.set_ylabel('Cumulative Return (%)')
        ax_regime.legend(loc='upper left')
        
        # Plot Moving Averages
        ax_ma = axs[current_plot]
        ax_ma.set_title('Moving Average Examples by Regime', fontsize=14)
        
        # High vol moving averages
        high_short_ma = price.rolling(window=high_short).mean()
        high_long_ma = price.rolling(window=high_long).mean()
        
        # Normal vol moving averages
        normal_short_ma = price.rolling(window=normal_short).mean()
        normal_long_ma = price.rolling(window=normal_long).mean()
        
        # Low vol moving averages
        low_short_ma = price.rolling(window=low_short).mean()
        low_long_ma = price.rolling(window=low_long).mean()
        
        # Plot price and all MAs (with alpha to avoid overcrowding)
        ax_ma.plot(price.index, price, color='gray', alpha=0.6, label='Price')
        
        # Only show a subset of data for clarity (last 30% of the data)
        start_idx = int(len(price) * 0.7)
        subset_idx = price.index[start_idx:]
        
        ax_ma.plot(subset_idx, high_short_ma.loc[subset_idx], 'r-', alpha=0.7, label=f'High Vol Short MA ({high_short})')
        ax_ma.plot(subset_idx, high_long_ma.loc[subset_idx], 'r--', alpha=0.7, label=f'High Vol Long MA ({high_long})')
        
        ax_ma.plot(subset_idx, normal_short_ma.loc[subset_idx], 'k-', alpha=0.7, label=f'Normal Vol Short MA ({normal_short})')
        ax_ma.plot(subset_idx, normal_long_ma.loc[subset_idx], 'k--', alpha=0.7, label=f'Normal Vol Long MA ({normal_long})')
        
        ax_ma.plot(subset_idx, low_short_ma.loc[subset_idx], 'g-', alpha=0.7, label=f'Low Vol Short MA ({low_short})')
        ax_ma.plot(subset_idx, low_long_ma.loc[subset_idx], 'g--', alpha=0.7, label=f'Low Vol Long MA ({low_long})')
        
        ax_ma.set_ylabel('Price & MAs')
        ax_ma.legend(loc='upper left')
        
        # Add strategy performance summary
        plt.figtext(0.1, 0.01, 
                f"Return: {results['Total Return']:.2%} | Annual: {results['Annual Return']:.2%} | Sharpe: {results['Sharpe Ratio']:.2f} | MaxDD: {results['Max Drawdown']:.2%}\n"
                f"Trades: {results['Number of Trades']} | Costs: {results['Total Trading Costs']:.2%} | B&H: {results['Buy & Hold Return']:.2%} | Alpha: {results['Outperformance']:.2%}\n"
                f"High Vol: {results['High Volatility % of Time']:.1%} of time | Normal Vol: {results['Normal Volatility % of Time']:.1%} of time | Low Vol: {results['Low Volatility % of Time']:.1%} of time",
                ha='left', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.06)
        
        # Save the figure
        strategy_type = "regime_position" if position_sizing_used else "regime"
        plt.savefig(os.path.join(RESULTS_DIR, f'test_results_{strategy_type}_{CURRENCY.replace("/", "_")}.png'), dpi=300, bbox_inches='tight')
        
    else:
        # Simple SMA strategy plotting
        # Extract parameters for better labeling
        short_window = params['short_window']
        long_window = params['long_window']
        
        # Prepare data for plotting
        price = df["close_price"]
        signal = df["signal"]
        position = df["position"] if position_sizing_used else signal
        short_ma = df["short_ma"]
        long_ma = df["long_ma"]
        strategy_cumulative = df["strategy_cumulative"] * INITIAL_CAPITAL
        
        # Calculate buy & hold equity curve
        buy_hold_cumulative = (1 + df["returns"]).cumprod() * INITIAL_CAPITAL
        
        # Drawdown calculation
        running_max = strategy_cumulative.cummax()
        drawdown = (strategy_cumulative / running_max - 1) * 100  # as percentage
        
        # Create figure with subplots
        fig, axs = plt.subplots(num_plots, 1, figsize=(14, 5 * num_plots))
        
        # Plot 1: Price and Performance
        ax1 = axs[0]
        ax1.set_title(f'Simple SMA Strategy ({short_window}/{long_window}) for {CURRENCY}', fontsize=16)
        ax1.plot(price.index, price, color='gray', alpha=0.6, label='Price')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(strategy_cumulative.index, strategy_cumulative, 'b-', label='Strategy')
        ax1_twin.plot(buy_hold_cumulative.index, buy_hold_cumulative, 'r--', label='Buy & Hold')
        ax1.set_ylabel('Price')
        ax1_twin.set_ylabel('Portfolio Value ($)')
        ax1_twin.legend(loc='upper left')
        
        # Plot 2: Trading Signals & Positions
        ax2 = axs[1]
        if position_sizing_used:
            ax2.set_title('Trading Signals & Position Sizes', fontsize=14)
            ax2.plot(signal.index, signal, 'k--', alpha=0.5, label='Signal')
            ax2.plot(position.index, position, 'b-', label='Position Size')
        else:
            ax2.set_title('Trading Signals', fontsize=14)
            ax2.plot(signal.index, signal, 'k-', label='Position Signal')
        
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Add horizontal lines at +/-1 if using position sizing
        if position_sizing_used:
            ax2.axhline(y=1, color='k', linestyle='--', alpha=0.3)
            ax2.axhline(y=-1, color='k', linestyle='--', alpha=0.3)
            ax2.set_ylabel('Position Size')
        else:
            ax2.set_ylim(-1.5, 1.5)
            ax2.set_yticks([-1, 0, 1])
            ax2.set_yticklabels(['Short', 'Neutral', 'Long'])
        
        ax2.legend(loc='upper left')
        
        # Plot additional position sizing information if enabled
        current_plot = 2
        if position_sizing_used:
            ax_vol = axs[current_plot]
            current_plot += 1
            ax_vol.set_title('Volatility and Position Sizing', fontsize=14)
            ax_vol.plot(df.index, df['actual_volatility'], 'b-', label='Volatility')
            ax_vol_twin = ax_vol.twinx()
            ax_vol_twin.plot(df.index, position.abs(), 'r-', label='Abs Position Size')
            ax_vol.set_ylabel('Volatility')
            ax_vol_twin.set_ylabel('Abs Position Size')
            ax_vol.legend(loc='upper left')
            ax_vol_twin.legend(loc='upper right')
        
        # Plot 3: Drawdown
        ax_dd = axs[current_plot]
        current_plot += 1
        ax_dd.set_title(f'Drawdown (Max: {results["Max Drawdown"]:.2%})', fontsize=14)
        ax_dd.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
        ax_dd.set_ylabel('Drawdown (%)')
        ax_dd.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Plot 4: Moving Averages
        ax_ma = axs[current_plot]
        ax_ma.set_title('Moving Averages', fontsize=14)
        ax_ma.plot(price.index, price, color='gray', alpha=0.6, label='Price')
        ax_ma.plot(short_ma.index, short_ma, 'g-', label=f'Short MA ({short_window})')
        ax_ma.plot(long_ma.index, long_ma, 'r-', label=f'Long MA ({long_window})')
        ax_ma.set_ylabel('Price & MAs')
        ax_ma.legend(loc='upper left')
        
        # Add strategy performance summary
        plt.figtext(0.1, 0.01, 
                f"Return: {results['Total Return']:.2%} | Annual: {results['Annual Return']:.2%} | Sharpe: {results['Sharpe Ratio']:.2f} | MaxDD: {results['Max Drawdown']:.2%}\n"
                f"Trades: {results['Number of Trades']} | Costs: {results['Total Trading Costs']:.2%} | B&H: {results['Buy & Hold Return']:.2%} | Alpha: {results['Outperformance']:.2%}",
                ha='left', fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.06)
        
        # Save the figure
        strategy_type = "simple_position" if position_sizing_used else "simple"
        plt.savefig(os.path.join(RESULTS_DIR, f'test_results_{strategy_type}_{CURRENCY.replace("/", "_")}.png'), dpi=300, bbox_inches='tight')
    
    plt.close()

def save_results(params, train_return, test_results, using_regimes=True):
    """Save the optimization and test results to file"""
    if not SAVE_RESULTS:
        return
        
    # Create output directory if needed
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    if using_regimes:
        # Save the optimal parameters for regime-based strategy
        filename = os.path.join(RESULTS_DIR, f'optimal_parameters_regime_{CURRENCY.replace("/", "_")}.txt')
        with open(filename, 'w') as f:
            f.write("===== Optimal Parameters =====\n\n")
            f.write(f"Volatility Window: {params['vol_window']} hours\n")
            f.write(f"Volatility Z-score Threshold: {params['vol_threshold']}\n")
            f.write(f"Trading Cost: {TRADING_FEE_PCT:.4%} per trade\n\n")
            
            f.write("High Volatility Parameters:\n")
            f.write(f"  Short MA: {params['high_vol_params'][0]}\n")
            f.write(f"  Long MA: {params['high_vol_params'][1]}\n\n")
            
            f.write("Normal Volatility Parameters:\n")
            f.write(f"  Short MA: {params['normal_vol_params'][0]}\n")
            f.write(f"  Long MA: {params['normal_vol_params'][1]}\n\n")
            
            f.write("Low Volatility Parameters:\n")
            f.write(f"  Short MA: {params['low_vol_params'][0]}\n")
            f.write(f"  Long MA: {params['low_vol_params'][1]}\n\n")
            
            f.write("===== Training Results =====\n")
            f.write(f"Training Return: {train_return:.4%}\n\n")
            
            f.write("===== Test Results =====\n")
            for key, value in test_results.items():
                if isinstance(value, float):
                    if "%" in key or "Return" in key or "Outperformance" in key or "Drawdown" in key or "Volatility" in key or "Costs" in key:
                        f.write(f"{key}: {value:.4%}\n")
                    elif "Sharpe" in key:
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value:.2f}\n")
                else:
                    f.write(f"{key}: {value}\n")
    else:
        # Save the optimal parameters for simple SMA strategy
        filename = os.path.join(RESULTS_DIR, f'optimal_parameters_simple_{CURRENCY.replace("/", "_")}.txt')
        with open(filename, 'w') as f:
            f.write("===== Optimal Parameters =====\n\n")
            f.write(f"Short MA: {params['short_window']}\n")
            f.write(f"Long MA: {params['long_window']}\n")
            f.write(f"Trading Cost: {TRADING_FEE_PCT:.4%} per trade\n\n")
            
            f.write("===== Training Results =====\n")
            f.write(f"Training Return: {train_return:.4%}\n\n")
            
            f.write("===== Test Results =====\n")
            for key, value in test_results.items():
                if isinstance(value, float):
                    if "%" in key or "Return" in key or "Outperformance" in key or "Drawdown" in key or "Volatility" in key or "Costs" in key:
                        f.write(f"{key}: {value:.4%}\n")
                    elif "Sharpe" in key:
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value:.2f}\n")
                else:
                    f.write(f"{key}: {value}\n")
    
    print(f"Results saved to {filename}")

# ==================== MAIN FUNCTION ====================
def main():
    start_time = time.time()
    print(f"Starting SMA Optimization for {CURRENCY}")
    print(f"Trading Cost per trade: {TRADING_FEE_PCT:.4%}")
    
    # Check if using position sizing
    if USE_POSITION_SIZING:
        print(f"Position sizing enabled: {POSITION_SIZING_METHOD}")
        print(f"Volatility calculation method: {VOLATILITY_METHOD}")
        print(f"Target volatility: {TARGET_VOLATILITY:.2%}")
    else:
        print("Position sizing disabled - using binary signals")
    
    # Run sectional testing if enabled (takes precedence)
    if 'USE_SECTIONAL_TESTING' in globals() and USE_SECTIONAL_TESTING:
        print(f"Running sectional testing from {TESTING_START} to {TESTING_END} in {SECTION_SIZE}-day sections")
        section_results = run_sectional_testing()
        
        # End execution after sectional testing
        end_time = time.time()
        print(f"\nTotal execution time: {(end_time - start_time) / 60:.2f} minutes")
        return
    
    # Run expanding window test if enabled
    if USE_EXPANDING_WINDOW:
        print(f"Running expanding window test from {INITIAL_WINDOW} to {TESTING_END}")
        window_results = run_expanding_window_test()
        
        # End execution after expanding window test
        end_time = time.time()
        print(f"\nTotal execution time: {(end_time - start_time) / 60:.2f} minutes")
        return
    
    # Regular single-period testing
    # Initialize database connection
    db = DatabaseHandler()
    
    # Fetch training data
    print(f"Fetching training data: {TRAINING_START} to {TRAINING_END}")
    train_df = db.get_historical_data(CURRENCY, TRAINING_START, TRAINING_END)
    
    if len(train_df) < 100:
        print(f"Insufficient training data for {CURRENCY}. Exiting.")
        db.close()
        return
    
    # Check if using volatility regimes or simple SMA
    if USE_VOLATILITY_REGIMES:
        print("Using volatility-based regime switching SMA strategy")
        # Find the optimal parameters for each volatility regime
        print(f"Finding optimal parameters for each volatility regime...")
        best_params, train_return, _ = find_optimal_volatility_settings(train_df)
    else:
        print("Using simple SMA strategy without volatility regimes")
        # Find the best single set of SMA parameters
        print(f"Finding optimal SMA parameters...")
        best_params, train_return, _ = find_best_simple_sma_parameters(train_df)
    
    # Fetch test data
    print(f"Fetching test data: {TESTING_START} to {TESTING_END}")
    test_df = db.get_historical_data(CURRENCY, TESTING_START, TESTING_END)
    
    if len(test_df) < 100:
        print(f"Insufficient test data for {CURRENCY}. Exiting.")
        db.close()
        return
    
    # Apply the strategy to the test data
    print(f"Applying strategy to test data...")
    if USE_VOLATILITY_REGIMES:
        result_df = apply_regime_specific_strategy(test_df, best_params)
        regime_data = True
    else:
        result_df = apply_simple_sma_strategy(test_df, best_params)
        regime_data = False
    
    # Analyze the test results
    test_results = analyze_performance(result_df, INITIAL_CAPITAL, regime_data)
    
    # Print the test results
    print("\n===== Test Results =====")
    for key, value in test_results.items():
        if isinstance(value, float):
            if "%" in key or "Return" in key or "Outperformance" in key or "Drawdown" in key or "Volatility" in key or "Costs" in key:
                print(f"{key}: {value:.4%}")
            elif "Sharpe" in key:
                print(f"{key}: {value:.4f}")
            else:
                print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")
    
    # Plot the test results
    plot_test_results(result_df, best_params, test_results)
    
    # Save the results
    save_results(best_params, train_return, test_results, using_regimes=USE_VOLATILITY_REGIMES)
    
    # Close database connection
    db.close()
    
    end_time = time.time()
    print(f"\nTotal execution time: {(end_time - start_time) / 60:.2f} minutes")

if __name__ == "__main__":
    main()