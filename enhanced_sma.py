# enhanced_sma.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import sys
import warnings
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.cluster import KMeans
from itertools import product
from datetime import datetime, timedelta
from statsmodels.nonparametric.kde import KDEUnivariate
from arch import arch_model
import joblib

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Import the database handler
try:
    from database import DatabaseHandler
    print("Successfully imported DatabaseHandler from database.py")
except ImportError as e:
    print(f"Error importing DatabaseHandler: {e}")
    sys.exit(1)  # Exit with error

# Try to import configuration
try:
    from enhanced_config import (
        TRADING_FREQUENCY,
        TRAINING_START,
        TRAINING_END,
        TESTING_START,
        TESTING_END,
        CURRENCY,
        INITIAL_CAPITAL,
        TRADING_FEE_PCT,
        RESULTS_DIR,
        SAVE_RESULTS,
        PLOT_RESULTS,
        STRATEGY_CONFIG
    )
    print("Successfully imported configuration from enhanced_config.py")
except ImportError as e:
    print(f"Error importing enhanced_config: {e}")
    try:
        # Try to import from config.py as fallback
        from config import (
            TRADING_FREQUENCY,
            TRAINING_START,
            TRAINING_END,
            TESTING_START,
            TESTING_END,
            CURRENCY,
            INITIAL_CAPITAL,
            TRADING_FEE_PCT,
            RESULTS_DIR,
            SAVE_RESULTS,
            PLOT_RESULTS,
            STRATEGY_CONFIG
        )
        print("Successfully imported configuration from config.py")
    except ImportError:
        print("Using default configuration values")
        
        # Default configuration if import fails
        TRADING_FREQUENCY = "1H"
        TRAINING_START = "2018-05-20"
        TRAINING_END = "2020-12-31"
        TESTING_START = "2021-01-01"
        TESTING_END = "2024-10-20"
        CURRENCY = "XRP/USD"
        INITIAL_CAPITAL = 10000
        TRADING_FEE_PCT = 0.001
        RESULTS_DIR = "enhanced_sma_results"
        SAVE_RESULTS = True
        PLOT_RESULTS = True

        # Default strategy config
        STRATEGY_CONFIG = {
            # Volatility calculation settings
            'volatility': {
                'methods': ['parkinson', 'garch', 'yang_zhang'],  # Different volatility calculation methods
                'lookback_periods': [20, 50, 100],  # Different lookback periods for volatility
                'regime_smoothing': 5,  # Days to smooth regime transitions
                'min_history_multiplier': 5,  # Minimum history required as multiplier of lookback
            },
            
            # Regime detection settings
            'regime_detection': {
                'method': 'kmeans',  # Options: 'kmeans', 'kde', 'quantile'
                'n_regimes': 3,  # Number of distinct volatility regimes
                'quantile_thresholds': [0.33, 0.67],  # Percentile thresholds for regime transitions
                'regime_stability_period': 48,  # Hours required before confirming regime change
            },
            
            # SMA strategy settings
            'sma': {
                'short_windows': [5, 8, 13, 21, 34],  # Fibonacci-based short windows
                'long_windows': [21, 34, 55, 89, 144],  # Fibonacci-based long windows
                'min_holding_period': 24,  # Minimum holding period in hours
                'trend_filter_period': 200,  # Period for trend strength calculation
                'trend_strength_threshold': 0.3,  # Minimum trend strength to take a position
            },
            
            # Risk management settings
            'risk_management': {
                'target_volatility': 0.15,  # Target annualized volatility
                'max_position_size': 1.0,  # Maximum position size
                'min_position_size': 0.1,  # Minimum position size
                'max_drawdown_exit': 0.15,  # Exit if drawdown exceeds this threshold
                'profit_taking_threshold': 0.05,  # Take profit at this threshold
                'trailing_stop_activation': 0.03,  # Activate trailing stop after this gain
                'trailing_stop_distance': 0.02,  # Trailing stop distance
            },
            
            # Cross-validation settings
            'cross_validation': {
                'n_splits': 5,  # Number of time series cross-validation splits
                'min_train_size': 90,  # Minimum training size in days
                'step_forward': 30,  # Step forward size in days for expanding window
                'validation_ratio': 0.3,  # Portion of training data to use for validation
            },
            
            # Parameter selection settings
            'parameter_selection': {
                'stability_weight': 0.5,  # Weight for parameter stability vs. performance
                'sharpe_weight': 0.4,  # Weight for Sharpe ratio in fitness function
                'sortino_weight': 0.3,  # Weight for Sortino ratio in fitness function
                'calmar_weight': 0.3,  # Weight for Calmar ratio in fitness function
            }
        }
# ==================== ADVANCED VOLATILITY CALCULATION ====================

def calculate_parkinson_volatility(df, window=20):
    """
    Calculate Parkinson volatility using high-low range.
    
    Parkinson volatility is more efficient than close-to-close volatility
    and is particularly useful for assets with significant intraday movement.
    
    Parameters:
        df (DataFrame): DataFrame with high_price and low_price columns
        window (int): Rolling window for volatility calculation
        
    Returns:
        Series: Parkinson volatility (annualized)
    """
    # Check if high/low data is available
    if 'high_price' not in df.columns or 'low_price' not in df.columns:
        print("Warning: high_price or low_price not found. Falling back to standard deviation.")
        return calculate_standard_volatility(df, window)
    
    # Constant factor in Parkinson volatility estimator
    factor = 1.0 / (4.0 * np.log(2.0))
    
    # Calculate log(high/low)^2
    log_hl_squared = np.log(df['high_price'] / df['low_price'])**2
    
    # Apply factor and calculate rolling mean
    parkinsons_var = factor * log_hl_squared.rolling(window=window).mean()
    
    # Convert variance to standard deviation (volatility)
    parkinsons_vol = np.sqrt(parkinsons_var)
    
    # Annualize the volatility based on trading frequency
    if TRADING_FREQUENCY == "1H":
        parkinsons_vol = parkinsons_vol * np.sqrt(24 * 365)
    elif TRADING_FREQUENCY == "1D":
        parkinsons_vol = parkinsons_vol * np.sqrt(365)
    
    return parkinsons_vol.fillna(method='bfill').fillna(method='ffill')

def calculate_standard_volatility(df, window=20):
    """
    Calculate standard volatility using close price returns.
    
    Parameters:
        df (DataFrame): DataFrame with close_price column
        window (int): Rolling window for volatility calculation
        
    Returns:
        Series: Standard deviation of returns (annualized)
    """
    # Calculate returns
    returns = df['close_price'].pct_change().fillna(0)
    
    # Calculate standard deviation
    volatility = returns.rolling(window=window).std()
    
    # Annualize volatility based on trading frequency
    if TRADING_FREQUENCY == "1H":
        volatility = volatility * np.sqrt(24 * 365)
    elif TRADING_FREQUENCY == "1D":
        volatility = volatility * np.sqrt(365)
    
    return volatility.fillna(method='bfill').fillna(method='ffill')

def calculate_yang_zhang_volatility(df, window=20):
    """
    Calculate Yang-Zhang volatility which combines overnight and intraday volatility.
    
    Parameters:
        df (DataFrame): DataFrame with open, high, low, close columns
        window (int): Rolling window for volatility calculation
        
    Returns:
        Series: Yang-Zhang volatility (annualized)
    """
    # Check if necessary data is available
    required_cols = ['open_price', 'high_price', 'low_price', 'close_price']
    if not all(col in df.columns for col in required_cols):
        print("Warning: Required columns for Yang-Zhang volatility not found. Using Parkinson.")
        return calculate_parkinson_volatility(df, window)
    
    # Calculate overnight returns (close to open)
    close_to_open = np.log(df['open_price'] / df['close_price'].shift(1))
    
    # Calculate open to close returns
    open_to_close = np.log(df['close_price'] / df['open_price'])
    
    # Calculate Rogers-Satchell volatility components
    rs_vol = (np.log(df['high_price'] / df['close_price']) * 
              np.log(df['high_price'] / df['open_price']) + 
              np.log(df['low_price'] / df['close_price']) * 
              np.log(df['low_price'] / df['open_price']))
    
    # Calculate the different variance components
    close_to_open_var = close_to_open.rolling(window=window).var()
    open_to_close_var = open_to_close.rolling(window=window).var()
    rs_var = rs_vol.rolling(window=window).mean()
    
    # Combine components with weights
    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    yang_zhang_var = close_to_open_var + k * open_to_close_var + (1 - k) * rs_var
    
    # Convert to volatility (standard deviation)
    yang_zhang_vol = np.sqrt(yang_zhang_var)
    
    # Annualize volatility based on trading frequency
    if TRADING_FREQUENCY == "1H":
        yang_zhang_vol = yang_zhang_vol * np.sqrt(24 * 365)
    elif TRADING_FREQUENCY == "1D":
        yang_zhang_vol = yang_zhang_vol * np.sqrt(365)
    
    return yang_zhang_vol.fillna(method='bfill').fillna(method='ffill')

def calculate_garch_volatility(df, forecast_horizon=1):
    """
    Calculate volatility using a GARCH(1,1) model.
    
    Parameters:
        df (DataFrame): DataFrame with close_price column
        forecast_horizon (int): Forecast horizon for volatility prediction
        
    Returns:
        Series: GARCH volatility forecast (annualized)
    """
    # Calculate returns
    returns = 100 * df['close_price'].pct_change().fillna(0)
    
    try:
        # Initialize GARCH model (reduced complexity for speed)
        model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
        
        # Fit the model with a fixed window to avoid recalculation for entire series
        # This uses a rolling estimation instead of expanding to improve performance
        window_size = min(1000, len(returns))
        
        # Create Series to store forecasted volatility
        forecasted_vol = pd.Series(index=returns.index, dtype=float)
        
        # For periods with enough data, estimate GARCH and forecast
        for i in range(window_size, len(returns), 100):  # Update every 100 steps for efficiency
            end_loc = min(i + 100, len(returns))
            try:
                # Get subseries for estimation
                subseries = returns.iloc[max(0, i-window_size):i]
                
                # Fit model
                res = model.fit(disp='off', show_warning=False, update_freq=0)
                
                # Forecast volatility
                forecast = res.forecast(horizon=forecast_horizon)
                conditional_vol = np.sqrt(forecast.variance.iloc[-1].values[0])
                
                # Assign forecasted volatility to next periods
                for j in range(i, min(end_loc, len(returns))):
                    if j < len(forecasted_vol):
                        forecasted_vol.iloc[j] = conditional_vol
            except:
                # Fall back to standard deviation on failure
                forecasted_vol.iloc[i:end_loc] = returns.iloc[max(0, i-window_size):i].std()
        
        # Fill any missing values with standard deviation
        forecasted_vol = forecasted_vol.fillna(returns.rolling(window=20).std())
        
        # Annualize volatility based on trading frequency
        if TRADING_FREQUENCY == "1H":
            annualized_vol = forecasted_vol * np.sqrt(24 * 365) / 100
        elif TRADING_FREQUENCY == "1D":
            annualized_vol = forecasted_vol * np.sqrt(365) / 100
        else:
            annualized_vol = forecasted_vol / 100
        
        return annualized_vol
    except Exception as e:
        print(f"GARCH estimation failed: {e}")
        print("Falling back to standard volatility")
        return calculate_standard_volatility(df)

def calculate_volatility(df, method='parkinson', window=20):
    """
    Calculate volatility using the specified method.
    
    Parameters:
        df (DataFrame): DataFrame with price data
        method (str): Method to use for volatility calculation
        window (int): Lookback period for volatility calculation
        
    Returns:
        Series: Volatility series
    """
    if method == 'parkinson':
        return calculate_parkinson_volatility(df, window)
    elif method == 'garch':
        return calculate_garch_volatility(df)
    elif method == 'yang_zhang':
        return calculate_yang_zhang_volatility(df, window)
    else:  # Use standard volatility as default
        return calculate_standard_volatility(df, window)

# ==================== ADVANCED REGIME DETECTION ====================

def detect_regimes_kmeans(volatility_series, n_regimes=3, smoothing_period=5):
    """
    Detect volatility regimes using K-means clustering.
    
    Parameters:
        volatility_series (Series): Volatility time series
        n_regimes (int): Number of regimes to identify
        smoothing_period (int): Period for smoothing regime transitions
        
    Returns:
        Series: Regime classifications (0 to n_regimes-1)
    """
    # Prepare data for clustering
    # Use log volatility to better capture distribution characteristics
    X = np.log(volatility_series.replace(0, np.nan).fillna(volatility_series.min())).values.reshape(-1, 1)
    
    # Train KMeans model
    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    # Sort regimes by volatility level (0=low, n_regimes-1=high)
    cluster_centers = kmeans.cluster_centers_.flatten()
    sorting_indices = np.argsort(cluster_centers)
    
    # Map original labels to sorted labels
    regime_map = {sorting_indices[i]: i for i in range(n_regimes)}
    sorted_labels = np.array([regime_map[label] for label in labels])
    
    # Create Series with regime labels
    regimes = pd.Series(sorted_labels, index=volatility_series.index)
    
    # Apply smoothing to prevent frequent regime transitions
    if smoothing_period > 1:
        regimes = regimes.rolling(window=smoothing_period, center=True).median().fillna(method='ffill').fillna(method='bfill')
        regimes = regimes.round().astype(int)
    
    return regimes

def detect_regimes_kde(volatility_series, n_regimes=3, smoothing_period=5):
    """
    Detect volatility regimes using Kernel Density Estimation.
    
    Parameters:
        volatility_series (Series): Volatility time series
        n_regimes (int): Number of regimes to identify
        smoothing_period (int): Period for smoothing regime transitions
        
    Returns:
        Series: Regime classifications (0 to n_regimes-1)
    """
    # Use log volatility to better capture distribution characteristics
    log_vol = np.log(volatility_series.replace(0, np.nan).fillna(volatility_series.min()))
    
    # Fit KDE
    kde = KDEUnivariate(log_vol.values)
    kde.fit()
    
    # Find local minima in the density to identify regime boundaries
    density = kde.density
    x_grid = kde.support
    
    # Use first and second derivatives to find local minima
    d_density = np.gradient(density)
    dd_density = np.gradient(d_density)
    
    # Find points where first derivative is close to zero and second derivative is positive
    # These are local minima which represent boundaries between regimes
    local_min_indices = []
    for i in range(1, len(d_density) - 1):
        if abs(d_density[i]) < 0.01 and dd_density[i] > 0:
            local_min_indices.append(i)
    
    # If we cannot find enough local minima, use quantiles instead
    if len(local_min_indices) < n_regimes - 1:
        print(f"Not enough local minima found. Using quantiles instead.")
        thresholds = [np.quantile(log_vol, q) for q in np.linspace(0, 1, n_regimes+1)[1:-1]]
    else:
        # Sort and select top n_regimes-1 boundaries
        local_min_indices.sort(key=lambda i: density[i])
        boundary_indices = local_min_indices[:n_regimes-1]
        boundary_indices.sort()  # Sort by x value for consistent thresholds
        thresholds = [x_grid[i] for i in boundary_indices]
    
    # Apply thresholds to classify regimes
    regimes = pd.Series(0, index=volatility_series.index)
    for i, threshold in enumerate(sorted(thresholds), 1):
        regimes[log_vol > threshold] = i
    
    # Apply smoothing to prevent frequent regime transitions
    if smoothing_period > 1:
        regimes = regimes.rolling(window=smoothing_period, center=True).median().fillna(method='ffill').fillna(method='bfill')
        regimes = regimes.round().astype(int)
    
    return regimes

def detect_regimes_quantile(volatility_series, quantile_thresholds=[0.33, 0.67], smoothing_period=5):
    """
    Detect volatility regimes using empirical quantiles.
    
    Parameters:
        volatility_series (Series): Volatility time series
        quantile_thresholds (list): List of quantile thresholds
        smoothing_period (int): Period for smoothing regime transitions
        
    Returns:
        Series: Regime classifications (0, 1, 2, etc.)
    """
    n_regimes = len(quantile_thresholds) + 1
    
    # Calculate expanding quantiles to avoid lookahead bias
    expanding_thresholds = []
    min_periods = 500  # Minimum data points for reliable quantile estimation
    
    for q in quantile_thresholds:
        threshold_series = volatility_series.expanding(min_periods=min_periods).quantile(q)
        threshold_series = threshold_series.fillna(volatility_series.median())  # Fill early periods
        expanding_thresholds.append(threshold_series)
    
    # Initialize regime series
    regimes = pd.Series(0, index=volatility_series.index)
    
    # Classify regimes based on thresholds
    for i, threshold_series in enumerate(expanding_thresholds, 1):
        mask = volatility_series > threshold_series
        regimes[mask] = i
    
    # Apply smoothing to prevent frequent regime transitions
    if smoothing_period > 1:
        regimes = regimes.rolling(window=smoothing_period, center=True).median().fillna(method='ffill').fillna(method='bfill')
        regimes = regimes.round().astype(int)
    
    return regimes

def detect_volatility_regimes(df, volatility, method='kmeans', n_regimes=3, 
                             quantile_thresholds=[0.33, 0.67], smoothing_period=5,
                             stability_period=48):
    """
    Detect volatility regimes using the specified method and apply stability constraints.
    
    Parameters:
        df (DataFrame): DataFrame with price data
        volatility (Series): Volatility series
        method (str): Method for regime detection
        n_regimes (int): Number of regimes to identify (for kmeans/kde)
        quantile_thresholds (list): Quantile thresholds (for quantile method)
        smoothing_period (int): Period for smoothing regime transitions
        stability_period (int): Hours required before confirming regime change
        
    Returns:
        Series: Regime classifications
    """
    # Detect regimes using selected method
    if method == 'kmeans':
        regimes = detect_regimes_kmeans(volatility, n_regimes, smoothing_period)
    elif method == 'kde':
        regimes = detect_regimes_kde(volatility, n_regimes, smoothing_period)
    else:  # Default to quantile method
        regimes = detect_regimes_quantile(volatility, quantile_thresholds, smoothing_period)
    
    # Apply stability constraint to prevent rapid regime transitions
    if stability_period > 1:
        stable_regimes = regimes.copy()
        
        # Track consecutive periods in the same regime
        last_regime = regimes.iloc[0]
        consecutive_periods = 0
        
        for i in range(len(regimes)):
            current_regime = regimes.iloc[i]
            
            if current_regime == last_regime:
                consecutive_periods += 1
            else:
                # Only change regime if it has persisted for stability_period
                if consecutive_periods >= stability_period:
                    # Confirmed regime change
                    last_regime = current_regime
                    consecutive_periods = 0
                else:
                    # Reject regime change, maintain previous regime
                    stable_regimes.iloc[i] = last_regime
                    consecutive_periods += 1
        
        return stable_regimes
    
    return regimes

# ==================== TREND STRENGTH & SIGNAL FILTERING ====================

def calculate_trend_strength(df, window=200):
    """
    Calculate trend strength indicator.
    
    Parameters:
        df (DataFrame): DataFrame with price data
        window (int): Window for trend calculation
        
    Returns:
        Series: Trend strength indicator
    """
    # Calculate moving average
    ma = df['close_price'].rolling(window=window).mean()
    
    # Calculate distance from MA normalized by volatility
    price_distance = (df['close_price'] - ma).abs()
    price_volatility = df['close_price'].rolling(window=window).std()
    
    # Calculate trend strength as normalized distance from MA
    trend_strength = price_distance / price_volatility
    
    # Determine trend direction (positive for uptrend, negative for downtrend)
    trend_direction = np.sign(df['close_price'] - ma)
    
    # Combine strength and direction
    directional_strength = trend_strength * trend_direction
    
    return directional_strength.fillna(0)

def calculate_momentum(df, window=20):
    """
    Calculate price momentum.
    
    Parameters:
        df (DataFrame): DataFrame with price data
        window (int): Window for momentum calculation
        
    Returns:
        Series: Momentum indicator
    """
    # Calculate momentum as percentage change over window
    momentum = df['close_price'].pct_change(periods=window).fillna(0)
    
    # Normalize momentum by dividing by volatility over same period
    volatility = df['close_price'].pct_change().rolling(window=window).std().fillna(0)
    
    # Avoid division by zero
    volatility = volatility.replace(0, volatility.median())
    
    # Calculate normalized momentum
    normalized_momentum = momentum / volatility
    
    return normalized_momentum

def filter_signals(signal, trend_strength, momentum, min_trend_strength=0.3):
    """
    Filter trading signals based on trend strength and momentum.
    
    Parameters:
        signal (Series): Raw trading signals (-1, 0, 1)
        trend_strength (Series): Trend strength indicator
        momentum (Series): Momentum indicator
        min_trend_strength (float): Minimum trend strength to take a position
        
    Returns:
        Series: Filtered trading signals
    """
    # Initialize filtered signal
    filtered_signal = pd.Series(0, index=signal.index)
    
    # Take long positions only in uptrend with sufficient strength
    long_condition = (signal > 0) & (trend_strength > min_trend_strength) & (momentum > 0)
    filtered_signal[long_condition] = 1
    
    # Take short positions only in downtrend with sufficient strength
    short_condition = (signal < 0) & (trend_strength < -min_trend_strength) & (momentum < 0)
    filtered_signal[short_condition] = -1
    
    return filtered_signal

def apply_min_holding_period(position, min_holding_hours=24):
    """
    Apply minimum holding period to reduce overtrading.
    
    Parameters:
        position (Series): Position series (-1, 0, 1)
        min_holding_hours (int): Minimum holding period in hours
        
    Returns:
        Series: Position with minimum holding period applied
    """
    if min_holding_hours <= 1:
        return position
    
    modified_position = position.copy()
    
    # Track last trade and holding period
    last_trade_time = None
    last_position = 0
    
    for i, (timestamp, current_position) in enumerate(position.items()):
        # Position change detected
        if current_position != last_position and current_position != 0:
            # Check if minimum holding period has passed
            if last_trade_time is not None:
                hours_since_last_trade = (timestamp - last_trade_time).total_seconds() / 3600
                
                if hours_since_last_trade < min_holding_hours:
                    # Reject this trade, keep previous position
                    modified_position.iloc[i] = last_position
                    continue
            
            # Update last trade time and position
            last_trade_time = timestamp
            last_position = current_position
        
        # Position closed (moved to neutral)
        elif current_position == 0 and last_position != 0:
            # Update position
            last_position = 0
    
    return modified_position

# ==================== POSITION SIZING & RISK MANAGEMENT ====================

def calculate_adaptive_position_size(volatility, target_vol=0.15, max_size=1.0, min_size=0.1):
    """
    Scale position size inversely with volatility using a continuous function.
    
    Parameters:
        volatility (Series): Volatility series
        target_vol (float): Target annualized volatility
        max_size (float): Maximum position size
        min_size (float): Minimum position size
        
    Returns:
        Series: Position size scaling factor
    """
    # Avoid division by zero
    safe_volatility = volatility.replace(0, volatility.median())
    
    # Calculate position scale based on volatility
    position_scale = target_vol / safe_volatility
    
    # Apply limits
    position_scale = position_scale.clip(lower=min_size, upper=max_size)
    
    return position_scale

def apply_trailing_stop(df, position, returns, activation_threshold=0.03, stop_distance=0.02):
    """
    Apply trailing stop loss to protect profits.
    
    Parameters:
        df (DataFrame): DataFrame with price data
        position (Series): Position series (-1, 0, 1)
        returns (Series): Returns series
        activation_threshold (float): Profit threshold to activate trailing stop
        stop_distance (float): Distance to maintain trailing stop
        
    Returns:
        Series: Position with trailing stop applied
    """
    modified_position = position.copy()
    
    # Calculate cumulative returns for each trade
    trade_returns = pd.Series(0.0, index=position.index)
    running_return = 0.0
    entry_price = None
    
    for i in range(len(position)):
        # New position
        if i > 0 and position.iloc[i] != 0 and position.iloc[i-1] == 0:
            entry_price = df['close_price'].iloc[i]
            running_return = 0.0
        
        # Maintaining position
        elif i > 0 and position.iloc[i] != 0 and position.iloc[i] == position.iloc[i-1]:
            running_return = (running_return + 1) * (1 + returns.iloc[i]) - 1
        
        # Closed position
        elif i > 0 and position.iloc[i] == 0 and position.iloc[i-1] != 0:
            running_return = 0.0
            entry_price = None
        
        trade_returns.iloc[i] = running_return
    
    # Track highest return achieved during each trade
    highest_return = trade_returns.copy()
    for i in range(1, len(trade_returns)):
        if position.iloc[i] == position.iloc[i-1] and position.iloc[i] != 0:
            highest_return.iloc[i] = max(highest_return.iloc[i-1], trade_returns.iloc[i])
    
    # Apply trailing stop
    for i in range(1, len(position)):
        # Check if in a position and trailing stop is activated
        if position.iloc[i] != 0 and highest_return.iloc[i] >= activation_threshold:
            # Calculate drawdown from highest point
            drawdown = (trade_returns.iloc[i] - highest_return.iloc[i])
            
            # Close position if trailing stop is hit
            if drawdown < -stop_distance:
                modified_position.iloc[i] = 0
    
    return modified_position

def apply_stop_loss(position, returns, trade_returns, max_drawdown=0.15):
    """
    Apply maximum drawdown stop loss.
    
    Parameters:
        position (Series): Position series (-1, 0, 1)
        returns (Series): Returns series
        trade_returns (Series): Cumulative returns for each trade
        max_drawdown (float): Maximum allowed drawdown
        
    Returns:
        Series: Position with stop loss applied
    """
    modified_position = position.copy()
    
    # Track highest return achieved during each trade
    highest_return = pd.Series(0.0, index=position.index)
    
    for i in range(1, len(position)):
        if position.iloc[i] == position.iloc[i-1] and position.iloc[i] != 0:
            highest_return.iloc[i] = max(highest_return.iloc[i-1], trade_returns.iloc[i])
        else:
            highest_return.iloc[i] = trade_returns.iloc[i]
    
    # Apply stop loss
    for i in range(1, len(position)):
        if position.iloc[i] != 0:
            # Calculate drawdown from highest point
            drawdown = (trade_returns.iloc[i] - highest_return.iloc[i])
            
            # Close position if drawdown exceeds maximum
            if drawdown < -max_drawdown:
                modified_position.iloc[i] = 0
    
    return modified_position

def apply_profit_taking(position, trade_returns, profit_threshold=0.05):
    """
    Apply profit taking rule.
    
    Parameters:
        position (Series): Position series (-1, 0, 1)
        trade_returns (Series): Cumulative returns for each trade
        profit_threshold (float): Profit threshold to take profits
        
    Returns:
        Series: Position with profit taking applied
    """
    modified_position = position.copy()
    
    # Apply profit taking
    for i in range(1, len(position)):
        if position.iloc[i] != 0 and trade_returns.iloc[i] >= profit_threshold:
            modified_position.iloc[i] = 0
    
    return modified_position

# ==================== ADVANCED BACKTEST & PERFORMANCE ====================

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, annualization_factor=None):
    """
    Calculate Sharpe ratio.
    
    Parameters:
        returns (Series): Returns series
        risk_free_rate (float): Risk-free rate
        annualization_factor (float): Factor to annualize returns
        
    Returns:
        float: Sharpe ratio
    """
    if annualization_factor is None:
        # Determine annualization factor from data frequency
        if TRADING_FREQUENCY == "1H":
            annualization_factor = np.sqrt(24 * 365)
        elif TRADING_FREQUENCY == "1D":
            annualization_factor = np.sqrt(365)
        else:
            annualization_factor = 1
    
    # Remove outliers
    clean_returns = returns.clip(returns.quantile(0.01), returns.quantile(0.99))
    
    mean_return = clean_returns.mean()
    std_return = clean_returns.std()
    
    if std_return == 0:
        return 0
    
    sharpe = ((mean_return - risk_free_rate) / std_return) * annualization_factor
    
    return sharpe

def calculate_sortino_ratio(returns, risk_free_rate=0.0, annualization_factor=None):
    """
    Calculate Sortino ratio (using only downside deviation).
    
    Parameters:
        returns (Series): Returns series
        risk_free_rate (float): Risk-free rate
        annualization_factor (float): Factor to annualize returns
        
    Returns:
        float: Sortino ratio
    """
    if annualization_factor is None:
        # Determine annualization factor from data frequency
        if TRADING_FREQUENCY == "1H":
            annualization_factor = np.sqrt(24 * 365)
        elif TRADING_FREQUENCY == "1D":
            annualization_factor = np.sqrt(365)
        else:
            annualization_factor = 1
    
    # Calculate downside returns (returns below target, typically 0)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf  # No downside risk
    
    # Calculate downside deviation
    downside_deviation = np.sqrt(np.mean(downside_returns**2))
    
    if downside_deviation == 0:
        return 0
    
    # Calculate Sortino ratio
    sortino = ((returns.mean() - risk_free_rate) / downside_deviation) * annualization_factor
    
    return sortino

def calculate_max_drawdown(equity_curve):
    """
    Calculate maximum drawdown from an equity curve.
    
    Parameters:
        equity_curve (Series): Equity curve
        
    Returns:
        float: Maximum drawdown
    """
    # Calculate running maximum
    running_max = equity_curve.cummax()
    
    # Calculate drawdown
    drawdown = (equity_curve / running_max) - 1
    
    # Find maximum drawdown
    max_drawdown = drawdown.min()
    
    return max_drawdown

def calculate_calmar_ratio(returns, max_drawdown, annualization_factor=None):
    """
    Calculate Calmar ratio (return / max drawdown).
    
    Parameters:
        returns (Series): Returns series
        max_drawdown (float): Maximum drawdown (positive number)
        annualization_factor (float): Factor to annualize returns
        
    Returns:
        float: Calmar ratio
    """
    if annualization_factor is None:
        # Determine annualization factor from data frequency
        if TRADING_FREQUENCY == "1H":
            annualization_factor = np.sqrt(24 * 365)
        elif TRADING_FREQUENCY == "1D":
            annualization_factor = np.sqrt(365)
        else:
            annualization_factor = 1
    
    if max_drawdown == 0:
        return np.inf  # No drawdown
    
    # Convert max_drawdown to positive value if needed
    abs_drawdown = abs(max_drawdown)
    
    # Calculate annualized return
    annual_return = returns.mean() * annualization_factor
    
    calmar = annual_return / abs_drawdown
    
    return calmar

def calculate_advanced_metrics(strategy_returns, equity_curve):
    """
    Calculate advanced performance metrics for the strategy.
    
    Parameters:
        strategy_returns (Series): Strategy returns series
        equity_curve (Series): Strategy equity curve
        
    Returns:
        dict: Performance metrics
    """
    # Basic metrics
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0] - 1
    annualized_return = ((1 + total_return) ** (365 / (equity_curve.index[-1] - equity_curve.index[0]).days)) - 1
    
    # Risk metrics
    volatility = strategy_returns.std() * np.sqrt(252)  # Annualized
    max_dd = calculate_max_drawdown(equity_curve)
    
    # Risk-adjusted metrics
    sharpe = calculate_sharpe_ratio(strategy_returns)
    sortino = calculate_sortino_ratio(strategy_returns)
    calmar = calculate_calmar_ratio(strategy_returns, max_dd)
    
    # Efficiency metrics - handle case with no trades
    non_zero_returns = strategy_returns[strategy_returns != 0]
    if len(non_zero_returns) > 0:
        win_rate = len(strategy_returns[strategy_returns > 0]) / len(non_zero_returns)
        gain_sum = strategy_returns[strategy_returns > 0].sum()
        loss_sum = abs(strategy_returns[strategy_returns < 0].sum())
        gain_to_pain = gain_sum / loss_sum if loss_sum > 0 else np.inf
    else:
        # No trades case
        win_rate = 0.0
        gain_to_pain = 0.0
    
    # Return dictionary of metrics
    metrics = {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'max_drawdown': max_dd,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'win_rate': win_rate,
        'gain_to_pain': gain_to_pain
    }
    
    return metrics

def calculate_parameter_stability(param_results):
    """
    Calculate parameter stability score across validation periods.
    
    Parameters:
        param_results (list): List of parameter results across validation periods
        
    Returns:
        float: Stability score (higher is better)
    """
    # Extract parameter values from results
    param_values = []
    for result in param_results:
        param_values.append(result['best_params'])
    
    # Check if we have enough results to measure stability
    if len(param_values) < 2:
        return 0.0
    
    # Initialize stability score
    stability_score = 0.0
    
    # For each parameter, calculate variation across validation periods
    for param_name in param_values[0].keys():
        param_series = [params[param_name] for params in param_values if param_name in params]
        
        # Skip if not enough data
        if len(param_series) < 2:
            continue
        
        # Handle different parameter types differently
        if all(isinstance(x, (int, float)) for x in param_series):
            # For numeric parameters: calculate coefficient of variation
            param_mean = np.mean(param_series)
            param_std = np.std(param_series)
            
            # Avoid division by zero
            if param_mean == 0:
                cv = 0
            else:
                cv = param_std / abs(param_mean)
                
            # Convert to stability score (higher is better)
            param_stability = 1 / (1 + cv)
        else:
            # For string/categorical parameters: calculate consistency
            # (fraction of values matching the most common value)
            from collections import Counter
            counts = Counter(param_series)
            most_common_count = counts.most_common(1)[0][1]
            param_stability = most_common_count / len(param_series)
        
        # Add to overall stability score
        stability_score += param_stability
    
    # Normalize by number of parameters
    stability_score /= len(param_values[0])
    
    return stability_score

# ==================== TIME SERIES CROSS VALIDATION ====================

def generate_time_series_cv_splits(start_date, end_date, n_splits=5, min_train_size=90, step_forward=30):
    """
    Generate time series cross-validation splits with expanding window.
    
    Parameters:
        start_date (str or datetime): Start date
        end_date (str or datetime): End date
        n_splits (int): Number of splits
        min_train_size (int): Minimum training size in days
        step_forward (int): Step forward size in days
        
    Returns:
        list: List of (train_start, train_end, val_start, val_end) tuples
    """
    # Convert dates to pandas datetime
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Calculate total days
    total_days = (end - start).days
    
    # Ensure we have enough data
    if total_days < min_train_size + step_forward:
        raise ValueError(f"Not enough data for cross-validation. Need at least {min_train_size + step_forward} days.")
    
    # Generate splits
    splits = []
    
    # Calculate validation period size
    val_size = min(step_forward, total_days // (n_splits + 1))
    
    for i in range(n_splits):
        # Calculate train end date
        train_size = min_train_size + i * step_forward
        if train_size > total_days - val_size:
            break
        
        train_start = start
        train_end = start + timedelta(days=train_size)
        val_start = train_end
        val_end = val_start + timedelta(days=val_size)
        
        # Ensure validation end date doesn't exceed overall end date
        if val_end > end:
            val_end = end
        
        splits.append((train_start, train_end, val_start, val_end))
    
    return splits

def calculate_fitness_score(metrics, config):
    """
    Calculate combined fitness score based on multiple metrics.
    
    Parameters:
        metrics (dict): Dictionary of performance metrics
        config (dict): Configuration with weights
        
    Returns:
        float: Combined fitness score
    """
    # Extract weights from config
    sharpe_weight = config['parameter_selection']['sharpe_weight']
    sortino_weight = config['parameter_selection']['sortino_weight']
    calmar_weight = config['parameter_selection']['calmar_weight']
    
    # Normalize weights
    total_weight = sharpe_weight + sortino_weight + calmar_weight
    sharpe_weight /= total_weight
    sortino_weight /= total_weight
    calmar_weight /= total_weight
    
    # Calculate weighted score
    fitness_score = (
        sharpe_weight * metrics['sharpe_ratio'] +
        sortino_weight * metrics['sortino_ratio'] +
        calmar_weight * metrics['calmar_ratio']
    )
    
    # Apply penalty for excessive drawdown
    if metrics['max_drawdown'] < -0.3:  # 30% drawdown threshold
        fitness_score *= (1 + metrics['max_drawdown'])  # Reduce score proportionally
    
    return fitness_score

def optimize_parameters_with_cv(df, config, splits):
    """
    Optimize strategy parameters using time series cross-validation.
    
    Parameters:
        df (DataFrame): DataFrame with price data
        config (dict): Strategy configuration
        splits (list): Cross-validation splits
        
    Returns:
        dict: Optimal parameters
        list: Results for each validation period
    """
    print("Optimizing parameters with time series cross-validation...")
    
    # Extract parameter ranges
    vol_methods = config['volatility']['methods']
    vol_lookbacks = config['volatility']['lookback_periods']
    regime_method = config['regime_detection']['method']
    n_regimes = config['regime_detection']['n_regimes']
    
    # Define parameter grid for optimization
    param_grid = []
    for vol_method in vol_methods:
        for vol_lookback in vol_lookbacks:
            for short_window in config['sma']['short_windows']:
                for long_window in config['sma']['long_windows']:
                    if short_window >= long_window:
                        continue
                    
                    param_set = {
                        'vol_method': vol_method,
                        'vol_lookback': vol_lookback,
                        'short_window': short_window,
                        'long_window': long_window
                    }
                    param_grid.append(param_set)
    
    # Store results for each validation period
    cv_results = []
    
    # Track best parameters across all CV folds
    all_fold_results = []
    
    # For each cross-validation split
    for i, (train_start, train_end, val_start, val_end) in enumerate(splits):
        print(f"\nCV Fold {i+1}/{len(splits)}:")
        print(f"  Training: {train_start.strftime('%Y-%m-%d')} to {train_end.strftime('%Y-%m-%d')}")
        print(f"  Validation: {val_start.strftime('%Y-%m-%d')} to {val_end.strftime('%Y-%m-%d')}")
        
        # Extract training and validation data
        train_df = df.loc[train_start:train_end].copy()
        val_df = df.loc[val_start:val_end].copy()
        
        # Skip if not enough data
        if len(train_df) < 100 or len(val_df) < 20:
            print("  Not enough data, skipping fold")
            continue
        
        # Store best parameters and score for this fold
        best_params = None
        best_score = -np.inf
        best_metrics = None
        
        # Track fold results for each parameter set
        fold_results = []
        
        # Test each parameter combination
        for j, params in enumerate(param_grid):
            # Extract parameters
            vol_method = params['vol_method']
            vol_lookback = params['vol_lookback']
            short_window = params['short_window']
            long_window = params['long_window']
            
            try:
                # Calculate volatility for training data
                train_volatility = calculate_volatility(train_df, method=vol_method, window=vol_lookback)
                
                # Detect regimes for training data
                train_regimes = detect_volatility_regimes(
                    train_df, 
                    train_volatility, 
                    method=regime_method,
                    n_regimes=n_regimes,
                    stability_period=config['regime_detection']['regime_stability_period']
                )
                
                # Calculate trend strength
                trend_strength = calculate_trend_strength(train_df, window=config['sma']['trend_filter_period'])
                
                # Calculate momentum
                momentum = calculate_momentum(train_df, window=vol_lookback)
                
                # Calculate SMA signals
                short_ma = train_df['close_price'].rolling(window=short_window).mean()
                long_ma = train_df['close_price'].rolling(window=long_window).mean()
                
                # Generate raw signal
                raw_signal = pd.Series(0, index=train_df.index)
                raw_signal[short_ma > long_ma] = 1
                raw_signal[short_ma < long_ma] = -1
                
                # Filter signals
                filtered_signal = filter_signals(
                    raw_signal, 
                    trend_strength, 
                    momentum,
                    min_trend_strength=config['sma']['trend_strength_threshold']
                )
                
                # Apply minimum holding period
                position = apply_min_holding_period(
                    filtered_signal,
                    min_holding_hours=config['sma']['min_holding_period']
                )
                
                # Calculate position size based on volatility
                position_size = calculate_adaptive_position_size(
                    train_volatility,
                    target_vol=config['risk_management']['target_volatility'],
                    max_size=config['risk_management']['max_position_size'],
                    min_size=config['risk_management']['min_position_size']
                )
                
                # Apply position sizing
                sized_position = position * position_size
                
                # Calculate returns
                train_returns = train_df['close_price'].pct_change().fillna(0)
                
                # Calculate trade returns for risk management
                trade_returns = pd.Series(0.0, index=sized_position.index)
                for k in range(1, len(sized_position)):
                    if sized_position.iloc[k] != 0:
                        if sized_position.iloc[k] == sized_position.iloc[k-1]:
                            # Continuing the same position
                            trade_returns.iloc[k] = (1 + trade_returns.iloc[k-1]) * (1 + train_returns.iloc[k] * sized_position.iloc[k-1]) - 1
                        else:
                            # New position
                            trade_returns.iloc[k] = train_returns.iloc[k] * sized_position.iloc[k]
                    else:
                        # No position
                        trade_returns.iloc[k] = 0
                
                # Apply risk management (stop loss and profit taking)
                managed_position = apply_stop_loss(
                    sized_position, 
                    train_returns, 
                    trade_returns,
                    max_drawdown=config['risk_management']['max_drawdown_exit']
                )
                
                managed_position = apply_profit_taking(
                    managed_position,
                    trade_returns,
                    profit_threshold=config['risk_management']['profit_taking_threshold']
                )
                
                # Calculate strategy returns
                strategy_returns = managed_position.shift(1).fillna(0) * train_returns
                
                # Calculate equity curve
                equity_curve = (1 + strategy_returns).cumprod()
                
                # Calculate performance metrics
                metrics = calculate_advanced_metrics(strategy_returns, equity_curve)
                
                # Calculate fitness score
                score = calculate_fitness_score(metrics, config)
                
                # Store fold result
                fold_result = {
                    'params': params,
                    'metrics': metrics,
                    'score': score
                }
                fold_results.append(fold_result)
                
                # Update best parameters if better
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    best_metrics = metrics.copy()
                
                # Print progress
                if (j+1) % 50 == 0 or (j+1) == len(param_grid):
                    print(f"  Tested {j+1}/{len(param_grid)} parameter combinations")
            
            except Exception as e:
                print(f"  Error testing parameters {params}: {e}")
                continue
        
        # Sort fold results by score
        fold_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Store top results for this fold
        cv_results.append({
            'fold': i,
            'best_params': best_params,
            'best_score': best_score,
            'best_metrics': best_metrics,
            'all_results': fold_results[:10]  # Store top 10 results
        })
        
        # Extend all fold results
        all_fold_results.extend(fold_results)
        
        # Print best parameters for this fold
        print(f"  Best parameters for fold {i+1}:")
        print(f"    {best_params}")
        print(f"  Best score: {best_score:.4f}")
        print(f"  Sharpe: {best_metrics['sharpe_ratio']:.4f}, Sortino: {best_metrics['sortino_ratio']:.4f}, Calmar: {best_metrics['calmar_ratio']:.4f}")
        print(f"  Return: {best_metrics['total_return']:.4%}, Max DD: {best_metrics['max_drawdown']:.4%}")
    
    # Calculate parameter stability across folds
    stability_score = calculate_parameter_stability([result for result in cv_results if 'best_params' in result])
    
    print(f"\nParameter stability score: {stability_score:.4f}")
    
    # Combine results across all folds
    combined_results = {}
    for param_set in param_grid:
        # Create parameter key
        param_key = f"{param_set['vol_method']}_{param_set['vol_lookback']}_{param_set['short_window']}_{param_set['long_window']}"
        
        # Find matching results
        matching_results = [result for result in all_fold_results if result['params'] == param_set]
        
        if matching_results:
            # Calculate average score across folds
            avg_score = np.mean([result['score'] for result in matching_results])
            avg_sharpe = np.mean([result['metrics']['sharpe_ratio'] for result in matching_results])
            avg_sortino = np.mean([result['metrics']['sortino_ratio'] for result in matching_results])
            avg_calmar = np.mean([result['metrics']['calmar_ratio'] for result in matching_results])
            avg_return = np.mean([result['metrics']['total_return'] for result in matching_results])
            avg_drawdown = np.mean([result['metrics']['max_drawdown'] for result in matching_results])
            
            # Count number of folds
            fold_count = len(matching_results)
            
            # Store combined result
            combined_results[param_key] = {
                'params': param_set,
                'avg_score': avg_score,
                'avg_sharpe': avg_sharpe,
                'avg_sortino': avg_sortino,
                'avg_calmar': avg_calmar,
                'avg_return': avg_return,
                'avg_drawdown': avg_drawdown,
                'fold_count': fold_count
            }
    
    # Convert to list and sort by average score
    combined_results_list = list(combined_results.values())
    combined_results_list.sort(key=lambda x: x['avg_score'], reverse=True)
    
    # Apply stability weight to score
    stability_weight = config['parameter_selection']['stability_weight']
    
    # Select best parameter set considering both performance and stability
    for result in combined_results_list:
        # Only consider parameter sets tested in most folds
        if result['fold_count'] >= len(cv_results) * 0.8:
            # Adjust score with stability
            result['final_score'] = result['avg_score'] * (1 - stability_weight) + stability_score * stability_weight
    
    # Sort by final score
    combined_results_list.sort(key=lambda x: x.get('final_score', 0), reverse=True)
    
    # Get best parameters
    best_overall_params = combined_results_list[0]['params'] if combined_results_list else None
    
    print("\nTop 5 parameter sets across all folds:")
    for i, result in enumerate(combined_results_list[:5]):
        print(f"{i+1}. {result['params']}")
        print(f"   Avg Score: {result['avg_score']:.4f}, Folds: {result['fold_count']}/{len(cv_results)}")
        print(f"   Avg Sharpe: {result['avg_sharpe']:.4f}, Avg Sortino: {result['avg_sortino']:.4f}, Avg Calmar: {result['avg_calmar']:.4f}")
        print(f"   Avg Return: {result['avg_return']:.4%}, Avg Max DD: {result['avg_drawdown']:.4%}")
        if 'final_score' in result:
            print(f"   Final Score (with stability): {result['final_score']:.4f}")
    
    print(f"\nSelected best parameters: {best_overall_params}")
    
    return best_overall_params, cv_results

# Add these improved functions to your enhanced_sma.py file

def calculate_adaptive_position_size_with_schedule(volatility, regime, timestamp, 
                                                  target_vol=0.15, max_size=1.0, min_size=0.1,
                                                  rebalance_frequency="daily", 
                                                  materiality_threshold=0.05,
                                                  regime_opt_out=None,
                                                  regime_position_factors=None,
                                                  use_dynamic_materiality=False,
                                                  dynamic_materiality=None):
    """
    Scale position size inversely with volatility using a scheduled rebalancing approach.
    Position sizing is regime-aware and can use different factors for different regimes.
    
    Parameters:
        volatility (Series): Volatility series
        regime (Series): Regime classifications
        timestamp (DatetimeIndex): Timestamps for the series
        target_vol (float): Target annualized volatility
        max_size (float): Maximum position size
        min_size (float): Minimum position size
        rebalance_frequency (str): How often to rebalance - "daily", "weekly", or "hourly"
        materiality_threshold (float): Only rebalance if position size change exceeds this percentage
        regime_opt_out (dict): Dictionary specifying which regimes to opt out from trading (True = opt out)
        regime_position_factors (dict): Dictionary specifying position size factors for each regime
        use_dynamic_materiality (bool): Whether to use regime-specific materiality thresholds
        dynamic_materiality (dict): Dictionary with regime-specific materiality thresholds
        
    Returns:
        Series: Position size scaling factor
    """
    # Avoid division by zero
    safe_volatility = volatility.replace(0, volatility.median())
    
    # Calculate base position scale based on volatility
    position_scale = target_vol / safe_volatility
    
    # Create regime adjustment factors using config values if provided
    if regime_position_factors is not None:
        regime_factors = pd.Series(1.0, index=regime.index)
        for r, factor in regime_position_factors.items():
            regime_factors[regime == r] = factor
    else:
        # Default regime factors if not provided
        regime_factors = pd.Series(1.0, index=regime.index)
        regime_factors[regime == 1] = 0.8  # Medium volatility: 80% position size
        regime_factors[regime == 2] = 0.2  # High volatility: 20% position size
    
    # Apply opt-out for specific regimes if specified
    if regime_opt_out is not None:
        for r, opt_out in regime_opt_out.items():
            if opt_out:
                # Set position size to 0 for opted-out regimes
                regime_factors[regime == r] = 0.0
    
    # Apply regime adjustment
    position_scale = position_scale * regime_factors
    
    # Apply limits
    position_scale = position_scale.clip(lower=min_size, upper=max_size)
    
    # Set position size to 0 for opted-out regimes (need to do this after clipping)
    if regime_opt_out is not None:
        for r, opt_out in regime_opt_out.items():
            if opt_out:
                position_scale[regime == r] = 0.0
    
    # Determine materiality threshold for each timestamp based on regime
    if use_dynamic_materiality and dynamic_materiality is not None:
        materiality_thresholds = pd.Series(materiality_threshold, index=regime.index)
        for r, threshold in dynamic_materiality.items():
            materiality_thresholds[regime == r] = threshold
    else:
        # Use the same threshold for all regimes
        materiality_thresholds = pd.Series(materiality_threshold, index=regime.index)
    
    # Apply scheduled rebalancing to reduce trading frequency
    rebalanced_scale = apply_regime_aware_rebalancing(
        position_scale, 
        regime,
        timestamp, 
        rebalance_frequency, 
        materiality_thresholds
    )
    
    return rebalanced_scale

def apply_regime_aware_rebalancing(position_scale, regime, timestamp, frequency="daily", materiality_thresholds=None):
    """
    Apply a rebalancing schedule to position sizes with awareness of regime changes.
    
    Parameters:
        position_scale (Series): Original position scale series
        regime (Series): Regime classifications
        timestamp (DatetimeIndex): Timestamps for the series
        frequency (str): Rebalancing frequency - "daily", "weekly", or "hourly"
        materiality_thresholds (Series): Regime-specific materiality thresholds
        
    Returns:
        Series: Rebalanced position scale series
    """
    # Create a copy to avoid modifying the original
    rebalanced_scale = position_scale.copy()
    
    # Initialize with the first position
    current_position = position_scale.iloc[0]
    last_rebalance_time = None
    current_regime = regime.iloc[0]
    
    # Go through each timestamp
    for i, (ts, new_position) in enumerate(zip(timestamp, position_scale)):
        # Skip the first position as it's already set
        if i == 0:
            last_rebalance_time = ts
            continue
        
        # Get the current regime at this timestamp
        new_regime = regime.iloc[i]
        
        # Regime change always triggers rebalancing
        regime_change = new_regime != current_regime
        
        # Determine if it's time to rebalance
        rebalance_time = False
        
        if frequency == "daily":
            # Rebalance at the beginning of each day
            if ts.date() != last_rebalance_time.date():
                rebalance_time = True
        elif frequency == "weekly":
            # Rebalance at the beginning of each week
            if ts.isocalendar()[1] != last_rebalance_time.isocalendar()[1]:
                rebalance_time = True
        elif frequency == "hourly":
            # Rebalance every N hours
            rebalance_hours = 4  # Adjust as needed
            hour_diff = (ts - last_rebalance_time).total_seconds() / 3600
            if hour_diff >= rebalance_hours:
                rebalance_time = True
        
        # Get the appropriate materiality threshold for this timestamp
        if materiality_thresholds is not None:
            threshold = materiality_thresholds.iloc[i]
        else:
            threshold = 0.05  # Default
        
        # Special case: if BOTH current and new positions are zero, don't consider it a change
        if current_position == 0 and new_position == 0:
            material_change = False
        # Special case: if going from zero to non-zero, always rebalance
        elif current_position == 0 and new_position != 0:
            material_change = True
        # Special case: if going from non-zero to zero, always rebalance
        elif current_position != 0 and new_position == 0:
            material_change = True
        # Normal case: calculate percentage change
        else:
            position_change_pct = abs(new_position - current_position) / current_position
            material_change = position_change_pct > threshold
        
        # Rebalance if it's time and the change is material, or if regime changed
        if (rebalance_time and material_change) or regime_change:
            current_position = new_position
            last_rebalance_time = ts
            current_regime = new_regime
        else:
            # Keep the previous position
            rebalanced_scale.iloc[i] = current_position
    
    return rebalanced_scale

def calculate_rsi(prices, window=14):
    """
    Calculate the Relative Strength Index (RSI).
    
    Parameters:
        prices (Series): Price series
        window (int): RSI calculation period
        
    Returns:
        Series: RSI values
    """
    # Calculate price changes
    delta = prices.diff().dropna()
    
    # Separate gains and losses
    gains = delta.copy()
    losses = delta.copy()
    
    gains[gains < 0] = 0
    losses[losses > 0] = 0
    losses = abs(losses)
    
    # Calculate average gains and losses
    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def apply_rebalancing_schedule(position_scale, timestamp, frequency="daily", materiality_threshold=0.05):
    """
    Apply a rebalancing schedule to position sizes to reduce trading frequency.
    
    Parameters:
        position_scale (Series): Original position scale series
        timestamp (DatetimeIndex): Timestamps for the series
        frequency (str): Rebalancing frequency - "daily", "weekly", or "hourly"
        materiality_threshold (float): Only rebalance if position change exceeds this percentage
        
    Returns:
        Series: Rebalanced position scale series
    """
    # Create a copy to avoid modifying the original
    rebalanced_scale = position_scale.copy()
    
    # Initialize with the first position
    current_position = position_scale.iloc[0]
    last_rebalance_time = None
    
    # Go through each timestamp
    for i, (ts, new_position) in enumerate(zip(timestamp, position_scale)):
        # Skip the first position as it's already set
        if i == 0:
            last_rebalance_time = ts
            continue
        
        # Determine if it's time to rebalance
        rebalance_time = False
        
        if frequency == "daily":
            # Rebalance at the beginning of each day
            if ts.date() != last_rebalance_time.date():
                rebalance_time = True
        elif frequency == "weekly":
            # Rebalance at the beginning of each week
            if ts.isocalendar()[1] != last_rebalance_time.isocalendar()[1]:
                rebalance_time = True
        elif frequency == "hourly":
            # Rebalance every N hours
            rebalance_hours = 4  # Adjust as needed
            hour_diff = (ts - last_rebalance_time).total_seconds() / 3600
            if hour_diff >= rebalance_hours:
                rebalance_time = True
        
        # Special case: if BOTH current and new positions are zero, don't consider it a change
        if current_position == 0 and new_position == 0:
            material_change = False
        # Special case: if going from zero to non-zero, always rebalance
        elif current_position == 0 and new_position != 0:
            material_change = True
        # Special case: if going from non-zero to zero, always rebalance
        elif current_position != 0 and new_position == 0:
            material_change = True
        # Normal case: calculate percentage change
        else:
            position_change_pct = abs(new_position - current_position) / current_position
            material_change = position_change_pct > materiality_threshold
        
        # Rebalance if it's time and the change is material
        if rebalance_time and material_change:
            current_position = new_position
            last_rebalance_time = ts
        else:
            # Keep the previous position
            rebalanced_scale.iloc[i] = current_position
    
    return rebalanced_scale

def apply_enhanced_sma_strategy(df, params, config):
    """
    Apply enhanced SMA strategy with optimized parameters and advanced features.
    Features regime-specific parameters, counter-trend signals for high volatility,
    and dynamic risk management.
    
    Parameters:
        df (DataFrame): DataFrame with price data
        params (dict): Optimized parameters
        config (dict): Strategy configuration
        
    Returns:
        DataFrame: Results DataFrame
    """
    print("Applying enhanced SMA strategy with adaptive regime-based parameters...")
    
    # Extract parameters
    vol_method = params['vol_method']
    vol_lookback = params['vol_lookback']
    short_window = params['short_window']
    long_window = params['long_window']
    
    # Calculate volatility
    volatility = calculate_volatility(df, method=vol_method, window=vol_lookback)
    
    # Detect regimes
    regimes = detect_volatility_regimes(
        df, 
        volatility, 
        method=config['regime_detection']['method'],
        n_regimes=config['regime_detection']['n_regimes'],
        stability_period=config['regime_detection']['regime_stability_period']
    )
    
    # Print regime distribution
    regime_counts = regimes.value_counts()
    total_periods = len(df)
    print("\nRegime Distribution:")
    for regime, count in regime_counts.items():
        percentage = (count / total_periods) * 100
        print(f"Regime {regime}: {count} periods ({percentage:.2f}%)")
    
    # Implement regime-specific SMA windows
    # Base windows for each regime
    regime_windows = {
        0: {  # Low volatility regime - slower parameters
            'short': int(short_window * 1.5),
            'long': int(long_window * 1.5)
        },
        1: {  # Normal volatility regime - baseline parameters
            'short': short_window,
            'long': long_window
        },
        2: {  # High volatility regime - faster parameters
            'short': max(3, int(short_window * 0.7)),
            'long': max(8, int(long_window * 0.7))
        }
    }
    
    # Calculate SMA for each regime
    short_ma_regime = {}
    long_ma_regime = {}
    for regime_id, windows in regime_windows.items():
        short_ma_regime[regime_id] = df['close_price'].rolling(window=windows['short']).mean()
        long_ma_regime[regime_id] = df['close_price'].rolling(window=windows['long']).mean()
    
    # Create consolidated SMA series
    short_ma = pd.Series(0.0, index=df.index)
    long_ma = pd.Series(0.0, index=df.index)
    
    # Assign regime-specific MAs for each timestamp
    for i, regime in enumerate(regimes):
        short_ma.iloc[i] = short_ma_regime[regime].iloc[i]
        long_ma.iloc[i] = long_ma_regime[regime].iloc[i]
    
    # Calculate trend strength
    trend_strength = calculate_trend_strength(df, window=config['sma']['trend_filter_period'])
    
    # Calculate momentum
    momentum = calculate_momentum(df, window=vol_lookback)
    
    # Generate raw signal
    raw_signal = pd.Series(0, index=df.index)
    raw_signal[short_ma > long_ma] = 1
    raw_signal[short_ma < long_ma] = -1
    
    # Initialize series to store regime-specific parameters
    trend_threshold = pd.Series(config['sma']['trend_strength_threshold'], index=df.index)
    min_holding = pd.Series(config['sma']['min_holding_period'], index=df.index)
    profit_threshold = pd.Series(config['risk_management']['profit_taking_threshold'], index=df.index)
    trailing_stop = pd.Series(config['risk_management']['trailing_stop_distance'], index=df.index)
    
    # Apply regime-specific parameters if configured
    if 'regime_specific_parameters' in config:
        for regime_id, params in config['regime_specific_parameters'].items():
            # Apply trend strength threshold
            if 'trend_strength_threshold' in params:
                trend_threshold[regimes == regime_id] = params['trend_strength_threshold']
            
            # Apply min holding period
            if 'min_holding_period' in params:
                min_holding[regimes == regime_id] = params['min_holding_period']
            
            # Apply profit taking threshold
            if 'profit_taking_threshold' in params:
                profit_threshold[regimes == regime_id] = params['profit_taking_threshold']
            
            # Apply trailing stop distance
            if 'trailing_stop_distance' in params:
                trailing_stop[regimes == regime_id] = params['trailing_stop_distance']
    
    # Apply volatility-adjusted risk if configured
    if config['risk_management'].get('volatility_adjusted_risk', False):
        multipliers = config['risk_management']['volatility_risk_multiplier']
        
        # Adjust the max drawdown exit based on regime
        max_dd_exit = pd.Series(config['risk_management']['max_drawdown_exit'], index=df.index)
        for regime_id, multiplier in multipliers.items():
            max_dd_exit[regimes == regime_id] = config['risk_management']['max_drawdown_exit'] * multiplier
    else:
        # Use constant max drawdown
        max_dd_exit = pd.Series(config['risk_management']['max_drawdown_exit'], index=df.index)
    
    # Filter signals using regime-specific trend thresholds
    filtered_signal = pd.Series(0, index=df.index)
    
    for i in range(len(df)):
        current_regime = regimes.iloc[i]
        current_threshold = trend_threshold.iloc[i]
        
        # Apply regime-specific filters
        if current_regime == 2:  # High volatility regime - more selective
            # Only take long positions in strong uptrends, short positions in strong downtrends
            if raw_signal.iloc[i] > 0 and trend_strength.iloc[i] > current_threshold and momentum.iloc[i] > 0:
                filtered_signal.iloc[i] = 1
            elif raw_signal.iloc[i] < 0 and trend_strength.iloc[i] < -current_threshold and momentum.iloc[i] < 0:
                filtered_signal.iloc[i] = -1
        else:  # Low and medium volatility regimes - normal filtering
            if raw_signal.iloc[i] > 0 and trend_strength.iloc[i] > current_threshold:
                filtered_signal.iloc[i] = 1
            elif raw_signal.iloc[i] < 0 and trend_strength.iloc[i] < -current_threshold:
                filtered_signal.iloc[i] = -1
    
    # Add counter-trend signals if enabled
    if config.get('counter_trend', {}).get('enabled', False):
        # Calculate RSI
        rsi_period = config['counter_trend'].get('rsi_period', 14)
        rsi = calculate_rsi(df['close_price'], window=rsi_period)
        
        # Initialize counter-trend signal
        counter_trend_signal = pd.Series(0, index=df.index)
        
        # Get counter-trend parameters
        oversold = config['counter_trend'].get('oversold_threshold', 30)
        overbought = config['counter_trend'].get('overbought_threshold', 70)
        signal_strength = config['counter_trend'].get('signal_strength', 0.5)
        only_high_vol = config['counter_trend'].get('only_high_vol_regime', True)
        
        # Generate counter-trend signals
        for i in range(len(df)):
            if only_high_vol and regimes.iloc[i] != 2:
                # Skip if not in high volatility regime and only_high_vol is True
                continue
                
            if rsi.iloc[i] < oversold:
                # Oversold - add long signal
                counter_trend_signal.iloc[i] = signal_strength
            elif rsi.iloc[i] > overbought:
                # Overbought - add short signal
                counter_trend_signal.iloc[i] = -signal_strength
        
        # Blend with existing signal
        blended_signal = filtered_signal + counter_trend_signal
        # Normalize signal to keep within -1 to 1 range
        filtered_signal = blended_signal.clip(-1, 1)
    
    # Apply minimum holding period with regime-specific values
    position = pd.Series(0, index=df.index)
    in_position = 0
    position_start = 0
    
    for i in range(len(df)):
        # New potential position
        if filtered_signal.iloc[i] != 0 and filtered_signal.iloc[i] != in_position:
            # Check if minimum holding period has passed
            if in_position != 0:
                current_min_holding = min_holding.iloc[position_start]
                if i - position_start < current_min_holding:
                    # Minimum holding period not met, maintain current position
                    position.iloc[i] = in_position
                    continue
            
            # Update position
            in_position = filtered_signal.iloc[i]
            position_start = i
        
        # Maintain current position
        position.iloc[i] = in_position
    
    # Get regime position factors
    regime_position_factors = config['regime_detection'].get('regime_position_factors', None)
    
    # Get regime opt-out settings
    regime_opt_out = config['regime_detection'].get('regime_opt_out', None)
    
    # Get materiality threshold and dynamic settings
    materiality_threshold = config['risk_management'].get('materiality_threshold', 0.05)
    use_dynamic_materiality = config['risk_management'].get('use_dynamic_materiality', False)
    dynamic_materiality = config.get('dynamic_materiality', None)
    
    # Calculate position size based on volatility with advanced features
    position_size = calculate_adaptive_position_size_with_schedule(
        volatility,
        regimes,
        df.index,
        target_vol=config['risk_management']['target_volatility'],
        max_size=config['risk_management']['max_position_size'],
        min_size=config['risk_management']['min_position_size'],
        rebalance_frequency="daily",  # Rebalance daily
        materiality_threshold=materiality_threshold,
        regime_opt_out=regime_opt_out,
        regime_position_factors=regime_position_factors,
        use_dynamic_materiality=use_dynamic_materiality,
        dynamic_materiality=dynamic_materiality
    )
    
    # Apply position sizing
    sized_position = position * position_size
    
    # Calculate returns
    returns = df['close_price'].pct_change().fillna(0)
    
    # Calculate trade returns for risk management
    trade_returns = pd.Series(0.0, index=sized_position.index)
    for i in range(1, len(sized_position)):
        if sized_position.iloc[i] != 0:
            if sized_position.iloc[i] == sized_position.iloc[i-1]:
                # Continuing the same position
                trade_returns.iloc[i] = (1 + trade_returns.iloc[i-1]) * (1 + returns.iloc[i] * sized_position.iloc[i-1]) - 1
            else:
                # New position
                trade_returns.iloc[i] = returns.iloc[i] * sized_position.iloc[i]
        else:
            # No position
            trade_returns.iloc[i] = 0
    
    # Apply risk management with dynamic parameters
    managed_position = sized_position.copy()
    
    # Initialize for tracking running values
    current_position = 0
    running_return = 0
    highest_return = 0
    position_start_idx = None
    
    for i in range(1, len(managed_position)):
        current_regime = regimes.iloc[i]
        
        # Current position settings
        current_profit_threshold = profit_threshold.iloc[i]
        current_trailing_stop = trailing_stop.iloc[i]
        current_max_dd = max_dd_exit.iloc[i]
        
        # Position change detection
        if managed_position.iloc[i] != 0 and managed_position.iloc[i] != current_position:
            # New position
            current_position = managed_position.iloc[i]
            running_return = 0
            highest_return = 0
            position_start_idx = i
        elif managed_position.iloc[i] != 0 and managed_position.iloc[i] == current_position:
            # Continuing position - update running return
            running_return = (1 + running_return) * (1 + returns.iloc[i] * current_position) - 1
            
            # Update highest return
            if running_return > highest_return:
                highest_return = running_return
            
            # Apply profit taking
            if running_return >= current_profit_threshold:
                managed_position.iloc[i] = 0
                current_position = 0
                continue
                
            # Apply maximum drawdown stop loss
            current_drawdown = (running_return - highest_return)
            if current_drawdown <= -current_max_dd:
                managed_position.iloc[i] = 0
                current_position = 0
                continue
                
            # Apply trailing stop if activated
            trailing_activation = config['risk_management']['trailing_stop_activation']
            if highest_return >= trailing_activation:
                if current_drawdown <= -current_trailing_stop:
                    managed_position.iloc[i] = 0
                    current_position = 0
                    continue
        elif managed_position.iloc[i] == 0 and current_position != 0:
            # Position closed
            current_position = 0
            running_return = 0
            highest_return = 0
            position_start_idx = None
    
    # Calculate position changes (when a trade occurs)
    position_changes = managed_position.diff().fillna(0).abs()
    
    # Count actual trades (ignoring zero changes)
    num_trades = int((position_changes != 0).sum())
    
    # Calculate strategy returns
    strategy_returns = managed_position.shift(1).fillna(0) * returns
    
    # Calculate cumulative returns
    strategy_cumulative = (1 + strategy_returns).cumprod()
    
    # Calculate buy and hold returns
    buy_hold_returns = returns
    buy_hold_cumulative = (1 + buy_hold_returns).cumprod()
    
    # Create result DataFrame
    result_df = pd.DataFrame({
        'close_price': df['close_price'],
        'volatility': volatility,
        'regime': regimes,
        'trend_strength': trend_strength,
        'momentum': momentum,
        'short_ma': short_ma,
        'long_ma': long_ma,
        'raw_signal': raw_signal,
        'filtered_signal': filtered_signal,
        'position': position,
        'position_size': position_size,
        'sized_position': sized_position,
        'managed_position': managed_position,
        'returns': returns,
        'strategy_returns': strategy_returns,
        'strategy_cumulative': strategy_cumulative,
        'buy_hold_cumulative': buy_hold_cumulative,
        'trade_returns': trade_returns
    })
    
    # Add regime-specific parameters for analysis
    result_df['trend_threshold'] = trend_threshold
    result_df['profit_threshold'] = profit_threshold
    result_df['trailing_stop'] = trailing_stop
    result_df['max_drawdown_threshold'] = max_dd_exit
    
    # If counter-trend is enabled, add RSI for analysis
    if config.get('counter_trend', {}).get('enabled', False):
        result_df['rsi'] = rsi
    
    # Add regime-specific MA data for analysis
    for regime_id in regime_windows.keys():
        result_df[f'short_ma_regime_{regime_id}'] = short_ma_regime[regime_id]
        result_df[f'long_ma_regime_{regime_id}'] = long_ma_regime[regime_id]
    
    print(f"Strategy applied with {num_trades} trades")
    
    return result_df

def run_enhanced_backtest():
    """
    Run enhanced backtest of SMA strategy with cross-validation.
    """
    print(f"Starting enhanced SMA backtest for {CURRENCY}")
    start_time = time.time()
    
    try:
        # Initialize database connection
        db = DatabaseHandler()
        
        # Fetch complete data for the entire period
        print(f"Fetching data from {TRAINING_START} to {TESTING_END}")
        df = db.get_historical_data(CURRENCY, TRAINING_START, TESTING_END)
        
        if len(df) < 1000:
            print(f"Insufficient data for {CURRENCY} ({len(df)} data points). Exiting.")
            db.close()
            return
        
        # Generate cross-validation splits for parameter optimization
        cv_config = STRATEGY_CONFIG['cross_validation']
        splits = generate_time_series_cv_splits(
            TRAINING_START, 
            TRAINING_END,
            n_splits=cv_config['n_splits'],
            min_train_size=cv_config['min_train_size'],
            step_forward=cv_config['step_forward']
        )
        
        # Optimize parameters using cross-validation
        best_params, cv_results = optimize_parameters_with_cv(df, STRATEGY_CONFIG, splits)
        
        if best_params is None:
            print("Parameter optimization failed. Exiting.")
            db.close()
            return
        
        # Fetch test data
        test_df = df.loc[TESTING_START:TESTING_END].copy()
        
        # Apply enhanced strategy to test data
        result_df = apply_enhanced_sma_strategy(test_df, best_params, STRATEGY_CONFIG)
        
        # Calculate performance metrics
        metrics = calculate_advanced_metrics(result_df['strategy_returns'], result_df['strategy_cumulative'])
        
        # Calculate buy & hold metrics
        buy_hold_return = result_df['buy_hold_cumulative'].iloc[-1] - 1
        buy_hold_metrics = calculate_advanced_metrics(result_df['returns'], result_df['buy_hold_cumulative'])
        
        # Close database connection
        db.close()
        
        # Print test results
        print("\n===== Test Results =====")
        print(f"Total Return: {metrics['total_return']:.4%}")
        print(f"Annualized Return: {metrics['annualized_return']:.4%}")
        print(f"Volatility: {metrics['volatility']:.4%}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.4%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.4f}")
        print(f"Calmar Ratio: {metrics['calmar_ratio']:.4f}")
        print(f"Win Rate: {metrics['win_rate']:.4%}")
        print(f"Gain-to-Pain Ratio: {metrics['gain_to_pain']:.4f}")
        print(f"Buy & Hold Return: {buy_hold_return:.4%}")
        print(f"Outperformance: {metrics['total_return'] - buy_hold_return:.4%}")
        
        # Plot results
        if PLOT_RESULTS:
            plot_enhanced_results(result_df, best_params, metrics)
        
        # Save results
        if SAVE_RESULTS:
            save_enhanced_results(result_df, best_params, metrics, cv_results)
        
        end_time = time.time()
        print(f"\nTotal execution time: {(end_time - start_time) / 60:.2f} minutes")
        
        return result_df, best_params, metrics
    
    except Exception as e:
        print(f"Error in backtest: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def plot_enhanced_results(df, params, metrics):
    """
    Plot enhanced backtest results.
    
    Parameters:
        df (DataFrame): Results DataFrame
        params (dict): Strategy parameters
        metrics (dict): Performance metrics
    """
    if not PLOT_RESULTS:
        return
    
    # Create output directory if needed
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Check if there were any trades
    position_changes = df['managed_position'].diff().fillna(0).abs()
    num_trades = int((position_changes != 0).sum())
    
    # Create figure with subplots
    fig, axs = plt.subplots(5, 1, figsize=(14, 20), gridspec_kw={'height_ratios': [2, 1, 1, 1, 1]})
    
    # Plot 1: Price and Performance
    ax1 = axs[0]
    ax1.set_title(f'Enhanced SMA Strategy for {CURRENCY}', fontsize=16)
    ax1.plot(df.index, df['close_price'], color='gray', alpha=0.6, label='Price')
    ax1_twin = ax1.twinx()
    ax1_twin.plot(df.index, df['strategy_cumulative'] * INITIAL_CAPITAL, 'b-', label='Strategy')
    ax1_twin.plot(df.index, df['buy_hold_cumulative'] * INITIAL_CAPITAL, 'r--', label='Buy & Hold')
    ax1.set_ylabel('Price')
    ax1_twin.set_ylabel('Portfolio Value ($)')
    ax1_twin.legend(loc='upper left')
    
    # Plot 2: Volatility and Regimes
    ax2 = axs[1]
    ax2.set_title('Volatility and Regimes', fontsize=14)
    ax2.plot(df.index, df['volatility'], 'b-', label='Volatility')
    
    # Color background by regime and mark opt-out regimes
    regime_colors = ['green', 'gray', 'red']
    regime_labels = ['Low Vol', 'Medium Vol', 'High Vol']
    regime_opt_out = STRATEGY_CONFIG['regime_detection'].get('regime_opt_out', {})
    
    for regime in range(STRATEGY_CONFIG['regime_detection']['n_regimes']):
        regime_mask = df['regime'] == regime
        if regime_mask.any():
            color = regime_colors[regime]
            label = f'{regime_labels[regime]}'
            
            # Add "(Opt-out)" to label if this regime has opt-out enabled
            if regime_opt_out.get(regime, False):
                label += " (Opt-out)"
                
            ax2.fill_between(df.index, 0, df['volatility'].max(), where=regime_mask, 
                             color=color, alpha=0.2, label=label)
    
    ax2.set_ylabel('Volatility')
    ax2.legend(loc='upper left')
    
    # Plot 3: Trading Signals and Position
    ax3 = axs[2]
    ax3.set_title('Signals, Positions and Risk Management', fontsize=14)
    ax3.plot(df.index, df['raw_signal'], 'k--', alpha=0.5, label='Raw Signal')
    ax3.plot(df.index, df['filtered_signal'], 'g-', alpha=0.7, label='Filtered Signal')
    ax3.plot(df.index, df['managed_position'], 'b-', linewidth=1.5, label='Final Position')
    
    # Highlight trend strength
    ax3_twin = ax3.twinx()
    ax3_twin.plot(df.index, df['trend_strength'], 'r-', alpha=0.3, label='Trend Strength')
    
    # Add threshold lines
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.axhline(y=1, color='k', linestyle='--', alpha=0.3)
    ax3.axhline(y=-1, color='k', linestyle='--', alpha=0.3)
    ax3_twin.axhline(y=STRATEGY_CONFIG['sma']['trend_strength_threshold'], color='r', linestyle='--', alpha=0.3)
    ax3_twin.axhline(y=-STRATEGY_CONFIG['sma']['trend_strength_threshold'], color='r', linestyle='--', alpha=0.3)
    
    ax3.set_ylabel('Position')
    ax3_twin.set_ylabel('Trend Strength')
    
    # Create combined legend
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot 4: Drawdown
    ax4 = axs[3]
    drawdown = df['strategy_cumulative'] / df['strategy_cumulative'].cummax() - 1
    ax4.set_title(f'Drawdown (Max: {metrics["max_drawdown"]:.2%})', fontsize=14)
    ax4.fill_between(df.index, drawdown * 100, 0, color='red', alpha=0.3)
    ax4.set_ylabel('Drawdown (%)')
    ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax4.axhline(y=STRATEGY_CONFIG['risk_management']['max_drawdown_exit'] * 100, color='r', linestyle='--', alpha=0.5, 
                label=f'Stop Loss ({STRATEGY_CONFIG["risk_management"]["max_drawdown_exit"] * 100:.0f}%)')
    ax4.legend(loc='lower left')
    
    # Plot 5: Moving Averages
    ax5 = axs[4]
    ax5.set_title(f'Moving Averages (Short: {params["short_window"]}, Long: {params["long_window"]})', fontsize=14)
    
    # Use subset of data for clarity (last 30% of the data)
    start_idx = int(len(df) * 0.7)
    subset_idx = df.index[start_idx:]
    
    ax5.plot(subset_idx, df.loc[subset_idx, 'close_price'], color='gray', alpha=0.6, label='Price')
    ax5.plot(subset_idx, df.loc[subset_idx, 'short_ma'], 'g-', alpha=0.8, label=f'Short MA ({params["short_window"]})')
    ax5.plot(subset_idx, df.loc[subset_idx, 'long_ma'], 'r-', alpha=0.8, label=f'Long MA ({params["long_window"]})')
    
    # Color background by position
    for i in range(start_idx, len(df)):
        if df['managed_position'].iloc[i] > 0:
            ax5.axvspan(df.index[i-1], df.index[i], color='green', alpha=0.1)
        elif df['managed_position'].iloc[i] < 0:
            ax5.axvspan(df.index[i-1], df.index[i], color='red', alpha=0.1)
    
    ax5.set_ylabel('Price & MAs')
    ax5.legend(loc='upper left')
    
    # Add strategy performance summary
    # Calculate regime statistics for summary
    regime_stats = []
    for regime in range(STRATEGY_CONFIG['regime_detection']['n_regimes']):
        regime_mask = df['regime'] == regime
        if regime_mask.any():
            regime_returns = df.loc[regime_mask, 'strategy_returns']
            regime_return = (1 + regime_returns).prod() - 1 if len(regime_returns) > 0 else 0
            
            # Calculate buy and hold return for same regime periods
            regime_bh_returns = df.loc[regime_mask, 'returns']
            regime_bh_return = (1 + regime_bh_returns).prod() - 1 if len(regime_bh_returns) > 0 else 0
            
            # Calculate outperformance
            outperf = regime_return - regime_bh_return
            regime_stats.append(f"Regime {regime}: {regime_return:.1%} vs B&H {regime_bh_return:.1%} ({outperf:.1%})")
    
    # Join regime stats with line breaks
    regime_summary = " | ".join(regime_stats)
    
    # Add warning if no trades
    if num_trades == 0:
        warning_text = "WARNING: No trades executed! Check regime opt-out settings."
        plt.figtext(0.5, 0.5, warning_text, ha='center', va='center', fontsize=20, 
                   color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.figtext(0.1, 0.01, 
             f"Return: {metrics['total_return']:.2%} | Annual: {metrics['annualized_return']:.2%} | "
             f"Sharpe: {metrics['sharpe_ratio']:.2f} | Sortino: {metrics['sortino_ratio']:.2f} | "
             f"Calmar: {metrics['calmar_ratio']:.2f} | MaxDD: {metrics['max_drawdown']:.2%}\n"
             f"Win Rate: {metrics['win_rate']:.2%} | Gain/Pain: {metrics['gain_to_pain']:.2f} | "
             f"Volatility: {metrics['volatility']:.2%} | "
             f"Buy & Hold: {df['buy_hold_cumulative'].iloc[-1] - 1:.2%} | "
             f"Alpha: {metrics['total_return'] - (df['buy_hold_cumulative'].iloc[-1] - 1):.2%}\n"
             f"Regime Performance: {regime_summary}" +
             (f"\nWARNING: No trades executed during backtest period" if num_trades == 0 else ""),
             ha='left', fontsize=11, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.09)  # Adjust bottom margin to fit the additional regime info
    
    # Save the figure
    plt.savefig(os.path.join(RESULTS_DIR, f'enhanced_sma_results_{CURRENCY.replace("/", "_")}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Results plot saved to {os.path.join(RESULTS_DIR, f'enhanced_sma_results_{CURRENCY.replace('/', '_')}.png')}")

def save_enhanced_results(df, params, metrics, cv_results):
    """
    Save enhanced backtest results to files.
    
    Parameters:
        df (DataFrame): Results DataFrame
        params (dict): Strategy parameters
        metrics (dict): Performance metrics
        cv_results (list): Cross-validation results
    """
    if not SAVE_RESULTS:
        return
    
    # Create output directory if needed
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Calculate position changes and check if there are any trades
    position_changes = df['managed_position'].diff().fillna(0).abs()
    num_trades = int((position_changes != 0).sum())
    
    # Save parameters and metrics
    results_file = os.path.join(RESULTS_DIR, f'enhanced_sma_results_{CURRENCY.replace("/", "_")}.txt')
    with open(results_file, 'w') as f:
        f.write("===== ENHANCED SMA STRATEGY RESULTS =====\n\n")
        
        f.write("Strategy Configuration:\n")
        f.write(f"Trading Frequency: {TRADING_FREQUENCY}\n")
        f.write(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}\n")
        f.write(f"Trading Fee: {TRADING_FEE_PCT:.4%} per trade\n\n")
        
        f.write("Optimized Parameters:\n")
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        
        # Check if there are no trades
        if num_trades == 0:
            f.write("WARNING: No trades were executed during the backtest period.\n")
            f.write("This may be due to all regimes being set to opt-out or other restrictive settings.\n\n")
        
        f.write("Performance Metrics:\n")
        for key, value in metrics.items():
            if isinstance(value, float):
                if key in ['total_return', 'annualized_return', 'volatility', 'max_drawdown', 'win_rate']:
                    f.write(f"{key}: {value:.4%}\n")
                else:
                    f.write(f"{key}: {value:.4f}\n")
            else:
                f.write(f"{key}: {value}\n")
        
        # Add buy & hold comparison
        buy_hold_return = df['buy_hold_cumulative'].iloc[-1] - 1
        f.write(f"\nBuy & Hold Return: {buy_hold_return:.4%}\n")
        f.write(f"Outperformance: {metrics['total_return'] - buy_hold_return:.4%}\n")
        
        # Add trade statistics
        f.write(f"\nNumber of Trades: {num_trades}\n")
        
        # Calculate average trade duration if there are trades
        if num_trades > 0:
            trade_durations = []
            in_trade = False
            trade_start = None
            
            for i, (timestamp, position) in enumerate(df['managed_position'].items()):
                if not in_trade and position != 0:
                    # Trade entry
                    in_trade = True
                    trade_start = timestamp
                elif in_trade and position == 0:
                    # Trade exit
                    in_trade = False
                    if trade_start is not None:
                        duration = (timestamp - trade_start).total_seconds() / 3600  # hours
                        trade_durations.append(duration)
            
            if trade_durations:
                avg_duration = np.mean(trade_durations)
                f.write(f"Average Trade Duration: {avg_duration:.2f} hours\n")
                
            # Calculate win/loss ratio
            winning_trades = df['strategy_returns'] > 0
            losing_trades = df['strategy_returns'] < 0
            
            num_winning = winning_trades.sum()
            num_losing = losing_trades.sum()
            
            if num_losing > 0:
                win_loss_ratio = num_winning / num_losing
                f.write(f"Win/Loss Ratio: {win_loss_ratio:.2f}\n")
            
            # Calculate average profit per trade
            avg_profit = metrics['total_return'] / num_trades
            f.write(f"Average Profit per Trade: {avg_profit:.4%}\n")
        
        # Add regime statistics with enhanced metrics
        f.write("\n===== REGIME PERFORMANCE =====\n")
        
        # Get the regime position factors
        regime_position_factors = STRATEGY_CONFIG['regime_detection'].get('regime_position_factors', {})
        
        for regime in range(STRATEGY_CONFIG['regime_detection']['n_regimes']):
            regime_mask = df['regime'] == regime
            if regime_mask.any():
                regime_pct = regime_mask.mean()
                regime_returns = df.loc[regime_mask, 'strategy_returns']
                regime_return = (1 + regime_returns).prod() - 1 if len(regime_returns) > 0 else 0
                
                # Calculate buy and hold return for same regime periods
                regime_bh_returns = df.loc[regime_mask, 'returns']
                regime_bh_return = (1 + regime_bh_returns).prod() - 1 if len(regime_bh_returns) > 0 else 0
                
                # Calculate outperformance in this regime
                regime_outperformance = regime_return - regime_bh_return
                
                # Get regime description
                regime_descriptions = {
                    0: "Low Volatility",
                    1: "Medium Volatility",
                    2: "High Volatility"
                }
                regime_desc = regime_descriptions.get(regime, f"Regime {regime}")
                
                # Get regime position factor
                position_factor = regime_position_factors.get(regime, 1.0)
                
                # Check if this regime has opt-out enabled
                regime_opt_out = "Enabled" if STRATEGY_CONFIG['regime_detection'].get('regime_opt_out', {}).get(regime, False) else "Disabled"
                
                f.write(f"\n{regime_desc} (Regime {regime}):\n")
                f.write(f"  Opt-out: {regime_opt_out}\n")
                f.write(f"  Position Size Factor: {position_factor:.2f}\n")
                f.write(f"  Percentage of time: {regime_pct:.4%}\n")
                f.write(f"  Strategy return: {regime_return:.4%}\n")
                f.write(f"  Buy & Hold return: {regime_bh_return:.4%}\n")
                f.write(f"  Outperformance: {regime_outperformance:.4%}\n")
                
                # Get regime-specific parameters
                if 'regime_specific_parameters' in STRATEGY_CONFIG:
                    regime_params = STRATEGY_CONFIG['regime_specific_parameters'].get(regime, {})
                    if regime_params:
                        f.write(f"  Regime-specific parameters:\n")
                        for param_name, param_value in regime_params.items():
                            f.write(f"    {param_name}: {param_value}\n")
                
                # Only calculate these metrics if there are returns in this regime
                if len(regime_returns) > 0 and (regime_returns != 0).any():
                    regime_sharpe = calculate_sharpe_ratio(regime_returns)
                    regime_sortino = calculate_sortino_ratio(regime_returns)
                    f.write(f"  Sharpe: {regime_sharpe:.4f}\n")
                    f.write(f"  Sortino: {regime_sortino:.4f}\n")
                    
                    # Count trades in this regime
                    regime_position_changes = df.loc[regime_mask, 'managed_position'].diff().fillna(0).abs()
                    regime_trades = int((regime_position_changes != 0).sum())
                    regime_trade_pct = (regime_trades / num_trades) * 100 if num_trades > 0 else 0
                    f.write(f"  Trades in regime: {regime_trades} ({regime_trade_pct:.2f}% of all trades)\n")
                    
                    # Calculate win rate in this regime
                    if regime_trades > 0:
                        regime_winning = df.loc[regime_mask & (df['strategy_returns'] > 0), 'strategy_returns'].count()
                        regime_win_rate = regime_winning / regime_trades
                        f.write(f"  Win rate in regime: {regime_win_rate:.4%}\n")
                else:
                    f.write(f"  No active trades in this regime\n")
        
        # Add counter-trend strategy stats if enabled
        if STRATEGY_CONFIG.get('counter_trend', {}).get('enabled', True):
            f.write("\n===== COUNTER-TREND STRATEGY =====\n")
            f.write(f"Enabled: {STRATEGY_CONFIG['counter_trend']['enabled']}\n")
            f.write(f"Only High Volatility Regime: {STRATEGY_CONFIG['counter_trend'].get('only_high_vol_regime', True)}\n")
            f.write(f"RSI Period: {STRATEGY_CONFIG['counter_trend'].get('rsi_period', 14)}\n")
            f.write(f"Oversold Threshold: {STRATEGY_CONFIG['counter_trend'].get('oversold_threshold', 30)}\n")
            f.write(f"Overbought Threshold: {STRATEGY_CONFIG['counter_trend'].get('overbought_threshold', 70)}\n")
            f.write(f"Signal Strength: {STRATEGY_CONFIG['counter_trend'].get('signal_strength', 0.5)}\n")
        
        # Add volatility-adjusted risk info if enabled
        if STRATEGY_CONFIG['risk_management'].get('volatility_adjusted_risk', True):
            f.write("\n===== VOLATILITY-ADJUSTED RISK MANAGEMENT =====\n")
            f.write("Enabled: True\n")
            f.write("Risk multipliers by regime:\n")
            for regime, multiplier in STRATEGY_CONFIG['risk_management']['volatility_risk_multiplier'].items():
                f.write(f"  Regime {regime}: {multiplier:.2f}\n")
        
        # Add dynamic materiality info if enabled
        if STRATEGY_CONFIG['risk_management'].get('use_dynamic_materiality', True):
            f.write("\n===== DYNAMIC MATERIALITY THRESHOLDS =====\n")
            f.write("Enabled: True\n")
            f.write("Materiality thresholds by regime:\n")
            for regime, threshold in STRATEGY_CONFIG['dynamic_materiality'].items():
                f.write(f"  Regime {regime}: {threshold:.4f}\n")
        
        # Add cross-validation summary
        f.write("\n===== CROSS-VALIDATION RESULTS =====\n\n")
        for i, result in enumerate(cv_results):
            if 'best_params' in result and 'best_metrics' in result:
                f.write(f"Fold {i+1}:\n")
                f.write(f"  Parameters: {result['best_params']}\n")
                f.write(f"  Score: {result['best_score']:.4f}\n")
                f.write(f"  Sharpe: {result['best_metrics']['sharpe_ratio']:.4f}\n")
                f.write(f"  Sortino: {result['best_metrics']['sortino_ratio']:.4f}\n")
                f.write(f"  Return: {result['best_metrics']['total_return']:.4%}\n")
                f.write(f"  Max DD: {result['best_metrics']['max_drawdown']:.4%}\n\n")
    
    print(f"Results saved to {results_file}")
    
    # Save DataFrame to CSV
    csv_file = os.path.join(RESULTS_DIR, f'enhanced_sma_data_{CURRENCY.replace("/", "_")}.csv')
    df.to_csv(csv_file)
    print(f"Data saved to {csv_file}")
    
    # Save model parameters to pickle
    model_file = os.path.join(RESULTS_DIR, f'enhanced_sma_model_{CURRENCY.replace("/", "_")}.pkl')
    
    model_data = {
        'params': params,
        'config': STRATEGY_CONFIG,
        'metrics': metrics,
        'date_trained': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    joblib.dump(model_data, model_file)
    print(f"Model saved to {model_file}")

# ==================== MAIN FUNCTION ====================
def main():
    # Create results directory
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
    
    # Run enhanced backtest
    print("Running enhanced SMA backtest...")
    run_enhanced_backtest()
    
if __name__ == "__main__":
    main()