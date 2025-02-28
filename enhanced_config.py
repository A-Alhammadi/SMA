############################################################
#                   DATABASE SETTINGS                      #
############################################################

DB_CONFIG = {
    'dbname': 'cryptocurrencies',
    'user': 'myuser',
    'password': 'mypassword',
    'host': 'localhost',
    'port': '5432'
}

############################################################
#                   BACKTEST SETTINGS                      #
############################################################

# Data and Backtesting Settings
TRADING_FREQUENCY = "1H"  # Frequency of data (1H = hourly, 1D = daily)
TRAINING_START = "2018-05-20"
TRAINING_END = "2020-12-31"
TESTING_START = "2021-01-01"
TESTING_END = "2024-10-20"
CURRENCY = "BTC/USD"  # Base currency to analyze
INITIAL_CAPITAL = 10000
TRADING_FEE_PCT = 0.001  # Example: 0.1% trading fee per trade

############################################################
#                  ENHANCED STRATEGY SETTINGS              #
############################################################

# Strategy parameters for the enhanced SMA model
STRATEGY_CONFIG = {
    # Volatility calculation settings
    'volatility': {
        'methods': ['parkinson', 'garch', 'yang_zhang', 'standard'],  # Added standard method
        'lookback_periods': [80, 90, 110, 120, 130],  # Expanded lookback periods
        'regime_smoothing': 5,  # Days to smooth regime transitions
        'min_history_multiplier': 5,  # Minimum history required as multiplier of lookback
    },
    
    # Regime detection settings
    'regime_detection': {
        'method': 'hmm',  # Options: 'kmeans', 'kde', 'quantile', 'hmm'
        'n_regimes': 3,  # Number of distinct volatility regimes
        'quantile_thresholds': [0.33, 0.75],  # Percentile thresholds for regime transitions
        'regime_stability_period': 48,  # Hours required before confirming regime change
        'regime_opt_out': {
            0: False,  # Low volatility regime - False means trade normally in this regime
            1: False,  # Medium volatility regime - False means trade normally in this regime
            2: False   # High volatility regime - False means trade normally in this regime
                # Set to True to liquidate positions when entering this regime
        },
        'regime_buy_hold': {
            0: False,  # Low volatility regime - False means use strategy signals in this regime
            1: False,  # Medium volatility regime - False means use strategy signals in this regime
            2: True   # High volatility regime - False means use strategy signals in this regime
                # Set to True to switch to buy & hold when entering this regime
        }
    },
    
    # SMA strategy settings
    'sma': {
        'short_windows': [5, 8, 34, 55, 89],  # Extended Fibonacci-based short windows
        'long_windows': [21, 377],  # Extended Fibonacci-based long windows
        'min_holding_period': 24,  # Minimum holding period in hours
        'trend_filter_period': 200,  # Period for trend strength calculation
        'trend_strength_threshold': 0.3,  # Minimum trend strength to take a position
    },
    
    # Risk management settings
    'risk_management': {
        'target_volatility': 0.15,  # Target annualized volatility
        'max_position_size': 1.0,  # Maximum position size
        'min_position_size': 0.1,  # Minimum position size
        'max_drawdown_exit': 0.12,  # Exit if drawdown exceeds this threshold
        'profit_taking_threshold': 0.08,  # Take profit at this threshold
        'trailing_stop_activation': 0.06,  # Activate trailing stop after this gain
        'trailing_stop_distance': 0.03,  # Trailing stop distance
        'materiality_threshold': 0.05,  # Only rebalance if position size change exceeds this percentage
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

############################################################
#                   OUTPUT SETTINGS                        #
############################################################

SAVE_RESULTS = True
PLOT_RESULTS = True
RESULTS_DIR = "enhanced_sma_results"