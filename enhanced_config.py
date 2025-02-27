# enhanced_config.py
# Enhanced configuration file for the advanced SMA strategy

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
        'regime_opt_out': {
            0: False,  # Low volatility regime - False means trade normally in this regime
            1: False,  # Medium volatility regime - False means trade normally in this regime
            2: False   # High volatility regime - False means trade normally in this regime
        },
        'regime_position_factors': {
            0: 1.0,    # Low volatility - full position size
            1: 0.8,    # Medium volatility - 80% position size
            2: 0.2     # High volatility - 20% position size
        }
    },
    
    # SMA strategy settings
    'sma': {
        'short_windows': [5, 8, 13],  # Fibonacci-based short windows was [5, 8, 13, 21, 34]
        'long_windows': [21, 34, 55],  # Fibonacci-based long windows was [21, 34, 55, 89, 144]
        'min_holding_period': 12,  # Minimum holding period in hours
        'trend_filter_period': 200,  # Period for trend strength calculation
        'trend_strength_threshold': 0.3,  # Minimum trend strength to take a position
    },
    
    # Regime-specific parameters for adaptive trading
    'regime_specific_parameters': {
        0: {  # Low volatility
            'trend_strength_threshold': 0.2,  # More sensitive to trends in low vol
            'trailing_stop_distance': 0.03,   # Wider trailing stops
            'profit_taking_threshold': 0.07,  # Higher profit targets
            'min_holding_period': 24         # Longer minimum holding in low vol
        },
        1: {  # Medium volatility - use default parameters
            'trend_strength_threshold': 0.3,
            'trailing_stop_distance': 0.02,
            'profit_taking_threshold': 0.05,
            'min_holding_period': 12
        },
        2: {  # High volatility
            'trend_strength_threshold': 0.5,  # Require stronger trends to trade
            'trailing_stop_distance': 0.015,  # Tighter trailing stops
            'profit_taking_threshold': 0.03,  # Take profits more quickly
            'min_holding_period': 6           # Shorter holding period in high vol
        }
    },
    
    # Risk management settings
    'risk_management': {
        'target_volatility': 0.15,  # Target annualized volatility
        'max_position_size': 1.0,  # Maximum position size
        'min_position_size': 0.1,  # Minimum position size
        'max_drawdown_exit': 0.15,  # Exit if drawdown exceeds this threshold
        'profit_taking_threshold': 0.04,  # Take profit at this threshold
        'trailing_stop_activation': 0.02,  # Activate trailing stop after this gain
        'trailing_stop_distance': 0.02,  # Trailing stop distance
        'materiality_threshold': 0.05,  # Only rebalance if position size change exceeds this percentage
        'use_dynamic_materiality': True,  # Whether to use regime-specific materiality thresholds
        'volatility_adjusted_risk': True,  # Whether to adjust risk parameters based on volatility
        'volatility_risk_multiplier': {
            0: 1.2,  # Allow larger drawdowns in low volatility
            1: 1.0,  # Standard parameters
            2: 0.7   # Reduce maximum drawdown in high volatility
        }
    },
    
    # Dynamic materiality threshold settings
    'dynamic_materiality': {
        0: 0.07,  # Low volatility - less frequent rebalancing
        1: 0.05,  # Medium volatility - regular rebalancing
        2: 0.03   # High volatility - more frequent rebalancing
    },
    
    # Counter-trend strategy settings
    'counter_trend': {
        'enabled': True,  # Whether to use counter-trend strategy
        'only_high_vol_regime': True,  # Only apply in high volatility regime
        'rsi_period': 14,  # RSI calculation period
        'oversold_threshold': 30,  # RSI level considered oversold
        'overbought_threshold': 70,  # RSI level considered overbought
        'signal_strength': 0.5,  # Strength of counter-trend signals (0.0-1.0)
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