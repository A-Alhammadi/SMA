# config.py
# Configuration file for database connectivity

# Database connection settings
DB_CONFIG = {
    'dbname': 'cryptocurrencies',
    'user': 'myuser',
    'password': 'mypassword',
    'host': 'localhost',
    'port': '5432'
}

# Import other settings from enhanced_config
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
except ImportError:
    # Default configuration if enhanced_config import fails
    TRADING_FREQUENCY = "1H"
    TRAINING_START = "2018-05-20"
    TRAINING_END = "2020-12-31"
    TESTING_START = "2021-01-01"
    TESTING_END = "2024-10-20"
    CURRENCY = "BTC/USD"
    INITIAL_CAPITAL = 10000
    TRADING_FEE_PCT = 0.001
    RESULTS_DIR = "enhanced_sma_results"
    SAVE_RESULTS = True
    PLOT_RESULTS = True