# config.py
# Configuration file for the SMA strategy

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
TRAINING_START = "2021-01-01"
TRAINING_END = "2022-12-31"
TESTING_START = "2020-01-01" 
TESTING_END = "2025-01-01"
CURRENCY = "BTC/USD"  # Base currency to analyze
INITIAL_CAPITAL = 10000
TRADING_FEE_PCT = 0.001  # Example: 0.1% trading fee per trade

############################################################
#                   SMA STRATEGY SETTINGS                  #
############################################################

# Strategy Mode
USE_VOLATILITY_REGIMES = True  # Set to False to use a single set of SMA parameters

# Strategy Parameters to Grid Search
SMA_SHORT_WINDOWS = [3, 5, 8, 10, 15, 20, 30, 50]  # Short SMA window options
SMA_LONG_WINDOWS = [30, 50, 100, 150, 200, 300, 400, 500]  # Long SMA window options
SMA_VOLATILITY_THRESHOLDS = [0.5, 1.0, 1.5, 2.0]  # Z-score thresholds for volatility regimes
SMA_VOLATILITY_WINDOWS = [24, 48, 72, 120, 168]  # Windows for volatility calculation

# Output Settings
SAVE_RESULTS = True
PLOT_RESULTS = True
RESULTS_DIR = "sma_volatility_results"

# Add to your config.py file

############################################################
#               ENHANCED STRATEGY SETTINGS                 #
############################################################

# Position Sizing Settings
USE_POSITION_SIZING = True  # Set to False to use binary positions only
POSITION_SIZING_METHOD = "volatility"  # Options: "volatility", "fixed"
MAX_POSITION_SIZE = 1.0  # Maximum position size (1.0 = 100% of capital)
MIN_POSITION_SIZE = 0.1  # Minimum position size (0.1 = 10% of capital)
VOLATILITY_LOOKBACK = 20  # Lookback period for volatility calculation

# Volatility Calculation Settings
VOLATILITY_METHOD = "parkinson"  # Options: "standard", "parkinson", "garman-klass"
TARGET_VOLATILITY = 0.20  # Target annualized volatility (20% = 0.20)

# Expanding Window Settings
USE_EXPANDING_WINDOW = False  # Set to True to use expanding window testing
INITIAL_WINDOW = "2018-05-20"  # Start date for initial training window
STEP_SIZE = 90  # Step size in days for expanding window
MIN_TRAINING_SIZE = 365  # Minimum training window size in days
TEST_PERIOD_SIZE = 90  # Size of each test period in days

############################################################
#               TESTING WINDOW SETTINGS                    #
############################################################

# Sectional Testing Settings
USE_SECTIONAL_TESTING = True  # Set to True to divide testing period into sections
SECTION_SIZE = 90  # Size of each test section in days
AGGREGATE_RESULTS = True  # Whether to aggregate and report results across all sections