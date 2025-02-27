#!/usr/bin/env python
# run_strategy.py - Script to run the enhanced SMA strategy

import os
import sys
import time
from datetime import datetime

# Try to import our modules
try:
    from enhanced_sma import run_enhanced_backtest
except ImportError:
    print("Error: Could not import enhanced_sma module.")
    print("Please ensure enhanced_sma.py is in the current directory.")
    sys.exit(1)

# Set up logging to file
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, f"strategy_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Redirect stdout and stderr to log file
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")
        
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger(log_file)
sys.stderr = sys.stdout

# Print start information
print("=" * 80)
print(f"Enhanced SMA Strategy Runner")
print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Logging to: {log_file}")
print("=" * 80)
print()

# Run the enhanced backtest
try:
    start_time = time.time()
    print("Running enhanced SMA backtest...")
    
    result_df, best_params, metrics = run_enhanced_backtest()
    
    end_time = time.time()
    duration = (end_time - start_time) / 60
    
    print("\nExecution complete!")
    print(f"Total runtime: {duration:.2f} minutes")
    
    if result_df is not None and best_params is not None and metrics is not None:
        print("\nStrategy Summary:")
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
    
except Exception as e:
    print(f"ERROR: Strategy execution failed with error: {e}")
    import traceback
    traceback.print_exc()
    
print("\nLog file saved to:", log_file)
print("=" * 80)