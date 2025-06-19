#!/usr/bin/env python3
"""
Quick test to identify why no trades are being executed.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from backtest.engine import BacktestEngine
    from data.api import MarketDataAPI
    from config.settings import CONFIG
    from simple_logger import logger
    
    print("Starting quick debugging test...")
    
    # Initialize
    data_api = MarketDataAPI()
    engine = BacktestEngine(initial_capital=100000, data_api=data_api)
    
    # Test short backtest
    print(f"Entry Z-score threshold: {CONFIG['trading']['entry_z_score']}")
    print(f"Trading window: {CONFIG['strategy']['trading_window']} days")
    print(f"Lookback window: {CONFIG['strategy']['lookback_window']} days")
    
    # Run a very short test
    results = engine.run_rolling_backtest(
        start_date='2024-01-01',
        end_date='2024-03-31',
        universe='IBOV',
        save_results=False
    )
    
    print("\nQuick test results:")
    if results:
        print(f"Total trades: {results.get('total_trades', 0)}")
        print(f"Pairs found in history: {len(results.get('pair_history', []))}")
        if results.get('pair_history'):
            print(f"Example pair count per period: {[p['pairs_count'] for p in results['pair_history'][:3]]}")
    else:
        print("No results returned!")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()