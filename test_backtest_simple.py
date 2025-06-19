#!/usr/bin/env python3
"""
Simple test to verify basic backtest functionality.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Test basic imports
try:
    from backtest.engine import BacktestEngine
    from data.api import MarketDataAPI
    from config.settings import CONFIG
    from simple_logger import logger
    
    print("✅ All imports successful")
    
    # Test basic initialization
    data_api = MarketDataAPI()
    engine = BacktestEngine(initial_capital=100000, data_api=data_api)
    
    print("✅ BacktestEngine initialized successfully")
    
    # Test configuration
    print(f"Config loaded - Initial capital: {CONFIG['trading']['initial_capital']}")
    print(f"Entry Z-score: {CONFIG['trading']['entry_z_score']}")
    print(f"Max position size: {CONFIG['trading']['max_position_size']}")
    
    # Test basic data API functions
    date_range = data_api.storage.get_date_range()
    print(f"Database date range: {date_range}")
    
    available_symbols = data_api.get_available_symbols()
    print(f"Available symbols: {len(available_symbols)}")
    
    if len(available_symbols) > 0:
        print(f"First few symbols: {available_symbols[:5]}")
    
    print("✅ Basic functionality test completed successfully")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Error: {e}")