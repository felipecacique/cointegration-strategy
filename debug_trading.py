#!/usr/bin/env python3
"""
Debug script to identify why backtest finds pairs but shows 0% returns.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from simple_logger import logger

# Import modules
from backtest.engine import BacktestEngine
from data.api import MarketDataAPI
from strategy.signals import TradingSignalGenerator, SignalType, PositionSide
from backtest.positions import PositionManager
from config.settings import CONFIG

def debug_trading_execution():
    """Debug the trading execution flow."""
    logger.info("=== DEBUGGING TRADING EXECUTION ===")
    
    # Initialize components
    data_api = MarketDataAPI()
    signal_generator = TradingSignalGenerator(data_api)
    position_manager = PositionManager(100000)
    
    # Get a sample of data to work with
    available_symbols = data_api.get_available_symbols()
    logger.info(f"Available symbols in DB: {len(available_symbols)}")
    
    if len(available_symbols) < 2:
        logger.error("Insufficient symbols in database")
        return
    
    # Pick two symbols that have data
    symbol1, symbol2 = available_symbols[:2]
    logger.info(f"Testing with symbols: {symbol1}, {symbol2}")
    
    # Test date range - recent data
    end_date = "2024-12-31"
    start_date = "2024-12-01"  # 1 month
    
    logger.info(f"Testing date range: {start_date} to {end_date}")
    
    # Step 1: Check if we can get price data
    logger.info("Step 1: Testing price data retrieval...")
    price_data = data_api.get_pairs_data(symbol1, symbol2, start_date, end_date)
    logger.info(f"Price data shape: {price_data.shape}")
    logger.info(f"Price data head:\n{price_data.head() if not price_data.empty else 'EMPTY'}")
    
    if price_data.empty:
        logger.error("No price data available - this explains 0% returns!")
        return
    
    # Step 2: Test spread calculation
    logger.info("Step 2: Testing spread calculation...")
    hedge_ratio = 1.0  # Simple 1:1 ratio for testing
    intercept = 0.0
    
    spread = signal_generator.calculate_spread(
        symbol1, symbol2, hedge_ratio, intercept, start_date, end_date
    )
    logger.info(f"Spread calculated: {len(spread)} points")
    logger.info(f"Spread sample: {spread.tail(5) if not spread.empty else 'EMPTY'}")
    
    if spread.empty:
        logger.error("Spread calculation failed - this explains 0% returns!")
        return
    
    # Step 3: Test z-score calculation
    logger.info("Step 3: Testing z-score calculation...")
    z_score = signal_generator.calculate_z_score(spread)
    logger.info(f"Z-score calculated: {len(z_score)} points")
    logger.info(f"Z-score sample: {z_score.tail(5) if not z_score.empty else 'EMPTY'}")
    
    if z_score.empty:
        logger.error("Z-score calculation failed - this explains 0% returns!")
        return
    
    # Step 4: Test signal generation
    logger.info("Step 4: Testing signal generation...")
    signals = signal_generator.generate_signals(z_score, current_position=PositionSide.FLAT)
    logger.info(f"Signals generated: {len(signals)} points")
    
    # Count signal types
    signal_counts = signals.value_counts()
    logger.info(f"Signal counts:\n{signal_counts}")
    
    # Check if we have any actionable signals
    actionable_signals = signals[signals != SignalType.NO_SIGNAL]
    logger.info(f"Actionable signals: {len(actionable_signals)}")
    
    if len(actionable_signals) == 0:
        logger.error("No actionable signals generated - this explains 0% returns!")
        logger.info(f"Z-score thresholds: entry={CONFIG['trading']['entry_z_score']}, exit={CONFIG['trading']['exit_z_score']}")
        logger.info(f"Max absolute z-score: {abs(z_score).max()}")
        return
    
    # Step 5: Test position opening
    logger.info("Step 5: Testing position opening...")
    
    # Find first entry signal
    entry_signals = signals[signals.isin([SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT])]
    if len(entry_signals) == 0:
        logger.error("No entry signals found!")
        return
    
    first_entry_date = entry_signals.index[0]
    first_entry_signal = entry_signals.iloc[0]
    
    logger.info(f"First entry signal: {first_entry_signal} on {first_entry_date}")
    
    # Get prices for that date
    date_str = first_entry_date.strftime('%Y-%m-%d')
    entry_price_data = data_api.get_pairs_data(symbol1, symbol2, date_str, date_str)
    
    if entry_price_data.empty:
        logger.error(f"No price data for entry date {date_str} - this explains 0% returns!")
        return
    
    price1 = entry_price_data[symbol1].iloc[0]
    price2 = entry_price_data[symbol2].iloc[0]
    
    logger.info(f"Entry prices: {symbol1}={price1}, {symbol2}={price2}")
    
    # Try to open position
    pair_id = f"{symbol1}-{symbol2}"
    side = PositionSide.LONG if first_entry_signal == SignalType.ENTRY_LONG else PositionSide.SHORT
    capital_allocation = position_manager.initial_capital * CONFIG['trading']['max_position_size']
    
    logger.info(f"Attempting to open position: {pair_id}, side={side}, capital={capital_allocation}")
    
    trade_result = position_manager.open_position(
        pair_id, symbol1, symbol2, side, capital_allocation,
        price1, price2, hedge_ratio, date_str
    )
    
    if trade_result:
        logger.info(f"Position opened successfully: {trade_result}")
        logger.info(f"Portfolio value after opening: {position_manager.get_portfolio_value()}")
        
        # Try to close position on next exit signal
        exit_signals = signals[signals.isin([SignalType.EXIT_LONG, SignalType.EXIT_SHORT])]
        exit_signals = exit_signals[exit_signals.index > first_entry_date]
        
        if len(exit_signals) > 0:
            first_exit_date = exit_signals.index[0]
            logger.info(f"Found exit signal on {first_exit_date}")
            
            exit_date_str = first_exit_date.strftime('%Y-%m-%d')
            exit_price_data = data_api.get_pairs_data(symbol1, symbol2, exit_date_str, exit_date_str)
            
            if not exit_price_data.empty:
                exit_price1 = exit_price_data[symbol1].iloc[0]
                exit_price2 = exit_price_data[symbol2].iloc[0]
                
                close_result = position_manager.close_position(pair_id, exit_price1, exit_price2, exit_date_str)
                
                if close_result:
                    logger.info(f"Position closed successfully: P&L = {close_result['pnl']}")
                    logger.info(f"Portfolio value after closing: {position_manager.get_portfolio_value()}")
                else:
                    logger.error("Failed to close position")
            else:
                logger.error(f"No price data for exit date {exit_date_str}")
        else:
            logger.warning("No exit signals found")
    else:
        logger.error("Failed to open position - this explains 0% returns!")
        logger.info(f"Current cash: {position_manager.cash}")
        logger.info(f"Required capital: {capital_allocation}")
        logger.info(f"Active positions: {len(position_manager.positions)}")

def check_backtest_data_availability():
    """Check what data is actually available for backtesting."""
    logger.info("=== CHECKING DATA AVAILABILITY ===")
    
    data_api = MarketDataAPI()
    date_range = data_api.storage.get_date_range()
    logger.info(f"Database date range: {date_range}")
    
    available_symbols = data_api.get_available_symbols()
    logger.info(f"Available symbols: {len(available_symbols)}")
    
    # Check recent data availability
    end_date = "2024-12-31"
    start_date = "2024-12-01"
    
    symbols_with_recent_data = []
    for symbol in available_symbols[:10]:  # Check first 10
        data = data_api.get_price_data([symbol], start_date, end_date)
        if not data.empty and len(data) > 10:
            symbols_with_recent_data.append(symbol)
    
    logger.info(f"Symbols with recent data ({start_date} to {end_date}): {len(symbols_with_recent_data)}")
    logger.info(f"Sample symbols: {symbols_with_recent_data[:5]}")

if __name__ == "__main__":
    logger.info("Starting trading debug analysis...")
    
    check_backtest_data_availability()
    debug_trading_execution()
    
    logger.info("Debug analysis complete.")