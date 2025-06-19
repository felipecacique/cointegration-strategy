#!/usr/bin/env python3
"""
Detailed debugging script to trace trade execution flow and identify blocking points.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from backtest.engine import BacktestEngine
from data.api import MarketDataAPI
from strategy.signals import TradingSignalGenerator, SignalType, PositionSide
from config.settings import CONFIG
from simple_logger import logger

def debug_trade_execution_flow():
    """Debug the complete trade execution flow."""
    
    print("=== DEBUGGING TRADE EXECUTION FLOW ===\n")
    
    # Initialize components
    data_api = MarketDataAPI()
    engine = BacktestEngine(initial_capital=100000, data_api=data_api)
    signal_generator = TradingSignalGenerator(data_api)
    
    # Test period - focusing on a short period where we know there's data
    start_date = '2024-01-01'
    end_date = '2024-03-31'
    
    print(f"Testing period: {start_date} to {end_date}")
    print(f"Config settings:")
    print(f"  - Entry Z-score: {CONFIG['trading']['entry_z_score']}")
    print(f"  - Exit Z-score: {CONFIG['trading']['exit_z_score']}")
    print(f"  - Stop-loss Z-score: {CONFIG['trading']['stop_loss_z_score']}")
    print(f"  - Max position size: {CONFIG['trading']['max_position_size']}")
    print(f"  - Max active pairs: {CONFIG['trading']['max_active_pairs']}")
    print()
    
    # Step 1: Check data availability
    print("=== STEP 1: DATA AVAILABILITY CHECK ===")
    date_range = data_api.storage.get_date_range()
    print(f"Database date range: {date_range.get('min_date')} to {date_range.get('max_date')}")
    
    available_symbols = data_api.get_available_symbols()
    print(f"Available symbols in database: {len(available_symbols)}")
    print(f"First 10 symbols: {available_symbols[:10]}")
    print()
    
    # Step 2: Find pairs for a specific period
    print("=== STEP 2: FINDING COINTEGRATED PAIRS ===")
    formation_start = '2024-01-01'
    formation_end = '2024-01-31'
    
    pairs = engine._find_pairs_for_period(formation_start, formation_end, 'IBOV')
    print(f"Found {len(pairs)} cointegrated pairs for formation period {formation_start} to {formation_end}")
    
    if pairs:
        print("Top 3 pairs found:")
        for i, pair in enumerate(pairs[:3]):
            print(f"  {i+1}. {pair['symbol1']}-{pair['symbol2']}: "
                  f"p_value={pair.get('p_value', 'N/A'):.4f}, "
                  f"hedge_ratio={pair.get('hedge_ratio', 'N/A'):.4f}")
    else:
        print("NO PAIRS FOUND - This is the first blocking point!")
        return
    print()
    
    # Step 3: Test signal generation for the first pair
    print("=== STEP 3: SIGNAL GENERATION TEST ===")
    test_pair = pairs[0]
    symbol1 = test_pair['symbol1']
    symbol2 = test_pair['symbol2']
    hedge_ratio = test_pair['hedge_ratio']
    intercept = test_pair.get('intercept', 0)
    
    print(f"Testing signals for pair: {symbol1}-{symbol2}")
    print(f"Hedge ratio: {hedge_ratio:.4f}, Intercept: {intercept:.4f}")
    
    # Test spread calculation over a longer period
    spread_start = '2024-01-15'
    spread_end = '2024-02-15'
    
    print(f"Calculating spread for period {spread_start} to {spread_end}")
    spread = signal_generator.calculate_spread(symbol1, symbol2, hedge_ratio, intercept, spread_start, spread_end)
    
    if spread.empty:
        print("❌ SPREAD CALCULATION FAILED - No data available for spread calculation")
        
        # Check individual symbol data
        print("Checking individual symbol data availability:")
        data1 = data_api.get_price_data([symbol1], spread_start, spread_end)
        data2 = data_api.get_price_data([symbol2], spread_start, spread_end)
        print(f"  {symbol1}: {len(data1)} data points")
        print(f"  {symbol2}: {len(data2)} data points")
        return
    else:
        print(f"✅ Spread calculated successfully: {len(spread)} data points")
        print(f"Spread stats: mean={spread.mean():.4f}, std={spread.std():.4f}")
    
    # Test z-score calculation
    z_score = signal_generator.calculate_z_score(spread)
    if z_score.empty:
        print("❌ Z-SCORE CALCULATION FAILED")
        return
    else:
        print(f"✅ Z-score calculated successfully: {len(z_score)} data points")
        print(f"Z-score stats: mean={z_score.mean():.4f}, std={z_score.std():.4f}")
        print(f"Z-score range: {z_score.min():.4f} to {z_score.max():.4f}")
    
    # Test signal generation
    entry_threshold = CONFIG['trading']['entry_z_score']
    signals = signal_generator.generate_signals(z_score, current_position=PositionSide.FLAT)
    
    if signals.empty:
        print("❌ SIGNAL GENERATION FAILED")
        return
    else:
        print(f"✅ Signals generated successfully: {len(signals)} signals")
        
        # Count signal types
        signal_counts = signals.value_counts()
        print("Signal distribution:")
        for signal_type, count in signal_counts.items():
            print(f"  {signal_type}: {count}")
        
        # Check for entry signals
        entry_signals = signals[signals.isin([SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT])]
        print(f"Entry signals found: {len(entry_signals)}")
        
        if len(entry_signals) == 0:
            print(f"❌ NO ENTRY SIGNALS GENERATED")
            print(f"This could be because z-scores never exceeded entry threshold of ±{entry_threshold}")
            
            # Check z-score extremes
            max_abs_z = z_score.abs().max()
            print(f"Maximum absolute z-score: {max_abs_z:.4f}")
            if max_abs_z < entry_threshold:
                print(f"❌ BLOCKING POINT: Z-scores never exceeded entry threshold of ±{entry_threshold}")
                print(f"Consider lowering entry threshold or checking pair selection criteria")
            return
        else:
            print(f"✅ {len(entry_signals)} entry signals generated")
            print("First few entry signals:")
            for date, signal in entry_signals.head(3).items():
                z_val = z_score.loc[date]
                print(f"  {date}: {signal} (z-score: {z_val:.4f})")
    print()
    
    # Step 4: Test position opening
    print("=== STEP 4: POSITION OPENING TEST ===")
    
    # Get price data for position opening
    test_date = entry_signals.index[0].strftime('%Y-%m-%d')
    print(f"Testing position opening on date: {test_date}")
    
    price_data = data_api.get_pairs_data(symbol1, symbol2, test_date, test_date)
    if price_data.empty:
        print(f"❌ NO PRICE DATA AVAILABLE for {test_date}")
        return
    
    price1 = price_data[symbol1].iloc[-1]
    price2 = price_data[symbol2].iloc[-1]
    print(f"Prices on {test_date}: {symbol1}={price1:.2f}, {symbol2}={price2:.2f}")
    
    # Test position opening
    pair_id = f"{symbol1}-{symbol2}"
    portfolio_value = engine.position_manager.get_portfolio_value()
    max_position_size = CONFIG['trading']['max_position_size']
    position_value = portfolio_value * max_position_size
    
    print(f"Portfolio value: {portfolio_value:.2f}")
    print(f"Max position size: {max_position_size:.1%}")
    print(f"Position value: {position_value:.2f}")
    
    # Check risk management
    risk_check = engine.risk_manager.check_position_size(pair_id, position_value, portfolio_value)
    print(f"Risk management check: {'PASS' if risk_check else 'FAIL'}")
    
    if not risk_check:
        print("❌ BLOCKING POINT: Risk management rejected position")
        return
    
    # Test actual position opening
    signal_type = entry_signals.iloc[0]
    side = PositionSide.LONG if signal_type == SignalType.ENTRY_LONG else PositionSide.SHORT
    
    print(f"Attempting to open {side.value} position...")
    
    trade_result = engine.position_manager.open_position(
        pair_id, symbol1, symbol2, side, position_value,
        price1, price2, hedge_ratio, test_date
    )
    
    if trade_result:
        print(f"✅ POSITION OPENED SUCCESSFULLY")
        print(f"Trade result: {trade_result}")
        print()
        
        # Step 5: Test position closing
        print("=== STEP 5: POSITION CLOSING TEST ===")
        
        # Find an exit signal
        exit_signals = signals[signals.isin([SignalType.EXIT_LONG, SignalType.EXIT_SHORT])]
        if len(exit_signals) > 0:
            exit_date = exit_signals.index[0].strftime('%Y-%m-%d')
            print(f"Testing position closing on date: {exit_date}")
            
            exit_price_data = data_api.get_pairs_data(symbol1, symbol2, exit_date, exit_date)
            if not exit_price_data.empty:
                exit_price1 = exit_price_data[symbol1].iloc[-1]
                exit_price2 = exit_price_data[symbol2].iloc[-1]
                
                close_result = engine.position_manager.close_position(pair_id, exit_price1, exit_price2, exit_date, 'EXIT')
                
                if close_result:
                    print(f"✅ POSITION CLOSED SUCCESSFULLY")
                    print(f"P&L: {close_result['pnl']:.2f}")
                    print(f"Return: {close_result['return_pct']:.2f}%")
                else:
                    print("❌ POSITION CLOSING FAILED")
            else:
                print(f"❌ No price data available for exit date {exit_date}")
        else:
            print("No exit signals found in test period")
    else:
        print("❌ BLOCKING POINT: Position opening failed")
        
        # Diagnostic checks
        print("Diagnostic checks:")
        print(f"  - Position already exists: {'Yes' if pair_id in engine.position_manager.positions else 'No'}")
        print(f"  - Max positions reached: {'Yes' if len(engine.position_manager.positions) >= engine.position_manager.max_positions else 'No'}")
        print(f"  - Insufficient cash: {'Yes' if position_value > engine.position_manager.cash else 'No'}")
        print(f"  - Available cash: {engine.position_manager.cash:.2f}")
    
    print()
    
    # Step 6: Full backtest with detailed logging
    print("=== STEP 6: MINI BACKTEST WITH DETAILED LOGGING ===")
    
    # Set up detailed logging
    original_log_level = logger.level
    logger.setLevel(10)  # DEBUG level
    
    try:
        # Run a short backtest
        mini_results = engine.run_rolling_backtest(
            start_date='2024-02-01',
            end_date='2024-02-29',
            universe='IBOV',
            save_results=False
        )
        
        if mini_results:
            print(f"Mini backtest results:")
            print(f"  - Total return: {mini_results.get('total_return', 0):.2%}")
            print(f"  - Total trades: {mini_results.get('total_trades', 0)}")
            print(f"  - Final capital: {mini_results.get('final_capital', 0):.2f}")
            
            trades_history = mini_results.get('trades_history', [])
            if trades_history:
                print(f"✅ TRADES EXECUTED: {len(trades_history)}")
                for i, trade in enumerate(trades_history[:3]):
                    print(f"  Trade {i+1}: {trade.get('pair_id', 'Unknown')} - "
                          f"{trade.get('action', 'Unknown')} - P&L: {trade.get('pnl', 0):.2f}")
            else:
                print("❌ NO TRADES EXECUTED IN MINI BACKTEST")
                print("This confirms there's a blocking point in the trading logic")
        else:
            print("❌ MINI BACKTEST FAILED TO RETURN RESULTS")
    
    finally:
        logger.setLevel(original_log_level)
    
    print()
    print("=== SUMMARY ===")
    print("Check the detailed output above to identify where the trade execution is being blocked.")
    print("Common blocking points:")
    print("1. No cointegrated pairs found")
    print("2. Insufficient spread data for signal generation")
    print("3. Z-scores not exceeding entry thresholds")
    print("4. Risk management rejecting positions")
    print("5. Position opening failures")
    print("6. Price data unavailable on trading dates")

if __name__ == "__main__":
    debug_trade_execution_flow()