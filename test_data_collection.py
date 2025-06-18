"""
Simple test script to verify data collection.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data.api import MarketDataAPI
    from config.universe import get_universe_tickers
    print("✅ Imports successful!")
    
    # Test data collection
    print("🔄 Testing data initialization...")
    api = MarketDataAPI()
    
    # Get IBOV symbols
    symbols = get_universe_tickers('IBOV')
    print(f"📊 IBOV universe: {len(symbols)} symbols")
    print(f"First 5 symbols: {symbols[:5]}")
    
    # Test single symbol download
    test_symbol = symbols[0] if symbols else 'PETR4.SA'
    print(f"🧪 Testing download for {test_symbol}...")
    
    from datetime import datetime, timedelta
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    data = api.get_price_data([test_symbol], start_date, end_date)
    
    if not data.empty:
        print(f"✅ Data collected successfully!")
        print(f"📈 Data shape: {data.shape}")
        print(f"📅 Date range: {data.index[0]} to {data.index[-1]}")
        print(f"💹 Columns: {list(data.columns)}")
        print("\n🔍 Sample data:")
        print(data.head())
        
        # Try to store in database
        print("\n💾 Testing database storage...")
        result = api.initialize_system('IBOV')
        print(f"🎯 Initialize result: {result}")
        
    else:
        print("❌ No data collected!")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()