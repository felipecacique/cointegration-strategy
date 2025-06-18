"""
Simple test to verify data collection works.
Run this from Windows Python where packages are installed.
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Test basic Yahoo Finance download
print("Testing basic Yahoo Finance download...")

# Test PETR4 (Petrobras)
symbol = "PETR4.SA"
start_date = "2024-01-01"
end_date = "2024-06-18"

try:
    ticker = yf.Ticker(symbol)
    data = ticker.history(start=start_date, end=end_date)
    
    if not data.empty:
        print(f"✅ SUCCESS: Downloaded {len(data)} days of data for {symbol}")
        print(f"📅 Date range: {data.index[0].date()} to {data.index[-1].date()}")
        print(f"💹 Columns: {list(data.columns)}")
        print("\n📊 Sample data:")
        print(data.head())
        print(f"\n💰 Latest close price: R$ {data['Close'].iloc[-1]:.2f}")
        
        # Save to CSV for verification
        data.to_csv(f"test_{symbol.replace('.SA', '')}_data.csv")
        print(f"💾 Data saved to test_{symbol.replace('.SA', '')}_data.csv")
        
    else:
        print("❌ FAILED: No data returned")
        
except Exception as e:
    print(f"❌ ERROR: {e}")

print("\n" + "="*50)
print("🔍 Test multiple IBOV symbols...")

# Test multiple symbols
ibov_symbols = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA']

for symbol in ibov_symbols:
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(start="2024-06-01", end="2024-06-18", period="1mo")
        
        if not data.empty:
            print(f"✅ {symbol}: {len(data)} days, latest close: R$ {data['Close'].iloc[-1]:.2f}")
        else:
            print(f"❌ {symbol}: No data")
            
    except Exception as e:
        print(f"❌ {symbol}: Error - {e}")

print("\n🎯 Test completed. If you see ✅ symbols, Yahoo Finance is working!")
print("Next step: Run 'python main.py --init' to populate the database.")