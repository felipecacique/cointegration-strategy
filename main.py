"""
Main entry point for the pairs trading system.
"""
import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.api import MarketDataAPI
from strategy.pairs import PairSelector
from strategy.signals import TradingSignalGenerator
from simple_logger import logger
from config.settings import CONFIG

def initialize_data(universe='IBOV'):
    """Initialize system with market data."""
    print(f"Initializing data for {universe} universe...")
    
    api = MarketDataAPI()
    result = api.initialize_system(universe)
    
    if result['status'] == 'success':
        print("✅ Data initialization successful!")
        print(f"📊 Data summary: {result['data_summary']}")
    else:
        print(f"❌ Data initialization failed: {result.get('error', 'Unknown error')}")
        return False
    
    return True

def find_pairs(universe='IBOV'):
    """Find and analyze cointegrated pairs."""
    print(f"Finding cointegrated pairs in {universe} universe...")
    
    selector = PairSelector()
    pairs = selector.get_pair_universe(universe)
    
    if not pairs:
        print("❌ No cointegrated pairs found!")
        return []
    
    print(f"✅ Found {len(pairs)} cointegrated pairs!")
    
    # Display top pairs
    print("\n🔝 Top 5 pairs:")
    for i, pair in enumerate(pairs[:5]):
        print(f"{i+1}. {pair['symbol1']}-{pair['symbol2']} "
              f"(p-value: {pair['coint_pvalue']:.4f}, "
              f"half-life: {pair['half_life']:.1f} days, "
              f"correlation: {pair['correlation']:.3f})")
    
    # Store results in database
    selector.update_pair_database(pairs)
    
    return pairs

def generate_signals(pairs, days_back=252):
    """Generate current trading signals."""
    print(f"Generating signals for {len(pairs)} pairs...")
    
    signal_gen = TradingSignalGenerator()
    signals = signal_gen.get_current_signals(pairs, lookback_days=days_back)
    
    if not signals:
        print("❌ No signals generated!")
        return []
    
    # Filter for actionable signals
    actionable = signal_gen.filter_actionable_signals(signals)
    
    print(f"✅ Generated {len(signals)} signals, {len(actionable)} actionable!")
    
    # Display actionable signals
    if actionable:
        print("\n🚨 Actionable signals:")
        for signal in actionable:
            print(f"• {signal['pair_id']}: {signal['signal'].value} "
                  f"(z-score: {signal['z_score']:.2f})")
    
    return signals

def run_dashboard():
    """Launch Streamlit dashboard."""
    print("🚀 Launching Streamlit dashboard...")
    print("📱 Dashboard will be available at: http://localhost:8501")
    
    import subprocess
    try:
        subprocess.run([
            "streamlit", "run", "frontend/dashboard.py",
            "--server.port", "8501"
        ])
    except FileNotFoundError:
        print("❌ Streamlit not found. Please install: pip install streamlit")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description="Pairs Trading System")
    parser.add_argument("--init", action="store_true", 
                       help="Initialize system with market data")
    parser.add_argument("--pairs", action="store_true",
                       help="Find cointegrated pairs")
    parser.add_argument("--signals", action="store_true",
                       help="Generate trading signals")
    parser.add_argument("--dashboard", action="store_true",
                       help="Launch Streamlit dashboard")
    parser.add_argument("--universe", default="IBOV",
                       help="Stock universe (IBOV, IBRX100)")
    parser.add_argument("--all", action="store_true",
                       help="Run complete analysis (init + pairs + signals)")
    
    args = parser.parse_args()
    
    if not any([args.init, args.pairs, args.signals, args.dashboard, args.all]):
        parser.print_help()
        return
    
    print("🎯 Pairs Trading System - Brazilian Market")
    print("=" * 50)
    
    try:
        if args.all or args.init:
            if not initialize_data(args.universe):
                return
        
        pairs = []
        if args.all or args.pairs:
            pairs = find_pairs(args.universe)
            if not pairs:
                return
        
        if args.all or args.signals:
            if not pairs:
                # Load pairs from database if not found in current run
                selector = PairSelector()
                historical_pairs = selector.get_historical_pair_results()
                if not historical_pairs.empty:
                    # Convert to expected format
                    pairs = []
                    for _, row in historical_pairs.iterrows():
                        if row['is_cointegrated']:
                            pairs.append({
                                'symbol1': row['symbol1'],
                                'symbol2': row['symbol2'],
                                'hedge_ratio': row['hedge_ratio'],
                                'coint_pvalue': row['p_value'],
                                'half_life': row['half_life'],
                                'correlation': row['correlation']
                            })
            
            if pairs:
                generate_signals(pairs)
        
        if args.dashboard:
            run_dashboard()
        
        print("\n✅ Analysis complete!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
    except Exception as e:
        logger.error(f"Error in main: {e}")
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()