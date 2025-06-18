"""
Main Streamlit dashboard for pairs trading system.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.api import MarketDataAPI
from strategy.pairs import PairSelector
from strategy.signals import TradingSignalGenerator
from backtest.engine import BacktestEngine
from config.settings import CONFIG
from frontend.pages.home import render_home_page
from frontend.pages.live_trading import render_live_trading_page
from frontend.pages.pair_analysis import render_pair_analysis_page
from frontend.pages.backtesting import render_backtesting_page
from frontend.pages.performance import render_performance_page
from frontend.pages.configuration import render_configuration_page
from frontend.pages.data_management import render_data_management_page
from frontend.pages.logs_alerts import render_logs_alerts_page
from frontend.utils import load_css, initialize_session_state

# Page configuration
st.set_page_config(
    page_title="Pairs Trading System - Brazilian Market",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom CSS
load_css()

# Initialize session state
initialize_session_state()

def main():
    """Main dashboard function."""
    
    # Sidebar
    st.sidebar.title("🎯 Pairs Trading System")
    st.sidebar.markdown("**Brazilian Market Analytics**")
    
    # Navigation
    pages = {
        "🏠 Home/Overview": "home",
        "📊 Live Trading": "live_trading", 
        "🔍 Pair Analysis": "pair_analysis",
        "📈 Backtesting": "backtesting",
        "📋 Performance": "performance",
        "⚙️ Configuration": "configuration",
        "💾 Data Management": "data_management",
        "📝 Logs & Alerts": "logs_alerts"
    }
    
    selected_page = st.sidebar.selectbox(
        "Select Page",
        list(pages.keys())
    )
    
    # System status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### System Status")
    
    try:
        # Quick system health check
        api = MarketDataAPI()
        summary = api.get_data_summary()
        
        if summary:
            st.sidebar.success("🟢 System Online")
            st.sidebar.metric("Total Symbols", summary.get('total_symbols', 0))
            st.sidebar.metric("Total Records", f"{summary.get('total_records', 0):,}")
            st.sidebar.caption(f"Last Update: {summary.get('last_update', 'Unknown')}")
        else:
            st.sidebar.warning("🟡 Limited Data")
    except Exception as e:
        st.sidebar.error("🔴 System Error")
        st.sidebar.caption(str(e))
    
    # Quick actions
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Quick Actions")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🔄 Update Data", help="Update market data"):
            with st.spinner("Updating data..."):
                try:
                    api = MarketDataAPI()
                    result = api.update_data()
                    if result['status'] == 'success':
                        st.sidebar.success("Data updated!")
                    else:
                        st.sidebar.error("Update failed")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
    
    with col2:
        if st.button("🔍 Find Pairs", help="Find cointegrated pairs"):
            with st.spinner("Finding pairs..."):
                try:
                    selector = PairSelector()
                    pairs = selector.get_pair_universe('IBOV')
                    st.session_state.current_pairs = pairs
                    st.sidebar.success(f"Found {len(pairs)} pairs!")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
    
    # Main content area
    page_key = pages[selected_page]
    
    if page_key == "home":
        render_home_page()
    elif page_key == "live_trading":
        render_live_trading_page()
    elif page_key == "pair_analysis":
        render_pair_analysis_page()
    elif page_key == "backtesting":
        render_backtesting_page()
    elif page_key == "performance":
        render_performance_page()
    elif page_key == "configuration":
        render_configuration_page()
    elif page_key == "data_management":
        render_data_management_page()
    elif page_key == "logs_alerts":
        render_logs_alerts_page()

if __name__ == "__main__":
    main()