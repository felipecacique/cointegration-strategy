"""
Home/Overview page for the dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

from data.api import MarketDataAPI
from strategy.pairs import PairSelector
from frontend.utils import (
    create_metric_card, create_status_card, create_equity_curve_chart,
    format_currency, format_percentage, show_loading_spinner
)

def render_home_page():
    """Render the home/overview page."""
    
    st.markdown('<h1 class="main-header">🎯 Pairs Trading System - Overview</h1>', 
                unsafe_allow_html=True)
    
    # Quick stats row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        with show_loading_spinner("Loading data..."):
            try:
                api = MarketDataAPI()
                summary = api.get_data_summary()
                st.metric(
                    "Total Symbols", 
                    summary.get('total_symbols', 0),
                    help="Number of stocks in database"
                )
            except Exception as e:
                st.metric("Total Symbols", "Error")
    
    with col2:
        try:
            st.metric(
                "Data Records", 
                f"{summary.get('total_records', 0):,}",
                help="Total price records in database"
            )
        except:
            st.metric("Data Records", "Error")
    
    with col3:
        try:
            if 'backtest_results' in st.session_state and st.session_state.backtest_results:
                total_return = st.session_state.backtest_results.get('total_return', 0)
                st.metric(
                    "Total Return", 
                    format_percentage(total_return),
                    help="Strategy total return"
                )
            else:
                st.metric("Total Return", "N/A")
        except:
            st.metric("Total Return", "Error")
    
    with col4:
        try:
            if 'backtest_results' in st.session_state and st.session_state.backtest_results:
                sharpe = st.session_state.backtest_results.get('sharpe_ratio', 0)
                st.metric(
                    "Sharpe Ratio", 
                    f"{sharpe:.2f}",
                    help="Risk-adjusted return metric"
                )
            else:
                st.metric("Sharpe Ratio", "N/A")
        except:
            st.metric("Sharpe Ratio", "Error")
    
    st.markdown("---")
    
    # Main content area
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("📈 Portfolio Performance")
        
        # Performance chart
        if 'backtest_results' in st.session_state and st.session_state.backtest_results:
            try:
                equity_curve = st.session_state.backtest_results.get('equity_curve')
                
                if equity_curve is not None and not equity_curve.empty:
                    # Create benchmark comparison
                    fig = create_equity_curve_chart(
                        equity_curve, 
                        title="Strategy vs Benchmark"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Performance metrics
                    st.subheader("📊 Key Metrics")
                    
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                    
                    with metrics_col1:
                        performance = st.session_state.backtest_results.get('performance_metrics', {})
                        st.metric("Annualized Return", format_percentage(performance.get('annualized_return', 0)))
                        st.metric("Volatility", format_percentage(performance.get('volatility', 0)))
                    
                    with metrics_col2:
                        st.metric("Max Drawdown", format_percentage(performance.get('max_drawdown', 0)))
                        st.metric("Win Rate", format_percentage(performance.get('win_rate', 0) / 100))
                    
                    with metrics_col3:
                        st.metric("Total Trades", f"{performance.get('total_trades', 0):,}")
                        st.metric("Calmar Ratio", f"{performance.get('calmar_ratio', 0):.2f}")
                
                else:
                    st.info("📊 No performance data available. Run a backtest to see results.")
            
            except Exception as e:
                st.error(f"Error displaying performance data: {e}")
        else:
            # Sample chart for demonstration
            st.info("📊 No backtest results available. Run a backtest to see portfolio performance.")
            
            # Show sample/demo chart
            sample_dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
            sample_returns = np.random.normal(0.0005, 0.02, len(sample_dates))
            sample_equity = (1 + pd.Series(sample_returns, index=sample_dates)).cumprod() * 100000
            
            fig = create_equity_curve_chart(sample_equity, title="Sample Portfolio Performance")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("*This is sample data for demonstration purposes*")
    
    with col_right:
        st.subheader("🚨 System Status")
        
        # System health checks
        try:
            api = MarketDataAPI()
            summary = api.get_data_summary()
            
            # Data freshness check
            last_update = summary.get('last_update')
            if last_update:
                try:
                    last_update_dt = pd.to_datetime(last_update)
                    hours_since_update = (datetime.now() - last_update_dt).total_seconds() / 3600
                    
                    if hours_since_update < 24:
                        create_status_card("Data Status", "success", f"Updated {hours_since_update:.1f} hours ago")
                    elif hours_since_update < 72:
                        create_status_card("Data Status", "warning", f"Updated {hours_since_update:.1f} hours ago")
                    else:
                        create_status_card("Data Status", "error", "Data is stale")
                except:
                    create_status_card("Data Status", "warning", "Unable to parse update time")
            else:
                create_status_card("Data Status", "warning", "No update information")
            
            # Database size check
            db_size = summary.get('database_size_mb', 0)
            if db_size > 0:
                create_status_card("Database", "success", f"Size: {db_size:.1f} MB")
            else:
                create_status_card("Database", "warning", "Size unknown")
            
            # Recent data availability
            recent_symbols = summary.get('symbols_with_recent_data', 0)
            total_symbols = summary.get('total_symbols', 0)
            
            if total_symbols > 0:
                recent_pct = recent_symbols / total_symbols
                if recent_pct > 0.8:
                    create_status_card("Data Coverage", "success", f"{recent_pct:.1%} symbols current")
                elif recent_pct > 0.5:
                    create_status_card("Data Coverage", "warning", f"{recent_pct:.1%} symbols current")
                else:
                    create_status_card("Data Coverage", "error", f"Only {recent_pct:.1%} symbols current")
        
        except Exception as e:
            create_status_card("System Error", "error", str(e))
        
        st.markdown("---")
        
        # Quick pair analysis
        st.subheader("🔍 Recent Pairs")
        
        if st.session_state.current_pairs:
            pairs_df = pd.DataFrame(st.session_state.current_pairs[:5])
            
            for _, pair in pairs_df.iterrows():
                with st.expander(f"{pair['symbol1']}-{pair['symbol2']}"):
                    col_a, col_b = st.columns(2)
                    with col_a:
                        st.metric("P-Value", f"{pair['coint_pvalue']:.4f}")
                        st.metric("Correlation", f"{pair['correlation']:.3f}")
                    with col_b:
                        st.metric("Half-Life", f"{pair['half_life']:.1f}d")
                        st.metric("Rank", f"#{pair.get('rank', 'N/A')}")
        else:
            st.info("No pairs found yet. Use 'Find Pairs' to discover cointegrated pairs.")
            
            if st.button("🔍 Find Pairs Now", key="find_pairs_home"):
                with show_loading_spinner("Finding cointegrated pairs..."):
                    try:
                        selector = PairSelector()
                        pairs = selector.get_pair_universe('IBOV')
                        st.session_state.current_pairs = pairs
                        st.success(f"Found {len(pairs)} pairs!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error finding pairs: {e}")
        
        st.markdown("---")
        
        # Live signals preview
        st.subheader("📊 Live Signals")
        
        if st.session_state.live_signals:
            signals_df = pd.DataFrame(st.session_state.live_signals[:3])
            
            for _, signal in signals_df.iterrows():
                signal_type = signal.get('signal', 'NO_SIGNAL')
                z_score = signal.get('z_score', 0)
                
                if hasattr(signal_type, 'value'):
                    signal_str = signal_type.value
                else:
                    signal_str = str(signal_type)
                
                if signal_str in ['ENTRY_LONG', 'ENTRY_SHORT']:
                    color = "green" if signal_str == 'ENTRY_LONG' else "red"
                    st.markdown(f"""
                    <div style="
                        background-color: {color}; 
                        color: white; 
                        padding: 0.5rem; 
                        border-radius: 0.25rem; 
                        margin: 0.25rem 0;
                        font-size: 0.8rem;
                    ">
                        {signal['pair_id']}: {signal_str} (Z: {z_score:.2f})
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No live signals available. Generate signals from pair analysis.")
    
    # Recent activity footer
    st.markdown("---")
    
    st.subheader("📋 Recent Activity")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔄 Latest Updates**")
        
        activity_items = [
            f"• Data updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"• Pairs analyzed: {len(st.session_state.current_pairs)} found",
            f"• Signals generated: {len(st.session_state.live_signals)} active",
        ]
        
        for item in activity_items:
            st.markdown(item)
    
    with col2:
        st.markdown("**⚡ Quick Actions**")
        
        action_col1, action_col2 = st.columns(2)
        
        with action_col1:
            if st.button("🔄 Refresh Data", key="refresh_data_home"):
                with show_loading_spinner("Refreshing..."):
                    try:
                        api = MarketDataAPI()
                        result = api.update_data()
                        if result['status'] == 'success':
                            st.success("Data refreshed!")
                        else:
                            st.error("Refresh failed")
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with action_col2:
            if st.button("📈 Run Backtest", key="run_backtest_home"):
                st.info("Navigate to Backtesting page to run full analysis")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.8rem;">
        🎯 Pairs Trading System for Brazilian Market | 
        Data Source: Yahoo Finance | 
        Last Updated: {timestamp}
    </div>
    """.format(timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')), 
    unsafe_allow_html=True)