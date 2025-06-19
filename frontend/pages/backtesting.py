"""
Backtesting page for the dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time

from backtest.engine import BacktestEngine
from data.api import MarketDataAPI
from frontend.utils import (
    create_equity_curve_chart, create_drawdown_chart, create_monthly_returns_heatmap,
    create_trade_analysis_chart, format_currency, format_percentage,
    show_loading_spinner, download_dataframe_as_csv
)

def render_backtesting_page():
    """Render the backtesting page."""
    
    st.title("📈 Backtesting")
    st.markdown("Historical simulation and performance analysis of the pairs trading strategy")
    
    # Load default config values
    from config.settings import CONFIG
    
    # Check if reset was requested
    if st.session_state.get('reset_requested', False):
        # Clear the reset flag
        st.session_state.reset_requested = False
    
    # Backtest configuration
    with st.expander("🔧 Backtest Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📅 Time Period")
            
            # Get available date range from database
            try:
                from data.api import MarketDataAPI
                api = MarketDataAPI()
                date_range = api.storage.get_date_range()
                
                if date_range.get('min_date') and date_range.get('max_date'):
                    min_date = pd.to_datetime(date_range['min_date']).date()
                    max_date = pd.to_datetime(date_range['max_date']).date()
                    
                    # Set default dates within available range
                    default_start = max(min_date, datetime(2023, 1, 1).date())
                    default_end = min(max_date, datetime(2024, 12, 31).date())
                    
                    st.caption(f"📊 Available data: {min_date} to {max_date}")
                else:
                    default_start = datetime(2023, 1, 1).date()
                    default_end = datetime(2024, 12, 31).date()
                    
            except Exception as e:
                default_start = datetime(2023, 1, 1).date()
                default_end = datetime(2024, 12, 31).date()
                st.warning(f"Could not load date range: {e}")
            
            start_date = st.date_input("Start Date", default_start)
            end_date = st.date_input("End Date", default_end)
            
            universe = st.selectbox("Universe", 
                                  ["IBOV", "IBRX100", "ALL"])
        
        with col2:
            st.subheader("💰 Capital Settings")
            initial_capital = st.number_input("Initial Capital (R$)", 
                                            10000, 10000000, 
                                            value=CONFIG['trading']['initial_capital'], 
                                            step=10000, key='initial_capital')
            
            max_position_size = st.slider("Max Position Size (%)", 
                                        1, 20, 
                                        value=int(CONFIG['trading']['max_position_size'] * 100),
                                        step=1, key='max_position_size_slider') / 100
            
            max_active_pairs = st.number_input("Max Active Pairs", 
                                             1, 50, 
                                             value=CONFIG['trading']['max_active_pairs'],
                                             step=1, key='max_active_pairs')
        
        with col3:
            st.subheader("📊 Strategy Parameters")
            entry_z_score = st.number_input("Entry Z-Score", 
                                          1.0, 5.0, 
                                          value=CONFIG['trading']['entry_z_score'],
                                          step=0.1, key='entry_z_score')
            
            exit_z_score = st.number_input("Exit Z-Score", 
                                         0.1, 2.0, 
                                         value=CONFIG['trading']['exit_z_score'],
                                         step=0.1, key='exit_z_score')
            
            stop_loss_z_score = st.number_input("Stop-Loss Z-Score", 
                                               2.0, 10.0, 
                                               value=CONFIG['trading']['stop_loss_z_score'],
                                               step=0.1, key='stop_loss_z_score')
        
        # Advanced settings
        with st.expander("⚙️ Advanced Settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                lookback_window = st.number_input("Lookback Window (days)", 
                                                100, 500, 
                                                value=CONFIG['strategy']['lookback_window'],
                                                step=10, key='lookback_window')
                trading_window = st.number_input("Trading Window (days)", 
                                               10, 200, 
                                               value=CONFIG['strategy']['trading_window'],
                                               step=5, key='trading_window')
            
            with col2:
                rebalance_frequency = st.number_input("Rebalance Frequency (days)", 
                                                    1, 100, 
                                                    value=CONFIG['strategy']['rebalance_frequency'],
                                                    step=1, key='rebalance_frequency')
                commission_rate = st.number_input("Commission Rate (%)", 
                                                0.0, 1.0, 
                                                value=CONFIG['trading']['commission_rate'] * 100,
                                                step=0.1, key='commission_rate_input') / 100
        
        # Cointegration criteria section
        with st.expander("🔬 Cointegration Criteria"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                p_value_threshold = st.number_input("P-value Threshold", 
                                                   0.001, 0.20, 
                                                   value=CONFIG['strategy']['p_value_threshold'],
                                                   step=0.005, key='p_value_threshold',
                                                   help="Maximum p-value for cointegration test (0.05 = 95% confidence)")
                min_correlation = st.number_input("Min Correlation", 
                                                 0.3, 0.95, 
                                                 value=CONFIG['strategy']['min_correlation'],
                                                 step=0.05, key='min_correlation',
                                                 help="Minimum absolute correlation between pairs")
            
            with col2:
                min_half_life = st.number_input("Min Half-Life (days)", 
                                               1, 50, 
                                               value=CONFIG['strategy']['min_half_life'],
                                               step=1, key='min_half_life',
                                               help="Minimum mean reversion speed")
                max_half_life = st.number_input("Max Half-Life (days)", 
                                               10, 200, 
                                               value=CONFIG['strategy']['max_half_life'],
                                               step=5, key='max_half_life',
                                               help="Maximum mean reversion speed")
            
            with col3:
                top_pairs = st.number_input("Top Pairs to Select", 
                                           5, 50, 
                                           value=CONFIG['strategy']['top_pairs'],
                                           step=1, key='top_pairs',
                                           help="Number of best pairs to trade")
                sector_matching = st.checkbox("Same Sector Only", 
                                             value=CONFIG['strategy']['sector_matching'],
                                             key='sector_matching',
                                             help="Only trade pairs from the same sector")
        
        # Risk management section
        with st.expander("⚠️ Risk Management"):
            col1, col2 = st.columns(2)
            
            with col1:
                force_close_missing_days = st.number_input("Force Close Missing Data (days)", 
                                                         3, 30, 
                                                         value=CONFIG['trading']['force_close_missing_days'],
                                                         step=1, key='force_close_missing_days',
                                                         help="Force close position after X consecutive days without data")
            
            with col2:
                max_holding_period = st.number_input("Max Holding Period (days)", 
                                                    7, 365, 
                                                    value=CONFIG['trading']['max_holding_period'],
                                                    step=1, key='max_holding_period',
                                                    help="Maximum days to hold any position (timeout)")
        
        # Run backtest and reset buttons
        col1, col2, col3, col4 = st.columns([1, 1.5, 1, 1])
        
        # Check if backtest is running
        backtest_running = st.session_state.get('backtest_running', False)
        
        with col2:
            if not backtest_running:
                if st.button("🚀 Run Backtest", type="primary", use_container_width=True):
                    st.session_state.backtest_running = True
                    st.session_state.backtest_stopped = False
                    run_backtest(start_date, end_date, universe, initial_capital,
                               max_position_size, max_active_pairs, entry_z_score,
                               exit_z_score, stop_loss_z_score, lookback_window,
                               trading_window, rebalance_frequency, commission_rate,
                               p_value_threshold, min_correlation, min_half_life, 
                               max_half_life, top_pairs, sector_matching,
                               force_close_missing_days, max_holding_period)
            else:
                st.button("🚀 Running...", disabled=True, use_container_width=True)
        
        with col3:
            if backtest_running:
                if st.button("⏹️ Stop Backtest", type="secondary", use_container_width=True):
                    st.session_state.backtest_stopped = True
                    st.session_state.backtest_running = False
                    st.warning("⚠️ Backtest stopped by user")
                    st.rerun()
            else:
                if st.button("🔄 Reset to Defaults", use_container_width=True):
                    reset_to_defaults()
    
    st.markdown("---")
    
    # Results section
    if 'backtest_results' in st.session_state and st.session_state.backtest_results:
        display_backtest_results()
    else:
        st.info("📊 Configure parameters above and click 'Run Backtest' to analyze strategy performance.")
        
        # Show sample results for demonstration
        show_sample_backtest_results()

def reset_to_defaults():
    """Reset all parameters to their default values."""
    # Clear all widget keys to force reset
    keys_to_clear = [
        'initial_capital', 'max_position_size_slider', 'max_active_pairs',
        'entry_z_score', 'exit_z_score', 'stop_loss_z_score',
        'lookback_window', 'trading_window', 'rebalance_frequency', 'commission_rate_input',
        'p_value_threshold', 'min_correlation', 'min_half_life', 'max_half_life', 
        'top_pairs', 'sector_matching', 'force_close_missing_days', 'max_holding_period'
    ]
    
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]
    
    # Set a flag to indicate reset was requested
    st.session_state.reset_requested = True
    
    st.success("✅ Parameters reset to defaults!")
    st.rerun()

def run_backtest(start_date, end_date, universe, initial_capital, max_position_size,
                max_active_pairs, entry_z_score, exit_z_score, stop_loss_z_score,
                lookback_window, trading_window, rebalance_frequency, commission_rate,
                p_value_threshold, min_correlation, min_half_life, max_half_life, 
                top_pairs, sector_matching, force_close_missing_days, max_holding_period):
    """Run the backtest with specified parameters."""
    
    # Create progress indicators
    progress_container = st.container()
    with progress_container:
        progress_bar = st.progress(0)
        status_text = st.empty()
        time_estimate = st.empty()
    
    # Check if user stopped the backtest
    if st.session_state.get('backtest_stopped', False):
        st.session_state.backtest_running = False
        return
    
    with show_loading_spinner("Running backtest... This may take several minutes."):
        try:
            # Create API and backtest engine with same instance
            from data.api import MarketDataAPI
            api = MarketDataAPI()
            engine = BacktestEngine(initial_capital=initial_capital, data_api=api)
            
            # Calculate total periods for progress tracking
            import pandas as pd
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            total_periods = int((end_dt - start_dt).days / rebalance_frequency)
            
            def update_progress(current_period, current_date, message=""):
                # Check if user stopped the backtest
                if st.session_state.get('backtest_stopped', False):
                    raise InterruptedError("Backtest stopped by user")
                
                if current_period <= total_periods:
                    progress = current_period / total_periods
                    progress_bar.progress(progress)
                    
                    # Time estimation
                    if current_period > 0:
                        elapsed = time.time() - start_time
                        estimated_total = elapsed / progress
                        remaining = estimated_total - elapsed
                        remaining_min = int(remaining / 60)
                        remaining_sec = int(remaining % 60)
                        time_estimate.text(f"⏱️ Estimated remaining: {remaining_min}m {remaining_sec}s")
                    
                    status_text.text(f"📊 Period {current_period}/{total_periods}: {current_date} - {message}")
                    
                    # Force UI update
                    time.sleep(0.1)
            
            # Store update function in engine for callbacks
            engine.progress_callback = update_progress
            start_time = time.time()
            
            # Update configuration temporarily
            from config.settings import CONFIG
            original_config = CONFIG.copy()
            
            CONFIG['strategy']['lookback_window'] = lookback_window
            CONFIG['strategy']['trading_window'] = trading_window
            CONFIG['strategy']['rebalance_frequency'] = rebalance_frequency
            CONFIG['strategy']['top_pairs'] = top_pairs
            CONFIG['strategy']['p_value_threshold'] = p_value_threshold
            CONFIG['strategy']['min_correlation'] = min_correlation
            CONFIG['strategy']['min_half_life'] = min_half_life
            CONFIG['strategy']['max_half_life'] = max_half_life
            CONFIG['strategy']['sector_matching'] = sector_matching
            CONFIG['trading']['entry_z_score'] = entry_z_score
            CONFIG['trading']['exit_z_score'] = exit_z_score
            CONFIG['trading']['stop_loss_z_score'] = stop_loss_z_score
            CONFIG['trading']['max_position_size'] = max_position_size
            CONFIG['trading']['max_active_pairs'] = max_active_pairs
            CONFIG['trading']['commission_rate'] = commission_rate
            CONFIG['trading']['force_close_missing_days'] = force_close_missing_days
            CONFIG['trading']['max_holding_period'] = max_holding_period
            
            # Run backtest
            results = engine.run_rolling_backtest(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                universe=universe
            )
            
            if results:
                st.session_state.backtest_results = results
                st.success("✅ Backtest completed successfully!")
            else:
                st.error("❌ Backtest failed. Please check your parameters and try again.")
            
            # Restore original config
            CONFIG.update(original_config)
        
        except InterruptedError:
            st.warning("⚠️ Backtest was stopped by user")
            # Restore original config
            CONFIG.update(original_config)
        
        except Exception as e:
            st.error(f"Error running backtest: {e}")
            # Restore original config
            CONFIG.update(original_config)
        
        finally:
            # Always reset the running state
            st.session_state.backtest_running = False

def display_backtest_results():
    """Display comprehensive backtest results."""
    
    results = st.session_state.backtest_results
    
    # Performance summary
    st.subheader("📊 Performance Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_return = results.get('total_return', 0)
        st.metric("Total Return", format_percentage(total_return),
                 help="Total strategy return over backtest period")
    
    with col2:
        annualized_return = results.get('annualized_return', 0)
        st.metric("Annualized Return", format_percentage(annualized_return),
                 help="Annualized return")
    
    with col3:
        sharpe_ratio = results.get('sharpe_ratio', 0)
        st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}",
                 help="Risk-adjusted return metric")
    
    with col4:
        max_drawdown = results.get('max_drawdown', 0)
        st.metric("Max Drawdown", format_percentage(max_drawdown),
                 help="Maximum peak-to-trough decline")
    
    with col5:
        win_rate = results.get('win_rate', 0)
        st.metric("Win Rate", format_percentage(win_rate / 100),
                 help="Percentage of profitable trades")
    
    # Equity curve and drawdown
    st.subheader("📈 Equity Curve")
    
    equity_curve = results.get('equity_curve')
    
    # Get benchmark comparison if available (moved outside if block)
    benchmark_comparison = get_benchmark_comparison(results)
    
    if equity_curve is not None and not equity_curve.empty:
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Equity curve chart
            fig = create_equity_curve_chart(equity_curve, title="Portfolio Performance")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Key metrics
            final_value = equity_curve.iloc[-1]
            initial_value = equity_curve.iloc[0]
            
            st.metric("Initial Capital", format_currency(initial_value))
            st.metric("Final Value", format_currency(final_value))
            st.metric("Profit/Loss", format_currency(final_value - initial_value))
            
            if benchmark_comparison:
                excess_return = benchmark_comparison.get('excess_return', 0)
                st.metric("Excess Return", format_percentage(excess_return),
                         help="Return above benchmark")
        
        # Drawdown chart
        st.subheader("📉 Drawdown Analysis")
        fig = create_drawdown_chart(equity_curve)
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance metrics table
    st.subheader("📋 Detailed Metrics")
    
    performance_metrics = results.get('performance_metrics', {})
    
    if performance_metrics:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Return Metrics**")
            metrics_df = pd.DataFrame([
                ["Total Return", format_percentage(performance_metrics.get('total_return', 0))],
                ["Annualized Return", format_percentage(performance_metrics.get('annualized_return', 0))],
                ["Volatility", format_percentage(performance_metrics.get('volatility', 0))],
                ["Sharpe Ratio", f"{performance_metrics.get('sharpe_ratio', 0):.2f}"],
                ["Sortino Ratio", f"{performance_metrics.get('sortino_ratio', 0):.2f}"],
                ["Calmar Ratio", f"{performance_metrics.get('calmar_ratio', 0):.2f}"]
            ], columns=["Metric", "Value"])
            
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**Trade Metrics**")
            trade_metrics_df = pd.DataFrame([
                ["Total Trades", f"{performance_metrics.get('total_trades', 0):,}"],
                ["Win Rate", format_percentage(performance_metrics.get('win_rate', 0) / 100)],
                ["Average Win", format_currency(performance_metrics.get('avg_win', 0))],
                ["Average Loss", format_currency(performance_metrics.get('avg_loss', 0))],
                ["Best Trade", format_currency(performance_metrics.get('best_trade', 0))],
                ["Worst Trade", format_currency(performance_metrics.get('worst_trade', 0))]
            ], columns=["Metric", "Value"])
            
            st.dataframe(trade_metrics_df, use_container_width=True, hide_index=True)
    
    # Trade analysis
    st.subheader("💼 Trade Analysis")
    
    trades_history = results.get('trades_history', [])
    
    if trades_history:
        trades_df = pd.DataFrame(trades_history)
        
        # Trade statistics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Trades", len(trades_df))
        
        with col2:
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            st.metric("Winning Trades", winning_trades)
        
        with col3:
            avg_trade_duration = trades_df['holding_days'].mean() if 'holding_days' in trades_df.columns else 0
            st.metric("Avg Duration", f"{avg_trade_duration:.1f} days")
        
        with col4:
            total_pnl = trades_df['pnl'].sum()
            st.metric("Total P&L", format_currency(total_pnl))
        
        # Trade analysis charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Cumulative P&L
            fig = create_trade_analysis_chart(trades_df, "Cumulative Trade P&L")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Trade duration distribution
            if 'holding_days' in trades_df.columns:
                fig = px.histogram(trades_df, x='holding_days', nbins=20,
                                 title="Trade Duration Distribution",
                                 labels={'holding_days': 'Holding Days', 'count': 'Number of Trades'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Monthly returns heatmap
        if equity_curve is not None and not equity_curve.empty:
            st.subheader("📅 Monthly Returns")
            returns = equity_curve.pct_change().dropna()
            fig = create_monthly_returns_heatmap(returns, "Monthly Returns Heatmap")
            st.plotly_chart(fig, use_container_width=True)
        
        # Trade details table
        st.subheader("📊 Trade Details")
        
        if st.checkbox("Show all trades"):
            # Prepare display dataframe
            display_trades_df = trades_df.copy()
            
            if 'entry_date' in display_trades_df.columns:
                display_trades_df['Entry Date'] = pd.to_datetime(display_trades_df['entry_date']).dt.strftime('%Y-%m-%d')
            if 'exit_date' in display_trades_df.columns:
                display_trades_df['Exit Date'] = pd.to_datetime(display_trades_df['exit_date']).dt.strftime('%Y-%m-%d')
            
            display_trades_df['P&L'] = display_trades_df['pnl'].apply(format_currency)
            display_trades_df['Return %'] = (display_trades_df['pnl'] / display_trades_df['capital_allocated'] * 100).apply(lambda x: f"{x:.2f}%")
            
            columns_to_show = ['pair_id', 'side', 'Entry Date', 'Exit Date', 'holding_days', 'P&L', 'Return %', 'exit_reason']
            available_columns = [col for col in columns_to_show if col in display_trades_df.columns]
            
            st.dataframe(display_trades_df[available_columns], use_container_width=True)
            
            # Download trades
            download_dataframe_as_csv(display_trades_df, "trade_history.csv")
    
    else:
        st.info("No trade data available in backtest results.")
    
    # Pair performance analysis
    st.subheader("🔍 Pair Performance")
    
    pair_history = results.get('pair_history', [])
    
    if pair_history:
        # Show pairs used over time
        pair_dates = [p['date'] for p in pair_history]
        pair_counts = [p['pairs_count'] for p in pair_history]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pair_dates, y=pair_counts,
                               mode='lines+markers', name='Active Pairs'))
        
        fig.update_layout(title="Number of Active Pairs Over Time",
                        xaxis_title="Date", yaxis_title="Number of Pairs")
        st.plotly_chart(fig, use_container_width=True)
    
    # Benchmark comparison
    if benchmark_comparison:
        st.subheader("📊 Benchmark Comparison")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            strategy_return = benchmark_comparison.get('strategy_total_return', 0)
            st.metric("Strategy Return", format_percentage(strategy_return))
        
        with col2:
            benchmark_return = benchmark_comparison.get('benchmark_total_return', 0)
            st.metric("Benchmark Return", format_percentage(benchmark_return))
        
        with col3:
            excess_return = benchmark_comparison.get('excess_return', 0)
            st.metric("Excess Return", format_percentage(excess_return))
        
        # Additional benchmark metrics
        benchmark_metrics_df = pd.DataFrame([
            ["Beta", f"{benchmark_comparison.get('beta', 0):.2f}"],
            ["Alpha", format_percentage(benchmark_comparison.get('alpha', 0))],
            ["Information Ratio", f"{benchmark_comparison.get('information_ratio', 0):.2f}"],
            ["Tracking Error", format_percentage(benchmark_comparison.get('tracking_error', 0))],
            ["Correlation", f"{benchmark_comparison.get('correlation', 0):.3f}"]
        ], columns=["Metric", "Value"])
        
        st.dataframe(benchmark_metrics_df, use_container_width=True, hide_index=True)

def get_benchmark_comparison(results):
    """Get benchmark comparison if available."""
    try:
        # In a real implementation, you would calculate benchmark comparison
        # For now, return mock data
        return {
            'strategy_total_return': results.get('total_return', 0),
            'benchmark_total_return': 0.08,  # Mock 8% benchmark return
            'excess_return': results.get('total_return', 0) - 0.08,
            'beta': 0.85,
            'alpha': 0.02,
            'information_ratio': 0.75,
            'tracking_error': 0.12,
            'correlation': 0.65
        }
    except:
        return None

def show_sample_backtest_results():
    """Show sample backtest results for demonstration."""
    
    st.subheader("📊 Sample Results (Demo)")
    st.caption("*These are sample results for demonstration. Run a backtest to see actual performance.*")
    
    # Sample metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", "15.3%")
    
    with col2:
        st.metric("Sharpe Ratio", "1.25")
    
    with col3:
        st.metric("Max Drawdown", "-8.2%")
    
    with col4:
        st.metric("Win Rate", "62.5%")
    
    # Sample equity curve
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    returns = np.random.normal(0.0005, 0.015, len(dates))
    equity_curve = (1 + pd.Series(returns, index=dates)).cumprod() * 100000
    
    fig = create_equity_curve_chart(equity_curve, title="Sample Portfolio Performance")
    st.plotly_chart(fig, use_container_width=True)
    
    # Sample trade statistics
    st.subheader("Sample Trade Statistics")
    
    sample_trades = pd.DataFrame({
        'Metric': ['Total Trades', 'Winning Trades', 'Average Win', 'Average Loss', 'Profit Factor'],
        'Value': ['156', '98', 'R$ 1,250', 'R$ -890', '1.42']
    })
    
    st.dataframe(sample_trades, use_container_width=True, hide_index=True)