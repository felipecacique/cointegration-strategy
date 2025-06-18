"""
Performance analysis page for the dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from frontend.utils import (
    create_equity_curve_chart, format_currency, format_percentage,
    show_loading_spinner
)

def render_performance_page():
    """Render the performance analysis page."""
    
    st.title("📋 Performance Analysis")
    st.markdown("Comprehensive performance metrics and risk analysis")
    
    # Performance period selector
    col1, col2, col3 = st.columns(3)
    
    with col1:
        analysis_period = st.selectbox("Analysis Period", 
                                     ["Last 1 Month", "Last 3 Months", "Last 6 Months", 
                                      "Last 1 Year", "Since Inception", "Custom"])
    
    with col2:
        if analysis_period == "Custom":
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
        else:
            start_date = get_period_start_date(analysis_period)
    
    with col3:
        if analysis_period == "Custom":
            end_date = st.date_input("End Date", datetime.now())
        else:
            end_date = datetime.now()
    
    # Check if backtest results are available
    if 'backtest_results' not in st.session_state or not st.session_state.backtest_results:
        st.info("📊 No performance data available. Please run a backtest first.")
        
        if st.button("🚀 Run Sample Backtest"):
            show_sample_performance()
        return
    
    results = st.session_state.backtest_results
    
    # Strategy vs Benchmark comparison
    st.subheader("📈 Strategy vs Benchmark")
    
    render_strategy_benchmark_comparison(results)
    
    st.markdown("---")
    
    # Risk metrics
    st.subheader("⚠️ Risk Analysis")
    
    render_risk_analysis(results)
    
    st.markdown("---")
    
    # Factor analysis
    st.subheader("🔍 Factor Analysis")
    
    render_factor_analysis(results)
    
    st.markdown("---")
    
    # Rolling performance
    st.subheader("📊 Rolling Performance")
    
    render_rolling_performance(results)
    
    st.markdown("---")
    
    # Performance attribution
    st.subheader("🎯 Performance Attribution")
    
    render_performance_attribution(results)

def render_strategy_benchmark_comparison(results):
    """Render strategy vs benchmark comparison."""
    
    # Performance comparison table
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Return Comparison**")
        
        performance_metrics = results.get('performance_metrics', {})
        
        comparison_df = pd.DataFrame({
            'Metric': [
                'Total Return',
                'Annualized Return', 
                'Volatility',
                'Sharpe Ratio',
                'Max Drawdown'
            ],
            'Strategy': [
                format_percentage(performance_metrics.get('total_return', 0)),
                format_percentage(performance_metrics.get('annualized_return', 0)),
                format_percentage(performance_metrics.get('volatility', 0)),
                f"{performance_metrics.get('sharpe_ratio', 0):.2f}",
                format_percentage(performance_metrics.get('max_drawdown', 0))
            ],
            'Benchmark (IBOV)': [
                "8.5%",  # Mock benchmark data
                "7.8%",
                "22.3%", 
                "0.35",
                "-15.2%"
            ]
        })
        
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("**📈 Risk-Return Profile**")
        
        # Risk-return scatter plot
        strategy_return = performance_metrics.get('annualized_return', 0)
        strategy_vol = performance_metrics.get('volatility', 0)
        
        fig = go.Figure()
        
        # Strategy point
        fig.add_trace(go.Scatter(
            x=[strategy_vol], 
            y=[strategy_return],
            mode='markers',
            marker=dict(size=15, color='blue'),
            name='Strategy'
        ))
        
        # Benchmark point (mock)
        fig.add_trace(go.Scatter(
            x=[0.223], 
            y=[0.078],
            mode='markers',
            marker=dict(size=15, color='red'),
            name='Benchmark'
        ))
        
        fig.update_layout(
            title="Risk-Return Profile",
            xaxis_title="Volatility",
            yaxis_title="Annualized Return",
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Relative performance chart
    equity_curve = results.get('equity_curve')
    if equity_curve is not None and not equity_curve.empty:
        # Create mock benchmark for comparison
        benchmark_returns = np.random.normal(0.0003, 0.018, len(equity_curve))
        benchmark_curve = (1 + pd.Series(benchmark_returns, index=equity_curve.index)).cumprod() * equity_curve.iloc[0]
        
        fig = go.Figure()
        
        # Strategy line
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Strategy',
            line=dict(color='blue', width=2)
        ))
        
        # Benchmark line
        fig.add_trace(go.Scatter(
            x=benchmark_curve.index,
            y=benchmark_curve.values,
            mode='lines',
            name='IBOV Benchmark',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.update_layout(
            title="Strategy vs Benchmark Performance",
            xaxis_title="Date",
            yaxis_title="Portfolio Value",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_risk_analysis(results):
    """Render risk analysis metrics."""
    
    performance_metrics = results.get('performance_metrics', {})
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📉 Drawdown Metrics**")
        
        max_drawdown = performance_metrics.get('max_drawdown', 0)
        st.metric("Maximum Drawdown", format_percentage(max_drawdown))
        
        # Mock additional drawdown metrics
        st.metric("Average Drawdown", "-3.2%")
        st.metric("Drawdown Duration", "18 days")
        st.metric("Recovery Time", "12 days")
    
    with col2:
        st.markdown("**📊 Risk Ratios**")
        
        sharpe = performance_metrics.get('sharpe_ratio', 0)
        sortino = performance_metrics.get('sortino_ratio', 0)
        calmar = performance_metrics.get('calmar_ratio', 0)
        
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        st.metric("Sortino Ratio", f"{sortino:.2f}")
        st.metric("Calmar Ratio", f"{calmar:.2f}")
        st.metric("Information Ratio", "0.85")  # Mock
    
    with col3:
        st.markdown("**⚡ Volatility Analysis**")
        
        volatility = performance_metrics.get('volatility', 0)
        st.metric("Annualized Volatility", format_percentage(volatility))
        
        # Mock VaR metrics
        st.metric("Daily VaR (95%)", "-1.8%")
        st.metric("Expected Shortfall", "-2.7%")
        st.metric("Skewness", "0.12")
    
    # Value at Risk chart
    equity_curve = results.get('equity_curve')
    if equity_curve is not None and not equity_curve.empty:
        returns = equity_curve.pct_change().dropna()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Returns distribution
            fig = px.histogram(x=returns.values, nbins=30,
                             title="Daily Returns Distribution")
            
            # Add VaR lines
            var_95 = np.percentile(returns, 5)
            var_99 = np.percentile(returns, 1)
            
            fig.add_vline(x=var_95, line_dash="dash", line_color="orange",
                         annotation_text="VaR 95%")
            fig.add_vline(x=var_99, line_dash="dash", line_color="red",
                         annotation_text="VaR 99%")
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Rolling volatility
            rolling_vol = returns.rolling(30).std() * np.sqrt(252)
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol.values,
                mode='lines',
                name='30-Day Rolling Volatility'
            ))
            
            fig.update_layout(
                title="Rolling Volatility (30-day)",
                xaxis_title="Date",
                yaxis_title="Annualized Volatility"
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_factor_analysis(results):
    """Render factor analysis."""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Factor Exposure**")
        
        # Mock factor loadings
        factors_df = pd.DataFrame({
            'Factor': ['Market', 'Size', 'Value', 'Momentum', 'Quality', 'Volatility'],
            'Exposure': [0.85, -0.23, 0.12, 0.08, 0.31, -0.45],
            'T-Stat': [8.2, -2.1, 1.1, 0.7, 2.8, -4.1]
        })
        
        fig = px.bar(factors_df, x='Factor', y='Exposure',
                    title="Factor Exposures",
                    color='Exposure',
                    color_continuous_scale='RdYlBu')
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(factors_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("**🎯 Attribution Analysis**")
        
        # Mock attribution data
        attribution_df = pd.DataFrame({
            'Source': ['Factor Selection', 'Pair Selection', 'Timing', 'Residual'],
            'Contribution': [0.045, 0.032, -0.008, 0.016],
            'Percentage': [52.3, 37.2, -9.3, 18.6]
        })
        
        fig = px.pie(attribution_df, values='Contribution', names='Source',
                    title="Return Attribution")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Attribution table
        attribution_display = attribution_df.copy()
        attribution_display['Contribution'] = attribution_display['Contribution'].apply(format_percentage)
        attribution_display['Percentage'] = attribution_display['Percentage'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(attribution_display, use_container_width=True, hide_index=True)

def render_rolling_performance(results):
    """Render rolling performance metrics."""
    
    equity_curve = results.get('equity_curve')
    if equity_curve is None or equity_curve.empty:
        st.info("No equity curve data available.")
        return
    
    returns = equity_curve.pct_change().dropna()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Rolling Sharpe ratio
        rolling_sharpe = returns.rolling(252).mean() / returns.rolling(252).std() * np.sqrt(252)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_sharpe.index,
            y=rolling_sharpe.values,
            mode='lines',
            name='Rolling Sharpe Ratio (1Y)'
        ))
        
        fig.add_hline(y=1.0, line_dash="dash", line_color="green",
                     annotation_text="Sharpe = 1.0")
        
        fig.update_layout(
            title="Rolling Sharpe Ratio",
            xaxis_title="Date",
            yaxis_title="Sharpe Ratio"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Rolling correlation with benchmark
        # Mock benchmark returns
        benchmark_returns = np.random.normal(0.0003, 0.018, len(returns))
        benchmark_series = pd.Series(benchmark_returns, index=returns.index)
        
        rolling_corr = returns.rolling(252).corr(benchmark_series)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling_corr.index,
            y=rolling_corr.values,
            mode='lines',
            name='Rolling Correlation with Benchmark'
        ))
        
        fig.update_layout(
            title="Rolling Correlation (1Y)",
            xaxis_title="Date",
            yaxis_title="Correlation"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Rolling performance table
    st.markdown("**📊 Rolling Performance Summary**")
    
    # Calculate rolling metrics
    rolling_periods = [30, 90, 180, 252]
    rolling_metrics = []
    
    for period in rolling_periods:
        if len(returns) >= period:
            period_returns = returns.tail(period)
            total_return = (1 + period_returns).prod() - 1
            volatility = period_returns.std() * np.sqrt(252)
            sharpe = period_returns.mean() / period_returns.std() * np.sqrt(252) if period_returns.std() > 0 else 0
            
            rolling_metrics.append({
                'Period': f"{period} days",
                'Return': format_percentage(total_return),
                'Volatility': format_percentage(volatility),
                'Sharpe': f"{sharpe:.2f}"
            })
    
    if rolling_metrics:
        rolling_df = pd.DataFrame(rolling_metrics)
        st.dataframe(rolling_df, use_container_width=True, hide_index=True)

def render_performance_attribution(results):
    """Render performance attribution analysis."""
    
    trades_history = results.get('trades_history', [])
    
    if not trades_history:
        st.info("No trade data available for attribution analysis.")
        return
    
    trades_df = pd.DataFrame(trades_history)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Performance by Pair**")
        
        if 'pair_id' in trades_df.columns and 'pnl' in trades_df.columns:
            pair_performance = trades_df.groupby('pair_id')['pnl'].agg(['sum', 'count', 'mean']).round(2)
            pair_performance.columns = ['Total P&L', 'Trades', 'Avg P&L']
            pair_performance = pair_performance.sort_values('Total P&L', ascending=False)
            
            # Show top 10 pairs
            top_pairs = pair_performance.head(10)
            
            fig = px.bar(x=top_pairs.index, y=top_pairs['Total P&L'],
                        title="Top 10 Pairs by P&L")
            st.plotly_chart(fig, use_container_width=True)
            
            # Format for display
            display_pairs = top_pairs.copy()
            display_pairs['Total P&L'] = display_pairs['Total P&L'].apply(format_currency)
            display_pairs['Avg P&L'] = display_pairs['Avg P&L'].apply(format_currency)
            
            st.dataframe(display_pairs, use_container_width=True)
    
    with col2:
        st.markdown("**⏱️ Performance by Holding Period**")
        
        if 'holding_days' in trades_df.columns and 'pnl' in trades_df.columns:
            # Create holding period bins
            trades_df['holding_period'] = pd.cut(trades_df['holding_days'], 
                                               bins=[0, 5, 10, 20, 30, float('inf')],
                                               labels=['1-5 days', '6-10 days', '11-20 days', 
                                                     '21-30 days', '30+ days'])
            
            period_performance = trades_df.groupby('holding_period')['pnl'].agg(['sum', 'count', 'mean']).round(2)
            period_performance.columns = ['Total P&L', 'Trades', 'Avg P&L']
            
            fig = px.bar(x=period_performance.index, y=period_performance['Total P&L'],
                        title="Performance by Holding Period")
            st.plotly_chart(fig, use_container_width=True)
            
            # Format for display
            display_periods = period_performance.copy()
            display_periods['Total P&L'] = display_periods['Total P&L'].apply(format_currency)
            display_periods['Avg P&L'] = display_periods['Avg P&L'].apply(format_currency)
            
            st.dataframe(display_periods, use_container_width=True)
    
    # Monthly performance breakdown
    st.markdown("**📅 Monthly Performance Breakdown**")
    
    if 'entry_date' in trades_df.columns:
        trades_df['entry_month'] = pd.to_datetime(trades_df['entry_date']).dt.to_period('M')
        monthly_performance = trades_df.groupby('entry_month')['pnl'].agg(['sum', 'count']).round(2)
        monthly_performance.columns = ['Monthly P&L', 'Trades']
        
        if len(monthly_performance) > 0:
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=monthly_performance.index.astype(str),
                y=monthly_performance['Monthly P&L'],
                name='Monthly P&L',
                yaxis='y'
            ))
            
            fig.add_trace(go.Scatter(
                x=monthly_performance.index.astype(str),
                y=monthly_performance['Trades'],
                mode='lines+markers',
                name='Number of Trades',
                yaxis='y2'
            ))
            
            fig.update_layout(
                title="Monthly P&L and Trade Count",
                xaxis_title="Month",
                yaxis=dict(title="P&L", side="left"),
                yaxis2=dict(title="Number of Trades", side="right", overlaying="y"),
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

def get_period_start_date(period):
    """Get start date for analysis period."""
    now = datetime.now()
    
    if period == "Last 1 Month":
        return now - timedelta(days=30)
    elif period == "Last 3 Months":
        return now - timedelta(days=90)
    elif period == "Last 6 Months":
        return now - timedelta(days=180)
    elif period == "Last 1 Year":
        return now - timedelta(days=365)
    elif period == "Since Inception":
        return datetime(2022, 1, 1)  # Default inception
    else:
        return now - timedelta(days=365)

def show_sample_performance():
    """Show sample performance analysis."""
    st.info("📊 Showing sample performance data for demonstration.")
    
    # Create sample data and show simplified performance analysis
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    returns = np.random.normal(0.0008, 0.015, len(dates))
    equity_curve = (1 + pd.Series(returns, index=dates)).cumprod() * 100000
    
    # Mock results
    mock_results = {
        'equity_curve': equity_curve,
        'performance_metrics': {
            'total_return': 0.185,
            'annualized_return': 0.172,
            'volatility': 0.142,
            'sharpe_ratio': 1.21,
            'max_drawdown': -0.087
        },
        'trades_history': []
    }
    
    st.session_state.backtest_results = mock_results
    st.success("Sample performance data loaded!")
    st.rerun()