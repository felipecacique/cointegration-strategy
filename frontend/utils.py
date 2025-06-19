"""
Utility functions for Streamlit dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import numpy as np

def load_css():
    """Load custom CSS for the dashboard."""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    
    .success-card {
        background-color: #d4edda;
        border-color: #28a745;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .warning-card {
        background-color: #fff3cd;
        border-color: #ffc107;
        color: #856404;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .error-card {
        background-color: #f8d7da;
        border-color: #dc3545;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    
    .stMetric {
        background-color: white;
        border: 1px solid #e6e6e6;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .sidebar .stSelectbox {
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if 'current_pairs' not in st.session_state:
        st.session_state.current_pairs = []
    
    if 'backtest_results' not in st.session_state:
        st.session_state.backtest_results = {}
    
    if 'live_signals' not in st.session_state:
        st.session_state.live_signals = []
    
    if 'selected_pair' not in st.session_state:
        st.session_state.selected_pair = None
    
    if 'last_data_update' not in st.session_state:
        st.session_state.last_data_update = None

def format_currency(value, prefix="R$"):
    """Format currency values."""
    if pd.isna(value):
        return "N/A"
    
    if abs(value) >= 1e9:
        return f"{prefix} {value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"{prefix} {value/1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"{prefix} {value/1e3:.2f}K"
    else:
        return f"{prefix} {value:.2f}"

def format_percentage(value, decimals=2):
    """Format percentage values."""
    if pd.isna(value):
        return "N/A"
    return f"{value*100:.{decimals}f}%"

def create_metric_card(title, value, delta=None, delta_color="normal"):
    """Create a custom metric card."""
    delta_html = ""
    if delta is not None:
        color = "green" if delta_color == "normal" and delta > 0 else "red" if delta_color == "normal" and delta < 0 else delta_color
        delta_html = f'<span style="color: {color}; font-size: 0.8rem;">({delta:+.2f})</span>'
    
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="margin: 0; color: #666;">{title}</h4>
        <h2 style="margin: 0.5rem 0 0 0; color: #333;">{value} {delta_html}</h2>
    </div>
    """, unsafe_allow_html=True)

def create_status_card(title, status, message=""):
    """Create a status card with color coding."""
    if status == "success":
        card_class = "success-card"
        icon = "✅"
    elif status == "warning":
        card_class = "warning-card"
        icon = "⚠️"
    elif status == "error":
        card_class = "error-card"
        icon = "❌"
    else:
        card_class = "metric-card"
        icon = "ℹ️"
    
    st.markdown(f"""
    <div class="{card_class}">
        <h4 style="margin: 0;">{icon} {title}</h4>
        <p style="margin: 0.5rem 0 0 0;">{message}</p>
    </div>
    """, unsafe_allow_html=True)

def create_equity_curve_chart(equity_curve, benchmark=None, title="Portfolio Performance"):
    """Create an equity curve chart."""
    fig = go.Figure()
    
    # Add strategy line
    fig.add_trace(go.Scatter(
        x=equity_curve.index,
        y=equity_curve.values,
        mode='lines',
        name='Strategy',
        line=dict(color='#1f77b4', width=2)
    ))
    
    # Add benchmark if provided
    if benchmark is not None:
        fig.add_trace(go.Scatter(
            x=benchmark.index,
            y=benchmark.values,
            mode='lines',
            name='Benchmark',
            line=dict(color='#ff7f0e', width=2, dash='dash')
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Portfolio Value",
        hovermode='x unified',
        showlegend=True,
        height=400
    )
    
    return fig

def create_drawdown_chart(equity_curve, title="Drawdown Analysis"):
    """Create a drawdown chart."""
    # Calculate drawdown
    cumulative = equity_curve / equity_curve.iloc[0]
    rolling_max = cumulative.expanding().max()
    drawdown = (cumulative - rolling_max) / rolling_max * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=drawdown.index,
        y=drawdown.values,
        fill='tozeroy',
        mode='lines',
        name='Drawdown %',
        line=dict(color='red', width=1),
        fillcolor='rgba(255, 0, 0, 0.3)'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x unified',
        height=300
    )
    
    return fig

def create_correlation_heatmap(correlation_matrix, title="Correlation Matrix"):
    """Create a correlation heatmap."""
    fig = px.imshow(
        correlation_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title=title
    )
    
    fig.update_layout(height=600)
    return fig

def create_scatter_plot(x_data, y_data, title="Scatter Plot", x_label="X", y_label="Y", 
                       color_data=None, size_data=None):
    """Create a scatter plot."""
    fig = go.Figure()
    
    scatter_kwargs = {
        'x': x_data,
        'y': y_data,
        'mode': 'markers',
        'name': 'Data Points'
    }
    
    if color_data is not None:
        scatter_kwargs['marker'] = dict(
            color=color_data,
            colorscale='Viridis',
            showscale=True
        )
    
    if size_data is not None:
        if 'marker' not in scatter_kwargs:
            scatter_kwargs['marker'] = {}
        scatter_kwargs['marker']['size'] = size_data
    
    fig.add_trace(go.Scatter(**scatter_kwargs))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        hovermode='closest',
        height=400
    )
    
    return fig

def create_z_score_chart(z_score_series, entry_threshold=2.0, exit_threshold=0.5, 
                        stop_loss_threshold=3.0, title="Z-Score Evolution"):
    """Create a z-score chart with trading thresholds."""
    fig = go.Figure()
    
    # Z-score line
    fig.add_trace(go.Scatter(
        x=z_score_series.index,
        y=z_score_series.values,
        mode='lines',
        name='Z-Score',
        line=dict(color='blue', width=2)
    ))
    
    # Threshold lines
    fig.add_hline(y=entry_threshold, line_dash="dash", line_color="green", 
                  annotation_text="Entry Long")
    fig.add_hline(y=-entry_threshold, line_dash="dash", line_color="green", 
                  annotation_text="Entry Short")
    fig.add_hline(y=exit_threshold, line_dash="dot", line_color="orange", 
                  annotation_text="Exit")
    fig.add_hline(y=-exit_threshold, line_dash="dot", line_color="orange")
    fig.add_hline(y=stop_loss_threshold, line_dash="solid", line_color="red", 
                  annotation_text="Stop Loss")
    fig.add_hline(y=-stop_loss_threshold, line_dash="solid", line_color="red")
    
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Z-Score",
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_monthly_returns_heatmap(returns, title="Monthly Returns Heatmap"):
    """Create a monthly returns heatmap."""
    if returns.empty:
        return go.Figure()
    
    # Resample to monthly returns
    monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    
    # Create pivot table
    monthly_returns.index = pd.to_datetime(monthly_returns.index)
    pivot_data = monthly_returns.groupby([
        monthly_returns.index.year,
        monthly_returns.index.month
    ]).first().unstack()
    
    # Month labels
    month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    fig = px.imshow(
        pivot_data.values,
        x=month_labels,
        y=pivot_data.index,
        text_auto='.2%',
        aspect="auto",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=0,
        title=title
    )
    
    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Year",
        height=400
    )
    
    return fig

def create_trade_analysis_chart(trades_df, title="Trade Analysis"):
    """Create trade analysis visualization."""
    if trades_df.empty:
        return go.Figure()
    
    fig = go.Figure()
    
    # Cumulative P&L
    cumulative_pnl = trades_df['pnl'].cumsum()
    
    fig.add_trace(go.Scatter(
        x=range(len(cumulative_pnl)),
        y=cumulative_pnl.values,
        mode='lines+markers',
        name='Cumulative P&L',
        line=dict(color='blue', width=2)
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Trade Number",
        yaxis_title="Cumulative P&L",
        hovermode='x unified',
        height=400
    )
    
    return fig

def display_pair_info(pair_data):
    """Display pair information in a formatted way."""
    if not pair_data:
        st.warning("No pair data available")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("P-Value", f"{pair_data.get('coint_pvalue', 0):.4f}")
    
    with col2:
        st.metric("Half-Life", f"{pair_data.get('half_life', 0):.1f} days")
    
    with col3:
        st.metric("Correlation", f"{pair_data.get('correlation', 0):.3f}")
    
    with col4:
        st.metric("Hedge Ratio", f"{pair_data.get('hedge_ratio', 0):.3f}")

def create_signal_indicator(signal_type, z_score):
    """Create a visual signal indicator."""
    if signal_type == "ENTRY_LONG":
        color = "green"
        icon = "📈"
        text = "LONG ENTRY"
    elif signal_type == "ENTRY_SHORT":
        color = "red"
        icon = "📉"
        text = "SHORT ENTRY"
    elif signal_type in ["EXIT_LONG", "EXIT_SHORT"]:
        color = "orange"
        icon = "🔄"
        text = "EXIT"
    elif signal_type == "STOP_LOSS":
        color = "darkred"
        icon = "🛑"
        text = "STOP LOSS"
    else:
        color = "gray"
        icon = "⚪"
        text = "NO SIGNAL"
    
    st.markdown(f"""
    <div style="
        background-color: {color}; 
        color: white; 
        padding: 0.5rem 1rem; 
        border-radius: 0.5rem; 
        text-align: center;
        font-weight: bold;
        margin: 0.5rem 0;
    ">
        {icon} {text}<br>
        <small>Z-Score: {z_score:.2f}</small>
    </div>
    """, unsafe_allow_html=True)

def format_large_number(number):
    """Format large numbers with appropriate suffixes."""
    if abs(number) >= 1e9:
        return f"{number/1e9:.1f}B"
    elif abs(number) >= 1e6:
        return f"{number/1e6:.1f}M"
    elif abs(number) >= 1e3:
        return f"{number/1e3:.1f}K"
    else:
        return f"{number:.0f}"

@st.cache_data(ttl=300)  # Cache for 5 minutes
def cached_data_loader(func, *args, **kwargs):
    """Generic cached data loader."""
    return func(*args, **kwargs)

def show_loading_spinner(message="Loading..."):
    """Show a loading spinner with message."""
    return st.spinner(message)

def download_dataframe_as_csv(df, filename="data.csv"):
    """Create download button for DataFrame as CSV."""
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=filename,
        mime="text/csv"
    )

def create_metric_card(title, value, delta=None, help_text=None):
    """Create a metric card."""
    st.metric(
        label=title,
        value=value,
        delta=delta,
        help=help_text
    )

def create_status_card(title, status, color="green"):
    """Create a status card."""
    colors = {
        "green": "#28a745",
        "red": "#dc3545", 
        "yellow": "#ffc107",
        "blue": "#007bff"
    }
    
    bg_color = colors.get(color, "#28a745")
    
    st.markdown(f"""
    <div style="
        background-color: {bg_color}; 
        color: white; 
        padding: 1rem; 
        border-radius: 0.5rem; 
        text-align: center;
        margin: 0.5rem 0;
    ">
        <h4 style="margin: 0; color: white;">{title}</h4>
        <p style="margin: 0.5rem 0 0 0; color: white;">{status}</p>
    </div>
    """, unsafe_allow_html=True)

def create_trade_analysis_chart(trades_df, title="Trade Analysis"):
    """Create trade analysis chart."""
    if trades_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No trade data available", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig
    
    # Create cumulative P&L
    trades_df_sorted = trades_df.sort_values('entry_date' if 'entry_date' in trades_df.columns else trades_df.index)
    cumulative_pnl = trades_df_sorted['pnl'].cumsum() if 'pnl' in trades_df_sorted.columns else pd.Series([0])
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(len(cumulative_pnl))),
        y=cumulative_pnl,
        mode='lines+markers',
        name='Cumulative P&L'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Trade Number",
        yaxis_title="Cumulative P&L",
        height=400
    )
    
    return fig

def create_monthly_returns_heatmap(returns, title="Monthly Returns"):
    """Create monthly returns heatmap."""
    try:
        # Convert to monthly returns
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # Create pivot table for heatmap
        monthly_returns_df = monthly_returns.to_frame('returns')
        monthly_returns_df['year'] = monthly_returns_df.index.year
        monthly_returns_df['month'] = monthly_returns_df.index.month
        
        pivot_table = monthly_returns_df.pivot(index='year', columns='month', values='returns')
        
        fig = px.imshow(
            pivot_table,
            text_auto='.2%',
            aspect="auto",
            color_continuous_scale="RdYlGn",
            color_continuous_midpoint=0,
            title=title,
            labels=dict(x="Month", y="Year", color="Return")
        )
        
        return fig
    except:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for heatmap", 
                          xref="paper", yref="paper",
                          x=0.5, y=0.5, showarrow=False)
        return fig