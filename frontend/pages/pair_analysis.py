"""
Enhanced Pair Analysis page with individual pair testing capabilities.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from data.api import MarketDataAPI
from strategy.pairs import PairSelector
from strategy.cointegration import CointegrationTester
from config.universe import get_universe_tickers, SECTOR_MAPPING
from frontend.utils import (
    create_scatter_plot, create_correlation_heatmap, format_percentage,
    show_loading_spinner, download_dataframe_as_csv
)

def render_pair_analysis_page():
    """Render the enhanced pair analysis page."""
    
    st.title("🔍 Pair Analysis")
    st.markdown("Discover and analyze cointegrated pairs for trading opportunities")
    
    # Create tabs for different analysis types
    tab1, tab2, tab3 = st.tabs(["🧮 Individual Pair Testing", "📊 Bulk Pair Analysis", "🔍 Existing Pairs"])
    
    with tab1:
        render_individual_pair_analysis()
    
    with tab2:
        render_bulk_pair_analysis()
    
    with tab3:
        render_existing_pairs_analysis()

def render_individual_pair_analysis():
    """Render individual pair testing similar to simple_dashboard."""
    
    st.header("🧮 Test Individual Pair")
    st.markdown("Select two stocks and test their cointegration relationship in detail")
    
    # Get available symbols
    try:
        api = MarketDataAPI()
        all_symbols = api.get_available_symbols()
        
        if len(all_symbols) < 2:
            st.warning("⚠️ Need at least 2 symbols in database. Please update data first.")
            return
            
    except Exception as e:
        st.error(f"Error loading symbols: {e}")
        return
    
    # Symbol selection
    col1, col2 = st.columns(2)
    
    with col1:
        symbol1 = st.selectbox("First Stock", all_symbols, index=0, key="symbol1")
    
    with col2:
        symbol2 = st.selectbox("Second Stock", all_symbols, 
                              index=1 if len(all_symbols) > 1 else 0, key="symbol2")
    
    if symbol1 == symbol2:
        st.warning("⚠️ Please select two different stocks")
        return
    
    # Analysis period
    col1, col2, col3 = st.columns(3)
    
    with col1:
        days_back = st.slider("Analysis Period (days)", 252, 1000, 500)
    
    with col2:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        st.write(f"**Period:** {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    with col3:
        st.write(f"**Selected Pair:** {symbol1} vs {symbol2}")
    
    # Test button
    if st.button("🧮 Test Cointegration", type="primary", use_container_width=True):
        test_individual_pair(symbol1, symbol2, start_date, end_date)

def test_individual_pair(symbol1, symbol2, start_date, end_date):
    """Test cointegration for individual pair and display detailed results."""
    
    with st.spinner(f"Testing cointegration {symbol1} vs {symbol2}..."):
        try:
            # Load data
            api = MarketDataAPI()
            pair_data = api.get_pairs_data(
                symbol1, symbol2, 
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )
            
            if pair_data.empty:
                st.error("❌ Could not load data for selected pair")
                return
            
            if len(pair_data) < 100:
                st.warning(f"⚠️ Limited data available: {len(pair_data)} days")
                return
            
            # Test cointegration
            tester = CointegrationTester()
            result = tester.test_pair_cointegration(pair_data[symbol1], pair_data[symbol2])
            
            # Display results
            display_detailed_cointegration_results(result, symbol1, symbol2, pair_data)
            
        except Exception as e:
            st.error(f"Error in cointegration test: {e}")

def display_detailed_cointegration_results(result, symbol1, symbol2, pair_data):
    """Display detailed cointegration results with charts."""
    
    st.subheader(f"📊 Cointegration Results: {symbol1} vs {symbol2}")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        is_coint = result.get('is_cointegrated', False)
        color = "green" if is_coint else "red"
        st.markdown(f"""
        <div style="background-color: {color}; color: white; padding: 1rem; border-radius: 0.5rem; text-align: center;">
            <h4>{'✅ COINTEGRATED' if is_coint else '❌ NOT COINTEGRATED'}</h4>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        pvalue = result.get('coint_pvalue', 1.0)
        st.metric("P-Value", f"{pvalue:.4f}", 
                 help="Lower is better (< 0.05 indicates cointegration)")
    
    with col3:
        half_life = result.get('half_life', 0)
        st.metric("Half-Life", f"{half_life:.1f} days",
                 help="Speed of mean reversion")
    
    with col4:
        correlation = result.get('correlation', 0)
        st.metric("Correlation", f"{correlation:.3f}",
                 help="Linear relationship strength")
    
    # Additional metrics
    st.subheader("📋 Detailed Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Cointegration Test Results**")
        metrics_df = pd.DataFrame([
            ["ADF Statistic", f"{result.get('coint_statistic', 0):.4f}"],
            ["P-Value", f"{result.get('coint_pvalue', 1):.4f}"],
            ["Critical Value (1%)", f"{result.get('critical_1', 0):.4f}"],
            ["Critical Value (5%)", f"{result.get('critical_5', 0):.4f}"],
            ["Hedge Ratio", f"{result.get('hedge_ratio', 1):.4f}"],
            ["Intercept", f"{result.get('intercept', 0):.4f}"]
        ], columns=["Metric", "Value"])
        
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.markdown("**Relationship Metrics**")
        relationship_df = pd.DataFrame([
            ["Correlation", f"{result.get('correlation', 0):.4f}"],
            ["Half-Life", f"{result.get('half_life', 0):.2f} days"],
            ["Mean Reversion Speed", f"{1/result.get('half_life', 1)*100:.2f}% per day"],
            ["Data Points", f"{len(pair_data):,}"],
            ["Analysis Period", f"{len(pair_data)/252:.1f} years"],
            ["Volatility Ratio", f"{pair_data[symbol1].std()/pair_data[symbol2].std():.3f}"]
        ], columns=["Metric", "Value"])
        
        st.dataframe(relationship_df, use_container_width=True, hide_index=True)
    
    # Charts
    st.subheader("📈 Visual Analysis")
    
    # Price comparison
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        
        # Normalize prices to base 100
        norm1 = pair_data[symbol1] / pair_data[symbol1].iloc[0] * 100
        norm2 = pair_data[symbol2] / pair_data[symbol2].iloc[0] * 100
        
        fig.add_trace(go.Scatter(x=norm1.index, y=norm1, name=symbol1, 
                               line=dict(color='blue', width=2)))
        fig.add_trace(go.Scatter(x=norm2.index, y=norm2, name=symbol2, 
                               line=dict(color='red', width=2)))
        
        fig.update_layout(
            title="Normalized Prices (Base 100)",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=pair_data[symbol1], 
            y=pair_data[symbol2],
            mode='markers',
            name='Price Relationship',
            marker=dict(color='blue', size=4, opacity=0.6)
        ))
        
        # Add trend line
        hedge_ratio = result.get('hedge_ratio', 1.0)
        intercept = result.get('intercept', 0)
        x_range = [pair_data[symbol1].min(), pair_data[symbol1].max()]
        y_trend = [hedge_ratio * x + intercept for x in x_range]
        
        fig.add_trace(go.Scatter(
            x=x_range,
            y=y_trend,
            mode='lines',
            name=f'Hedge Ratio: {hedge_ratio:.3f}',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title="Price Relationship & Hedge Ratio",
            xaxis_title=f"{symbol1} Price",
            yaxis_title=f"{symbol2} Price",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Spread analysis
    hedge_ratio = result.get('hedge_ratio', 1.0)
    intercept = result.get('intercept', 0)
    spread = pair_data[symbol1] - hedge_ratio * pair_data[symbol2] - intercept
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spread.index, y=spread, name='Spread', 
                               line=dict(color='green', width=1)))
        fig.add_hline(y=spread.mean(), line_dash="dash", line_color="red", 
                     annotation_text="Mean")
        fig.add_hline(y=spread.mean() + spread.std(), line_dash="dot", 
                     line_color="orange", annotation_text="+1σ")
        fig.add_hline(y=spread.mean() - spread.std(), line_dash="dot", 
                     line_color="orange", annotation_text="-1σ")
        
        fig.update_layout(
            title="Spread (Residuals)",
            xaxis_title="Date",
            yaxis_title="Spread Value",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Z-Score
        z_score = (spread - spread.mean()) / spread.std()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=z_score.index, y=z_score, name='Z-Score', 
                               line=dict(color='purple', width=1)))
        
        # Trading signals
        fig.add_hline(y=2, line_dash="dash", line_color="green", 
                     annotation_text="Entry Long Threshold")
        fig.add_hline(y=-2, line_dash="dash", line_color="green", 
                     annotation_text="Entry Short Threshold")
        fig.add_hline(y=0.5, line_dash="dot", line_color="orange", 
                     annotation_text="Exit Long")
        fig.add_hline(y=-0.5, line_dash="dot", line_color="orange", 
                     annotation_text="Exit Short")
        fig.add_hline(y=0, line_dash="solid", line_color="gray", 
                     annotation_text="Mean")
        
        fig.update_layout(
            title="Z-Score (Trading Signals)",
            xaxis_title="Date",
            yaxis_title="Z-Score",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Current signal
    if not z_score.empty:
        current_z = z_score.iloc[-1]
        
        st.subheader("🚨 Current Trading Signal")
        
        if current_z <= -2:
            st.markdown(f"""
            <div style="background-color: green; color: white; padding: 1rem; border-radius: 0.5rem; text-align: center; font-weight: bold;">
                🟢 ENTRY LONG SIGNAL<br>
                Buy {symbol1}, Sell {symbol2}<br>
                Z-Score: {current_z:.2f}
            </div>
            """, unsafe_allow_html=True)
        elif current_z >= 2:
            st.markdown(f"""
            <div style="background-color: red; color: white; padding: 1rem; border-radius: 0.5rem; text-align: center; font-weight: bold;">
                🔴 ENTRY SHORT SIGNAL<br>
                Sell {symbol1}, Buy {symbol2}<br>
                Z-Score: {current_z:.2f}
            </div>
            """, unsafe_allow_html=True)
        elif abs(current_z) <= 0.5:
            st.info(f"💭 **NO SIGNAL** - Z-Score neutral: {current_z:.2f}")
        else:
            st.warning(f"⏳ **WAITING** - Z-Score: {current_z:.2f}")
    
    # Download data option
    st.subheader("💾 Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Prepare export data
        export_data = pair_data.copy()
        export_data['spread'] = spread
        export_data['z_score'] = z_score
        export_data['hedge_ratio'] = hedge_ratio
        
        download_dataframe_as_csv(export_data, f"pair_analysis_{symbol1}_{symbol2}.csv")
    
    with col2:
        # Summary stats
        st.write("**Analysis Summary:**")
        st.write(f"- Period: {len(pair_data)} trading days")
        st.write(f"- Correlation: {result.get('correlation', 0):.3f}")
        st.write(f"- Cointegrated: {'Yes' if result.get('is_cointegrated', False) else 'No'}")
        st.write(f"- Current Z-Score: {z_score.iloc[-1]:.2f}")

def render_bulk_pair_analysis():
    """Render bulk pair analysis (existing functionality)."""
    
    st.header("📊 Bulk Pair Discovery")
    st.markdown("Analyze multiple pairs simultaneously to find cointegration opportunities")
    
    # Configuration panel
    with st.expander("🔧 Analysis Configuration", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            universe = st.selectbox("Stock Universe", 
                                   ["IBOV", "IBRX100", "ALL"], 
                                   help="Select stock universe to analyze")
            
            min_correlation = st.slider("Minimum Correlation", 0.0, 1.0, 0.7, 0.05,
                                      help="Minimum correlation between pairs")
        
        with col2:
            min_half_life = st.number_input("Min Half-Life (days)", 1, 100, 5,
                                          help="Minimum mean reversion speed")
            
            max_half_life = st.number_input("Max Half-Life (days)", 1, 100, 30,
                                          help="Maximum mean reversion speed")
        
        with col3:
            p_value_threshold = st.number_input("P-Value Threshold", 0.001, 0.1, 0.05, 0.001,
                                              help="Maximum p-value for cointegration")
            
            same_sector_only = st.checkbox("Same Sector Only", False,
                                         help="Only analyze pairs from same sector")
    
    # Analysis button
    if st.button("🔍 Find Cointegrated Pairs", type="primary"):
        run_bulk_pair_analysis(universe, min_correlation, min_half_life, 
                             max_half_life, p_value_threshold, same_sector_only)

def run_bulk_pair_analysis(universe, min_correlation, min_half_life, 
                          max_half_life, p_value_threshold, same_sector_only):
    """Run bulk pair analysis with progress tracking."""
    
    try:
        # Get symbols
        symbols = get_universe_tickers(universe)
        total_combinations = len(symbols) * (len(symbols) - 1) // 2
        
        st.info(f"📊 Analyzing {universe} universe: {len(symbols)} symbols = {total_combinations:,} pair combinations")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(progress_pct, message):
            progress_bar.progress(progress_pct)
            status_text.text(message)
        
        # Create pair selector
        selector = PairSelector()
        
        # Find pairs with progress tracking
        status_text.text("🔍 Starting cointegration analysis...")
        pairs = selector.find_cointegrated_pairs(
            symbols, 
            progress_callback=update_progress
        )
        
        # Final progress update
        progress_bar.progress(1.0)
        status_text.text(f"✅ Analysis complete! Found {len(pairs)} cointegrated pairs")
        
        if pairs:
            # Filter pairs
            filtered_pairs = selector.filter_by_criteria(
                pairs, 
                min_correlation=min_correlation,
                min_half_life=min_half_life,
                max_half_life=max_half_life,
                p_value_threshold=p_value_threshold
            )
            
            st.success(f"🎯 {len(filtered_pairs)} pairs meet your criteria!")
            
            # Display results
            display_bulk_analysis_results(filtered_pairs)
        else:
            st.error("❌ No cointegrated pairs found")
            
    except Exception as e:
        st.error(f"Error in bulk analysis: {e}")
    finally:
        # Clean up progress indicators
        try:
            progress_bar.empty()
            status_text.empty()
        except:
            pass

def display_bulk_analysis_results(pairs):
    """Display results from bulk pair analysis."""
    
    if not pairs:
        st.warning("No pairs found matching the criteria")
        return
    
    st.subheader(f"🎯 Found {len(pairs)} Cointegrated Pairs")
    
    # Convert to DataFrame
    pairs_df = pd.DataFrame(pairs)
    
    # Sort by p-value (best first)
    pairs_df = pairs_df.sort_values('coint_pvalue')
    
    # Display top pairs
    st.subheader("🔝 Top 10 Pairs")
    
    for i, pair in pairs_df.head(10).iterrows():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(f"#{i+1}: {pair['symbol1']}-{pair['symbol2']}", 
                     "✅ Cointegrated")
        
        with col2:
            st.metric("P-Value", f"{pair['coint_pvalue']:.4f}")
        
        with col3:
            st.metric("Half-Life", f"{pair['half_life']:.1f} days")
        
        with col4:
            st.metric("Correlation", f"{pair['correlation']:.3f}")
    
    # Full results table
    st.subheader("📋 All Results")
    
    # Prepare display columns
    display_df = pairs_df[['symbol1', 'symbol2', 'coint_pvalue', 'half_life', 
                          'correlation', 'hedge_ratio']].copy()
    
    display_df.columns = ['Symbol 1', 'Symbol 2', 'P-Value', 'Half-Life (days)', 
                         'Correlation', 'Hedge Ratio']
    
    st.dataframe(display_df, use_container_width=True)
    
    # Download option
    download_dataframe_as_csv(pairs_df, "cointegrated_pairs.csv")

def render_existing_pairs_analysis():
    """Render analysis of existing pairs in database."""
    
    st.header("🔍 Existing Pairs Analysis")
    st.markdown("View and analyze pairs already identified in the system")
    
    try:
        selector = PairSelector()
        historical_pairs = selector.get_historical_pair_results()
        
        if historical_pairs.empty:
            st.info("📝 No historical pairs found. Run pair discovery first.")
            return
        
        # Filter cointegrated pairs
        cointegrated = historical_pairs[historical_pairs['is_cointegrated'] == True]
        
        if cointegrated.empty:
            st.warning("⚠️ No cointegrated pairs in database")
            return
        
        st.subheader(f"📊 {len(cointegrated)} Cointegrated Pairs in Database")
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Pairs", len(cointegrated))
        
        with col2:
            avg_pvalue = cointegrated['p_value'].mean()
            st.metric("Avg P-Value", f"{avg_pvalue:.4f}")
        
        with col3:
            avg_half_life = cointegrated['half_life'].mean()
            st.metric("Avg Half-Life", f"{avg_half_life:.1f} days")
        
        with col4:
            avg_correlation = cointegrated['correlation'].mean()
            st.metric("Avg Correlation", f"{avg_correlation:.3f}")
        
        # Pairs list
        display_pairs = cointegrated.sort_values('p_value').head(20)
        
        st.subheader("🔝 Top 20 Pairs (by P-Value)")
        
        display_df = display_pairs[['symbol1', 'symbol2', 'p_value', 'half_life', 
                                  'correlation', 'hedge_ratio', 'test_date']].copy()
        
        display_df.columns = ['Symbol 1', 'Symbol 2', 'P-Value', 'Half-Life', 
                             'Correlation', 'Hedge Ratio', 'Test Date']
        
        st.dataframe(display_df, use_container_width=True)
        
        # Download option
        download_dataframe_as_csv(cointegrated, "historical_pairs.csv")
        
    except Exception as e:
        st.error(f"Error loading existing pairs: {e}")