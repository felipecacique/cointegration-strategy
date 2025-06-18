"""
Live Trading page for the dashboard.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

from strategy.signals import TradingSignalGenerator, SignalType
from frontend.utils import (
    create_signal_indicator, create_z_score_chart, format_currency,
    format_percentage, show_loading_spinner, download_dataframe_as_csv
)

def render_live_trading_page():
    """Render the live trading page."""
    
    st.title("📊 Live Trading")
    st.markdown("Real-time monitoring and signal generation for active pairs")
    
    # Control panel
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            auto_refresh = st.checkbox("🔄 Auto Refresh", value=False, 
                                     help="Automatically refresh signals every 5 minutes")
        
        with col2:
            refresh_interval = st.selectbox("Refresh Interval", 
                                          [1, 5, 15, 30], index=1,
                                          help="Minutes between auto refresh")
        
        with col3:
            signal_filter = st.selectbox("Signal Filter", 
                                       ["All", "Entry Signals", "Exit Signals", "Stop Loss"])
        
        with col4:
            if st.button("🔄 Refresh Now", help="Manually refresh all data"):
                with show_loading_spinner("Refreshing signals..."):
                    generate_live_signals()
                st.success("Signals refreshed!")
    
    st.markdown("---")
    
    # Generate signals if not available
    if not st.session_state.live_signals:
        if st.session_state.current_pairs:
            with show_loading_spinner("Generating live signals..."):
                generate_live_signals()
        else:
            st.info("🔍 No pairs available for signal generation. Please find pairs first.")
            if st.button("Find Pairs"):
                st.switch_page("pages/pair_analysis.py")
            return
    
    # Active positions summary
    st.subheader("💼 Active Positions")
    
    # Simulated active positions (in a real system, this would come from a position manager)
    active_positions = get_mock_active_positions()
    
    if active_positions:
        positions_df = pd.DataFrame(active_positions)
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Pairs", len(positions_df))
        
        with col2:
            total_pnl = positions_df['unrealized_pnl'].sum()
            st.metric("Total P&L", format_currency(total_pnl), 
                     delta=total_pnl, delta_color="normal")
        
        with col3:
            total_capital = positions_df['capital_allocated'].sum()
            st.metric("Capital Deployed", format_currency(total_capital))
        
        with col4:
            if total_capital > 0:
                total_return = total_pnl / total_capital
                st.metric("Return %", format_percentage(total_return))
            else:
                st.metric("Return %", "0.00%")
        
        # Positions table
        st.subheader("📋 Position Details")
        
        # Format the positions dataframe for display
        display_df = positions_df.copy()
        display_df['Entry Date'] = pd.to_datetime(display_df['entry_date']).dt.strftime('%Y-%m-%d')
        display_df['Capital'] = display_df['capital_allocated'].apply(format_currency)
        display_df['P&L'] = display_df['unrealized_pnl'].apply(format_currency)
        display_df['Return %'] = (display_df['unrealized_pnl'] / display_df['capital_allocated'] * 100).apply(lambda x: f"{x:.2f}%")
        display_df['Z-Score'] = display_df['current_z_score'].apply(lambda x: f"{x:.2f}")
        
        columns_to_show = ['pair_id', 'side', 'Entry Date', 'Capital', 'P&L', 'Return %', 'Z-Score']
        st.dataframe(display_df[columns_to_show], use_container_width=True)
        
        # Download positions
        download_dataframe_as_csv(display_df, "active_positions.csv")
    
    else:
        st.info("📊 No active positions currently.")
    
    st.markdown("---")
    
    # Live signals
    st.subheader("🚨 Live Signals")
    
    if st.session_state.live_signals:
        signals_df = pd.DataFrame(st.session_state.live_signals)
        
        # Filter signals based on selection
        if signal_filter != "All":
            if signal_filter == "Entry Signals":
                signals_df = signals_df[signals_df['signal'].isin(['ENTRY_LONG', 'ENTRY_SHORT'])]
            elif signal_filter == "Exit Signals":
                signals_df = signals_df[signals_df['signal'].isin(['EXIT_LONG', 'EXIT_SHORT'])]
            elif signal_filter == "Stop Loss":
                signals_df = signals_df[signals_df['signal'] == 'STOP_LOSS']
        
        if not signals_df.empty:
            # Sort by signal strength (absolute z-score)
            signals_df['abs_z_score'] = signals_df['z_score'].abs()
            signals_df = signals_df.sort_values('abs_z_score', ascending=False)
            
            # Display signals in cards
            for _, signal in signals_df.head(10).iterrows():  # Show top 10 signals
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{signal['pair_id']}**")
                        
                        # Signal indicator
                        signal_type = signal['signal']
                        z_score = signal['z_score']
                        
                        if hasattr(signal_type, 'value'):
                            signal_str = signal_type.value
                        else:
                            signal_str = str(signal_type)
                        
                        create_signal_indicator(signal_str, z_score)
                    
                    with col2:
                        st.metric("Current Prices", "")
                        st.write(f"**{signal['symbol1']}:** {signal['price1']:.2f}")
                        st.write(f"**{signal['symbol2']}:** {signal['price2']:.2f}")
                    
                    with col3:
                        st.metric("Spread Info", "")
                        st.write(f"**Spread:** {signal['spread']:.4f}")
                        st.write(f"**Hedge Ratio:** {signal['hedge_ratio']:.3f}")
                        
                        # Action button
                        if signal_str in ['ENTRY_LONG', 'ENTRY_SHORT']:
                            if st.button(f"📈 Execute {signal_str}", key=f"execute_{signal['pair_id']}"):
                                st.success(f"Signal logged for {signal['pair_id']}")
                
                st.divider()
        
        else:
            st.info(f"No {signal_filter.lower()} available at this time.")
    
    else:
        st.info("No signals generated yet.")
    
    st.markdown("---")
    
    # Z-Score monitoring
    st.subheader("📈 Z-Score Evolution")
    
    # Select pair for detailed monitoring
    if st.session_state.live_signals:
        available_pairs = [signal['pair_id'] for signal in st.session_state.live_signals]
        selected_pair = st.selectbox("Select Pair for Monitoring", available_pairs)
        
        if selected_pair:
            # Get historical z-score data for the selected pair
            pair_signal = next((s for s in st.session_state.live_signals if s['pair_id'] == selected_pair), None)
            
            if pair_signal:
                # Generate sample z-score history (in a real system, this would come from database)
                dates = pd.date_range(end=datetime.now(), periods=60, freq='D')
                z_scores = generate_sample_z_score_series(pair_signal['z_score'], len(dates))
                z_score_series = pd.Series(z_scores, index=dates)
                
                # Create z-score chart
                fig = create_z_score_chart(z_score_series, title=f"Z-Score Evolution - {selected_pair}")
                st.plotly_chart(fig, use_container_width=True)
                
                # Current status
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Z-Score", f"{pair_signal['z_score']:.2f}")
                
                with col2:
                    signal_strength = calculate_signal_strength(pair_signal['z_score'])
                    st.metric("Signal Strength", f"{signal_strength:.0f}%")
                
                with col3:
                    days_in_position = calculate_days_in_position(selected_pair)
                    st.metric("Days in Range", days_in_position)
    
    else:
        st.info("Generate signals to enable Z-score monitoring.")
    
    # Auto-refresh logic
    if auto_refresh:
        # In a real implementation, you would use st.rerun() with a timer
        st.info(f"🔄 Auto-refresh enabled (every {refresh_interval} minutes)")
        # Note: Streamlit doesn't support true auto-refresh without user interaction
        # In production, you might use st.empty() with a loop or external triggers

def generate_live_signals():
    """Generate live trading signals."""
    try:
        if not st.session_state.current_pairs:
            return
        
        signal_gen = TradingSignalGenerator()
        signals = signal_gen.get_current_signals(st.session_state.current_pairs, lookback_days=252)
        
        # Filter for actionable signals
        actionable_signals = signal_gen.filter_actionable_signals(signals)
        
        st.session_state.live_signals = signals
        
    except Exception as e:
        st.error(f"Error generating signals: {e}")

def get_mock_active_positions():
    """Get mock active positions for demonstration."""
    # In a real system, this would query the position manager
    if not st.session_state.live_signals:
        return []
    
    # Create some mock positions based on signals
    positions = []
    
    for i, signal in enumerate(st.session_state.live_signals[:3]):  # Take first 3 as "active"
        if hasattr(signal['signal'], 'value'):
            signal_str = signal['signal'].value
        else:
            signal_str = str(signal['signal'])
        
        if signal_str in ['ENTRY_LONG', 'ENTRY_SHORT']:
            side = 'LONG' if signal_str == 'ENTRY_LONG' else 'SHORT'
        else:
            side = 'LONG'  # Default for demonstration
        
        # Mock position data
        capital = 10000.0
        entry_days_ago = np.random.randint(1, 30)
        entry_date = datetime.now() - timedelta(days=entry_days_ago)
        
        # Mock P&L calculation
        z_score = signal['z_score']
        if side == 'LONG':
            # Long profits when z-score moves toward zero from negative
            pnl = capital * (-z_score * 0.1)
        else:
            # Short profits when z-score moves toward zero from positive
            pnl = capital * (z_score * 0.1)
        
        position = {
            'pair_id': signal['pair_id'],
            'side': side,
            'entry_date': entry_date,
            'capital_allocated': capital,
            'unrealized_pnl': pnl,
            'current_z_score': z_score
        }
        
        positions.append(position)
    
    return positions

def generate_sample_z_score_series(current_z_score, length):
    """Generate sample z-score time series ending at current value."""
    # Create a mean-reverting series
    z_scores = [current_z_score]
    
    for i in range(length - 1):
        # Mean reversion with some randomness
        prev_z = z_scores[-1]
        mean_reversion = -0.05 * prev_z  # 5% reversion to mean
        random_shock = np.random.normal(0, 0.2)
        next_z = prev_z + mean_reversion + random_shock
        z_scores.append(next_z)
    
    return list(reversed(z_scores))

def calculate_signal_strength(z_score):
    """Calculate signal strength as a percentage."""
    abs_z = abs(z_score)
    entry_threshold = 2.0
    stop_loss_threshold = 3.0
    
    if abs_z < entry_threshold:
        return 0
    elif abs_z >= stop_loss_threshold:
        return 100
    else:
        return (abs_z - entry_threshold) / (stop_loss_threshold - entry_threshold) * 100

def calculate_days_in_position(pair_id):
    """Calculate mock days in current position range."""
    # In a real system, this would check actual position history
    return np.random.randint(1, 15)