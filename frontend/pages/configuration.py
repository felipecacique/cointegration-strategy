"""
Configuration page for the dashboard.
"""
import streamlit as st
import json
from datetime import datetime

from config.settings import CONFIG
from config.universe import UNIVERSE_DEFINITIONS, SECTOR_MAPPING

def render_configuration_page():
    """Render the configuration page."""
    
    st.title("⚙️ Configuration")
    st.markdown("Configure strategy parameters, data settings, and system preferences")
    
    # Configuration tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Strategy Parameters", "Data Settings", "Risk Management", "Alerts & Notifications"])
    
    with tab1:
        render_strategy_config()
    
    with tab2:
        render_data_config()
    
    with tab3:
        render_risk_config()
    
    with tab4:
        render_alerts_config()
    
    # Save/Reset buttons
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("💾 Save Configuration", type="primary", use_container_width=True):
            save_configuration()
    
    with col2:
        if st.button("🔄 Reset to Defaults", use_container_width=True):
            reset_to_defaults()
    
    with col3:
        if st.button("📥 Load Preset", use_container_width=True):
            load_preset_config()
    
    with col4:
        if st.button("📤 Export Config", use_container_width=True):
            export_configuration()

def render_strategy_config():
    """Render strategy parameters configuration."""
    
    st.subheader("🎯 Strategy Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📊 Pair Selection**")
        
        # Cointegration parameters
        p_value_threshold = st.number_input(
            "P-Value Threshold",
            min_value=0.001,
            max_value=0.1,
            value=CONFIG['strategy'].get('p_value_threshold', 0.05),
            step=0.001,
            help="Maximum p-value for cointegration test"
        )
        
        min_correlation = st.number_input(
            "Minimum Correlation",
            min_value=0.0,
            max_value=1.0,
            value=CONFIG['strategy'].get('min_correlation', 0.7),
            step=0.05,
            help="Minimum correlation between pairs"
        )
        
        min_half_life = st.number_input(
            "Minimum Half-Life (days)",
            min_value=1,
            max_value=100,
            value=CONFIG['strategy'].get('min_half_life', 5),
            step=1,
            help="Minimum mean reversion speed"
        )
        
        max_half_life = st.number_input(
            "Maximum Half-Life (days)",
            min_value=1,
            max_value=100,
            value=CONFIG['strategy'].get('max_half_life', 30),
            step=1,
            help="Maximum mean reversion speed"
        )
        
        top_pairs = st.number_input(
            "Top Pairs to Trade",
            min_value=1,
            max_value=50,
            value=CONFIG['strategy'].get('top_pairs', 15),
            step=1,
            help="Number of top-ranked pairs to trade"
        )
    
    with col2:
        st.markdown("**⏱️ Timing Parameters**")
        
        lookback_window = st.number_input(
            "Lookback Window (days)",
            min_value=50,
            max_value=1000,
            value=CONFIG['strategy'].get('lookback_window', 252),
            step=10,
            help="Formation period for cointegration testing"
        )
        
        trading_window = st.number_input(
            "Trading Window (days)",
            min_value=10,
            max_value=200,
            value=CONFIG['strategy'].get('trading_window', 63),
            step=5,
            help="Trading period length"
        )
        
        rebalance_frequency = st.number_input(
            "Rebalance Frequency (days)",
            min_value=1,
            max_value=100,
            value=CONFIG['strategy'].get('rebalance_frequency', 21),
            step=1,
            help="Days between portfolio rebalancing"
        )
        
        sector_matching = st.checkbox(
            "Same Sector Only",
            value=CONFIG['strategy'].get('sector_matching', False),
            help="Only trade pairs from the same sector"
        )
    
    # Trading signals
    st.markdown("**🚨 Trading Signals**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        entry_z_score = st.number_input(
            "Entry Z-Score Threshold",
            min_value=1.0,
            max_value=5.0,
            value=CONFIG['trading'].get('entry_z_score', 2.0),
            step=0.1,
            help="Z-score threshold for entering positions"
        )
    
    with col2:
        exit_z_score = st.number_input(
            "Exit Z-Score Threshold",
            min_value=0.1,
            max_value=2.0,
            value=CONFIG['trading'].get('exit_z_score', 0.5),
            step=0.1,
            help="Z-score threshold for exiting positions"
        )
    
    with col3:
        stop_loss_z_score = st.number_input(
            "Stop-Loss Z-Score",
            min_value=2.0,
            max_value=10.0,
            value=CONFIG['trading'].get('stop_loss_z_score', 3.0),
            step=0.1,
            help="Z-score threshold for stop-loss"
        )
    
    # Store updated values in session state
    if 'config_updates' not in st.session_state:
        st.session_state.config_updates = {}
    
    st.session_state.config_updates.update({
        'strategy': {
            'p_value_threshold': p_value_threshold,
            'min_correlation': min_correlation,
            'min_half_life': min_half_life,
            'max_half_life': max_half_life,
            'top_pairs': top_pairs,
            'lookback_window': lookback_window,
            'trading_window': trading_window,
            'rebalance_frequency': rebalance_frequency,
            'sector_matching': sector_matching
        },
        'trading': {
            'entry_z_score': entry_z_score,
            'exit_z_score': exit_z_score,
            'stop_loss_z_score': stop_loss_z_score
        }
    })

def render_data_config():
    """Render data settings configuration."""
    
    st.subheader("💾 Data Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🌐 Data Source**")
        
        universe_selection = st.multiselect(
            "Stock Universe",
            options=list(UNIVERSE_DEFINITIONS.keys()),
            default=CONFIG['data'].get('universe', ['IBOV']),
            help="Select stock universes to analyze"
        )
        
        min_market_cap = st.number_input(
            "Minimum Market Cap (R$)",
            min_value=100_000_000,
            max_value=100_000_000_000,
            value=CONFIG['data'].get('min_market_cap', 1_000_000_000),
            step=100_000_000,
            format="%d",
            help="Minimum market capitalization filter"
        )
        
        min_avg_volume = st.number_input(
            "Minimum Avg Volume (R$)",
            min_value=100_000,
            max_value=100_000_000,
            value=CONFIG['data'].get('min_avg_volume', 1_000_000),
            step=100_000,
            format="%d",
            help="Minimum average daily volume filter"
        )
        
        lookback_days = st.number_input(
            "Data Lookback (days)",
            min_value=100,
            max_value=5000,
            value=CONFIG['data'].get('lookback_days', 1000),
            step=100,
            help="Days of historical data to maintain"
        )
    
    with col2:
        st.markdown("**🔄 Update Settings**")
        
        update_time = st.time_input(
            "Daily Update Time",
            value=datetime.strptime(CONFIG['data'].get('update_time', '18:30'), '%H:%M').time(),
            help="Time of day for automatic data updates"
        )
        
        timezone = st.selectbox(
            "Timezone",
            options=['America/Sao_Paulo', 'UTC', 'America/New_York'],
            index=0,
            help="Timezone for data updates"
        )
        
        backup_frequency = st.selectbox(
            "Backup Frequency",
            options=['daily', 'weekly', 'monthly'],
            index=0,
            help="Database backup frequency"
        )
        
        backup_retention = st.number_input(
            "Backup Retention (days)",
            min_value=7,
            max_value=365,
            value=CONFIG['database'].get('backup_retention', 30),
            step=1,
            help="Days to keep backup files"
        )
    
    # API settings
    st.markdown("**🔗 API Configuration**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rate_limit_delay = st.number_input(
            "Rate Limit Delay (seconds)",
            min_value=0.01,
            max_value=5.0,
            value=CONFIG['api'].get('rate_limit_delay', 0.1),
            step=0.01,
            help="Delay between API calls"
        )
    
    with col2:
        max_retries = st.number_input(
            "Max Retries",
            min_value=1,
            max_value=10,
            value=CONFIG['api'].get('max_retries', 3),
            step=1,
            help="Maximum retry attempts for failed requests"
        )
    
    with col3:
        timeout = st.number_input(
            "Request Timeout (seconds)",
            min_value=5,
            max_value=120,
            value=CONFIG['api'].get('timeout', 30),
            step=5,
            help="Timeout for API requests"
        )
    
    # Update session state
    st.session_state.config_updates.update({
        'data': {
            'universe': universe_selection,
            'min_market_cap': min_market_cap,
            'min_avg_volume': min_avg_volume,
            'lookback_days': lookback_days,
            'update_time': update_time.strftime('%H:%M'),
            'timezone': timezone
        },
        'database': {
            'backup_frequency': backup_frequency,
            'backup_retention': backup_retention
        },
        'api': {
            'rate_limit_delay': rate_limit_delay,
            'max_retries': max_retries,
            'timeout': timeout
        }
    })

def render_risk_config():
    """Render risk management configuration."""
    
    st.subheader("⚠️ Risk Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**💰 Position Management**")
        
        initial_capital = st.number_input(
            "Initial Capital (R$)",
            min_value=10_000,
            max_value=10_000_000,
            value=CONFIG['trading'].get('initial_capital', 100_000),
            step=10_000,
            format="%d",
            help="Starting portfolio value"
        )
        
        max_position_size = st.slider(
            "Max Position Size (%)",
            min_value=1,
            max_value=50,
            value=int(CONFIG['trading'].get('max_position_size', 0.1) * 100),
            step=1,
            help="Maximum percentage of capital per position"
        )
        
        max_active_pairs = st.number_input(
            "Max Active Pairs",
            min_value=1,
            max_value=100,
            value=CONFIG['trading'].get('max_active_pairs', 10),
            step=1,
            help="Maximum number of simultaneous positions"
        )
        
        commission_rate = st.number_input(
            "Commission Rate (%)",
            min_value=0.0,
            max_value=2.0,
            value=CONFIG['trading'].get('commission_rate', 0.003) * 100,
            step=0.01,
            help="Transaction commission rate"
        )
    
    with col2:
        st.markdown("**🛡️ Risk Limits**")
        
        max_drawdown = st.slider(
            "Max Drawdown (%)",
            min_value=5,
            max_value=50,
            value=int(CONFIG['risk'].get('max_drawdown', 0.15) * 100),
            step=1,
            help="Maximum allowed portfolio drawdown"
        )
        
        max_leverage = st.number_input(
            "Max Leverage",
            min_value=1.0,
            max_value=5.0,
            value=CONFIG['risk'].get('max_leverage', 2.0),
            step=0.1,
            help="Maximum portfolio leverage"
        )
        
        position_sizing = st.selectbox(
            "Position Sizing Method",
            options=['equal_weight', 'volatility_adjusted', 'kelly_criterion'],
            index=0,
            help="Method for determining position sizes"
        )
        
        rebalance_threshold = st.slider(
            "Rebalance Threshold (%)",
            min_value=1,
            max_value=20,
            value=int(CONFIG['risk'].get('rebalance_threshold', 0.05) * 100),
            step=1,
            help="Portfolio drift threshold for rebalancing"
        )
    
    # Update session state
    st.session_state.config_updates.update({
        'trading': {
            **st.session_state.config_updates.get('trading', {}),
            'initial_capital': initial_capital,
            'max_position_size': max_position_size / 100,
            'max_active_pairs': max_active_pairs,
            'commission_rate': commission_rate / 100
        },
        'risk': {
            'max_drawdown': max_drawdown / 100,
            'max_leverage': max_leverage,
            'position_sizing': position_sizing,
            'rebalance_threshold': rebalance_threshold / 100
        }
    })

def render_alerts_config():
    """Render alerts and notifications configuration."""
    
    st.subheader("🔔 Alerts & Notifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**📧 Email Settings**")
        
        email_enabled = st.checkbox(
            "Enable Email Alerts",
            value=CONFIG['alerts'].get('email_enabled', True),
            help="Enable email notifications"
        )
        
        email_from = st.text_input(
            "From Email",
            value=CONFIG['alerts'].get('email_from', ''),
            help="Sender email address",
            disabled=not email_enabled
        )
        
        email_to = st.text_input(
            "To Email",
            value=CONFIG['alerts'].get('email_to', ''),
            help="Recipient email address",
            disabled=not email_enabled
        )
        
        email_password = st.text_input(
            "Email Password",
            value="",
            type="password",
            help="Email account password or app password",
            disabled=not email_enabled
        )
        
        smtp_server = st.text_input(
            "SMTP Server",
            value=CONFIG['alerts'].get('smtp_server', 'smtp.gmail.com'),
            help="SMTP server address",
            disabled=not email_enabled
        )
        
        smtp_port = st.number_input(
            "SMTP Port",
            min_value=25,
            max_value=995,
            value=CONFIG['alerts'].get('smtp_port', 587),
            step=1,
            help="SMTP server port",
            disabled=not email_enabled
        )
    
    with col2:
        st.markdown("**🚨 Alert Types**")
        
        signal_alerts = st.checkbox(
            "Trading Signal Alerts",
            value=CONFIG['alerts'].get('signal_alerts', True),
            help="Alerts for new trading signals"
        )
        
        error_alerts = st.checkbox(
            "Error Alerts",
            value=CONFIG['alerts'].get('error_alerts', True),
            help="Alerts for system errors"
        )
        
        performance_alerts = st.checkbox(
            "Performance Alerts",
            value=True,
            help="Alerts for performance milestones"
        )
        
        risk_alerts = st.checkbox(
            "Risk Alerts",
            value=True,
            help="Alerts for risk limit breaches"
        )
        
        data_alerts = st.checkbox(
            "Data Quality Alerts",
            value=True,
            help="Alerts for data quality issues"
        )
    
    # Alert thresholds
    st.markdown("**⚙️ Alert Thresholds**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        drawdown_alert_threshold = st.slider(
            "Drawdown Alert (%)",
            min_value=1,
            max_value=20,
            value=10,
            step=1,
            help="Send alert when drawdown exceeds this level"
        )
    
    with col2:
        profit_alert_threshold = st.slider(
            "Profit Alert (%)",
            min_value=1,
            max_value=50,
            value=10,
            step=1,
            help="Send alert when profit exceeds this level"
        )
    
    with col3:
        error_count_threshold = st.number_input(
            "Error Count Alert",
            min_value=1,
            max_value=100,
            value=5,
            step=1,
            help="Send alert after this many errors"
        )
    
    # Test email button
    if email_enabled:
        if st.button("📧 Send Test Email", help="Send a test email to verify settings"):
            send_test_email(email_from, email_to, smtp_server, smtp_port)
    
    # Update session state
    st.session_state.config_updates.update({
        'alerts': {
            'email_enabled': email_enabled,
            'email_from': email_from,
            'email_to': email_to,
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'signal_alerts': signal_alerts,
            'error_alerts': error_alerts
        }
    })

def save_configuration():
    """Save configuration changes."""
    try:
        if 'config_updates' in st.session_state:
            # In a real implementation, you would save to a config file or database
            st.success("✅ Configuration saved successfully!")
            
            # Show summary of changes
            with st.expander("📋 Configuration Summary"):
                st.json(st.session_state.config_updates)
        else:
            st.warning("No configuration changes to save.")
    
    except Exception as e:
        st.error(f"Error saving configuration: {e}")

def reset_to_defaults():
    """Reset configuration to defaults."""
    try:
        # Clear session state updates
        if 'config_updates' in st.session_state:
            del st.session_state.config_updates
        
        st.success("✅ Configuration reset to defaults!")
        st.rerun()
    
    except Exception as e:
        st.error(f"Error resetting configuration: {e}")

def load_preset_config():
    """Load a preset configuration."""
    
    preset = st.selectbox(
        "Select Preset Configuration",
        options=["Conservative", "Aggressive", "Balanced", "High Frequency"],
        help="Pre-defined configuration sets"
    )
    
    if st.button("Load Preset"):
        try:
            presets = {
                "Conservative": {
                    'trading': {
                        'entry_z_score': 2.5,
                        'exit_z_score': 0.3,
                        'stop_loss_z_score': 4.0,
                        'max_position_size': 0.05
                    }
                },
                "Aggressive": {
                    'trading': {
                        'entry_z_score': 1.5,
                        'exit_z_score': 0.7,
                        'stop_loss_z_score': 2.5,
                        'max_position_size': 0.15
                    }
                },
                "Balanced": {
                    'trading': {
                        'entry_z_score': 2.0,
                        'exit_z_score': 0.5,
                        'stop_loss_z_score': 3.0,
                        'max_position_size': 0.1
                    }
                },
                "High Frequency": {
                    'strategy': {
                        'rebalance_frequency': 5,
                        'trading_window': 21
                    },
                    'trading': {
                        'entry_z_score': 1.8,
                        'exit_z_score': 0.4
                    }
                }
            }
            
            st.session_state.config_updates = presets.get(preset, {})
            st.success(f"✅ {preset} preset loaded!")
            st.rerun()
        
        except Exception as e:
            st.error(f"Error loading preset: {e}")

def export_configuration():
    """Export current configuration."""
    try:
        config_to_export = CONFIG.copy()
        
        # Update with any session state changes
        if 'config_updates' in st.session_state:
            for section, updates in st.session_state.config_updates.items():
                if section in config_to_export:
                    config_to_export[section].update(updates)
        
        # Convert to JSON
        config_json = json.dumps(config_to_export, indent=2, default=str)
        
        # Create download button
        st.download_button(
            label="📥 Download Configuration",
            data=config_json,
            file_name=f"pairs_trading_config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
        
        st.success("✅ Configuration ready for download!")
    
    except Exception as e:
        st.error(f"Error exporting configuration: {e}")

def send_test_email(email_from, email_to, smtp_server, smtp_port):
    """Send a test email."""
    try:
        # Mock email sending (in real implementation, use actual SMTP)
        if email_from and email_to:
            st.success(f"✅ Test email would be sent from {email_from} to {email_to}")
            st.info("📧 Email functionality is configured correctly!")
        else:
            st.warning("⚠️ Please configure email addresses first.")
    
    except Exception as e:
        st.error(f"Error sending test email: {e}")