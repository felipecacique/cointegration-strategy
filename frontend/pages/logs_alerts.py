"""
Logs & Alerts page for the dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from frontend.utils import (
    create_status_card, format_percentage, show_loading_spinner,
    download_dataframe_as_csv
)

def render_logs_alerts_page():
    """Render the logs & alerts page."""
    
    st.title("📝 Logs & Alerts")
    st.markdown("Monitor system events, alerts, and operational history")
    
    # Alert status overview
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            active_alerts = get_active_alerts_count()
            st.metric("Active Alerts", active_alerts, 
                     delta=-2 if active_alerts < 5 else 1,
                     delta_color="inverse")
        
        with col2:
            critical_alerts = get_critical_alerts_count()
            st.metric("Critical Alerts", critical_alerts,
                     delta_color="inverse")
        
        with col3:
            resolved_today = get_resolved_alerts_today()
            st.metric("Resolved Today", resolved_today)
        
        with col4:
            system_health = get_system_health_score()
            st.metric("System Health", f"{system_health}%",
                     delta=2 if system_health > 95 else -5,
                     delta_color="normal")
    
    st.markdown("---")
    
    # Tabs for different log types
    tab1, tab2, tab3, tab4 = st.tabs(["🚨 Active Alerts", "📋 System Logs", "💼 Trading Events", "📊 Performance Logs"])
    
    with tab1:
        render_active_alerts()
    
    with tab2:
        render_system_logs()
    
    with tab3:
        render_trading_events()
    
    with tab4:
        render_performance_logs()

def render_active_alerts():
    """Render active alerts section."""
    
    st.subheader("🚨 Active Alerts")
    
    # Alert filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        severity_filter = st.selectbox("Filter by Severity", 
                                     ["All", "Critical", "Warning", "Info"])
    
    with col2:
        category_filter = st.selectbox("Filter by Category",
                                     ["All", "Trading", "Data", "System", "Risk"])
    
    with col3:
        time_filter = st.selectbox("Time Range",
                                 ["Last 24h", "Last 7d", "Last 30d", "All"])
    
    # Mock active alerts data
    alerts_data = get_mock_active_alerts()
    
    # Apply filters
    filtered_alerts = filter_alerts(alerts_data, severity_filter, category_filter, time_filter)
    
    if filtered_alerts:
        # Display alerts
        for alert in filtered_alerts:
            with st.container():
                severity = alert['severity']
                
                # Color coding based on severity
                if severity == 'Critical':
                    border_color = "red"
                    bg_color = "#ffebee"
                elif severity == 'Warning':
                    border_color = "orange"
                    bg_color = "#fff3e0"
                else:
                    border_color = "blue"
                    bg_color = "#e3f2fd"
                
                st.markdown(f"""
                <div style="
                    border-left: 4px solid {border_color};
                    background-color: {bg_color};
                    padding: 1rem;
                    margin: 0.5rem 0;
                    border-radius: 0.25rem;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <strong>{alert['title']}</strong>
                            <span style="
                                background-color: {border_color};
                                color: white;
                                padding: 0.2rem 0.5rem;
                                border-radius: 0.25rem;
                                font-size: 0.8rem;
                                margin-left: 1rem;
                            ">{severity}</span>
                        </div>
                        <small>{alert['timestamp']}</small>
                    </div>
                    <p style="margin: 0.5rem 0 0 0; color: #666;">
                        {alert['description']}
                    </p>
                    <small><strong>Category:</strong> {alert['category']}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Action buttons for each alert
                col1, col2, col3 = st.columns([1, 1, 6])
                
                with col1:
                    if st.button("✅ Resolve", key=f"resolve_{alert['id']}"):
                        resolve_alert(alert['id'])
                        st.success("Alert resolved!")
                        st.rerun()
                
                with col2:
                    if st.button("🔕 Snooze", key=f"snooze_{alert['id']}"):
                        snooze_alert(alert['id'])
                        st.info("Alert snoozed for 1 hour")
        
        # Bulk actions
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✅ Resolve All Warnings", type="secondary"):
                resolve_all_warnings()
                st.success("All warning alerts resolved!")
        
        with col2:
            if st.button("📧 Send Alert Summary", type="secondary"):
                send_alert_summary()
                st.info("Alert summary sent!")
        
        with col3:
            if st.button("📥 Export Alerts", type="secondary"):
                export_alerts(filtered_alerts)
    
    else:
        st.success("🎉 No active alerts! System is running smoothly.")

def render_system_logs():
    """Render system logs section."""
    
    st.subheader("📋 System Logs")
    
    # Log level filter
    col1, col2, col3 = st.columns(3)
    
    with col1:
        log_level = st.selectbox("Log Level", ["All", "ERROR", "WARNING", "INFO", "DEBUG"])
    
    with col2:
        log_source = st.selectbox("Source", ["All", "Data Collector", "Strategy Engine", "Risk Manager", "Dashboard"])
    
    with col3:
        lines_to_show = st.number_input("Lines to Show", min_value=10, max_value=1000, value=100)
    
    # Mock system logs
    system_logs = get_mock_system_logs(lines_to_show)
    
    # Filter logs
    filtered_logs = filter_logs(system_logs, log_level, log_source)
    
    # Display logs in a code block style
    st.markdown("**Recent System Logs:**")
    
    log_container = st.container()
    
    with log_container:
        for log in filtered_logs[:50]:  # Show max 50 lines
            level = log['level']
            
            # Color coding
            if level == 'ERROR':
                color = "red"
            elif level == 'WARNING':
                color = "orange"
            elif level == 'INFO':
                color = "blue"
            else:
                color = "gray"
            
            st.markdown(f"""
            <div style="
                font-family: monospace;
                font-size: 0.8rem;
                padding: 0.2rem 0.5rem;
                border-left: 3px solid {color};
                margin: 0.1rem 0;
                background-color: #f8f9fa;
            ">
                <span style="color: {color}; font-weight: bold;">[{log['timestamp']}]</span>
                <span style="color: {color};">{level}</span>
                <span style="color: #666;">{log['source']}</span>
                - {log['message']}
            </div>
            """, unsafe_allow_html=True)
    
    # Download logs
    if st.button("📥 Download Full Logs"):
        download_system_logs()

def render_trading_events():
    """Render trading events section."""
    
    st.subheader("💼 Trading Events")
    
    # Event type filter
    col1, col2 = st.columns(2)
    
    with col1:
        event_type = st.selectbox("Event Type", 
                                ["All", "Position Opened", "Position Closed", "Signal Generated", "Risk Alert"])
    
    with col2:
        date_range = st.selectbox("Date Range",
                                ["Today", "Last 7 days", "Last 30 days", "Custom"])
    
    if date_range == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", datetime.now() - timedelta(days=30))
        with col2:
            end_date = st.date_input("End Date", datetime.now())
    
    # Mock trading events
    trading_events = get_mock_trading_events()
    
    # Filter events
    filtered_events = filter_trading_events(trading_events, event_type, date_range)
    
    # Display events
    if filtered_events:
        events_df = pd.DataFrame(filtered_events)
        
        # Format the dataframe for display
        display_df = events_df.copy()
        display_df['Timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Color code by event type
        def color_event_type(val):
            color_map = {
                'Position Opened': 'background-color: #e8f5e8',
                'Position Closed': 'background-color: #ffe8e8',
                'Signal Generated': 'background-color: #e8f0ff',
                'Risk Alert': 'background-color: #fff3e0'
            }
            return color_map.get(val, '')
        
        styled_df = display_df.style.applymap(color_event_type, subset=['event_type'])
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Event statistics
        st.markdown("**📊 Event Statistics**")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            positions_opened = len(events_df[events_df['event_type'] == 'Position Opened'])
            st.metric("Positions Opened", positions_opened)
        
        with col2:
            positions_closed = len(events_df[events_df['event_type'] == 'Position Closed'])
            st.metric("Positions Closed", positions_closed)
        
        with col3:
            signals_generated = len(events_df[events_df['event_type'] == 'Signal Generated'])
            st.metric("Signals Generated", signals_generated)
        
        with col4:
            risk_alerts = len(events_df[events_df['event_type'] == 'Risk Alert'])
            st.metric("Risk Alerts", risk_alerts)
        
        # Event timeline
        st.markdown("**📈 Event Timeline**")
        
        # Group events by hour
        events_df['hour'] = pd.to_datetime(events_df['timestamp']).dt.floor('H')
        hourly_events = events_df.groupby(['hour', 'event_type']).size().reset_index(name='count')
        
        fig = px.bar(hourly_events, x='hour', y='count', color='event_type',
                    title="Trading Events Timeline",
                    labels={'hour': 'Time', 'count': 'Event Count'})
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Download events
        download_dataframe_as_csv(display_df, "trading_events.csv")
    
    else:
        st.info("No trading events found for the selected criteria.")

def render_performance_logs():
    """Render performance logs section."""
    
    st.subheader("📊 Performance Logs")
    
    # Performance metrics over time
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🚀 System Performance**")
        
        # Mock performance data
        dates = pd.date_range(end=datetime.now(), periods=24, freq='H')
        cpu_usage = [20 + 30 * np.random.random() for _ in range(24)]
        memory_usage = [40 + 20 * np.random.random() for _ in range(24)]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates, y=cpu_usage,
            mode='lines', name='CPU Usage %',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatter(
            x=dates, y=memory_usage,
            mode='lines', name='Memory Usage %',
            line=dict(color='red'), yaxis='y2'
        ))
        
        fig.update_layout(
            title="System Resource Usage (24h)",
            xaxis_title="Time",
            yaxis=dict(title="CPU %", side="left"),
            yaxis2=dict(title="Memory %", side="right", overlaying="y"),
            height=300
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("**⚡ Processing Times**")
        
        # Mock processing times
        processes = ['Data Update', 'Pair Finding', 'Signal Generation', 'Risk Check']
        avg_times = [45, 180, 12, 8]  # seconds
        
        fig = px.bar(x=processes, y=avg_times,
                    title="Average Processing Times",
                    labels={'x': 'Process', 'y': 'Time (seconds)'},
                    color=avg_times,
                    color_continuous_scale='Viridis')
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance summary table
    st.markdown("**📋 Performance Summary**")
    
    performance_summary = pd.DataFrame([
        ["System Uptime", "99.8%", "Excellent"],
        ["Data Update Success Rate", "98.5%", "Good"],
        ["Average Response Time", "1.2s", "Good"],
        ["Error Rate", "0.3%", "Excellent"],
        ["Memory Usage", "65%", "Normal"],
        ["CPU Usage", "35%", "Normal"]
    ], columns=["Metric", "Value", "Status"])
    
    # Color code status
    def color_status(val):
        if val == 'Excellent':
            return 'color: green'
        elif val == 'Good':
            return 'color: blue'
        elif val == 'Normal':
            return 'color: orange'
        else:
            return 'color: red'
    
    styled_performance = performance_summary.style.applymap(color_status, subset=['Status'])
    st.dataframe(styled_performance, use_container_width=True, hide_index=True)

# Helper functions for mock data

def get_active_alerts_count():
    """Get count of active alerts."""
    return np.random.randint(3, 8)

def get_critical_alerts_count():
    """Get count of critical alerts."""
    return np.random.randint(0, 3)

def get_resolved_alerts_today():
    """Get count of alerts resolved today."""
    return np.random.randint(5, 15)

def get_system_health_score():
    """Get system health score."""
    return np.random.randint(92, 99)

def get_mock_active_alerts():
    """Generate mock active alerts."""
    alerts = [
        {
            'id': 1,
            'title': 'High Drawdown Alert',
            'description': 'Portfolio drawdown exceeded -10% threshold',
            'severity': 'Critical',
            'category': 'Risk',
            'timestamp': (datetime.now() - timedelta(minutes=30)).strftime('%Y-%m-%d %H:%M:%S')
        },
        {
            'id': 2,
            'title': 'Data Quality Issue',
            'description': 'Missing price data for 3 symbols in last update',
            'severity': 'Warning',
            'category': 'Data',
            'timestamp': (datetime.now() - timedelta(hours=2)).strftime('%Y-%m-%d %H:%M:%S')
        },
        {
            'id': 3,
            'title': 'New Trading Signal',
            'description': 'Strong entry signal detected for PETR4-VALE3 pair',
            'severity': 'Info',
            'category': 'Trading',
            'timestamp': (datetime.now() - timedelta(minutes=15)).strftime('%Y-%m-%d %H:%M:%S')
        },
        {
            'id': 4,
            'title': 'Position Size Alert',
            'description': 'Position size approaching maximum allocation limit',
            'severity': 'Warning',
            'category': 'Risk',
            'timestamp': (datetime.now() - timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
        }
    ]
    return alerts

def get_mock_system_logs(lines=100):
    """Generate mock system logs."""
    logs = []
    levels = ['INFO', 'WARNING', 'ERROR', 'DEBUG']
    sources = ['DataCollector', 'StrategyEngine', 'RiskManager', 'Dashboard']
    
    for i in range(lines):
        timestamp = (datetime.now() - timedelta(minutes=i)).strftime('%Y-%m-%d %H:%M:%S')
        level = np.random.choice(levels, p=[0.6, 0.2, 0.1, 0.1])
        source = np.random.choice(sources)
        
        messages = {
            'INFO': f"Successfully processed {np.random.randint(10, 100)} records",
            'WARNING': f"Rate limit approaching for {source}",
            'ERROR': f"Failed to connect to data source: timeout",
            'DEBUG': f"Processing batch {np.random.randint(1, 10)} of market data"
        }
        
        logs.append({
            'timestamp': timestamp,
            'level': level,
            'source': source,
            'message': messages[level]
        })
    
    return logs

def get_mock_trading_events():
    """Generate mock trading events."""
    events = []
    event_types = ['Position Opened', 'Position Closed', 'Signal Generated', 'Risk Alert']
    
    for i in range(50):
        timestamp = (datetime.now() - timedelta(hours=i/2)).strftime('%Y-%m-%d %H:%M:%S')
        event_type = np.random.choice(event_types, p=[0.3, 0.3, 0.3, 0.1])
        
        pair_symbols = ['PETR4-VALE3', 'ITUB4-BBDC4', 'ABEV3-BRFS3', 'WEGE3-EGIE3']
        pair = np.random.choice(pair_symbols)
        
        details = {
            'Position Opened': f"Opened LONG position for {pair} (Z-score: -2.1)",
            'Position Closed': f"Closed position for {pair} (P&L: R$ {np.random.randint(-500, 1500)})",
            'Signal Generated': f"Entry signal for {pair} (Z-score: {np.random.uniform(-3, 3):.2f})",
            'Risk Alert': f"Stop-loss triggered for {pair}"
        }
        
        events.append({
            'timestamp': timestamp,
            'event_type': event_type,
            'pair': pair,
            'details': details[event_type]
        })
    
    return events

def filter_alerts(alerts, severity_filter, category_filter, time_filter):
    """Filter alerts based on criteria."""
    filtered = alerts.copy()
    
    if severity_filter != "All":
        filtered = [a for a in filtered if a['severity'] == severity_filter]
    
    if category_filter != "All":
        filtered = [a for a in filtered if a['category'] == category_filter]
    
    # Time filtering would be implemented here
    
    return filtered

def filter_logs(logs, log_level, log_source):
    """Filter logs based on criteria."""
    filtered = logs.copy()
    
    if log_level != "All":
        filtered = [l for l in filtered if l['level'] == log_level]
    
    if log_source != "All":
        filtered = [l for l in filtered if log_source.replace(" ", "") in l['source']]
    
    return filtered

def filter_trading_events(events, event_type, date_range):
    """Filter trading events based on criteria."""
    filtered = events.copy()
    
    if event_type != "All":
        filtered = [e for e in filtered if e['event_type'] == event_type]
    
    # Date range filtering would be implemented here
    
    return filtered

def resolve_alert(alert_id):
    """Resolve an alert."""
    pass  # Implementation would update alert status

def snooze_alert(alert_id):
    """Snooze an alert."""
    pass  # Implementation would snooze alert

def resolve_all_warnings():
    """Resolve all warning alerts."""
    pass  # Implementation would resolve warnings

def send_alert_summary():
    """Send alert summary email."""
    pass  # Implementation would send email

def export_alerts(alerts):
    """Export alerts to CSV."""
    if alerts:
        alerts_df = pd.DataFrame(alerts)
        download_dataframe_as_csv(alerts_df, "active_alerts.csv")

def download_system_logs():
    """Download system logs."""
    logs = get_mock_system_logs(1000)
    logs_df = pd.DataFrame(logs)
    download_dataframe_as_csv(logs_df, "system_logs.csv")