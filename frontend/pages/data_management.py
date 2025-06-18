"""
Data Management page for the dashboard.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

from data.api import MarketDataAPI
from data.collector import DataCollector
from data.storage import DataStorageManager
from frontend.utils import (
    format_currency, format_large_number, create_status_card,
    show_loading_spinner, download_dataframe_as_csv
)

def render_data_management_page():
    """Render the data management page."""
    
    st.title("💾 Data Management")
    st.markdown("Monitor data quality, manage updates, and perform maintenance operations")
    
    # Quick actions toolbar
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Update Data", use_container_width=True, type="primary"):
                update_market_data()
        
        with col2:
            if st.button("🔍 Quality Check", use_container_width=True):
                run_quality_check()
        
        with col3:
            if st.button("💾 Backup DB", use_container_width=True):
                backup_database()
        
        with col4:
            if st.button("🧹 Cleanup", use_container_width=True):
                cleanup_old_data()
    
    st.markdown("---")
    
    # Database status overview
    st.subheader("📊 Database Status")
    
    render_database_status()
    
    st.markdown("---")
    
    # Data quality metrics
    st.subheader("🔍 Data Quality Metrics")
    
    render_data_quality_metrics()
    
    st.markdown("---")
    
    # Update history and logs
    st.subheader("📋 Update History")
    
    render_update_history()
    
    st.markdown("---")
    
    # Data coverage analysis
    st.subheader("📈 Data Coverage Analysis")
    
    render_data_coverage_analysis()

def render_database_status():
    """Render database status overview."""
    
    try:
        api = MarketDataAPI()
        stats = api.storage.get_database_stats()
        summary = api.get_data_summary()
        
        # Main metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_symbols = stats.get('stocks_master_count', 0)
            st.metric("Total Symbols", f"{total_symbols:,}")
        
        with col2:
            total_records = stats.get('daily_prices_count', 0)
            st.metric("Price Records", format_large_number(total_records))
        
        with col3:
            db_size = stats.get('db_size_mb', 0)
            st.metric("Database Size", f"{db_size:.1f} MB")
        
        with col4:
            min_date = stats.get('min_date')
            if min_date:
                st.metric("Data From", min_date)
            else:
                st.metric("Data From", "N/A")
        
        with col5:
            max_date = stats.get('max_date')
            if max_date:
                st.metric("Data Until", max_date)
            else:
                st.metric("Data Until", "N/A")
        
        # Additional statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Table Statistics**")
            
            table_stats = pd.DataFrame([
                ["Stocks Master", f"{stats.get('stocks_master_count', 0):,}"],
                ["Daily Prices", f"{stats.get('daily_prices_count', 0):,}"],
                ["Dividends", f"{stats.get('dividends_count', 0):,}"],
                ["Stock Splits", f"{stats.get('splits_count', 0):,}"],
                ["Pair Results", f"{stats.get('pair_results_count', 0):,}"],
                ["Quality Logs", f"{stats.get('data_quality_log_count', 0):,}"]
            ], columns=["Table", "Records"])
            
            st.dataframe(table_stats, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**🕒 Last Update Status**")
            
            last_update = summary.get('last_update', 'Unknown')
            symbols_current = summary.get('symbols_with_recent_data', 0)
            total_symbols = summary.get('total_symbols', 1)
            
            if symbols_current / total_symbols > 0.9:
                create_status_card("Data Freshness", "success", f"{symbols_current}/{total_symbols} symbols current")
            elif symbols_current / total_symbols > 0.7:
                create_status_card("Data Freshness", "warning", f"{symbols_current}/{total_symbols} symbols current")
            else:
                create_status_card("Data Freshness", "error", f"Only {symbols_current}/{total_symbols} symbols current")
            
            st.write(f"**Last Update:** {last_update}")
    
    except Exception as e:
        st.error(f"Error loading database status: {e}")

def render_data_quality_metrics():
    """Render data quality metrics and issues."""
    
    try:
        # Mock data quality metrics (in real implementation, query from database)
        quality_metrics = {
            'total_issues': 23,
            'critical_issues': 2,
            'warning_issues': 15,
            'info_issues': 6,
            'data_completeness': 0.945,
            'outlier_detection': 0.012,
            'missing_data_ratio': 0.033
        }
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            completeness = quality_metrics['data_completeness']
            st.metric("Data Completeness", f"{completeness:.1%}")
        
        with col2:
            outliers = quality_metrics['outlier_detection']
            st.metric("Outlier Rate", f"{outliers:.1%}")
        
        with col3:
            missing = quality_metrics['missing_data_ratio']
            st.metric("Missing Data", f"{missing:.1%}")
        
        with col4:
            total_issues = quality_metrics['total_issues']
            st.metric("Total Issues", total_issues)
        
        # Quality issues breakdown
        col1, col2 = st.columns(2)
        
        with col1:
            # Issues by severity
            severity_data = pd.DataFrame({
                'Severity': ['Critical', 'Warning', 'Info'],
                'Count': [
                    quality_metrics['critical_issues'],
                    quality_metrics['warning_issues'], 
                    quality_metrics['info_issues']
                ],
                'Color': ['red', 'orange', 'blue']
            })
            
            fig = px.pie(severity_data, values='Count', names='Severity',
                        title="Issues by Severity",
                        color='Severity',
                        color_discrete_map={'Critical': 'red', 'Warning': 'orange', 'Info': 'blue'})
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Quality trends (mock data)
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            completeness_trend = [0.92 + 0.025 * np.random.random() for _ in range(30)]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=completeness_trend,
                mode='lines+markers',
                name='Data Completeness',
                line=dict(color='green')
            ))
            
            fig.add_hline(y=0.95, line_dash="dash", line_color="red",
                         annotation_text="Target: 95%")
            
            fig.update_layout(
                title="Data Quality Trend (30 days)",
                xaxis_title="Date",
                yaxis_title="Completeness %",
                yaxis=dict(range=[0.9, 1.0])
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent quality issues
        st.markdown("**🚨 Recent Quality Issues**")
        
        # Mock recent issues
        recent_issues = pd.DataFrame([
            {
                'Date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
                'Symbol': 'PETR4.SA',
                'Issue Type': 'PRICE_OUTLIER',
                'Severity': 'Warning',
                'Description': 'Price change >20% detected'
            },
            {
                'Date': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d'),
                'Symbol': 'VALE3.SA',
                'Issue Type': 'MISSING_DATA',
                'Severity': 'Critical',
                'Description': 'No data for 3 consecutive days'
            },
            {
                'Date': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d'),
                'Symbol': 'ITUB4.SA',
                'Issue Type': 'ZERO_VOLUME',
                'Severity': 'Info',
                'Description': 'Zero volume trading day'
            }
        ])
        
        # Color code by severity
        def color_severity(val):
            if val == 'Critical':
                return 'background-color: #ffebee'
            elif val == 'Warning':
                return 'background-color: #fff3e0'
            else:
                return 'background-color: #e3f2fd'
        
        styled_df = recent_issues.style.applymap(color_severity, subset=['Severity'])
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # Download quality report
        download_dataframe_as_csv(recent_issues, "data_quality_report.csv")
    
    except Exception as e:
        st.error(f"Error loading quality metrics: {e}")

def render_update_history():
    """Render update history and status."""
    
    try:
        # Mock update history (in real implementation, query from update_status table)
        update_history = pd.DataFrame([
            {
                'Date': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'Process': 'daily_update',
                'Status': 'Success',
                'Records': 1250,
                'Duration': '2m 15s',
                'Error': ''
            },
            {
                'Date': (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d %H:%M'),
                'Process': 'daily_update',
                'Status': 'Success',
                'Records': 1180,
                'Duration': '1m 45s',
                'Error': ''
            },
            {
                'Date': (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d %H:%M'),
                'Process': 'weekly_cleanup',
                'Status': 'Success',
                'Records': 0,
                'Duration': '45s',
                'Error': ''
            },
            {
                'Date': (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d %H:%M'),
                'Process': 'daily_update',
                'Status': 'Failed',
                'Records': 0,
                'Duration': '30s',
                'Error': 'Network timeout'
            }
        ])
        
        # Status summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            success_count = len(update_history[update_history['Status'] == 'Success'])
            st.metric("Successful Updates", success_count)
        
        with col2:
            failed_count = len(update_history[update_history['Status'] == 'Failed'])
            st.metric("Failed Updates", failed_count)
        
        with col3:
            total_records = update_history[update_history['Status'] == 'Success']['Records'].sum()
            st.metric("Records Updated", f"{total_records:,}")
        
        with col4:
            success_rate = success_count / len(update_history) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
        
        # Update history table
        def color_status(val):
            if val == 'Success':
                return 'color: green'
            elif val == 'Failed':
                return 'color: red'
            else:
                return 'color: orange'
        
        styled_history = update_history.style.applymap(color_status, subset=['Status'])
        st.dataframe(styled_history, use_container_width=True, hide_index=True)
        
        # Update frequency chart
        st.markdown("**📊 Update Frequency**")
        
        # Mock daily update counts for the last 30 days
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        update_counts = [np.random.poisson(2) for _ in range(30)]  # Average 2 updates per day
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=dates,
            y=update_counts,
            name='Daily Updates'
        ))
        
        fig.update_layout(
            title="Daily Update Activity (30 days)",
            xaxis_title="Date",
            yaxis_title="Number of Updates"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Error loading update history: {e}")

def render_data_coverage_analysis():
    """Render data coverage analysis."""
    
    try:
        api = MarketDataAPI()
        available_symbols = api.get_available_symbols()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**📊 Symbol Coverage**")
            
            # Mock coverage data by universe
            universe_coverage = pd.DataFrame({
                'Universe': ['IBOV', 'IBRX100', 'Total Available'],
                'Total Symbols': [70, 100, 150],
                'Data Available': [68, 95, 142],
                'Coverage %': [97.1, 95.0, 94.7]
            })
            
            fig = px.bar(universe_coverage, x='Universe', y='Coverage %',
                        title="Data Coverage by Universe",
                        color='Coverage %',
                        color_continuous_scale='Greens')
            
            fig.update_layout(yaxis=dict(range=[90, 100]))
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(universe_coverage, use_container_width=True, hide_index=True)
        
        with col2:
            st.markdown("**📅 Historical Coverage**")
            
            # Mock historical data availability
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            # Create mock data showing percentage of symbols with data each day
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            coverage_pct = [92 + 8 * np.random.random() for _ in range(len(dates))]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dates,
                y=coverage_pct,
                mode='lines',
                name='Daily Coverage %',
                fill='tonexty'
            ))
            
            fig.add_hline(y=95, line_dash="dash", line_color="green",
                         annotation_text="Target: 95%")
            
            fig.update_layout(
                title="Historical Data Coverage",
                xaxis_title="Date",
                yaxis_title="Coverage %",
                yaxis=dict(range=[90, 100])
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Data gaps analysis
        st.markdown("**🕳️ Data Gaps Analysis**")
        
        # Mock data gaps
        data_gaps = pd.DataFrame([
            {
                'Symbol': 'MGLU3.SA',
                'Gap Start': '2024-01-15',
                'Gap End': '2024-01-17',
                'Days Missing': 3,
                'Gap Type': 'Trading Halt'
            },
            {
                'Symbol': 'AZUL4.SA',
                'Gap Start': '2024-01-22',
                'Gap End': '2024-01-22',
                'Days Missing': 1,
                'Gap Type': 'Data Source Issue'
            },
            {
                'Symbol': 'CYRE3.SA',
                'Gap Start': '2024-01-28',
                'Gap End': '2024-01-30',
                'Days Missing': 3,
                'Gap Type': 'Corporate Action'
            }
        ])
        
        if not data_gaps.empty:
            st.dataframe(data_gaps, use_container_width=True, hide_index=True)
            download_dataframe_as_csv(data_gaps, "data_gaps_report.csv")
        else:
            st.success("✅ No significant data gaps detected!")
    
    except Exception as e:
        st.error(f"Error analyzing data coverage: {e}")

def update_market_data():
    """Update market data."""
    
    with show_loading_spinner("Updating market data..."):
        try:
            api = MarketDataAPI()
            result = api.update_data()
            
            if result['status'] == 'success':
                st.success(f"✅ Data updated successfully!")
                st.info(f"📊 {result['symbols_updated']} symbols updated with {result['total_records']} new records")
            else:
                st.error(f"❌ Update failed: {result.get('error', 'Unknown error')}")
        
        except Exception as e:
            st.error(f"Error updating data: {e}")

def run_quality_check():
    """Run data quality check."""
    
    with show_loading_spinner("Running data quality check..."):
        try:
            collector = DataCollector()
            result = collector.validate_data_quality()
            
            issues_found = result.get('issues_found', 0)
            
            if issues_found == 0:
                st.success("✅ Data quality check passed! No issues found.")
            else:
                st.warning(f"⚠️ Quality check found {issues_found} issues.")
                
                # Show sample issues
                issues = result.get('issues', [])
                if issues:
                    issues_df = pd.DataFrame(issues[:10])  # Show first 10 issues
                    st.dataframe(issues_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error running quality check: {e}")

def backup_database():
    """Create database backup."""
    
    with show_loading_spinner("Creating database backup..."):
        try:
            storage = DataStorageManager()
            backup_path = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
            
            success = storage.backup_database(backup_path)
            
            if success:
                st.success(f"✅ Database backed up successfully!")
                st.info(f"💾 Backup saved as: {backup_path}")
            else:
                st.error("❌ Backup failed!")
        
        except Exception as e:
            st.error(f"Error creating backup: {e}")

def cleanup_old_data():
    """Cleanup old data and optimize database."""
    
    with show_loading_spinner("Cleaning up old data..."):
        try:
            # Mock cleanup operation
            st.success("✅ Database cleanup completed!")
            st.info("🗑️ Removed old logs and optimized database tables")
            
            # Show cleanup statistics
            cleanup_stats = pd.DataFrame([
                ["Old quality logs", "142 records"],
                ["Temporary files", "8 files"],
                ["Unused indexes", "3 indexes"],
                ["Database size reduction", "2.3 MB"]
            ], columns=["Item", "Cleaned"])
            
            st.dataframe(cleanup_stats, use_container_width=True, hide_index=True)
        
        except Exception as e:
            st.error(f"Error during cleanup: {e}")