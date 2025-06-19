import streamlit as st
import pandas as pd
import time
from datetime import datetime
from typing import Dict, List, Any
import json

class LiveTradeTimeline:
    """Real-time trade timeline display during backtesting"""
    
    def __init__(self):
        self.trades_log = []
        self.current_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'total_pnl': 0.0,
            'portfolio_value': 100000.0,
            'current_positions': 0,
            'best_trade': {'pair': '', 'pnl': 0.0},
            'worst_trade': {'pair': '', 'pnl': 0.0},
            'daily_summary': []
        }
    
    def initialize_display(self):
        """Initialize the live timeline display"""
        st.subheader("🚀 Live Backtest Timeline")
        
        # Create containers for different sections
        self.header_container = st.container()
        self.stats_container = st.container()
        self.timeline_container = st.container()
        
        with self.header_container:
            col1, col2, col3, col4 = st.columns(4)
            self.portfolio_metric = col1.empty()
            self.trades_metric = col2.empty()
            self.winrate_metric = col3.empty()
            self.pnl_metric = col4.empty()
        
        with self.stats_container:
            self.live_stats = st.empty()
        
        with self.timeline_container:
            self.timeline_display = st.empty()
    
    def log_backtest_start(self, start_date: str, end_date: str, initial_capital: float):
        """Log backtest initialization"""
        log_entry = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'type': 'BACKTEST_START',
            'message': f"🚀 BACKTEST INICIADO - {start_date} to {end_date}",
            'details': f"💰 Capital Inicial: ${initial_capital:,.2f}",
            'level': 'INFO'
        }
        self.trades_log.append(log_entry)
        self.current_stats['portfolio_value'] = initial_capital
        self._update_display()
    
    def log_period_start(self, period_num: int, start_date: str, end_date: str):
        """Log trading period start"""
        log_entry = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'type': 'PERIOD_START',
            'message': f"⏰ Período {period_num}: {start_date} to {end_date}",
            'level': 'INFO'
        }
        self.trades_log.append(log_entry)
        self._update_display()
    
    def log_trading_day(self, date: str, pairs_analyzed: int):
        """Log daily trading session start"""
        log_entry = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'type': 'TRADING_DAY',
            'message': f"📅 {date}",
            'details': f"├─ 🔍 Analisando {pairs_analyzed} pares...",
            'level': 'INFO'
        }
        self.trades_log.append(log_entry)
        self._update_display()
    
    def log_trade_entry(self, trade_num: int, pair_id: str, side: str, z_score: float, 
                       price1: float, price2: float, capital: float, symbol1: str, symbol2: str):
        """Log trade entry"""
        self.current_stats['total_trades'] += 1
        self.current_stats['current_positions'] += 1
        
        log_entry = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'type': 'TRADE_ENTRY',
            'message': f"├─ ✅ ENTRADA #{trade_num:03d}: {pair_id} | {side} | Z-score: {z_score:.2f}",
            'details': [
                f"│  ├─ 💰 Preços: {symbol1}=${price1:.2f}, {symbol2}=${price2:.2f}",
                f"│  ├─ 📊 Capital: ${capital:,.0f} ({capital/self.current_stats['portfolio_value']*100:.1f}% portfolio)",
                f"│  └─ 🎯 Target: Convergência z-score → 0"
            ],
            'level': 'SUCCESS'
        }
        self.trades_log.append(log_entry)
        self._update_display()
    
    def log_trade_exit(self, trade_num: int, pair_id: str, pnl: float, holding_days: int, 
                      z_score: float, exit_reason: str):
        """Log trade exit"""
        self.current_stats['current_positions'] -= 1
        self.current_stats['total_pnl'] += pnl
        self.current_stats['portfolio_value'] += pnl
        
        if pnl > 0:
            self.current_stats['winning_trades'] += 1
            status = "✅ PROFIT"
            level = 'SUCCESS'
        else:
            status = "❌ LOSS"
            level = 'ERROR'
        
        # Track best/worst trades
        if pnl > self.current_stats['best_trade']['pnl']:
            self.current_stats['best_trade'] = {'pair': pair_id, 'pnl': pnl}
        if pnl < self.current_stats['worst_trade']['pnl']:
            self.current_stats['worst_trade'] = {'pair': pair_id, 'pnl': pnl}
        
        log_entry = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'type': 'TRADE_EXIT',
            'message': f"├─ 🚪 SAÍDA #{trade_num:03d}: {pair_id} | {pnl:+.2f} | {holding_days} dias | {status}",
            'details': f"│  └─ 📈 Z-score: {z_score:.2f} | {exit_reason}",
            'level': level
        }
        self.trades_log.append(log_entry)
        self._update_display()
    
    def log_no_signal(self, pair_id: str, z_score: float):
        """Log when no trading signal is generated"""
        log_entry = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'type': 'NO_SIGNAL',
            'message': f"├─ ❌ SKIPPED: {pair_id} | NO_SIGNAL | Z-score: {z_score:.2f}",
            'level': 'WARNING'
        }
        self.trades_log.append(log_entry)
        self._update_display()
    
    def log_daily_summary(self, date: str, successful_trades: int, total_attempts: int, portfolio_value: float):
        """Log daily trading summary"""
        self.current_stats['portfolio_value'] = portfolio_value
        pnl_today = portfolio_value - 100000  # Assuming we track daily changes
        
        log_entry = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'type': 'DAILY_SUMMARY',
            'message': f"└─ 📊 Dia: {successful_trades}/{total_attempts} trades executados | Portfolio: ${portfolio_value:,.0f}",
            'details': f"   💰 P&L Total: {self.current_stats['total_pnl']:+,.2f} ({self.current_stats['total_pnl']/100000*100:+.2f}%)",
            'level': 'INFO'
        }
        self.trades_log.append(log_entry)
        self._update_display()
    
    def _update_display(self):
        """Update the live display with current stats and timeline"""
        # Update metrics
        with self.portfolio_metric:
            st.metric("Portfolio", f"${self.current_stats['portfolio_value']:,.0f}", 
                     f"{self.current_stats['total_pnl']:+,.0f}")
        
        with self.trades_metric:
            st.metric("Total Trades", self.current_stats['total_trades'])
        
        with self.winrate_metric:
            if self.current_stats['total_trades'] > 0:
                win_rate = self.current_stats['winning_trades'] / self.current_stats['total_trades'] * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
            else:
                st.metric("Win Rate", "0.0%")
        
        with self.pnl_metric:
            st.metric("P&L", f"${self.current_stats['total_pnl']:+,.0f}")
        
        # Update live stats
        with self.live_stats:
            if self.current_stats['total_trades'] > 0:
                win_rate = self.current_stats['winning_trades'] / self.current_stats['total_trades'] * 100
                
                stats_text = f"""
                📊 **LIVE PERFORMANCE:**
                - 🎯 Total Trades: {self.current_stats['total_trades']}
                - ✅ Win Rate: {win_rate:.1f}% ({self.current_stats['winning_trades']}/{self.current_stats['total_trades']})
                - 💰 Total P&L: ${self.current_stats['total_pnl']:+,.2f}
                - 🔄 Open Positions: {self.current_stats['current_positions']}
                """
                
                if self.current_stats['best_trade']['pnl'] > 0:
                    stats_text += f"- 📈 Best Trade: {self.current_stats['best_trade']['pair']} (${self.current_stats['best_trade']['pnl']:+,.2f})\n"
                if self.current_stats['worst_trade']['pnl'] < 0:
                    stats_text += f"- 📉 Worst Trade: {self.current_stats['worst_trade']['pair']} (${self.current_stats['worst_trade']['pnl']:+,.2f})"
                
                st.markdown(stats_text)
        
        # Update timeline (show last 20 entries)
        with self.timeline_display:
            timeline_text = "```\n"
            for entry in self.trades_log[-20:]:  # Show last 20 entries
                timeline_text += f"{entry['timestamp']} - {entry['message']}\n"
                if 'details' in entry:
                    if isinstance(entry['details'], list):
                        for detail in entry['details']:
                            timeline_text += f"{detail}\n"
                    else:
                        timeline_text += f"{entry['details']}\n"
            timeline_text += "```"
            st.markdown(timeline_text)
    
    def finalize_backtest(self, final_stats: Dict[str, Any]):
        """Display final backtest results"""
        log_entry = {
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'type': 'BACKTEST_COMPLETE',
            'message': "🏁 BACKTEST COMPLETO",
            'details': [
                f"💰 Capital Final: ${final_stats.get('final_capital', 0):,.2f}",
                f"📈 Retorno Total: {final_stats.get('total_return', 0):.2f}%",
                f"📊 Total Trades: {final_stats.get('total_trades', 0)}",
                f"✅ Win Rate: {final_stats.get('win_rate', 0):.1f}%"
            ],
            'level': 'SUCCESS'
        }
        self.trades_log.append(log_entry)
        self._update_display()