"""
Configuration settings for the pairs trading system.
"""
import os
from datetime import datetime, timedelta

CONFIG = {
    'data': {
        'universe': ['IBOV', 'IBRX100'],
        'min_market_cap': 1_000_000_000,
        'min_avg_volume': 1_000_000,
        'update_time': '18:30',
        'lookback_days': 1000,
        'data_source': 'yahoo',
        'timezone': 'America/Sao_Paulo',
    },
    'strategy': {
        'lookback_window': 252,  # 1 year (Gatev et al. 2006 standard)
        'trading_window': 63,  # 3 months (academic standard)
        'rebalance_frequency': 63,  # Quarterly rebalancing (academic standard)
        'top_pairs': 20,  # Academic studies typically use 15-20 pairs
        'min_half_life': 5,  # Minimum mean reversion speed (academic range: 5-30)
        'max_half_life': 30,  # Maximum mean reversion speed
        'min_correlation': 0.7,  # Strong correlation requirement (academic: 0.6-0.8)
        'p_value_threshold': 0.05,  # 95% confidence for cointegration (statistical standard)
        'sector_matching': False,  # Cross-sector pairs allowed (diversification)
    },
    'trading': {
        'entry_z_score': 1.5,  # Reduced from 2.0 for better signal generation
        'exit_z_score': 0.5,  # Exit before mean (conservative approach)
        'stop_loss_z_score': 3.0,  # Risk management (3 std devs)
        'max_position_size': 0.05,  # 5% per pair (conservative, academic standard)
        'initial_capital': 100000,  # R$ 100k baseline
        'commission_rate': 0.0025,  # 0.25% - realistic for Brazilian market
        'max_active_pairs': 20,  # Match with top_pairs for full utilization
        'force_close_missing_days': 14,  # Force close after X consecutive days without data
        'max_holding_period': 90,  # Maximum days to hold any position (timeout)
    },
    'risk': {
        'max_drawdown': 0.15,
        'max_leverage': 2.0,
        'position_sizing': 'equal_weight',
        'rebalance_threshold': 0.05,
    },
    'database': {
        'db_path': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'database', 'pairs_trading.db'),
        'backup_frequency': 'daily',
        'backup_retention': 30,
    },
    'alerts': {
        'email_enabled': True,
        'signal_alerts': True,
        'error_alerts': True,
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'email_from': os.getenv('EMAIL_FROM', ''),
        'email_password': os.getenv('EMAIL_PASSWORD', ''),
        'email_to': os.getenv('EMAIL_TO', ''),
    },
    'logging': {
        'log_level': 'INFO',
        'log_file': os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs', 'pairs_trading.log'),
        'max_file_size': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5,
    },
    'api': {
        'rate_limit_delay': 0.1,
        'max_retries': 3,
        'timeout': 30,
    }
}

# Environment-specific overrides
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

if ENVIRONMENT == 'production':
    CONFIG['database']['db_path'] = '/opt/pairs_trading/database/pairs_trading.db'
    CONFIG['logging']['log_file'] = '/opt/pairs_trading/logs/pairs_trading.log'
    CONFIG['alerts']['email_enabled'] = True
elif ENVIRONMENT == 'testing':
    CONFIG['database']['db_path'] = ':memory:'
    CONFIG['alerts']['email_enabled'] = False
    CONFIG['logging']['log_level'] = 'DEBUG'