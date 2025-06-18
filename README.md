# 🎯 Pairs Trading System - Brazilian Market

A comprehensive pairs trading system for the Brazilian stock market, implementing cointegration-based statistical arbitrage strategies.

## 🌟 Features

### Core Components
- **Data Management**: Automated data collection from Yahoo Finance for Brazilian stocks (.SA)
- **Cointegration Analysis**: Engle-Granger test, ADF testing, and half-life calculation
- **Signal Generation**: Z-score based entry/exit signals with position management
- **Backtesting Engine**: Rolling window backtesting with comprehensive performance metrics
- **Risk Management**: Position sizing, drawdown limits, and portfolio exposure controls
- **Interactive Dashboard**: 8-tab Streamlit interface for complete system monitoring

### Dashboard Features
1. **🏠 Home/Overview**: Portfolio performance summary and system status
2. **📊 Live Trading**: Real-time signal monitoring and position tracking
3. **🔍 Pair Analysis**: Cointegration testing and pair discovery tools
4. **📈 Backtesting**: Historical strategy simulation and optimization
5. **📋 Performance**: Detailed performance metrics and risk analysis
6. **⚙️ Configuration**: Strategy parameters and system settings
7. **💾 Data Management**: Database monitoring and data quality control
8. **📝 Logs & Alerts**: System monitoring and alert management

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd cointegration-strategy
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Initialize the system**:
```bash
python main.py --init --universe IBOV
```

### Basic Usage

#### Command Line Interface

```bash
# Initialize with market data
python main.py --init --universe IBOV

# Find cointegrated pairs
python main.py --pairs --universe IBOV

# Generate trading signals
python main.py --signals

# Run complete analysis
python main.py --all --universe IBOV

# Launch dashboard
python main.py --dashboard
```

#### Dashboard Interface

Launch the interactive dashboard:
```bash
streamlit run frontend/dashboard.py
```

The dashboard will be available at `http://localhost:8501`

## 📊 Strategy Overview

### Cointegration Testing
- **Engle-Granger Two-Step Method**: Tests for long-term equilibrium relationships
- **ADF Test**: Confirms stationarity of residuals
- **Half-Life Calculation**: Measures mean reversion speed using Ornstein-Uhlenbeck process

### Signal Generation
- **Entry Signals**: Z-score thresholds (default: ±2.0)
- **Exit Signals**: Mean reversion thresholds (default: ±0.5)
- **Stop-Loss**: Risk management thresholds (default: ±3.0)

### Portfolio Management
- **Rolling Window Approach**: 252-day formation, 63-day trading periods
- **Position Sizing**: Maximum 10% per position, up to 15 active pairs
- **Risk Controls**: Maximum drawdown limits, leverage constraints

## ⚙️ Configuration

### Strategy Parameters

Key parameters can be configured in `config/settings.py`:

```python
CONFIG = {
    'strategy': {
        'lookback_window': 252,      # Formation period (days)
        'trading_window': 63,        # Trading period (days)
        'rebalance_frequency': 21,   # Rebalancing interval (days)
        'top_pairs': 15,             # Maximum active pairs
        'p_value_threshold': 0.05,   # Cointegration significance
        'min_correlation': 0.7,      # Minimum pair correlation
        'min_half_life': 5,          # Minimum mean reversion speed
        'max_half_life': 30,         # Maximum mean reversion speed
    },
    'trading': {
        'entry_z_score': 2.0,        # Entry threshold
        'exit_z_score': 0.5,         # Exit threshold
        'stop_loss_z_score': 3.0,    # Stop-loss threshold
        'initial_capital': 100000,   # Starting capital (R$)
        'max_position_size': 0.1,    # Max position size (10%)
        'commission_rate': 0.003,    # Transaction costs (0.3%)
    }
}
```

### Universe Selection

Configure stock universes in `config/universe.py`:

- **IBOV**: Ibovespa index constituents (~70 stocks)
- **IBRX100**: IBrX-100 index constituents (~100 stocks)
- **Custom**: Define your own symbol lists

## 📈 Performance Metrics

The system calculates comprehensive performance metrics:

### Return Metrics
- Total Return
- Annualized Return
- Volatility (annualized)
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio

### Risk Metrics
- Maximum Drawdown
- Value at Risk (VaR)
- Expected Shortfall
- Beta vs benchmark
- Tracking Error

### Trade Statistics
- Total number of trades
- Win rate
- Average win/loss
- Profit factor
- Average holding period

## 🗄️ Database Schema

SQLite database with the following tables:

- **stocks_master**: Stock metadata and information
- **daily_prices**: OHLCV price data
- **dividends**: Dividend payments
- **splits**: Stock splits
- **pair_results**: Cointegration test results
- **data_quality_log**: Data quality monitoring
- **update_status**: System update tracking

## 🔧 Architecture

```
pairs_trading_system/
├── data/              # Data collection and storage
│   ├── collector.py   # Yahoo Finance data collection
│   ├── storage.py     # Database management
│   ├── quality.py     # Data quality control
│   └── api.py         # Unified data interface
├── strategy/          # Trading strategy implementation
│   ├── cointegration.py  # Statistical tests
│   ├── pairs.py       # Pair selection and ranking
│   └── signals.py     # Signal generation
├── backtest/          # Backtesting framework
│   ├── engine.py      # Main backtesting engine
│   ├── positions.py   # Position management
│   └── risk.py        # Risk management
├── frontend/          # Streamlit dashboard
│   ├── dashboard.py   # Main dashboard
│   ├── pages/         # Individual dashboard pages
│   └── utils.py       # Dashboard utilities
├── config/            # Configuration
│   ├── settings.py    # System parameters
│   └── universe.py    # Stock universes
└── utils/             # Shared utilities
    └── logger.py      # Logging configuration
```

## 📊 Data Sources

- **Primary**: Yahoo Finance (yfinance library)
- **Market**: Brazilian stocks (B3 exchange)
- **Coverage**: IBOV and IBrX-100 constituents
- **Frequency**: Daily OHLCV data with automatic updates

## 🛡️ Risk Management

### Position-Level Controls
- Maximum position size: 10% of portfolio
- Stop-loss at 3σ z-score levels
- Position correlation monitoring

### Portfolio-Level Controls
- Maximum drawdown: 15%
- Maximum leverage: 2.0x
- Concentration limits by sector
- Dynamic position sizing

### Data Quality Controls
- Outlier detection and filtering
- Missing data monitoring
- Corporate action adjustments
- Quality scoring and alerts

## 📱 Dashboard Usage

### Live Trading Tab
- Monitor active positions in real-time
- View current z-scores and signals
- Execute trading decisions
- Track P&L and performance

### Pair Analysis Tab
- Discover new cointegrated pairs
- Analyze pair statistics and relationships
- Visualize price relationships and spreads
- Export analysis results

### Backtesting Tab
- Configure strategy parameters
- Run historical simulations
- Analyze performance metrics
- Compare against benchmarks

### Configuration Tab
- Adjust strategy parameters
- Configure risk limits
- Set up alerts and notifications
- Export/import configurations

## 🔧 Troubleshooting

### Common Issues

1. **Data Download Failures**:
   - Check internet connection
   - Verify Yahoo Finance availability
   - Adjust rate limiting in configuration

2. **Cointegration Test Errors**:
   - Ensure sufficient data points (min 252 days)
   - Check for data quality issues
   - Verify symbol validity

3. **Dashboard Performance**:
   - Limit number of symbols in analysis
   - Use data caching where possible
   - Close unused browser tabs

### Logging

System logs are available in `logs/pairs_trading.log`:
- INFO: Normal operations
- WARNING: Non-critical issues
- ERROR: System errors requiring attention

## 📄 License

This project is for educational and research purposes. Please ensure compliance with your local regulations regarding algorithmic trading.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes
4. Add tests if applicable
5. Submit a pull request

## 📧 Support

For questions and support:
- Review the documentation
- Check the troubleshooting section
- Examine system logs for error details
- Create an issue with detailed information

---

**⚠️ Disclaimer**: This system is for educational purposes only. Past performance does not guarantee future results. Always conduct thorough testing and risk assessment before using any trading strategy with real capital.