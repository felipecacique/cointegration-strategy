# PROMPT PARA CLAUDE CODE - ESTRATÉGIA PAIRS TRADING BRASIL

## OBJETIVO GERAL
Criar um sistema completo de pairs trading por cointegração para o mercado brasileiro, com dados do Yahoo Finance, incluindo módulo de dados, backtesting, live trading e frontend dashboard.

## ARQUITETURA DO SISTEMA

### 1. MÓDULO DE DADOS
**Objetivo**: Coletar, armazenar e gerenciar dados de ações brasileiras

**Componentes**:
- **Database**: SQLite para desenvolvimento (estrutura preparada para PostgreSQL/TimescaleDB)
- **Schema Principal**:
  ```sql
  stocks_master (symbol, name, sector, market_cap, last_update)
  daily_prices (symbol, date, open, high, low, close, volume, adj_close)
  dividends (symbol, date, dividend)
  splits (symbol, date, ratio)
  market_calendar (date, is_trading_day, exchange)
  data_quality_log (date, symbol, issue_type, description)
  update_status (last_update, status, records_updated)
  ```

**Classes Principais**:
```python
class DataCollector:
    # Coleta dados do yfinance
    # Retry logic e error handling
    # Rate limiting
    
class DataStorageManager:
    # Interface unificada para DB
    # CRUD operations otimizadas
    # Cache layer
    
class DataQualityController:
    # Validação de dados
    # Detecção de outliers
    # Alertas de problemas
    
class MarketDataAPI:
    # Interface para outros módulos
    # get_price_data(), get_pairs_data(), etc.
```

**Funcionalidades**:
- Coleta inicial de histórico completo
- Atualizações diárias automatizadas (18:30 UTC-3)
- Validação de qualidade (gaps, outliers, inconsistências) 
- Backup automático
- Universo configurável (IBOV, IBRX100, filtros de liquidez)

### 2. ESTRATÉGIA DE PAIRS TRADING

**Testes de Cointegração**:
- Teste Engle-Granger (CADF)
- Teste ADF nos resíduos
- Cálculo de half-life (Ornstein-Uhlenbeck)
- Hedge ratio por regressão linear

**Classes Principais**:
```python
class CointegrationTester:
    def test_pair_cointegration(self, y1, y2)
    def calculate_half_life(self, residuals)
    def get_hedge_ratio(self, y1, y2)
    
class PairSelector:
    def find_cointegrated_pairs(self, price_data, min_pvalue=0.05)
    def rank_pairs(self, pairs_results)
    def filter_by_criteria(self, pairs, min_half_life=5, max_half_life=30)
    
class TradingSignalGenerator:
    def calculate_z_score(self, spread)
    def generate_signals(self, z_score, entry_threshold=2.0, exit_threshold=0.5)
```

**Critérios de Seleção**:
- P-value < 0.05 (significância estatística)
- Half-life entre 5-30 dias
- Correlação mínima > 0.7
- Volume mínimo R$ 1MM/dia
- Mesmo setor (opcional)

**Sinais de Trading**:
- Entrada LONG: z_score < -2.0
- Entrada SHORT: z_score > 2.0
- Saída: abs(z_score) < 0.5
- Stop-loss: abs(z_score) > 3.0

### 3. BACKTESTING ENGINE

**Janela Deslizante**:
- Look-back window: 252 dias (formação)
- Trading window: 63 dias (operação)
- Rebalance: 21 dias
- Top pairs: 10-20 selecionados

**Processo**:
1. **Período Formação**: Testa todos pares, calcula métricas, ranking
2. **Período Operação**: Opera top X pares, monitora sinais
3. **Rebalanceamento**: Atualiza ranking, fecha/abre posições

**Classes**:
```python
class BacktestEngine:
    def __init__(self, data_api, initial_capital=100000)
    def run_rolling_backtest(self, start_date, end_date, params)
    def calculate_performance_metrics(self)
    
class PositionManager:
    def open_position(self, pair, direction, capital_allocation)
    def close_position(self, pair, reason)
    def update_positions(self, current_prices)
    
class RiskManager:
    def check_position_size(self, pair, capital)
    def apply_stop_loss(self, positions, z_scores)
    def calculate_portfolio_exposure(self)
```

### 4. LIVE TRADING SYSTEM

**Funcionalidades**:
- Monitoramento em tempo real
- Geração automática de sinais
- Alertas de entrada/saída
- Dashboard de posições ativas

**Classes**:
```python
class LiveTrader:
    def monitor_active_pairs(self)
    def generate_live_signals(self)
    def send_alerts(self, signal_type, pair, z_score)
    
class AlertManager:
    def send_email_alert(self, message)
    def log_alert(self, alert_type, details)
```

### 5. FRONTEND DASHBOARD (STREAMLIT)

**Estrutura de Abas**:

**5.1 Home/Overview**:
- Cards de performance (Total Return, Sharpe, Drawdown, Win Rate)
- Gráfico de equity curve vs benchmark
- Status do sistema (online, última atualização)

**5.2 Live Trading**:
- Tabela de pares ativos (Entry Date, Z-Score, P&L, Action)
- Sinais de entrada/saída em tempo real
- Gráficos de z-score evolution

**5.3 Pair Analysis**:
- Ferramenta de busca de pares (filtros por setor, correlação, etc.)
- Tabela de resultados de cointegração (P-Value, Half-life, Hedge Ratio)
- Visualizações: scatter plot, residuals, z-score distribution

**5.4 Backtesting**:
- Configuração de parâmetros (datas, capital, janelas, thresholds)
- Resultados: equity curve, monthly returns heatmap, trade statistics
- Métricas detalhadas (Total Trades, Win Rate, Avg Win/Loss)

**5.5 Performance**:
- Tabela comparativa Strategy vs Benchmark
- Métricas avançadas (Sharpe, Calmar, Beta, Factor Exposure)
- Rolling performance metrics

**5.6 Configuration**:
- Parâmetros da estratégia (Z-Scores, Position Size, Max Pairs)
- Configurações de dados (Universo, Filtros, Update Frequency)
- Alertas (Email, Sinais, Erros)

**5.7 Data Management**:
- Status do banco de dados (Total Records, Size, Last Update)
- Métricas de qualidade (Outliers, Gaps, Success Rate)
- Ações manuais (Force Update, Quality Check, Cleanup)

**5.8 Logs & Alerts**:
- Log de eventos recentes
- Alertas ativos
- Histórico de operações

## TECNOLOGIAS E BIBLIOTECAS

```python
# Coleta de dados
import yfinance as yf

# Manipulação de dados
import pandas as pd
import numpy as np

# Testes estatísticos
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

# Banco de dados
import sqlite3
import sqlalchemy

# Visualizações
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Dashboard
import streamlit as st

# Utilidades
import warnings
import logging
from datetime import datetime, timedelta
import schedule
import time
```

## ESTRUTURA DE ARQUIVOS SUGERIDA

```
pairs_trading_system/
├── data/
│   ├── __init__.py
│   ├── collector.py          # DataCollector
│   ├── storage.py            # DataStorageManager  
│   ├── quality.py            # DataQualityController
│   └── api.py               # MarketDataAPI
├── strategy/
│   ├── __init__.py
│   ├── cointegration.py     # CointegrationTester
│   ├── pairs.py             # PairSelector
│   └── signals.py           # TradingSignalGenerator
├── backtest/
│   ├── __init__.py
│   ├── engine.py            # BacktestEngine
│   ├── positions.py         # PositionManager
│   └── risk.py              # RiskManager
├── live/
│   ├── __init__.py
│   ├── trader.py            # LiveTrader
│   └── alerts.py            # AlertManager
├── frontend/
│   ├── __init__.py
│   ├── dashboard.py         # Main Streamlit app
│   ├── pages/               # Individual pages
│   └── utils.py             # Helper functions
├── config/
│   ├── settings.py          # Configuration parameters
│   └── universe.py          # Stock universe definition
├── utils/
│   ├── __init__.py
│   ├── helpers.py           # Utility functions
│   └── logger.py            # Logging setup
├── database/
│   └── pairs_trading.db     # SQLite database
├── logs/
├── requirements.txt
├── main.py                  # Entry point
└── README.md
```

## PARÂMETROS DE CONFIGURAÇÃO

```python
CONFIG = {
    'data': {
        'universe': ['IBOV', 'IBRX100'],
        'min_market_cap': 1_000_000_000,
        'min_avg_volume': 1_000_000,
        'update_time': '18:30',
        'lookback_days': 1000,
    },
    'strategy': {
        'lookback_window': 252,
        'trading_window': 63,
        'rebalance_frequency': 21,
        'top_pairs': 15,
        'min_half_life': 5,
        'max_half_life': 30,
        'min_correlation': 0.7,
        'p_value_threshold': 0.05,
    },
    'trading': {
        'entry_z_score': 2.0,
        'exit_z_score': 0.5,
        'stop_loss_z_score': 3.0,
        'max_position_size': 0.1,
        'initial_capital': 100000,
    },
    'alerts': {
        'email_enabled': True,
        'signal_alerts': True,
        'error_alerts': True,
    }
}
```

## REQUISITOS ESPECÍFICOS

1. **Sistema deve funcionar com dados do Yahoo Finance** (yfinance)
2. **Mercado brasileiro** (ações .SA)
3. **Backtesting com janela deslizante** (rolling window)
4. **Interface visual completa** (Streamlit com múltiplas abas)
5. **Banco de dados local** (SQLite, expansível para PostgreSQL)
6. **Atualizações automáticas** (scheduling)
7. **Tratamento robusto de erros** e logging
8. **Métricas de performance completas**
9. **Sistema modular e extensível**
10. **Documentação clara** e comentários no código

## ENTREGÁVEIS ESPERADOS

1. **Sistema completo funcional**
2. **Base de dados populada** com histórico de ações brasileiras
3. **Dashboard interativo** com todas as abas especificadas
4. **Backtesting executável** com resultados visuais
5. **Sistema de alertas** configurado
6. **Documentação** de uso e configuração
7. **Logs detalhados** de operações
8. **Métricas de performance** calculadas e visualizadas

## PRIORIZAÇÃO DE DESENVOLVIMENTO

1. **Módulo de Dados** (base fundamental)
2. **Testes de Cointegração** (core da estratégia)  
3. **Backtesting Engine** (validação da estratégia)
4. **Frontend Dashboard** (visualização e controle)
5. **Live Trading System** (operação em tempo real)
6. **Refinamentos e otimizações**

Implemente o sistema de forma modular, testável e bem documentada, seguindo boas práticas de desenvolvimento Python.