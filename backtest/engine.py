"""
Backtesting engine for pairs trading strategy.
Implements rolling window backtesting with position management.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import warnings

from data.api import MarketDataAPI
from strategy.pairs import PairSelector
from strategy.signals import TradingSignalGenerator, SignalType, PositionSide
from backtest.positions import PositionManager
from backtest.risk import RiskManager
from config.settings import CONFIG
from simple_logger import logger

warnings.filterwarnings('ignore')

class BacktestEngine:
    """Main backtesting engine for pairs trading strategy."""
    
    def __init__(self, initial_capital: float = None, data_api: MarketDataAPI = None):
        self.initial_capital = initial_capital or CONFIG['trading']['initial_capital']
        self.data_api = data_api or MarketDataAPI()
        self.pair_selector = PairSelector(self.data_api)
        self.signal_generator = TradingSignalGenerator(self.data_api)
        self.position_manager = PositionManager(self.initial_capital)
        self.risk_manager = RiskManager()
        
        # Backtest parameters from config
        self.lookback_window = CONFIG['strategy']['lookback_window']
        self.trading_window = CONFIG['strategy']['trading_window']
        self.rebalance_frequency = CONFIG['strategy']['rebalance_frequency']
        self.top_pairs = CONFIG['strategy']['top_pairs']
        
        # Results storage
        self.results = {}
        self.equity_curve = pd.Series(dtype=float)
        self.trades_history = []
        self.pair_history = []
        
    def run_rolling_backtest(self, start_date: str, end_date: str,
                           universe: str = 'IBOV',
                           save_results: bool = True) -> Dict[str, Any]:
        """
        Run rolling window backtesting.
        
        Args:
            start_date: Backtest start date
            end_date: Backtest end date
            universe: Stock universe to use
            save_results: Whether to save detailed results
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting rolling backtest from {start_date} to {end_date}")
        
        try:
            # Convert dates
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # Initialize tracking variables
            self.equity_curve = pd.Series(dtype=float)
            self.trades_history = []
            self.pair_history = []
            
            # Create date range for backtest
            current_date = start_dt
            portfolio_values = []
            
            while current_date <= end_dt:
                try:
                    # Formation period: analyze pairs
                    formation_start = current_date - timedelta(days=self.lookback_window)
                    formation_end = current_date
                    
                    logger.debug(f"Processing date: {current_date.strftime('%Y-%m-%d')}")
                    
                    # Find cointegrated pairs for this period
                    pairs = self._find_pairs_for_period(
                        formation_start.strftime('%Y-%m-%d'),
                        formation_end.strftime('%Y-%m-%d'),
                        universe
                    )
                    
                    if pairs:
                        # Store pair information
                        self.pair_history.append({
                            'date': current_date,
                            'pairs_count': len(pairs),
                            'top_pairs': pairs[:self.top_pairs]
                        })
                        
                        # Trading period: generate signals and manage positions
                        trading_end = min(
                            current_date + timedelta(days=self.trading_window),
                            end_dt
                        )
                        
                        # Execute trading for this period
                        period_results = self._execute_trading_period(
                            pairs[:self.top_pairs],
                            current_date.strftime('%Y-%m-%d'),
                            trading_end.strftime('%Y-%m-%d')
                        )
                        
                        # Update portfolio value
                        portfolio_value = self.position_manager.get_portfolio_value()
                        portfolio_values.append({
                            'date': current_date,
                            'portfolio_value': portfolio_value,
                            'cash': self.position_manager.cash,
                            'positions_value': portfolio_value - self.position_manager.cash
                        })
                    
                    # Move to next rebalance date
                    current_date += timedelta(days=self.rebalance_frequency)
                    
                except Exception as e:
                    logger.error(f"Error processing date {current_date}: {e}")
                    current_date += timedelta(days=self.rebalance_frequency)
                    continue
            
            # Create equity curve
            if portfolio_values:
                portfolio_df = pd.DataFrame(portfolio_values)
                portfolio_df.set_index('date', inplace=True)
                self.equity_curve = portfolio_df['portfolio_value']
            
            # Calculate performance metrics
            performance_metrics = self.calculate_performance_metrics()
            
            # Compile results
            results = {
                'start_date': start_date,
                'end_date': end_date,
                'universe': universe,
                'initial_capital': self.initial_capital,
                'final_capital': self.position_manager.get_portfolio_value(),
                'total_return': performance_metrics.get('total_return', 0),
                'annualized_return': performance_metrics.get('annualized_return', 0),
                'volatility': performance_metrics.get('volatility', 0),
                'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
                'max_drawdown': performance_metrics.get('max_drawdown', 0),
                'total_trades': len(self.trades_history),
                'winning_trades': len([t for t in self.trades_history if t.get('pnl', 0) > 0]),
                'win_rate': performance_metrics.get('win_rate', 0),
                'performance_metrics': performance_metrics,
                'equity_curve': self.equity_curve,
                'trades_history': self.trades_history,
                'pair_history': self.pair_history
            }
            
            self.results = results
            
            logger.info(f"Backtest completed. Total return: {results['total_return']:.2%}, "
                       f"Sharpe ratio: {results['sharpe_ratio']:.2f}, "
                       f"Max drawdown: {results['max_drawdown']:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in rolling backtest: {e}")
            return {}
    
    def _find_pairs_for_period(self, start_date: str, end_date: str, 
                              universe: str) -> List[Dict[str, Any]]:
        """Find cointegrated pairs for a specific period."""
        try:
            from config.universe import get_universe_tickers
            
            # Get universe symbols
            symbols = get_universe_tickers(universe)
            
            # Get all data at once to avoid individual symbol warnings
            all_data = self.data_api.get_price_data(symbols, start_date, end_date)
            
            if all_data.empty:
                logger.debug(f"No data available for period {start_date} to {end_date}")
                return []
            
            # Filter symbols with sufficient data
            valid_symbols = []
            min_points = int(self.lookback_window * 0.8)  # 80% data availability
            
            for symbol in all_data.columns:
                symbol_data = all_data[symbol].dropna()
                if len(symbol_data) >= min_points:
                    valid_symbols.append(symbol)
            
            if len(valid_symbols) < 2:
                logger.debug(f"Insufficient valid symbols for period {start_date} to {end_date}: {len(valid_symbols)}")
                return []
            
            logger.debug(f"Found {len(valid_symbols)} valid symbols for period")
            
            # Find cointegrated pairs
            pairs = self.pair_selector.find_cointegrated_pairs(
                valid_symbols, start_date, end_date, max_workers=2
            )
            
            # Filter and rank pairs
            filtered_pairs = self.pair_selector.filter_by_criteria(pairs)
            ranked_pairs = self.pair_selector.rank_pairs(filtered_pairs)
            
            return ranked_pairs
            
        except Exception as e:
            logger.error(f"Error finding pairs for period {start_date} to {end_date}: {e}")
            return []
    
    def _execute_trading_period(self, pairs: List[Dict[str, Any]], 
                               start_date: str, end_date: str) -> Dict[str, Any]:
        """Execute trading for a specific period."""
        try:
            if not pairs:
                return {}
            
            # Get trading dates
            trading_dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            period_trades = []
            
            for date in trading_dates:
                try:
                    date_str = date.strftime('%Y-%m-%d')
                    
                    # Generate signals for all pairs
                    for pair in pairs:
                        symbol1 = pair['symbol1']
                        symbol2 = pair['symbol2']
                        hedge_ratio = pair['hedge_ratio']
                        intercept = pair.get('intercept', 0)
                        pair_id = f"{symbol1}-{symbol2}"
                        
                        # Get current position
                        current_position = self.position_manager.get_position(pair_id)
                        position_side = current_position.get('side', PositionSide.FLAT)
                        
                        # Calculate current spread and z-score
                        spread = self.signal_generator.calculate_spread(
                            symbol1, symbol2, hedge_ratio, intercept,
                            (date - timedelta(days=30)).strftime('%Y-%m-%d'),
                            date_str
                        )
                        
                        if spread.empty:
                            continue
                        
                        # Calculate z-score
                        z_score = self.signal_generator.calculate_z_score(spread)
                        
                        if z_score.empty:
                            continue
                        
                        current_z_score = z_score.iloc[-1]
                        
                        # Generate signal
                        signals = self.signal_generator.generate_signals(
                            z_score.tail(1), current_position=position_side
                        )
                        
                        if signals.empty:
                            continue
                        
                        current_signal = signals.iloc[-1]
                        
                        # Execute trades based on signals
                        trade_result = self._execute_signal(
                            pair_id, symbol1, symbol2, current_signal,
                            current_z_score, hedge_ratio, date_str
                        )
                        
                        if trade_result:
                            period_trades.append(trade_result)
                            self.trades_history.append(trade_result)
                
                except Exception as e:
                    logger.error(f"Error processing trading date {date}: {e}")
                    continue
            
            return {'trades': period_trades}
            
        except Exception as e:
            logger.error(f"Error executing trading period: {e}")
            return {}
    
    def _execute_signal(self, pair_id: str, symbol1: str, symbol2: str,
                       signal: SignalType, z_score: float, hedge_ratio: float,
                       date: str) -> Optional[Dict[str, Any]]:
        """Execute a trading signal."""
        try:
            if signal == SignalType.NO_SIGNAL:
                return None
            
            # Get current prices
            price_data = self.data_api.get_pairs_data(symbol1, symbol2, date, date)
            
            if price_data.empty:
                return None
            
            price1 = price_data[symbol1].iloc[-1]
            price2 = price_data[symbol2].iloc[-1]
            
            # Calculate position size
            portfolio_value = self.position_manager.get_portfolio_value()
            max_position_size = CONFIG['trading']['max_position_size']
            position_value = portfolio_value * max_position_size
            
            # Risk check
            if not self.risk_manager.check_position_size(pair_id, position_value):
                return None
            
            trade_result = None
            
            if signal in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT]:
                # Open new position
                side = PositionSide.LONG if signal == SignalType.ENTRY_LONG else PositionSide.SHORT
                
                trade_result = self.position_manager.open_position(
                    pair_id, symbol1, symbol2, side, position_value,
                    price1, price2, hedge_ratio, date
                )
                
                if trade_result:
                    trade_result.update({
                        'signal': signal,
                        'z_score': z_score,
                        'action': 'OPEN'
                    })
            
            elif signal in [SignalType.EXIT_LONG, SignalType.EXIT_SHORT, SignalType.STOP_LOSS]:
                # Close existing position
                exit_reason = 'STOP_LOSS' if signal == SignalType.STOP_LOSS else 'EXIT'
                
                trade_result = self.position_manager.close_position(
                    pair_id, price1, price2, date, exit_reason
                )
                
                if trade_result:
                    trade_result.update({
                        'signal': signal,
                        'z_score': z_score,
                        'action': 'CLOSE'
                    })
            
            return trade_result
            
        except Exception as e:
            logger.error(f"Error executing signal for {pair_id}: {e}")
            return None
    
    def calculate_performance_metrics(self) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        try:
            if self.equity_curve.empty:
                return {}
            
            # Basic returns
            returns = self.equity_curve.pct_change().dropna()
            
            if len(returns) == 0:
                return {}
            
            # Total return
            total_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1
            
            # Annualized return
            days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
            years = days / 365.25
            annualized_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
            
            # Volatility (annualized)
            volatility = returns.std() * np.sqrt(252)
            
            # Sharpe ratio
            risk_free_rate = 0.05  # 5% risk-free rate
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + returns).cumprod()
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Trade statistics
            trades_df = pd.DataFrame(self.trades_history)
            
            if not trades_df.empty and 'pnl' in trades_df.columns:
                winning_trades = trades_df[trades_df['pnl'] > 0]
                losing_trades = trades_df[trades_df['pnl'] <= 0]
                
                win_rate = len(winning_trades) / len(trades_df) * 100
                avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
                avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
                profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
                
                # Additional trade metrics
                avg_trade_duration = trades_df['holding_days'].mean() if 'holding_days' in trades_df.columns else 0
                best_trade = trades_df['pnl'].max()
                worst_trade = trades_df['pnl'].min()
            else:
                win_rate = 0
                avg_win = 0
                avg_loss = 0
                profit_factor = 0
                avg_trade_duration = 0
                best_trade = 0
                worst_trade = 0
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252)
            sortino_ratio = (annualized_return - risk_free_rate) / downside_std if downside_std > 0 else 0
            
            metrics = {
                'total_return': total_return,
                'annualized_return': annualized_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'calmar_ratio': calmar_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'avg_trade_duration': avg_trade_duration,
                'best_trade': best_trade,
                'worst_trade': worst_trade,
                'total_trades': len(self.trades_history),
                'trading_days': len(self.equity_curve),
                'trading_period_years': years
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            return {}
    
    def get_benchmark_comparison(self, benchmark_symbol: str = '^BVSP') -> Dict[str, Any]:
        """Compare strategy performance to benchmark."""
        try:
            if self.equity_curve.empty:
                return {}
            
            # Get benchmark data
            start_date = self.equity_curve.index[0].strftime('%Y-%m-%d')
            end_date = self.equity_curve.index[-1].strftime('%Y-%m-%d')
            
            benchmark_data = self.data_api.get_price_data([benchmark_symbol], start_date, end_date)
            
            if benchmark_data.empty:
                return {}
            
            # Align dates
            common_dates = self.equity_curve.index.intersection(benchmark_data.index)
            
            if len(common_dates) < 2:
                return {}
            
            strategy_returns = self.equity_curve[common_dates].pct_change().dropna()
            benchmark_returns = benchmark_data[benchmark_symbol][common_dates].pct_change().dropna()
            
            # Calculate benchmark metrics
            benchmark_total_return = (benchmark_data[benchmark_symbol].iloc[-1] / benchmark_data[benchmark_symbol].iloc[0]) - 1
            benchmark_vol = benchmark_returns.std() * np.sqrt(252)
            
            # Calculate beta and alpha
            covariance = np.cov(strategy_returns, benchmark_returns)[0][1]
            benchmark_variance = np.var(benchmark_returns)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
            
            strategy_total_return = (self.equity_curve.iloc[-1] / self.equity_curve.iloc[0]) - 1
            alpha = strategy_total_return - beta * benchmark_total_return
            
            # Information ratio
            active_returns = strategy_returns - benchmark_returns
            tracking_error = active_returns.std() * np.sqrt(252)
            information_ratio = active_returns.mean() * np.sqrt(252) / tracking_error if tracking_error > 0 else 0
            
            comparison = {
                'strategy_total_return': strategy_total_return,
                'benchmark_total_return': benchmark_total_return,
                'excess_return': strategy_total_return - benchmark_total_return,
                'strategy_volatility': strategy_returns.std() * np.sqrt(252),
                'benchmark_volatility': benchmark_vol,
                'beta': beta,
                'alpha': alpha,
                'information_ratio': information_ratio,
                'tracking_error': tracking_error,
                'correlation': strategy_returns.corr(benchmark_returns)
            }
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error calculating benchmark comparison: {e}")
            return {}
    
    def save_results(self, filepath: str = None):
        """Save backtest results to file."""
        try:
            if not filepath:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filepath = f"backtest_results_{timestamp}.pkl"
            
            import pickle
            
            with open(filepath, 'wb') as f:
                pickle.dump(self.results, f)
            
            logger.info(f"Backtest results saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving results: {e}")
    
    def load_results(self, filepath: str):
        """Load backtest results from file."""
        try:
            import pickle
            
            with open(filepath, 'rb') as f:
                self.results = pickle.load(f)
            
            # Restore equity curve and trade history
            self.equity_curve = self.results.get('equity_curve', pd.Series())
            self.trades_history = self.results.get('trades_history', [])
            
            logger.info(f"Backtest results loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading results: {e}")