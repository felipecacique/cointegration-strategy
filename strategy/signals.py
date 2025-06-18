"""
Trading signal generation module for pairs trading.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from enum import Enum

from strategy.cointegration import CointegrationTester
from data.api import MarketDataAPI
from config.settings import CONFIG
from simple_logger import logger

class SignalType(Enum):
    """Enumeration for signal types."""
    ENTRY_LONG = "ENTRY_LONG"
    ENTRY_SHORT = "ENTRY_SHORT"
    EXIT_LONG = "EXIT_LONG"
    EXIT_SHORT = "EXIT_SHORT"
    STOP_LOSS = "STOP_LOSS"
    NO_SIGNAL = "NO_SIGNAL"

class PositionSide(Enum):
    """Enumeration for position sides."""
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"

class TradingSignalGenerator:
    """Generates trading signals for pairs trading strategy."""
    
    def __init__(self, data_api: MarketDataAPI = None):
        self.data_api = data_api or MarketDataAPI()
        self.coint_tester = CointegrationTester()
        self.config = CONFIG['trading']
    
    def calculate_z_score(self, spread: pd.Series, 
                         window: int = 252,
                         method: str = 'rolling') -> pd.Series:
        """
        Calculate z-score of spread for signal generation.
        
        Args:
            spread: Time series of spread values
            window: Window for mean and std calculation
            method: 'rolling', 'expanding', or 'fixed'
            
        Returns:
            Z-score series
        """
        try:
            if len(spread) < 2:
                return pd.Series(dtype=float)
            
            if method == 'rolling':
                if len(spread) < window:
                    # Use expanding window if insufficient data
                    mean = spread.expanding().mean()
                    std = spread.expanding().std()
                else:
                    mean = spread.rolling(window=window).mean()
                    std = spread.rolling(window=window).std()
            
            elif method == 'expanding':
                mean = spread.expanding().mean()
                std = spread.expanding().std()
            
            elif method == 'fixed':
                # Use fixed mean and std from entire series
                mean = spread.mean()
                std = spread.std()
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Calculate z-score
            z_score = (spread - mean) / std
            
            # Replace inf and -inf with NaN
            z_score = z_score.replace([np.inf, -np.inf], np.nan)
            
            return z_score
            
        except Exception as e:
            logger.error(f"Error calculating z-score: {e}")
            return pd.Series(dtype=float)
    
    def calculate_spread(self, symbol1: str, symbol2: str,
                        hedge_ratio: float, intercept: float = 0,
                        start_date: str = None, end_date: str = None) -> pd.Series:
        """
        Calculate spread between two stocks.
        
        Args:
            symbol1, symbol2: Stock symbols
            hedge_ratio: Hedge ratio (β)
            intercept: Intercept (α)
            start_date, end_date: Date range
            
        Returns:
            Spread series: symbol1 - hedge_ratio * symbol2 - intercept
        """
        try:
            # Get pair data
            data = self.data_api.get_pairs_data(symbol1, symbol2, start_date, end_date)
            
            if data.empty:
                return pd.Series(dtype=float)
            
            # Calculate spread: y1 = α + β*y2 + ε
            # So spread = y1 - β*y2 - α
            spread = data[symbol1] - hedge_ratio * data[symbol2] - intercept
            
            return spread
            
        except Exception as e:
            logger.error(f"Error calculating spread for {symbol1}-{symbol2}: {e}")
            return pd.Series(dtype=float)
    
    def generate_signals(self, z_score: pd.Series,
                        entry_threshold: float = None,
                        exit_threshold: float = None,
                        stop_loss_threshold: float = None,
                        current_position: PositionSide = PositionSide.FLAT) -> pd.Series:
        """
        Generate trading signals based on z-score.
        
        Args:
            z_score: Z-score time series
            entry_threshold: Entry threshold (default from config)
            exit_threshold: Exit threshold (default from config)
            stop_loss_threshold: Stop loss threshold (default from config)
            current_position: Current position state
            
        Returns:
            Series of signals
        """
        try:
            if z_score.empty:
                return pd.Series(dtype=object)
            
            # Use config defaults if not specified
            entry_threshold = entry_threshold or self.config.get('entry_z_score', 2.0)
            exit_threshold = exit_threshold or self.config.get('exit_z_score', 0.5)
            stop_loss_threshold = stop_loss_threshold or self.config.get('stop_loss_z_score', 3.0)
            
            signals = pd.Series(SignalType.NO_SIGNAL, index=z_score.index)
            
            for i, (date, z_val) in enumerate(z_score.items()):
                if pd.isna(z_val):
                    continue
                
                # Determine signal based on z-score and current position
                if current_position == PositionSide.FLAT:
                    # No position - look for entry signals
                    if z_val <= -entry_threshold:
                        signals[date] = SignalType.ENTRY_LONG
                    elif z_val >= entry_threshold:
                        signals[date] = SignalType.ENTRY_SHORT
                
                elif current_position == PositionSide.LONG:
                    # Long position - look for exit or stop loss
                    if abs(z_val) <= exit_threshold:
                        signals[date] = SignalType.EXIT_LONG
                    elif z_val >= stop_loss_threshold:
                        signals[date] = SignalType.STOP_LOSS
                
                elif current_position == PositionSide.SHORT:
                    # Short position - look for exit or stop loss
                    if abs(z_val) <= exit_threshold:
                        signals[date] = SignalType.EXIT_SHORT
                    elif z_val <= -stop_loss_threshold:
                        signals[date] = SignalType.STOP_LOSS
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return pd.Series(dtype=object)
    
    def generate_pair_signals(self, symbol1: str, symbol2: str,
                             hedge_ratio: float, intercept: float = 0,
                             start_date: str = None, end_date: str = None,
                             current_position: PositionSide = PositionSide.FLAT) -> pd.DataFrame:
        """
        Generate complete signal history for a pair.
        
        Args:
            symbol1, symbol2: Stock symbols
            hedge_ratio: Hedge ratio
            intercept: Intercept
            start_date, end_date: Date range
            current_position: Current position
            
        Returns:
            DataFrame with prices, spread, z-score, and signals
        """
        try:
            # Get pair data
            data = self.data_api.get_pairs_data(symbol1, symbol2, start_date, end_date)
            
            if data.empty:
                return pd.DataFrame()
            
            # Calculate spread
            spread = data[symbol1] - hedge_ratio * data[symbol2] - intercept
            
            # Calculate z-score
            z_score = self.calculate_z_score(spread)
            
            # Generate signals
            signals = self.generate_signals(z_score, current_position=current_position)
            
            # Combine results
            result = pd.DataFrame({
                f'{symbol1}_price': data[symbol1],
                f'{symbol2}_price': data[symbol2],
                'spread': spread,
                'z_score': z_score,
                'signal': signals
            })
            
            # Add metadata
            result.attrs = {
                'symbol1': symbol1,
                'symbol2': symbol2,
                'hedge_ratio': hedge_ratio,
                'intercept': intercept
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error generating pair signals for {symbol1}-{symbol2}: {e}")
            return pd.DataFrame()
    
    def get_current_signals(self, pairs_data: List[Dict[str, Any]],
                           lookback_days: int = 252) -> List[Dict[str, Any]]:
        """
        Get current signals for multiple pairs.
        
        Args:
            pairs_data: List of pair dictionaries with cointegration results
            lookback_days: Days to look back for signal calculation
            
        Returns:
            List of current signals for each pair
        """
        try:
            signals = []
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=lookback_days)).strftime('%Y-%m-%d')
            
            for pair in pairs_data:
                try:
                    symbol1 = pair['symbol1']
                    symbol2 = pair['symbol2']
                    hedge_ratio = pair['hedge_ratio']
                    intercept = pair.get('intercept', 0)
                    
                    # Generate signals
                    signal_data = self.generate_pair_signals(
                        symbol1, symbol2, hedge_ratio, intercept,
                        start_date, end_date
                    )
                    
                    if signal_data.empty:
                        continue
                    
                    # Get latest signal info
                    latest_date = signal_data.index[-1]
                    latest_signal = signal_data['signal'].iloc[-1]
                    latest_z_score = signal_data['z_score'].iloc[-1]
                    latest_spread = signal_data['spread'].iloc[-1]
                    
                    signal_info = {
                        'symbol1': symbol1,
                        'symbol2': symbol2,
                        'date': latest_date,
                        'signal': latest_signal,
                        'z_score': latest_z_score,
                        'spread': latest_spread,
                        'price1': signal_data[f'{symbol1}_price'].iloc[-1],
                        'price2': signal_data[f'{symbol2}_price'].iloc[-1],
                        'hedge_ratio': hedge_ratio,
                        'intercept': intercept,
                        'pair_id': f"{symbol1}-{symbol2}"
                    }
                    
                    signals.append(signal_info)
                    
                except Exception as e:
                    logger.error(f"Error getting signal for pair {pair.get('symbol1', 'unknown')}-{pair.get('symbol2', 'unknown')}: {e}")
                    continue
            
            logger.info(f"Generated current signals for {len(signals)} pairs")
            return signals
            
        except Exception as e:
            logger.error(f"Error getting current signals: {e}")
            return []
    
    def filter_actionable_signals(self, signals: List[Dict[str, Any]],
                                 active_positions: Dict[str, Dict] = None) -> List[Dict[str, Any]]:
        """
        Filter signals to only actionable ones based on current positions.
        
        Args:
            signals: List of signal dictionaries
            active_positions: Dictionary of currently active positions
            
        Returns:
            List of actionable signals
        """
        try:
            if active_positions is None:
                active_positions = {}
            
            actionable_signals = []
            
            for signal in signals:
                pair_id = signal['pair_id']
                signal_type = signal['signal']
                
                # Skip if no signal
                if signal_type == SignalType.NO_SIGNAL:
                    continue
                
                # Check if signal is actionable based on current position
                current_position = active_positions.get(pair_id, {}).get('side', PositionSide.FLAT)
                
                is_actionable = False
                
                if current_position == PositionSide.FLAT:
                    # Can enter new positions
                    if signal_type in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT]:
                        is_actionable = True
                
                elif current_position == PositionSide.LONG:
                    # Can exit long positions
                    if signal_type in [SignalType.EXIT_LONG, SignalType.STOP_LOSS]:
                        is_actionable = True
                
                elif current_position == PositionSide.SHORT:
                    # Can exit short positions
                    if signal_type in [SignalType.EXIT_SHORT, SignalType.STOP_LOSS]:
                        is_actionable = True
                
                if is_actionable:
                    signal['current_position'] = current_position
                    actionable_signals.append(signal)
            
            logger.info(f"Found {len(actionable_signals)} actionable signals out of {len(signals)}")
            return actionable_signals
            
        except Exception as e:
            logger.error(f"Error filtering actionable signals: {e}")
            return []
    
    def calculate_signal_strength(self, z_score: float,
                                 entry_threshold: float = None,
                                 stop_loss_threshold: float = None) -> float:
        """
        Calculate signal strength as a percentage.
        
        Args:
            z_score: Current z-score
            entry_threshold: Entry threshold
            stop_loss_threshold: Stop loss threshold
            
        Returns:
            Signal strength (0-100%)
        """
        try:
            entry_threshold = entry_threshold or self.config.get('entry_z_score', 2.0)
            stop_loss_threshold = stop_loss_threshold or self.config.get('stop_loss_z_score', 3.0)
            
            abs_z = abs(z_score)
            
            if abs_z < entry_threshold:
                # No signal zone
                return 0.0
            
            elif abs_z >= stop_loss_threshold:
                # Maximum signal strength
                return 100.0
            
            else:
                # Scale between entry and stop loss thresholds
                strength = (abs_z - entry_threshold) / (stop_loss_threshold - entry_threshold) * 100
                return min(100.0, max(0.0, strength))
            
        except Exception as e:
            logger.error(f"Error calculating signal strength: {e}")
            return 0.0
    
    def backtest_signals(self, symbol1: str, symbol2: str,
                        hedge_ratio: float, intercept: float = 0,
                        start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """
        Backtest signals for a pair to evaluate performance.
        
        Args:
            symbol1, symbol2: Stock symbols
            hedge_ratio: Hedge ratio
            intercept: Intercept
            start_date, end_date: Date range
            
        Returns:
            Dictionary with backtest results
        """
        try:
            # Generate signal history
            signal_data = self.generate_pair_signals(
                symbol1, symbol2, hedge_ratio, intercept,
                start_date, end_date
            )
            
            if signal_data.empty:
                return {}
            
            # Simulate trading based on signals
            trades = []
            position = PositionSide.FLAT
            entry_date = None
            entry_z_score = None
            entry_spread = None
            
            for date, row in signal_data.iterrows():
                signal = row['signal']
                z_score = row['z_score']
                spread = row['spread']
                
                if position == PositionSide.FLAT and signal in [SignalType.ENTRY_LONG, SignalType.ENTRY_SHORT]:
                    # Enter position
                    position = PositionSide.LONG if signal == SignalType.ENTRY_LONG else PositionSide.SHORT
                    entry_date = date
                    entry_z_score = z_score
                    entry_spread = spread
                
                elif position != PositionSide.FLAT and signal in [SignalType.EXIT_LONG, SignalType.EXIT_SHORT, SignalType.STOP_LOSS]:
                    # Exit position
                    if entry_date is not None:
                        # Calculate trade return
                        if position == PositionSide.LONG:
                            trade_return = entry_spread - spread  # Long spread profits when spread decreases
                        else:
                            trade_return = spread - entry_spread  # Short spread profits when spread increases
                        
                        trade = {
                            'entry_date': entry_date,
                            'exit_date': date,
                            'position': position,
                            'entry_z_score': entry_z_score,
                            'exit_z_score': z_score,
                            'entry_spread': entry_spread,
                            'exit_spread': spread,
                            'trade_return': trade_return,
                            'exit_reason': signal,
                            'holding_days': (date - entry_date).days
                        }
                        trades.append(trade)
                    
                    # Reset position
                    position = PositionSide.FLAT
                    entry_date = None
                    entry_z_score = None
                    entry_spread = None
            
            # Calculate performance metrics
            if not trades:
                return {'trades': [], 'performance': {}}
            
            trades_df = pd.DataFrame(trades)
            
            performance = {
                'total_trades': len(trades),
                'winning_trades': len(trades_df[trades_df['trade_return'] > 0]),
                'losing_trades': len(trades_df[trades_df['trade_return'] < 0]),
                'win_rate': len(trades_df[trades_df['trade_return'] > 0]) / len(trades) * 100,
                'total_return': trades_df['trade_return'].sum(),
                'avg_return_per_trade': trades_df['trade_return'].mean(),
                'best_trade': trades_df['trade_return'].max(),
                'worst_trade': trades_df['trade_return'].min(),
                'avg_holding_days': trades_df['holding_days'].mean(),
                'return_std': trades_df['trade_return'].std(),
                'sharpe_ratio': trades_df['trade_return'].mean() / trades_df['trade_return'].std() if trades_df['trade_return'].std() > 0 else 0
            }
            
            return {
                'trades': trades,
                'performance': performance,
                'signal_data': signal_data
            }
            
        except Exception as e:
            logger.error(f"Error backtesting signals for {symbol1}-{symbol2}: {e}")
            return {}