"""
Position management for backtesting.
Handles opening, closing, and tracking of trading positions.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from strategy.signals import PositionSide
from config.settings import CONFIG
from simple_logger import logger

class PositionStatus(Enum):
    """Position status enumeration."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    PARTIAL = "PARTIAL"

class Position:
    """Represents a single pairs trading position."""
    
    def __init__(self, pair_id: str, symbol1: str, symbol2: str, 
                 side: PositionSide, quantity1: float, quantity2: float,
                 price1: float, price2: float, hedge_ratio: float,
                 entry_date: str, capital_allocated: float):
        self.pair_id = pair_id
        self.symbol1 = symbol1
        self.symbol2 = symbol2
        self.side = side
        self.quantity1 = quantity1  # Can be negative for short positions
        self.quantity2 = quantity2  # Can be negative for short positions
        self.entry_price1 = price1
        self.entry_price2 = price2
        self.hedge_ratio = hedge_ratio
        self.entry_date = pd.to_datetime(entry_date)
        self.capital_allocated = capital_allocated
        self.status = PositionStatus.OPEN
        self.exit_date = None
        self.exit_price1 = None
        self.exit_price2 = None
        self.pnl = 0.0
        self.commission_paid = 0.0
        self.missing_data_days = 0  # Track consecutive days without data
        self.last_monitoring_date = pd.to_datetime(entry_date)  # Track last successful monitoring
        
    def calculate_current_value(self, current_price1: float, current_price2: float) -> float:
        """Calculate current position value."""
        try:
            # Calculate P&L for each leg
            pnl1 = self.quantity1 * (current_price1 - self.entry_price1)
            pnl2 = self.quantity2 * (current_price2 - self.entry_price2)
            
            return pnl1 + pnl2
            
        except Exception as e:
            logger.error(f"Error calculating position value: {e}")
            return 0.0
    
    def close_position(self, exit_price1: float, exit_price2: float, 
                      exit_date: str, exit_reason: str = 'MANUAL') -> Dict[str, Any]:
        """Close the position and calculate final P&L."""
        try:
            self.exit_price1 = exit_price1
            self.exit_price2 = exit_price2
            self.exit_date = pd.to_datetime(exit_date)
            self.status = PositionStatus.CLOSED
            
            # Calculate final P&L
            pnl1 = self.quantity1 * (exit_price1 - self.entry_price1)
            pnl2 = self.quantity2 * (exit_price2 - self.entry_price2)
            self.pnl = pnl1 + pnl2
            
            # Calculate commissions
            commission_rate = CONFIG['trading'].get('commission_rate', 0.003)
            entry_value = abs(self.quantity1 * self.entry_price1) + abs(self.quantity2 * self.entry_price2)
            exit_value = abs(self.quantity1 * exit_price1) + abs(self.quantity2 * exit_price2)
            self.commission_paid = (entry_value + exit_value) * commission_rate
            
            # Net P&L after commissions
            net_pnl = self.pnl - self.commission_paid
            
            # Calculate holding period
            holding_days = (self.exit_date - self.entry_date).days
            
            trade_result = {
                'pair_id': self.pair_id,
                'symbol1': self.symbol1,
                'symbol2': self.symbol2,
                'side': self.side.value,
                'entry_date': self.entry_date,
                'exit_date': self.exit_date,
                'holding_days': holding_days,
                'entry_price1': self.entry_price1,
                'entry_price2': self.entry_price2,
                'exit_price1': self.exit_price1,
                'exit_price2': self.exit_price2,
                'quantity1': self.quantity1,
                'quantity2': self.quantity2,
                'hedge_ratio': self.hedge_ratio,
                'gross_pnl': self.pnl,
                'commission': self.commission_paid,
                'pnl': net_pnl,
                'capital_allocated': self.capital_allocated,
                'return_pct': net_pnl / self.capital_allocated * 100 if self.capital_allocated > 0 else 0,
                'exit_reason': exit_reason
            }
            
            return trade_result
            
        except Exception as e:
            logger.error(f"Error closing position {self.pair_id}: {e}")
            return {}
    
    def should_force_close_missing_data(self, max_missing_days: int) -> bool:
        """Check if position should be force-closed due to missing data."""
        return self.missing_data_days >= max_missing_days
    
    def should_force_close_timeout(self, current_date: str, max_holding_days: int) -> bool:
        """Check if position should be force-closed due to timeout."""
        current_dt = pd.to_datetime(current_date)
        holding_days = (current_dt - self.entry_date).days
        return holding_days >= max_holding_days
    
    def update_monitoring_status(self, current_date: str, has_data: bool):
        """Update position monitoring status."""
        if has_data:
            self.missing_data_days = 0  # Reset counter
            self.last_monitoring_date = pd.to_datetime(current_date)
        else:
            self.missing_data_days += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        return {
            'pair_id': self.pair_id,
            'symbol1': self.symbol1,
            'symbol2': self.symbol2,
            'side': self.side.value,
            'quantity1': self.quantity1,
            'quantity2': self.quantity2,
            'entry_price1': self.entry_price1,
            'entry_price2': self.entry_price2,
            'hedge_ratio': self.hedge_ratio,
            'entry_date': self.entry_date,
            'capital_allocated': self.capital_allocated,
            'status': self.status.value,
            'exit_date': self.exit_date,
            'exit_price1': self.exit_price1,
            'exit_price2': self.exit_price2,
            'pnl': self.pnl,
            'commission_paid': self.commission_paid
        }

class PositionManager:
    """Manages all trading positions for backtesting."""
    
    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.max_positions = CONFIG['trading'].get('max_active_pairs', 10)
        
    def open_position(self, pair_id: str, symbol1: str, symbol2: str,
                     side: PositionSide, capital_allocation: float,
                     price1: float, price2: float, hedge_ratio: float,
                     date: str) -> Optional[Dict[str, Any]]:
        """
        Open a new pairs trading position.
        
        Args:
            pair_id: Unique identifier for the pair
            symbol1, symbol2: Stock symbols
            side: LONG or SHORT (referring to the spread)
            capital_allocation: Amount of capital to allocate
            price1, price2: Current prices
            hedge_ratio: Hedge ratio between stocks
            date: Entry date
            
        Returns:
            Dictionary with trade information if successful
        """
        try:
            # Check if position already exists
            if pair_id in self.positions:
                logger.warning(f"Position already exists for {pair_id}")
                return None
            
            # Check maximum positions limit
            if len(self.positions) >= self.max_positions:
                logger.debug(f"Maximum positions limit reached ({self.max_positions})")
                return None
            
            # Check available cash
            if capital_allocation > self.cash:
                logger.debug(f"Insufficient cash for position {pair_id}: need {capital_allocation}, have {self.cash}")
                return None
            
            # Calculate position sizes
            # For pairs trading: buy stock1, sell hedge_ratio * stock2 (for LONG spread)
            # Total position value should equal capital_allocation
            
            if side == PositionSide.LONG:
                # Long spread: buy symbol1, short symbol2
                # position_value = price1 * qty1 + price2 * hedge_ratio * qty1
                # qty1 = capital_allocation / (price1 + price2 * hedge_ratio)
                
                total_cost_per_unit = price1 + price2 * hedge_ratio
                quantity1 = capital_allocation / total_cost_per_unit
                quantity2 = -quantity1 * hedge_ratio  # Negative = short
                
            else:  # SHORT spread
                # Short spread: short symbol1, buy symbol2
                total_cost_per_unit = price1 + price2 * hedge_ratio
                quantity1 = -capital_allocation / total_cost_per_unit  # Negative = short
                quantity2 = -quantity1 * hedge_ratio  # Positive = long
            
            # Create position
            position = Position(
                pair_id=pair_id,
                symbol1=symbol1,
                symbol2=symbol2,
                side=side,
                quantity1=quantity1,
                quantity2=quantity2,
                price1=price1,
                price2=price2,
                hedge_ratio=hedge_ratio,
                entry_date=date,
                capital_allocated=capital_allocation
            )
            
            # Update cash (reserve the allocated capital)
            self.cash -= capital_allocation
            
            # Store position
            self.positions[pair_id] = position
            
            logger.debug(f"Opened {side.value} position for {pair_id}: "
                        f"qty1={quantity1:.2f}, qty2={quantity2:.2f}, "
                        f"capital={capital_allocation:.2f}")
            
            # Return trade information
            return {
                'pair_id': pair_id,
                'symbol1': symbol1,
                'symbol2': symbol2,
                'side': side.value,
                'action': 'OPEN',
                'date': date,
                'quantity1': quantity1,
                'quantity2': quantity2,
                'price1': price1,
                'price2': price2,
                'capital_allocated': capital_allocation,
                'hedge_ratio': hedge_ratio
            }
            
        except Exception as e:
            logger.error(f"Error opening position {pair_id}: {e}")
            return None
    
    def close_position(self, pair_id: str, price1: float, price2: float,
                      date: str, exit_reason: str = 'MANUAL') -> Optional[Dict[str, Any]]:
        """Close an existing position."""
        try:
            if pair_id not in self.positions:
                logger.warning(f"Position {pair_id} not found")
                return None
            
            position = self.positions[pair_id]
            
            # Close the position
            trade_result = position.close_position(price1, price2, date, exit_reason)
            
            if trade_result:
                # Update cash (return allocated capital plus P&L)
                self.cash += position.capital_allocated + trade_result['pnl']
                
                # Move to closed positions
                self.closed_positions.append(position)
                del self.positions[pair_id]
                
                logger.debug(f"Closed position {pair_id}: P&L={trade_result['pnl']:.2f}, "
                           f"return={trade_result['return_pct']:.2f}%")
                
                return trade_result
            
            return None
            
        except Exception as e:
            logger.error(f"Error closing position {pair_id}: {e}")
            return None
    
    def update_positions(self, price_data: Dict[str, Dict[str, float]], 
                        current_date: str) -> Dict[str, float]:
        """
        Update all positions with current prices.
        
        Args:
            price_data: Dictionary with symbol -> price mapping
            current_date: Current date
            
        Returns:
            Dictionary with unrealized P&L for each position
        """
        try:
            unrealized_pnl = {}
            
            for pair_id, position in self.positions.items():
                try:
                    if position.symbol1 in price_data and position.symbol2 in price_data:
                        current_price1 = price_data[position.symbol1]
                        current_price2 = price_data[position.symbol2]
                        
                        current_value = position.calculate_current_value(current_price1, current_price2)
                        unrealized_pnl[pair_id] = current_value
                    
                except Exception as e:
                    logger.error(f"Error updating position {pair_id}: {e}")
                    unrealized_pnl[pair_id] = 0.0
            
            return unrealized_pnl
            
        except Exception as e:
            logger.error(f"Error updating positions: {e}")
            return {}
    
    def get_position(self, pair_id: str) -> Dict[str, Any]:
        """Get information about a specific position."""
        if pair_id in self.positions:
            position = self.positions[pair_id]
            return {
                'pair_id': pair_id,
                'side': position.side,
                'status': position.status,
                'entry_date': position.entry_date,
                'capital_allocated': position.capital_allocated,
                'symbol1': position.symbol1,
                'symbol2': position.symbol2,
                'quantity1': position.quantity1,
                'quantity2': position.quantity2,
                'hedge_ratio': position.hedge_ratio
            }
        return {}
    
    def get_portfolio_value(self, current_prices: Dict[str, float] = None) -> float:
        """
        Calculate total portfolio value.
        
        Args:
            current_prices: Optional current prices for position valuation
            
        Returns:
            Total portfolio value
        """
        try:
            portfolio_value = self.cash
            
            if current_prices:
                # Add unrealized P&L from open positions
                for position in self.positions.values():
                    if position.symbol1 in current_prices and position.symbol2 in current_prices:
                        current_value = position.calculate_current_value(
                            current_prices[position.symbol1],
                            current_prices[position.symbol2]
                        )
                        portfolio_value += current_value
                    
                    # Add allocated capital (since we reserved it when opening position)
                    portfolio_value += position.capital_allocated
            else:
                # Just add allocated capital if no current prices provided
                for position in self.positions.values():
                    portfolio_value += position.capital_allocated
            
            return portfolio_value
            
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
            return self.cash
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get summary of portfolio status."""
        try:
            total_allocated = sum(pos.capital_allocated for pos in self.positions.values())
            
            summary = {
                'total_value': self.get_portfolio_value(),
                'cash': self.cash,
                'allocated_capital': total_allocated,
                'free_cash': self.cash,
                'active_positions': len(self.positions),
                'closed_positions': len(self.closed_positions),
                'max_positions': self.max_positions,
                'capital_utilization': total_allocated / self.initial_capital * 100,
                'position_list': list(self.positions.keys())
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting portfolio summary: {e}")
            return {}
    
    def get_positions_requiring_update(self, symbols_to_check: List[str]) -> List[str]:
        """Get list of positions that need price updates."""
        positions_to_update = []
        
        for pair_id, position in self.positions.items():
            if position.symbol1 in symbols_to_check or position.symbol2 in symbols_to_check:
                positions_to_update.append(pair_id)
        
        return positions_to_update
    
    def check_force_close_conditions(self, current_date: str) -> List[str]:
        """Check which positions should be force-closed."""
        from config.settings import CONFIG
        
        positions_to_close = []
        max_missing_days = CONFIG['trading'].get('force_close_missing_days', 14)
        max_holding_days = CONFIG['trading'].get('max_holding_period', 90)
        
        for pair_id, position in self.positions.items():
            # Check missing data condition
            if position.should_force_close_missing_data(max_missing_days):
                positions_to_close.append((pair_id, 'MISSING_DATA'))
                logger.warning(f"Force closing {pair_id}: {position.missing_data_days} consecutive days without data")
            
            # Check timeout condition
            elif position.should_force_close_timeout(current_date, max_holding_days):
                positions_to_close.append((pair_id, 'TIMEOUT'))
                holding_days = (pd.to_datetime(current_date) - position.entry_date).days
                logger.warning(f"Force closing {pair_id}: {holding_days} days holding period exceeded")
        
        return positions_to_close
    
    def force_close_positions(self, positions_to_close: List[Tuple[str, str]], 
                            current_prices: Dict[str, float], date: str) -> List[Dict[str, Any]]:
        """Force close specific positions."""
        closed_trades = []
        
        for pair_id, reason in positions_to_close:
            if pair_id in self.positions:
                position = self.positions[pair_id]
                
                # Get current prices or use last known prices
                price1 = current_prices.get(position.symbol1, position.entry_price1)
                price2 = current_prices.get(position.symbol2, position.entry_price2)
                
                trade_result = self.close_position(pair_id, price1, price2, date, reason)
                if trade_result:
                    closed_trades.append(trade_result)
                    logger.info(f"Force closed position {pair_id} due to {reason}")
        
        return closed_trades
    
    def close_all_positions(self, current_prices: Dict[str, float], 
                           date: str, reason: str = 'FORCE_CLOSE') -> List[Dict[str, Any]]:
        """Close all open positions."""
        try:
            closed_trades = []
            
            positions_to_close = list(self.positions.keys())
            
            for pair_id in positions_to_close:
                position = self.positions[pair_id]
                
                if (position.symbol1 in current_prices and 
                    position.symbol2 in current_prices):
                    
                    trade_result = self.close_position(
                        pair_id,
                        current_prices[position.symbol1],
                        current_prices[position.symbol2],
                        date,
                        reason
                    )
                    
                    if trade_result:
                        closed_trades.append(trade_result)
            
            logger.info(f"Force closed {len(closed_trades)} positions")
            return closed_trades
            
        except Exception as e:
            logger.error(f"Error closing all positions: {e}")
            return []
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get complete trade history."""
        trades = []
        
        for position in self.closed_positions:
            if position.status == PositionStatus.CLOSED:
                trades.append(position.to_dict())
        
        return trades
    
    def reset(self):
        """Reset position manager to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.closed_positions.clear()