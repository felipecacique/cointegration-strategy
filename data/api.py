"""
Market Data API - Unified interface for data access.
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

from data.storage import DataStorageManager
from data.collector import DataCollector
from simple_logger import logger

class MarketDataAPI:
    """Unified API for market data access."""
    
    def __init__(self):
        self.storage = DataStorageManager()
        self.collector = DataCollector(self.storage)
    
    def get_price_data(self, symbols: List[str], 
                      start_date: str = None, 
                      end_date: str = None,
                      adjust_prices: bool = True) -> pd.DataFrame:
        """Get price data for multiple symbols."""
        try:
            if isinstance(symbols, str):
                symbols = [symbols]
            
            data_dict = {}
            
            for symbol in symbols:
                df = self.storage.get_price_data(symbol, start_date, end_date)
                
                if not df.empty:
                    # Use adjusted close if requested
                    price_col = 'adj_close' if adjust_prices and 'adj_close' in df.columns else 'close'
                    data_dict[symbol] = df[price_col]
                else:
                    # Debug: check if symbol has any data
                    if start_date or end_date:
                        all_data = self.storage.get_price_data(symbol)
                        if not all_data.empty:
                            date_range = self.storage.get_date_range(symbol)
                            logger.debug(f"Symbol {symbol} has data from {date_range.get('min_date')} to {date_range.get('max_date')}, but none in period {start_date} to {end_date}")
                        else:
                            logger.debug(f"Symbol {symbol} has no data at all in database")
            
            if not data_dict:
                # Check if this is a backtest/bulk analysis context
                missing_symbols = [s for s in symbols if s not in data_dict]
                if len(missing_symbols) <= 5:  # Only log if few symbols missing
                    logger.debug(f"No data found for symbols: {missing_symbols}")
                else:
                    logger.warning(f"No data found for {len(missing_symbols)} symbols")
                return pd.DataFrame()
            
            # Combine into single DataFrame
            result = pd.DataFrame(data_dict)
            result = result.dropna(how='all')  # Remove rows where all symbols are NaN
            
            logger.debug(f"Retrieved price data for {len(symbols)} symbols, {len(result)} records")
            return result
            
        except Exception as e:
            logger.error(f"Error getting price data: {e}")
            return pd.DataFrame()
    
    def get_pairs_data(self, symbol1: str, symbol2: str,
                      start_date: str = None, end_date: str = None,
                      adjust_prices: bool = True) -> pd.DataFrame:
        """Get synchronized price data for a pair of symbols."""
        try:
            data = self.get_price_data([symbol1, symbol2], start_date, end_date, adjust_prices)
            
            if data.empty or len(data.columns) < 2:
                return pd.DataFrame()
            
            # Drop rows with any NaN values for pairs analysis
            data = data.dropna()
            
            if len(data) < 30:  # Minimum data points for analysis
                logger.warning(f"Insufficient data for pair {symbol1}-{symbol2}: {len(data)} points")
                return pd.DataFrame()
            
            return data
            
        except Exception as e:
            logger.error(f"Error getting pairs data for {symbol1}-{symbol2}: {e}")
            return pd.DataFrame()
    
    def get_returns_data(self, symbols: List[str],
                        start_date: str = None, end_date: str = None,
                        period: str = 'daily') -> pd.DataFrame:
        """Get returns data for symbols."""
        try:
            prices = self.get_price_data(symbols, start_date, end_date)
            
            if prices.empty:
                return pd.DataFrame()
            
            if period == 'daily':
                returns = prices.pct_change().dropna()
            elif period == 'weekly':
                returns = prices.resample('W').last().pct_change().dropna()
            elif period == 'monthly':
                returns = prices.resample('M').last().pct_change().dropna()
            else:
                raise ValueError(f"Unsupported period: {period}")
            
            return returns
            
        except Exception as e:
            logger.error(f"Error calculating returns: {e}")
            return pd.DataFrame()
    
    def get_volume_data(self, symbols: List[str],
                       start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get volume data for symbols."""
        try:
            if isinstance(symbols, str):
                symbols = [symbols]
            
            data_dict = {}
            
            for symbol in symbols:
                df = self.storage.get_price_data(symbol, start_date, end_date)
                if not df.empty and 'volume' in df.columns:
                    data_dict[symbol] = df['volume']
            
            if not data_dict:
                return pd.DataFrame()
            
            result = pd.DataFrame(data_dict)
            return result.dropna(how='all')
            
        except Exception as e:
            logger.error(f"Error getting volume data: {e}")
            return pd.DataFrame()
    
    def get_available_symbols(self, min_data_points: int = 252) -> List[str]:
        """Get list of symbols with sufficient data."""
        try:
            all_symbols = self.storage.get_available_symbols()
            valid_symbols = []
            
            for symbol in all_symbols:
                df = self.storage.get_price_data(symbol)
                if len(df) >= min_data_points:
                    valid_symbols.append(symbol)
            
            logger.info(f"Found {len(valid_symbols)} symbols with >= {min_data_points} data points")
            return valid_symbols
            
        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            return []
    
    def get_universe_data(self, universe_symbols: List[str],
                         start_date: str = None, end_date: str = None,
                         min_data_points: int = 252) -> pd.DataFrame:
        """Get price data for a universe of symbols with filtering."""
        try:
            # Filter symbols with sufficient data
            valid_symbols = []
            
            for symbol in universe_symbols:
                df = self.storage.get_price_data(symbol, start_date, end_date)
                if len(df) >= min_data_points:
                    valid_symbols.append(symbol)
            
            if not valid_symbols:
                logger.warning("No symbols with sufficient data found")
                return pd.DataFrame()
            
            # Get price data
            data = self.get_price_data(valid_symbols, start_date, end_date)
            
            # Further filter by data completeness
            if not data.empty:
                # Keep symbols with at least 80% data completeness
                completeness = data.count() / len(data)
                valid_cols = completeness[completeness >= 0.8].index.tolist()
                data = data[valid_cols]
            
            logger.info(f"Universe data: {len(data)} records for {len(data.columns)} symbols")
            return data
            
        except Exception as e:
            logger.error(f"Error getting universe data: {e}")
            return pd.DataFrame()
    
    def calculate_basic_stats(self, symbol: str,
                             start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Calculate basic statistics for a symbol."""
        try:
            df = self.storage.get_price_data(symbol, start_date, end_date)
            
            if df.empty:
                return {}
            
            prices = df['adj_close'] if 'adj_close' in df.columns else df['close']
            returns = prices.pct_change().dropna()
            
            stats = {
                'symbol': symbol,
                'data_points': len(df),
                'start_date': df.index.min().strftime('%Y-%m-%d'),
                'end_date': df.index.max().strftime('%Y-%m-%d'),
                'current_price': float(prices.iloc[-1]),
                'min_price': float(prices.min()),
                'max_price': float(prices.max()),
                'avg_price': float(prices.mean()),
                'volatility': float(returns.std() * np.sqrt(252)),  # Annualized
                'avg_return': float(returns.mean() * 252),  # Annualized
                'avg_volume': float(df['volume'].mean()) if 'volume' in df.columns else 0,
                'total_return': float((prices.iloc[-1] / prices.iloc[0]) - 1),
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating stats for {symbol}: {e}")
            return {}
    
    def get_correlation_matrix(self, symbols: List[str],
                              start_date: str = None, end_date: str = None,
                              method: str = 'pearson') -> pd.DataFrame:
        """Calculate correlation matrix for symbols."""
        try:
            returns = self.get_returns_data(symbols, start_date, end_date)
            
            if returns.empty:
                return pd.DataFrame()
            
            correlation_matrix = returns.corr(method=method)
            return correlation_matrix
            
        except Exception as e:
            logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
    
    def update_data(self, symbols: List[str] = None, force_full_update: bool = False) -> Dict[str, Any]:
        """Update data for symbols."""
        try:
            if force_full_update:
                # Full historical update
                if symbols is None:
                    from config.universe import get_universe_tickers
                    symbols = get_universe_tickers('IBOV')
                
                results = self.collector.download_historical_data(symbols)
            else:
                # Incremental daily update
                results = self.collector.update_daily_data(symbols)
            
            return {
                'status': 'success',
                'symbols_updated': len([r for r in results.values() if r > 0]),
                'total_records': sum(results.values()),
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error updating data: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'symbols_updated': 0,
                'total_records': 0
            }
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of available data."""
        try:
            stats = self.storage.get_database_stats()
            symbols = self.storage.get_available_symbols()
            date_range = self.storage.get_date_range()
            
            summary = {
                'total_symbols': len(symbols),
                'total_records': stats.get('daily_prices_count', 0),
                'date_range': date_range,
                'database_size_mb': stats.get('db_size_mb', 0),
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'symbols_with_recent_data': len([
                    s for s in symbols 
                    if len(self.storage.get_price_data(s, 
                        (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))) > 0
                ])
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return {}
    
    def initialize_system(self, universe: str = 'IBOV') -> Dict[str, Any]:
        """Initialize the system with data for a universe."""
        try:
            logger.info(f"Initializing system with {universe} universe")
            
            results = self.collector.initialize_universe_data(universe)
            
            return {
                'status': 'success',
                'universe': universe,
                'initialization_results': results,
                'data_summary': self.get_data_summary()
            }
            
        except Exception as e:
            logger.error(f"Error initializing system: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }