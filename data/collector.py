"""
Data collection module for pairs trading system.
Handles downloading and updating stock data from Yahoo Finance.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
from typing import List, Dict, Optional, Tuple, Any
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.settings import CONFIG
from config.universe import get_universe_tickers
from data.storage import DataStorageManager
from simple_logger import logger

warnings.filterwarnings('ignore')

class DataCollector:
    """Collects stock data from Yahoo Finance."""
    
    def __init__(self, storage_manager: DataStorageManager = None):
        self.storage = storage_manager or DataStorageManager()
        self.rate_limit_delay = CONFIG['api']['rate_limit_delay']
        self.max_retries = CONFIG['api']['max_retries']
        self.timeout = CONFIG['api']['timeout']
    
    def _rate_limit(self):
        """Apply rate limiting between API calls."""
        time.sleep(self.rate_limit_delay)
    
    def _download_single_stock(self, symbol: str, start_date: str, 
                              end_date: str, retries: int = 0) -> Tuple[str, pd.DataFrame]:
        """Download data for a single stock with retry logic."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                auto_adjust=False,
                back_adjust=False,
                actions=True
            )
            
            if data.empty:
                logger.warning(f"No data returned for {symbol}")
                return symbol, pd.DataFrame()
            
            # Clean data
            data = self._clean_price_data(data)
            self._rate_limit()
            
            logger.debug(f"Downloaded {len(data)} records for {symbol}")
            return symbol, data
            
        except Exception as e:
            if retries < self.max_retries and ("401" not in str(e) and "404" not in str(e)):
                logger.warning(f"Retry {retries + 1} for {symbol}: {e}")
                time.sleep(2 ** retries)  # Exponential backoff
                return self._download_single_stock(symbol, start_date, end_date, retries + 1)
            else:
                if "401" in str(e):
                    logger.debug(f"Rate limited for {symbol}, continuing...")
                elif "404" in str(e):
                    logger.debug(f"Symbol {symbol} not found, skipping...")
                else:
                    logger.error(f"Failed to download {symbol} after {self.max_retries} retries: {e}")
                return symbol, pd.DataFrame()
    
    def _clean_price_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate price data."""
        if data.empty:
            return data
        
        # Remove invalid data
        data = data.dropna(subset=['Close'])
        
        # Remove zero or negative prices
        price_cols = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        for col in price_cols:
            if col in data.columns:
                data = data[data[col] > 0]
        
        # Check for obvious errors (High < Low, etc.)
        if 'High' in data.columns and 'Low' in data.columns:
            invalid_rows = data['High'] < data['Low']
            if invalid_rows.any():
                logger.warning(f"Found {invalid_rows.sum()} rows with High < Low, removing")
                data = data[~invalid_rows]
        
        # Sort by date
        data = data.sort_index()
        
        return data
    
    def download_historical_data(self, symbols: List[str], 
                               start_date: str = None, 
                               end_date: str = None,
                               max_workers: int = 5) -> Dict[str, int]:
        """Download historical data for multiple symbols."""
        
        if not start_date:
            start_date = (datetime.now() - timedelta(days=CONFIG['data']['lookback_days'])).strftime('%Y-%m-%d')
        
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        logger.info(f"Downloading data for {len(symbols)} symbols from {start_date} to {end_date}")
        
        results = {}
        successful_downloads = 0
        failed_downloads = 0
        
        # Use ThreadPoolExecutor for parallel downloads
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_symbol = {
                executor.submit(self._download_single_stock, symbol, start_date, end_date): symbol
                for symbol in symbols
            }
            
            # Process completed downloads
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    symbol_result, data = future.result()
                    
                    if not data.empty:
                        # Store in database
                        records_inserted = self.storage.insert_stock_data(symbol_result, data)
                        results[symbol_result] = records_inserted
                        successful_downloads += 1
                        
                        # Log stock info
                        self._update_stock_info(symbol_result, data)
                        
                    else:
                        results[symbol_result] = 0
                        failed_downloads += 1
                        self.storage.log_data_quality_issue(
                            symbol_result, 'NO_DATA', 
                            f'No data available for period {start_date} to {end_date}'
                        )
                
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    results[symbol] = 0
                    failed_downloads += 1
        
        # Update process status
        self.storage.update_process_status(
            'historical_download',
            'completed',
            sum(results.values()),
            f"Success: {successful_downloads}, Failed: {failed_downloads}"
        )
        
        logger.info(f"Download complete. Success: {successful_downloads}, Failed: {failed_downloads}")
        return results
    
    def _update_stock_info(self, symbol: str, data: pd.DataFrame):
        """Update stock master information."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}
            
            stock_info = {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'currency': info.get('currency', 'BRL'),
                'exchange': info.get('exchange', 'SAO')
            }
            
            self.storage.update_stock_master([stock_info])
            
        except Exception as e:
            logger.debug(f"Could not update stock info for {symbol}: {e}")
    
    def update_daily_data(self, symbols: List[str] = None) -> Dict[str, int]:
        """Update daily data for symbols (incremental update)."""
        
        if symbols is None:
            symbols = self.storage.get_available_symbols()
            if not symbols:
                logger.warning("No symbols in database for daily update")
                return {}
        
        logger.info(f"Starting daily update for {len(symbols)} symbols")
        
        # Get last available date for each symbol
        results = {}
        
        for symbol in symbols:
            try:
                # Get last date in database
                date_range = self.storage.get_date_range(symbol)
                last_date = date_range.get('max_date')
                
                if last_date:
                    # Start from day after last date
                    start_date = (pd.to_datetime(last_date) + timedelta(days=1)).strftime('%Y-%m-%d')
                else:
                    # No data exists, get last 30 days
                    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                
                end_date = datetime.now().strftime('%Y-%m-%d')
                
                # Skip if start_date is today or later
                if pd.to_datetime(start_date) >= pd.to_datetime(end_date):
                    logger.debug(f"No update needed for {symbol}, already up to date")
                    results[symbol] = 0
                    continue
                
                # Download and store data
                symbol_result, data = self._download_single_stock(symbol, start_date, end_date)
                
                if not data.empty:
                    records_inserted = self.storage.insert_stock_data(symbol_result, data)
                    results[symbol_result] = records_inserted
                    logger.debug(f"Updated {symbol_result} with {records_inserted} new records")
                else:
                    results[symbol] = 0
                
            except Exception as e:
                logger.error(f"Error updating {symbol}: {e}")
                results[symbol] = 0
        
        # Update process status
        total_records = sum(results.values())
        self.storage.update_process_status(
            'daily_update',
            'completed',
            total_records,
            f"Updated {len([r for r in results.values() if r > 0])} symbols with {total_records} records"
        )
        
        logger.info(f"Daily update complete. Total new records: {total_records}")
        return results
    
    def download_dividends_and_splits(self, symbols: List[str]) -> Dict[str, Dict[str, int]]:
        """Download dividend and split data for symbols."""
        logger.info(f"Downloading dividends and splits for {len(symbols)} symbols")
        
        results = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                # Get dividends
                dividends = ticker.dividends
                splits = ticker.splits
                
                symbol_results = {'dividends': 0, 'splits': 0}
                
                # Process dividends
                if not dividends.empty:
                    div_df = dividends.reset_index()
                    div_df.columns = ['date', 'dividend']
                    div_df['symbol'] = symbol
                    
                    # Insert dividends
                    try:
                        div_records = div_df.to_sql(
                            'dividends', 
                            self.storage.engine, 
                            if_exists='append', 
                            index=False
                        )
                    except Exception:
                        # Handle duplicates silently
                        div_records = 0
                    symbol_results['dividends'] = len(div_df)
                
                # Process splits
                if not splits.empty:
                    splits_df = splits.reset_index()
                    splits_df.columns = ['date', 'ratio']
                    splits_df['symbol'] = symbol
                    
                    # Insert splits
                    try:
                        split_records = splits_df.to_sql(
                            'splits', 
                            self.storage.engine, 
                            if_exists='append', 
                            index=False
                        )
                    except Exception:
                        # Handle duplicates silently
                        split_records = 0
                    symbol_results['splits'] = len(splits_df)
                
                results[symbol] = symbol_results
                self._rate_limit()
                
            except Exception as e:
                logger.error(f"Error downloading dividends/splits for {symbol}: {e}")
                results[symbol] = {'dividends': 0, 'splits': 0}
        
        logger.info("Dividends and splits download complete")
        return results
    
    def validate_data_quality(self, symbol: str = None) -> Dict[str, Any]:
        """Validate data quality and log issues."""
        logger.info("Starting data quality validation")
        
        issues = []
        
        if symbol:
            symbols = [symbol]
        else:
            symbols = self.storage.get_available_symbols()
        
        for sym in symbols:
            try:
                data = self.storage.get_price_data(sym)
                
                if data.empty:
                    issues.append({
                        'symbol': sym,
                        'issue': 'NO_DATA',
                        'description': 'No price data available'
                    })
                    continue
                
                # Check for gaps in data
                date_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq='D')
                missing_dates = date_range.difference(data.index)
                
                if len(missing_dates) > len(date_range) * 0.1:  # More than 10% missing
                    issues.append({
                        'symbol': sym,
                        'issue': 'MISSING_DATES',
                        'description': f'{len(missing_dates)} missing dates out of {len(date_range)}'
                    })
                
                # Check for outliers (price changes > 50%)
                if 'close' in data.columns:
                    pct_change = data['close'].pct_change().abs()
                    outliers = pct_change[pct_change > 0.5]
                    
                    if len(outliers) > 0:
                        issues.append({
                            'symbol': sym,
                            'issue': 'PRICE_OUTLIERS',
                            'description': f'{len(outliers)} price changes > 50%'
                        })
                
                # Check for zero volume days
                if 'volume' in data.columns:
                    zero_volume = data[data['volume'] == 0]
                    
                    if len(zero_volume) > 0:
                        issues.append({
                            'symbol': sym,
                            'issue': 'ZERO_VOLUME',
                            'description': f'{len(zero_volume)} days with zero volume'
                        })
                
            except Exception as e:
                logger.error(f"Error validating {sym}: {e}")
                issues.append({
                    'symbol': sym,
                    'issue': 'VALIDATION_ERROR',
                    'description': str(e)
                })
        
        # Log all issues
        for issue in issues:
            self.storage.log_data_quality_issue(
                issue['symbol'],
                issue['issue'],
                issue['description'],
                'WARNING' if issue['issue'] != 'VALIDATION_ERROR' else 'ERROR'
            )
        
        logger.info(f"Data quality validation complete. Found {len(issues)} issues")
        return {'issues_found': len(issues), 'issues': issues}
    
    def initialize_universe_data(self, universe: str = 'IBOV') -> Dict[str, int]:
        """Initialize data for a stock universe."""
        symbols = get_universe_tickers(universe)
        logger.info(f"Initializing data for {universe} universe ({len(symbols)} symbols)")
        
        # Download historical data
        results = self.download_historical_data(symbols)
        
        # Download dividends and splits
        div_split_results = self.download_dividends_and_splits(symbols)
        
        # Validate data quality
        quality_results = self.validate_data_quality()
        
        return {
            'price_data': results,
            'dividends_splits': div_split_results,
            'quality_issues': quality_results['issues_found']
        }