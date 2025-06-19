"""
Database storage management for pairs trading system.
"""
import sqlite3
import pandas as pd
import sqlalchemy as sa
from sqlalchemy import create_engine, text
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
import os
from datetime import datetime

from config.settings import CONFIG
from simple_logger import logger

class DataStorageManager:
    """Manages database operations for the pairs trading system."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or CONFIG['database']['db_path']
        self.engine = None
        self._create_directories()
        self._initialize_database()
    
    def _create_directories(self):
        """Create database directory if it doesn't exist."""
        if self.db_path != ':memory:':
            db_dir = os.path.dirname(self.db_path)
            if not os.path.exists(db_dir):
                os.makedirs(db_dir)
                logger.info(f"Created database directory: {db_dir}")
    
    def _initialize_database(self):
        """Initialize database connection and create tables."""
        try:
            self.engine = create_engine(f'sqlite:///{self.db_path}')
            self._create_tables()
            logger.info(f"Database initialized: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
    
    def _create_tables(self):
        """Create all required tables."""
        tables_sql = {
            'stocks_master': '''
                CREATE TABLE IF NOT EXISTS stocks_master (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    sector TEXT,
                    market_cap REAL,
                    currency TEXT DEFAULT 'BRL',
                    exchange TEXT DEFAULT 'SAO',
                    active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'daily_prices': '''
                CREATE TABLE IF NOT EXISTS daily_prices (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    adj_close REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (symbol) REFERENCES stocks_master (symbol),
                    UNIQUE(symbol, date)
                )
            ''',
            'dividends': '''
                CREATE TABLE IF NOT EXISTS dividends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    dividend REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (symbol) REFERENCES stocks_master (symbol),
                    UNIQUE(symbol, date)
                )
            ''',
            'splits': '''
                CREATE TABLE IF NOT EXISTS splits (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    ratio REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (symbol) REFERENCES stocks_master (symbol),
                    UNIQUE(symbol, date)
                )
            ''',
            'market_calendar': '''
                CREATE TABLE IF NOT EXISTS market_calendar (
                    date DATE PRIMARY KEY,
                    is_trading_day BOOLEAN NOT NULL,
                    exchange TEXT DEFAULT 'BOVESPA',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'data_quality_log': '''
                CREATE TABLE IF NOT EXISTS data_quality_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    symbol TEXT,
                    issue_type TEXT NOT NULL,
                    description TEXT,
                    severity TEXT DEFAULT 'INFO',
                    resolved BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'update_status': '''
                CREATE TABLE IF NOT EXISTS update_status (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    process_name TEXT NOT NULL,
                    last_update TIMESTAMP,
                    status TEXT,
                    records_updated INTEGER DEFAULT 0,
                    error_message TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''',
            'pair_results': '''
                CREATE TABLE IF NOT EXISTS pair_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol1 TEXT NOT NULL,
                    symbol2 TEXT NOT NULL,
                    test_date DATE NOT NULL,
                    p_value REAL,
                    hedge_ratio REAL,
                    half_life REAL,
                    correlation REAL,
                    adf_statistic REAL,
                    is_cointegrated BOOLEAN,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol1, symbol2, test_date)
                )
            '''
        }
        
        with self.engine.connect() as conn:
            for table_name, sql in tables_sql.items():
                try:
                    conn.execute(text(sql))
                    logger.debug(f"Table created/verified: {table_name}")
                except Exception as e:
                    logger.error(f"Error creating table {table_name}: {e}")
                    raise
            conn.commit()
        
        # Create indexes
        self._create_indexes()
    
    def _create_indexes(self):
        """Create database indexes for performance."""
        indexes_sql = [
            'CREATE INDEX IF NOT EXISTS idx_daily_prices_symbol ON daily_prices(symbol)',
            'CREATE INDEX IF NOT EXISTS idx_daily_prices_date ON daily_prices(date)',
            'CREATE INDEX IF NOT EXISTS idx_daily_prices_symbol_date ON daily_prices(symbol, date)',
            'CREATE INDEX IF NOT EXISTS idx_dividends_symbol ON dividends(symbol)',
            'CREATE INDEX IF NOT EXISTS idx_splits_symbol ON splits(symbol)',
            'CREATE INDEX IF NOT EXISTS idx_pair_results_symbols ON pair_results(symbol1, symbol2)',
            'CREATE INDEX IF NOT EXISTS idx_pair_results_date ON pair_results(test_date)',
            'CREATE INDEX IF NOT EXISTS idx_data_quality_date ON data_quality_log(date)',
        ]
        
        with self.engine.connect() as conn:
            for sql in indexes_sql:
                try:
                    conn.execute(text(sql))
                except Exception as e:
                    logger.warning(f"Error creating index: {e}")
            conn.commit()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()
    
    def insert_stock_data(self, symbol: str, data: pd.DataFrame, 
                         table: str = 'daily_prices') -> int:
        """Insert stock price data into database."""
        try:
            if data.empty:
                logger.warning(f"No data to insert for {symbol}")
                return 0
            
            # Prepare data
            data = data.copy()
            data['symbol'] = symbol
            data.reset_index(inplace=True)
            
            # Rename columns to match database schema
            column_mapping = {
                'Date': 'date',
                'Open': 'open',
                'High': 'high', 
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Adj Close': 'adj_close'
            }
            data.rename(columns=column_mapping, inplace=True)
            
            # Remove extra columns that aren't in our schema (like Dividends, Stock Splits)
            expected_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'adj_close', 'symbol']
            # Only keep columns that exist in both data and expected_columns
            available_columns = [col for col in expected_columns if col in data.columns]
            data = data[available_columns]
            
            # Insert data with duplicate handling
            try:
                records = data.to_sql(
                    table, 
                    self.engine, 
                    if_exists='append', 
                    index=False
                )
            except Exception as e:
                if "UNIQUE constraint failed" in str(e) or "duplicate" in str(e).lower():
                    # Handle duplicates by inserting only new records
                    logger.debug(f"Handling duplicates for {symbol}")
                    records = self._insert_new_records_only(data, table, symbol)
                else:
                    raise e
            
            logger.info(f"Inserted {records} records for {symbol}")
            return records or len(data)
            
        except Exception as e:
            logger.error(f"Error inserting data for {symbol}: {e}")
            return 0
    
    def _insert_new_records_only(self, data: pd.DataFrame, table: str, symbol: str) -> int:
        """Insert only new records that don't already exist."""
        try:
            # Get existing dates for this symbol
            existing_query = f"""
                SELECT date FROM {table} 
                WHERE symbol = :symbol 
            """
            existing_dates = pd.read_sql_query(existing_query, self.engine, params={'symbol': symbol})
            
            if not existing_dates.empty:
                existing_dates['date'] = pd.to_datetime(existing_dates['date'])
                data['date'] = pd.to_datetime(data['date'])
                
                # Filter out existing dates
                new_data = data[~data['date'].isin(existing_dates['date'])]
                
                if not new_data.empty:
                    records = new_data.to_sql(table, self.engine, if_exists='append', index=False)
                    return len(new_data)
                else:
                    return 0
            else:
                # No existing data, insert all
                records = data.to_sql(table, self.engine, if_exists='append', index=False)
                return len(data)
                
        except Exception as e:
            logger.error(f"Error inserting new records for {symbol}: {e}")
            return 0
    
    def get_price_data(self, symbol: str, start_date: str = None, 
                      end_date: str = None) -> pd.DataFrame:
        """Retrieve price data for a symbol."""
        try:
            # Temporarily disable overlap checking for debugging
            # if start_date and end_date:
            #     date_range = self.get_date_range(symbol)
            #     symbol_start = pd.to_datetime(date_range.get('min_date'))
            #     symbol_end = pd.to_datetime(date_range.get('max_date'))
            #     request_start = pd.to_datetime(start_date)
            #     request_end = pd.to_datetime(end_date)
            #     
            #     if (symbol_start is None or symbol_end is None or 
            #         request_end < symbol_start or request_start > symbol_end):
            #         logger.debug(f"No overlap for {symbol}: requested {start_date} to {end_date}, available {date_range.get('min_date')} to {date_range.get('max_date')}")
            #         return pd.DataFrame()
            
            query = """
                SELECT date, open, high, low, close, volume, adj_close
                FROM daily_prices 
                WHERE symbol = :symbol
            """
            params = {'symbol': symbol}
            
            if start_date:
                query += " AND date >= :start_date"
                params['start_date'] = start_date
            
            if end_date:
                query += " AND date <= :end_date"
                params['end_date'] = end_date
            
            query += " ORDER BY date"
            
            df = pd.read_sql_query(query, self.engine, params=params)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols in database."""
        try:
            query = "SELECT DISTINCT symbol FROM daily_prices ORDER BY symbol"
            df = pd.read_sql_query(query, self.engine)
            return df['symbol'].tolist()
        except Exception as e:
            logger.error(f"Error getting available symbols: {e}")
            return []
    
    def get_date_range(self, symbol: str = None) -> Dict[str, str]:
        """Get date range of available data."""
        try:
            if symbol:
                query = "SELECT MIN(date) as min_date, MAX(date) as max_date FROM daily_prices WHERE symbol = :symbol"
                params = {'symbol': symbol}
            else:
                query = "SELECT MIN(date) as min_date, MAX(date) as max_date FROM daily_prices"
                params = {}
            
            df = pd.read_sql_query(query, self.engine, params=params)
            
            return {
                'min_date': df['min_date'].iloc[0] if not df.empty else None,
                'max_date': df['max_date'].iloc[0] if not df.empty else None
            }
        except Exception as e:
            logger.error(f"Error getting date range: {e}")
            return {'min_date': None, 'max_date': None}
    
    def update_stock_master(self, stocks_info: List[Dict[str, Any]]) -> int:
        """Update stocks master table."""
        try:
            df = pd.DataFrame(stocks_info)
            if df.empty:
                return 0
            
            records = df.to_sql(
                'stocks_master', 
                self.engine, 
                if_exists='replace', 
                index=False
            )
            
            logger.info(f"Updated {len(df)} stocks in master table")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error updating stocks master: {e}")
            return 0
    
    def log_data_quality_issue(self, symbol: str, issue_type: str, 
                              description: str, severity: str = 'INFO'):
        """Log data quality issue."""
        try:
            with self.get_connection() as conn:
                conn.execute(text("""
                    INSERT INTO data_quality_log 
                    (date, symbol, issue_type, description, severity)
                    VALUES (?, ?, ?, ?, ?)
                """), (datetime.now().date(), symbol, issue_type, description, severity))
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging data quality issue: {e}")
    
    def update_process_status(self, process_name: str, status: str, 
                            records_updated: int = 0, error_message: str = None):
        """Update process status."""
        try:
            with self.get_connection() as conn:
                conn.execute(text("""
                    INSERT INTO update_status 
                    (process_name, last_update, status, records_updated, error_message)
                    VALUES (?, ?, ?, ?, ?)
                """), (process_name, datetime.now(), status, records_updated, error_message))
                conn.commit()
        except Exception as e:
            logger.error(f"Error updating process status: {e}")
    
    def backup_database(self, backup_path: str = None) -> bool:
        """Create database backup."""
        try:
            if not backup_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                backup_path = f"{self.db_path}.backup_{timestamp}"
            
            # Simple SQLite backup
            if self.db_path != ':memory:':
                import shutil
                shutil.copy2(self.db_path, backup_path)
                logger.info(f"Database backed up to: {backup_path}")
                return True
            
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            return False
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            stats = {}
            
            # Table record counts
            tables = ['stocks_master', 'daily_prices', 'dividends', 'splits', 
                     'pair_results', 'data_quality_log']
            
            for table in tables:
                query = f"SELECT COUNT(*) as count FROM {table}"
                result = pd.read_sql_query(query, self.engine)
                stats[f"{table}_count"] = result['count'].iloc[0]
            
            # Database size
            if self.db_path != ':memory:' and os.path.exists(self.db_path):
                stats['db_size_mb'] = os.path.getsize(self.db_path) / (1024 * 1024)
            
            # Date ranges
            date_range = self.get_date_range()
            stats.update(date_range)
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {}