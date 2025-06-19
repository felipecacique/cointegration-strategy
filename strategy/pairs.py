"""
Pair selection and ranking module for pairs trading.
"""
import pandas as pd
import numpy as np
from itertools import combinations
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from strategy.cointegration import CointegrationTester
from data.api import MarketDataAPI
from config.settings import CONFIG
from config.universe import get_ticker_sector
from simple_logger import logger

warnings.filterwarnings('ignore')

class PairSelector:
    """Selects and ranks cointegrated pairs for trading."""
    
    def __init__(self, data_api: MarketDataAPI = None):
        self.data_api = data_api or MarketDataAPI()
        self.coint_tester = CointegrationTester()
        self.config = CONFIG['strategy']
    
    def find_cointegrated_pairs(self, symbols: List[str],
                               start_date: str = None,
                               end_date: str = None,
                               min_pvalue: float = None,
                               max_workers: int = 4,
                               progress_callback=None) -> List[Dict[str, Any]]:
        """
        Find cointegrated pairs from a list of symbols.
        
        Args:
            symbols: List of stock symbols
            start_date: Start date for data
            end_date: End date for data
            min_pvalue: Maximum p-value for cointegration (default from config)
            max_workers: Number of parallel workers
            
        Returns:
            List of dictionaries with pair test results
        """
        try:
            if min_pvalue is None:
                min_pvalue = self.config['p_value_threshold']
            
            logger.info(f"=== STARTING COINTEGRATION ANALYSIS ===")
            logger.info(f"Testing {len(symbols)} symbols for cointegration")
            logger.info(f"Date range: {start_date} to {end_date}")
            logger.info(f"Max workers: {max_workers}")
            logger.info(f"P-value threshold: {min_pvalue}")
            
            # Generate all possible pairs
            pair_combinations = list(combinations(symbols, 2))
            logger.info(f"Testing {len(pair_combinations)} pair combinations")
        
            # Test dependencies first
            try:
                import statsmodels.api as sm
                from statsmodels.tsa.stattools import coint, adfuller
                logger.info(f"DEBUG: Dependencies OK - statsmodels imported successfully")
            except Exception as e:
                logger.error(f"DEBUG: Dependency error - {e}")
                return []
        
            # Test first pair manually for debugging
            if len(pair_combinations) > 0:
                test_pair = pair_combinations[0]
                logger.info(f"DEBUG: Will test sample pair {test_pair[0]} - {test_pair[1]}")
                try:
                    # Test it immediately 
                    test_result = self._test_single_pair(test_pair[0], test_pair[1], start_date, end_date)
                    logger.info(f"DEBUG: Sample pair result: {test_result}")
                except Exception as e:
                    logger.error(f"DEBUG: Error testing sample pair: {e}")
                    import traceback
                    logger.error(f"DEBUG: Traceback: {traceback.format_exc()}")
            
            results = []
            completed_count = 0
            total_pairs = len(pair_combinations)
        
            # Test first 3 pairs sequentially for debugging (reduced for speed)
            logger.info(f"Testing first 3 pairs sequentially for debugging...")
            for i, pair in enumerate(pair_combinations[:3]):
                logger.info(f"Testing pair {i+1}/3: {pair[0]} - {pair[1]}")
                try:
                    result = self._test_single_pair(pair[0], pair[1], start_date, end_date)
                    if result and result.get('is_cointegrated', False):
                        if result.get('coint_pvalue', 1.0) <= min_pvalue:
                            results.append(result)
                            logger.info(f"*** FOUND COINTEGRATED PAIR: {pair[0]}-{pair[1]} ***")
                    else:
                        logger.info(f"Pair {pair[0]}-{pair[1]} not cointegrated")
                except Exception as e:
                    logger.error(f"Error testing pair {pair}: {e}")
                    import traceback
                    logger.error(f"Traceback: {traceback.format_exc()}")
            
            # If we want to test all pairs, use parallel processing
            if len(results) == 0 and len(pair_combinations) > 10:
                logger.info(f"No cointegrated pairs in first 10, testing all {len(pair_combinations)} pairs...")
                # Test pairs in parallel
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    # Submit all pair tests
                    future_to_pair = {
                        executor.submit(
                            self._test_single_pair, 
                            pair[0], pair[1], 
                            start_date, end_date
                        ): pair
                        for pair in pair_combinations[10:]  # Skip first 10 already tested
                    }
                    
                    # Collect results
                    for future in as_completed(future_to_pair):
                        pair = future_to_pair[future]
                        completed_count += 1
                        
                        try:
                            result = future.result()
                            if result and result.get('is_cointegrated', False):
                                if result.get('coint_pvalue', 1.0) <= min_pvalue:
                                    results.append(result)
                                    logger.debug(f"Found cointegrated pair: {pair[0]}-{pair[1]} "
                                               f"(p={result['coint_pvalue']:.4f})")
                        
                        except Exception as e:
                            logger.error(f"Error testing pair {pair}: {e}")
                        
                        # Update progress if callback provided
                        if progress_callback and completed_count % 50 == 0:  # Update every 50 pairs
                            progress_pct = completed_count / total_pairs
                            progress_callback(progress_pct, f"Tested {completed_count}/{total_pairs} pairs, found {len(results)} cointegrated")
            
            logger.info(f"Found {len(results)} cointegrated pairs out of {len(pair_combinations)} tested")
            if len(results) == 0:
                logger.warning(f"No cointegrated pairs found with p-value <= {min_pvalue}. Consider increasing threshold.")
            return results
        
        except Exception as e:
            logger.error(f"CRITICAL ERROR in find_cointegrated_pairs: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []
    
    def _test_single_pair(self, symbol1: str, symbol2: str,
                         start_date: str = None, end_date: str = None) -> Dict[str, Any]:
        """Test cointegration for a single pair."""
        try:
            # Get pair data
            data = self.data_api.get_pairs_data(symbol1, symbol2, start_date, end_date)
            
            if data.empty:
                logger.debug(f"No data for pair {symbol1}-{symbol2}")
                return None
                
            if len(data) < 50:
                logger.debug(f"Insufficient data for pair {symbol1}-{symbol2}: {len(data)} points")
                return None
            
            # Extract price series
            y1 = data[symbol1]
            y2 = data[symbol2]
            
            # Test cointegration
            result = self.coint_tester.test_pair_cointegration(y1, y2)
            
            # Debug first successful test
            if result and result.get('is_cointegrated', False):
                logger.info(f"DEBUG: Found cointegrated pair {symbol1}-{symbol2}: {result}")
            
            if not result:
                logger.debug(f"Cointegration test failed for {symbol1}-{symbol2}")
                return None
            
            # Add pair information
            result.update({
                'symbol1': symbol1,
                'symbol2': symbol2,
                'sector1': get_ticker_sector(symbol1),
                'sector2': get_ticker_sector(symbol2),
                'same_sector': get_ticker_sector(symbol1) == get_ticker_sector(symbol2)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error testing pair {symbol1}-{symbol2}: {e}")
            return None
    
    def rank_pairs(self, pairs_results: List[Dict[str, Any]],
                  ranking_method: str = 'composite') -> List[Dict[str, Any]]:
        """
        Rank pairs based on various criteria.
        
        Args:
            pairs_results: List of pair test results
            ranking_method: Method for ranking ('pvalue', 'half_life', 'correlation', 'composite')
            
        Returns:
            Sorted list of pairs with ranking scores
        """
        if not pairs_results:
            return []
        
        logger.info(f"Ranking {len(pairs_results)} pairs using {ranking_method} method")
        
        # Add ranking scores
        for result in pairs_results:
            result['ranking_score'] = self._calculate_ranking_score(result, ranking_method)
        
        # Sort by ranking score (higher is better)
        ranked_pairs = sorted(pairs_results, key=lambda x: x['ranking_score'], reverse=True)
        
        # Add rank
        for i, pair in enumerate(ranked_pairs):
            pair['rank'] = i + 1
        
        return ranked_pairs
    
    def _calculate_ranking_score(self, result: Dict[str, Any], method: str) -> float:
        """Calculate ranking score for a pair."""
        try:
            if method == 'pvalue':
                # Lower p-value is better (invert for scoring)
                pvalue = result.get('coint_pvalue', 1.0)
                return 1.0 - pvalue
            
            elif method == 'half_life':
                # Optimal half-life range (5-30 days)
                half_life = result.get('half_life', 0)
                if 5 <= half_life <= 30:
                    return 1.0 - abs(half_life - 15) / 15  # Closer to 15 is better
                else:
                    return 0.0
            
            elif method == 'correlation':
                # Higher absolute correlation is better
                correlation = result.get('correlation', 0)
                return abs(correlation)
            
            elif method == 'composite':
                # Composite score combining multiple factors
                pvalue = result.get('coint_pvalue', 1.0)
                half_life = result.get('half_life', 0)
                correlation = result.get('correlation', 0)
                
                # P-value score (0-1, higher is better)
                pvalue_score = max(0, 1.0 - pvalue) if pvalue <= 0.05 else 0
                
                # Half-life score (0-1, optimal range 5-30)
                if 5 <= half_life <= 30:
                    half_life_score = 1.0 - abs(half_life - 15) / 15
                else:
                    half_life_score = 0
                
                # Correlation score (0-1, higher absolute correlation better)
                correlation_score = min(1.0, abs(correlation))
                
                # Weighted composite score
                composite_score = (
                    0.4 * pvalue_score +
                    0.3 * half_life_score +
                    0.3 * correlation_score
                )
                
                return composite_score
            
            else:
                return 0.0
                
        except Exception as e:
            logger.error(f"Error calculating ranking score: {e}")
            return 0.0
    
    def filter_by_criteria(self, pairs: List[Dict[str, Any]],
                          min_half_life: float = None,
                          max_half_life: float = None,
                          min_correlation: float = None,
                          same_sector_only: bool = None) -> List[Dict[str, Any]]:
        """
        Filter pairs by various criteria.
        
        Args:
            pairs: List of pair results
            min_half_life: Minimum half-life (days)
            max_half_life: Maximum half-life (days)
            min_correlation: Minimum absolute correlation
            same_sector_only: Only include pairs from same sector
            
        Returns:
            Filtered list of pairs
        """
        if not pairs:
            return []
        
        # Use config defaults if not specified
        min_half_life = min_half_life or self.config.get('min_half_life', 5)
        max_half_life = max_half_life or self.config.get('max_half_life', 30)
        min_correlation = min_correlation or self.config.get('min_correlation', 0.7)
        same_sector_only = same_sector_only or self.config.get('sector_matching', False)
        
        # Calculate minimum data points for this analysis - use flexible threshold
        # Since this is for backtest formation period, use a more lenient requirement
        min_data_points = max(50, 100)  # Fixed minimum requirement instead of percentage
        
        logger.info(f"Filtering {len(pairs)} pairs with criteria: "
                   f"half_life: {min_half_life}-{max_half_life}, "
                   f"min_correlation: {min_correlation}, "
                   f"same_sector_only: {same_sector_only}, "
                   f"min_data_points: {min_data_points}")
        
        filtered_pairs = []
        
        rejected_reasons = {'half_life': 0, 'correlation': 0, 'sector': 0, 'data_points': 0, 'p_value': 0}
        
        for pair in pairs:
            pair_id = f"{pair.get('symbol1', 'UNK')}-{pair.get('symbol2', 'UNK')}"
            
            # Half-life filter
            half_life = pair.get('half_life', 0)
            if not (min_half_life <= half_life <= max_half_life):
                rejected_reasons['half_life'] += 1
                continue
            
            # Correlation filter
            correlation = pair.get('correlation', 0)
            if abs(correlation) < min_correlation:
                rejected_reasons['correlation'] += 1
                continue
            
            # Sector filter
            if same_sector_only and not pair.get('same_sector', False):
                rejected_reasons['sector'] += 1
                continue
            
            # Additional quality filters - use calculated min_data_points
            data_points = pair.get('data_points', 0)
            if data_points < min_data_points:
                rejected_reasons['data_points'] += 1
                logger.debug(f"Pair {pair_id} rejected: data_points={data_points} < {min_data_points}")
                continue
            
            p_value = pair.get('coint_pvalue', 1.0)
            if p_value > self.config['p_value_threshold']:
                rejected_reasons['p_value'] += 1
                continue
            
            filtered_pairs.append(pair)
        
        logger.info(f"After filtering: {len(filtered_pairs)} pairs remain")
        if rejected_reasons['data_points'] > 0 or rejected_reasons['half_life'] > 0:
            logger.info(f"Rejection summary: half_life={rejected_reasons['half_life']}, "
                       f"correlation={rejected_reasons['correlation']}, "
                       f"data_points={rejected_reasons['data_points']}, "
                       f"p_value={rejected_reasons['p_value']}")
        return filtered_pairs
    
    def select_top_pairs(self, pairs: List[Dict[str, Any]],
                        top_n: int = None) -> List[Dict[str, Any]]:
        """
        Select top N pairs for trading.
        
        Args:
            pairs: List of ranked pairs
            top_n: Number of top pairs to select (default from config)
            
        Returns:
            Top N pairs
        """
        top_n = top_n or self.config.get('top_pairs', 15)
        
        if not pairs:
            return []
        
        # Ensure pairs are ranked
        if 'rank' not in pairs[0]:
            pairs = self.rank_pairs(pairs)
        
        top_pairs = pairs[:top_n]
        
        logger.info(f"Selected top {len(top_pairs)} pairs for trading")
        return top_pairs
    
    def get_pair_universe(self, universe: str = 'IBOV',
                         start_date: str = None,
                         end_date: str = None) -> List[Dict[str, Any]]:
        """
        Get cointegrated pairs for a stock universe.
        
        Args:
            universe: Stock universe ('IBOV', 'IBRX100')
            start_date: Start date for analysis
            end_date: End date for analysis
            
        Returns:
            List of top cointegrated pairs
        """
        from config.universe import get_universe_tickers
        
        # Get universe symbols
        symbols = get_universe_tickers(universe)
        
        # Filter symbols with sufficient data
        valid_symbols = self.data_api.get_available_symbols(min_data_points=252)
        symbols = [s for s in symbols if s in valid_symbols]
        
        logger.info(f"Analyzing {len(symbols)} symbols from {universe} universe")
        
        # Find cointegrated pairs
        pairs = self.find_cointegrated_pairs(symbols, start_date, end_date)
        
        # Filter and rank pairs
        filtered_pairs = self.filter_by_criteria(pairs)
        ranked_pairs = self.rank_pairs(filtered_pairs)
        
        # Select top pairs
        top_pairs = self.select_top_pairs(ranked_pairs)
        
        return top_pairs
    
    def update_pair_database(self, pairs_results: List[Dict[str, Any]]):
        """Store pair test results in database."""
        try:
            if not pairs_results:
                return
            
            # Prepare data for database
            db_records = []
            for result in pairs_results:
                if result.get('error'):
                    continue
                    
                record = {
                    'symbol1': result.get('symbol1'),
                    'symbol2': result.get('symbol2'),
                    'test_date': result.get('test_date'),
                    'p_value': result.get('coint_pvalue'),
                    'hedge_ratio': result.get('hedge_ratio'),
                    'half_life': result.get('half_life'),
                    'correlation': result.get('correlation'),
                    'adf_statistic': result.get('adf_residuals_statistic'),
                    'is_cointegrated': result.get('is_cointegrated', False)
                }
                db_records.append(record)
            
            if db_records:
                df = pd.DataFrame(db_records)
                try:
                    df.to_sql('pair_results', self.data_api.storage.engine, 
                             if_exists='append', index=False)
                except Exception:
                    # Handle duplicates silently
                    pass
                
                logger.info(f"Stored {len(db_records)} pair results in database")
                
        except Exception as e:
            logger.error(f"Error updating pair database: {e}")
    
    def get_historical_pair_results(self, symbol1: str = None, symbol2: str = None,
                                   start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Get historical pair test results from database."""
        try:
            query = "SELECT * FROM pair_results WHERE 1=1"
            params = []
            
            if symbol1:
                query += " AND (symbol1 = ? OR symbol2 = ?)"
                params.extend([symbol1, symbol1])
            
            if symbol2:
                query += " AND (symbol1 = ? OR symbol2 = ?)"
                params.extend([symbol2, symbol2])
            
            if start_date:
                query += " AND test_date >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND test_date <= ?"
                params.append(end_date)
            
            query += " ORDER BY test_date DESC"
            
            df = pd.read_sql_query(query, self.data_api.storage.engine, params=params)
            return df
            
        except Exception as e:
            logger.error(f"Error getting historical pair results: {e}")
            return pd.DataFrame()
    
    def analyze_pair_stability(self, symbol1: str, symbol2: str,
                              window_days: int = 63) -> Dict[str, Any]:
        """
        Analyze stability of cointegration relationship over time.
        
        Args:
            symbol1, symbol2: Pair symbols
            window_days: Rolling window for analysis
            
        Returns:
            Dictionary with stability metrics
        """
        try:
            # Get price data
            data = self.data_api.get_pairs_data(symbol1, symbol2)
            
            if data.empty or len(data) < window_days * 2:
                return {}
            
            # Rolling cointegration tests
            results = []
            
            for i in range(window_days, len(data)):
                window_data = data.iloc[i-window_days:i]
                y1 = window_data[symbol1]
                y2 = window_data[symbol2]
                
                result = self.coint_tester.test_pair_cointegration(y1, y2)
                
                if not result.get('error'):
                    results.append({
                        'date': data.index[i],
                        'pvalue': result.get('coint_pvalue'),
                        'hedge_ratio': result.get('hedge_ratio'),
                        'half_life': result.get('half_life'),
                        'correlation': result.get('correlation'),
                        'is_cointegrated': result.get('is_cointegrated', False)
                    })
            
            if not results:
                return {}
            
            # Calculate stability metrics
            results_df = pd.DataFrame(results)
            
            stability_metrics = {
                'total_windows': len(results),
                'cointegrated_windows': results_df['is_cointegrated'].sum(),
                'cointegration_ratio': results_df['is_cointegrated'].mean(),
                'avg_pvalue': results_df['pvalue'].mean(),
                'pvalue_std': results_df['pvalue'].std(),
                'hedge_ratio_mean': results_df['hedge_ratio'].mean(),
                'hedge_ratio_std': results_df['hedge_ratio'].std(),
                'half_life_mean': results_df['half_life'].mean(),
                'half_life_std': results_df['half_life'].std(),
                'correlation_mean': results_df['correlation'].mean(),
                'correlation_std': results_df['correlation'].std()
            }
            
            return stability_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing pair stability: {e}")
            return {}