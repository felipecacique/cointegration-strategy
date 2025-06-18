"""
Cointegration testing module for pairs trading.
Implements Engle-Granger test, ADF test, and half-life calculation.
"""
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from typing import Tuple, Dict, Any, Optional
import warnings

from simple_logger import logger

warnings.filterwarnings('ignore')

class CointegrationTester:
    """Tests for cointegration between pairs of stocks."""
    
    def __init__(self, confidence_level: float = 0.05):
        self.confidence_level = confidence_level
    
    def test_pair_cointegration(self, y1: pd.Series, y2: pd.Series) -> Dict[str, Any]:
        """
        Test cointegration between two price series using Engle-Granger method.
        
        Args:
            y1, y2: Price series for the two stocks
            
        Returns:
            Dictionary with test results
        """
        try:
            if len(y1) != len(y2):
                raise ValueError("Series must have same length")
            
            if len(y1) < 30:
                raise ValueError("Insufficient data points for cointegration test")
            
            # Align series and remove NaN values
            data = pd.DataFrame({'y1': y1, 'y2': y2}).dropna()
            
            if len(data) < 30:
                raise ValueError("Insufficient valid data points after cleaning")
            
            y1_clean = data['y1']
            y2_clean = data['y2']
            
            # Step 1: Test for stationarity of individual series
            adf_y1 = adfuller(y1_clean, autolag='AIC')
            adf_y2 = adfuller(y2_clean, autolag='AIC')
            
            # Step 2: Estimate cointegrating relationship (hedge ratio)
            hedge_ratio, intercept = self._estimate_hedge_ratio(y1_clean, y2_clean)
            
            # Step 3: Calculate residuals (spread)
            spread = y1_clean - hedge_ratio * y2_clean - intercept
            
            # Step 4: Test residuals for stationarity (Engle-Granger test)
            coint_test = coint(y1_clean, y2_clean, trend='c', method='aeg', maxlag=None, autolag='AIC')
            adf_residuals = adfuller(spread, autolag='AIC')
            
            # Step 5: Calculate half-life of mean reversion
            half_life = self.calculate_half_life(spread)
            
            # Step 6: Calculate correlation
            correlation = y1_clean.corr(y2_clean)
            
            # Determine if pair is cointegrated
            is_cointegrated = (
                coint_test[1] < self.confidence_level and  # Cointegration test p-value
                adf_residuals[1] < self.confidence_level and  # Residuals stationarity
                half_life > 0 and half_life < 60 and  # Reasonable mean reversion speed
                abs(correlation) > 0.3  # Minimum correlation
            )
            
            results = {
                'is_cointegrated': is_cointegrated,
                'coint_pvalue': coint_test[1],
                'coint_statistic': coint_test[0],
                'coint_critical_values': dict(zip(['1%', '5%', '10%'], coint_test[2])),
                'adf_residuals_pvalue': adf_residuals[1],
                'adf_residuals_statistic': adf_residuals[0],
                'adf_y1_pvalue': adf_y1[1],
                'adf_y2_pvalue': adf_y2[1],
                'hedge_ratio': hedge_ratio,
                'intercept': intercept,
                'half_life': half_life,
                'correlation': correlation,
                'spread_mean': spread.mean(),
                'spread_std': spread.std(),
                'data_points': len(data),
                'test_date': pd.Timestamp.now().date()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in cointegration test: {e}")
            return {
                'is_cointegrated': False,
                'error': str(e),
                'test_date': pd.Timestamp.now().date()
            }
    
    def _estimate_hedge_ratio(self, y1: pd.Series, y2: pd.Series) -> Tuple[float, float]:
        """
        Estimate hedge ratio using OLS regression: y1 = α + β*y2 + ε
        
        Returns:
            Tuple of (hedge_ratio, intercept)
        """
        try:
            # Add constant term for intercept
            X = sm.add_constant(y2)
            y = y1
            
            # Fit OLS model
            model = OLS(y, X).fit()
            
            intercept = model.params[0]  # α (constant)
            hedge_ratio = model.params[1]  # β (slope)
            
            return hedge_ratio, intercept
            
        except Exception as e:
            logger.error(f"Error estimating hedge ratio: {e}")
            return 1.0, 0.0
    
    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate half-life of mean reversion using Ornstein-Uhlenbeck process.
        
        Fits the model: Δspread(t) = α + β*spread(t-1) + ε(t)
        Half-life = -ln(2) / ln(1 + β)
        
        Args:
            spread: Time series of spread values
            
        Returns:
            Half-life in days (periods)
        """
        try:
            if len(spread) < 20:  # Need minimum data
                return np.nan
            
            # Remove any infinite or NaN values
            spread_clean = spread.dropna()
            if len(spread_clean) < 20:
                return np.nan
            
            # Calculate first differences more robustly
            spread_lag = spread_clean.shift(1).dropna()
            spread_diff = spread_clean.diff().dropna()
            
            # Ensure we have the same indices
            common_index = spread_lag.index.intersection(spread_diff.index)
            if len(common_index) < 10:
                return np.nan
                
            spread_lag = spread_lag[common_index]
            spread_diff = spread_diff[common_index]
            
            # Additional data quality checks
            if spread_lag.std() == 0 or spread_diff.std() == 0:
                return np.nan
            
            try:
                # Fit AR(1) model: Δspread(t) = α + β*spread(t-1) + ε(t)
                X = sm.add_constant(spread_lag)
                y = spread_diff
                
                # Check for multicollinearity or other issues
                if np.linalg.cond(X.T @ X) > 1e12:
                    return np.nan
                
                model = OLS(y, X).fit()
                
                if len(model.params) < 2 or not model.converged:
                    return np.nan
                    
                beta = float(model.params[1])  # Coefficient of lagged spread
                
                # Robust checks for beta
                if not np.isfinite(beta) or beta >= 0 or beta <= -2:
                    return np.nan
                
                # Calculate half-life with additional safety
                lambda_val = -beta
                if lambda_val <= 0:
                    return np.nan
                    
                half_life = np.log(2) / lambda_val
                
                # Final sanity checks
                if not np.isfinite(half_life) or half_life <= 0 or half_life > 500:
                    return np.nan
                
                return half_life
                
            except (np.linalg.LinAlgError, ValueError, ZeroDivisionError):
                return np.nan
            
        except Exception as e:
            logger.debug(f"Half-life calculation failed: {str(e)[:50]}")
            return np.nan
    
    def calculate_z_score(self, spread: pd.Series, window: int = 252) -> pd.Series:
        """
        Calculate rolling z-score of spread for signal generation.
        
        Args:
            spread: Time series of spread values
            window: Rolling window for mean and std calculation
            
        Returns:
            Z-score series
        """
        try:
            if len(spread) < window:
                # Use expanding window if insufficient data
                mean = spread.expanding().mean()
                std = spread.expanding().std()
            else:
                # Use rolling window
                mean = spread.rolling(window=window).mean()
                std = spread.rolling(window=window).std()
            
            z_score = (spread - mean) / std
            
            return z_score
            
        except Exception as e:
            logger.error(f"Error calculating z-score: {e}")
            return pd.Series(dtype=float)
    
    def test_stationarity(self, series: pd.Series, max_lag: int = None) -> Dict[str, Any]:
        """
        Test for stationarity using Augmented Dickey-Fuller test.
        
        Args:
            series: Time series to test
            max_lag: Maximum lag for ADF test
            
        Returns:
            Dictionary with ADF test results
        """
        try:
            if max_lag is None:
                max_lag = int(12 * (len(series) / 100) ** 0.25)  # Rule of thumb
            
            adf_result = adfuller(series, maxlag=max_lag, autolag='AIC')
            
            results = {
                'adf_statistic': adf_result[0],
                'pvalue': adf_result[1],
                'lags_used': adf_result[2],
                'nobs': adf_result[3],
                'critical_values': adf_result[4],
                'is_stationary': adf_result[1] < self.confidence_level
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in stationarity test: {e}")
            return {'is_stationary': False, 'error': str(e)}
    
    def johansen_cointegration_test(self, data: pd.DataFrame, 
                                   det_order: int = -1) -> Dict[str, Any]:
        """
        Johansen cointegration test for multiple time series.
        
        Args:
            data: DataFrame with price series as columns
            det_order: Deterministic order (-1: no deterministic, 0: constant, 1: linear trend)
            
        Returns:
            Dictionary with Johansen test results
        """
        try:
            from statsmodels.tsa.vector_ar.vecm import coint_johansen
            
            if len(data.columns) < 2:
                raise ValueError("Need at least 2 series for Johansen test")
            
            # Remove NaN values
            data_clean = data.dropna()
            
            if len(data_clean) < 30:
                raise ValueError("Insufficient data for Johansen test")
            
            # Perform Johansen test
            result = coint_johansen(data_clean.values, det_order=det_order, k_ar_diff=1)
            
            # Extract results
            trace_stats = result.lr1  # Trace statistics
            max_eigen_stats = result.lr2  # Maximum eigenvalue statistics
            critical_values_90 = result.cvt[:, 0]  # 90% critical values for trace
            critical_values_95 = result.cvt[:, 1]  # 95% critical values for trace
            critical_values_99 = result.cvt[:, 2]  # 99% critical values for trace
            
            # Count cointegrating relationships
            n_coint_90 = sum(trace_stats > critical_values_90)
            n_coint_95 = sum(trace_stats > critical_values_95)
            n_coint_99 = sum(trace_stats > critical_values_99)
            
            results = {
                'trace_statistics': trace_stats.tolist(),
                'max_eigen_statistics': max_eigen_stats.tolist(),
                'critical_values_90': critical_values_90.tolist(),
                'critical_values_95': critical_values_95.tolist(),
                'critical_values_99': critical_values_99.tolist(),
                'n_cointegrating_90': n_coint_90,
                'n_cointegrating_95': n_coint_95,
                'n_cointegrating_99': n_coint_99,
                'eigenvalues': result.eig.tolist(),
                'eigenvectors': result.evec.tolist()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in Johansen test: {e}")
            return {'error': str(e)}
    
    def calculate_cointegration_metrics(self, y1: pd.Series, y2: pd.Series,
                                      hedge_ratio: float) -> Dict[str, float]:
        """
        Calculate various metrics for a cointegrated pair.
        
        Args:
            y1, y2: Price series
            hedge_ratio: Estimated hedge ratio
            
        Returns:
            Dictionary with various metrics
        """
        try:
            # Calculate spread
            spread = y1 - hedge_ratio * y2
            
            # Basic statistics
            spread_mean = spread.mean()
            spread_std = spread.std()
            spread_min = spread.min()
            spread_max = spread.max()
            
            # Z-score metrics
            z_score = (spread - spread_mean) / spread_std
            z_score_current = z_score.iloc[-1] if len(z_score) > 0 else 0
            
            # Mean reversion metrics
            half_life = self.calculate_half_life(spread)
            
            # Correlation and R-squared
            correlation = y1.corr(y2)
            
            # Linear regression for R-squared
            X = sm.add_constant(y2)
            model = OLS(y1, X).fit()
            r_squared = model.rsquared
            
            # Volatility metrics
            y1_vol = y1.pct_change().std() * np.sqrt(252)
            y2_vol = y2.pct_change().std() * np.sqrt(252)
            spread_vol = spread.pct_change().std() * np.sqrt(252)
            
            metrics = {
                'spread_mean': spread_mean,
                'spread_std': spread_std,
                'spread_min': spread_min,
                'spread_max': spread_max,
                'z_score_current': z_score_current,
                'half_life': half_life,
                'correlation': correlation,
                'r_squared': r_squared,
                'y1_volatility': y1_vol,
                'y2_volatility': y2_vol,
                'spread_volatility': spread_vol,
                'hedge_ratio': hedge_ratio
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating cointegration metrics: {e}")
            return {}