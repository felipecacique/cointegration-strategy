"""
Risk management module for backtesting.
Implements position sizing, exposure limits, and risk controls.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta

from config.settings import CONFIG
from simple_logger import logger

class RiskManager:
    """Risk management for pairs trading strategy."""
    
    def __init__(self):
        self.config = CONFIG['risk']
        self.trading_config = CONFIG['trading']
        
        # Risk limits
        self.max_position_size = self.trading_config.get('max_position_size', 0.1)
        self.max_leverage = self.config.get('max_leverage', 2.0)
        self.max_drawdown = self.config.get('max_drawdown', 0.15)
        self.max_active_pairs = self.trading_config.get('max_active_pairs', 10)
        
        # Tracking variables
        self.daily_pnl_history = []
        self.position_history = []
        self.risk_breaches = []
    
    def check_position_size(self, pair_id: str, position_value: float,
                           portfolio_value: float = None) -> bool:
        """
        Check if position size is within risk limits.
        
        Args:
            pair_id: Pair identifier
            position_value: Value of the position to open
            portfolio_value: Current portfolio value
            
        Returns:
            True if position size is acceptable
        """
        try:
            if portfolio_value is None:
                return True  # Can't check without portfolio value
            
            # Check individual position size limit
            position_pct = position_value / portfolio_value
            
            if position_pct > self.max_position_size:
                logger.warning(f"Position size limit breach for {pair_id}: "
                             f"{position_pct:.2%} > {self.max_position_size:.2%}")
                
                self.risk_breaches.append({
                    'date': datetime.now(),
                    'pair_id': pair_id,
                    'breach_type': 'POSITION_SIZE',
                    'value': position_pct,
                    'limit': self.max_position_size
                })
                
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking position size: {e}")
            return False
    
    def check_portfolio_leverage(self, total_exposure: float, 
                               portfolio_value: float) -> bool:
        """
        Check if portfolio leverage is within limits.
        
        Args:
            total_exposure: Total position exposure
            portfolio_value: Current portfolio value
            
        Returns:
            True if leverage is acceptable
        """
        try:
            leverage = total_exposure / portfolio_value if portfolio_value > 0 else 0
            
            if leverage > self.max_leverage:
                logger.warning(f"Leverage limit breach: {leverage:.2f}x > {self.max_leverage:.2f}x")
                
                self.risk_breaches.append({
                    'date': datetime.now(),
                    'breach_type': 'LEVERAGE',
                    'value': leverage,
                    'limit': self.max_leverage
                })
                
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking leverage: {e}")
            return False
    
    def check_drawdown(self, equity_curve: pd.Series) -> Dict[str, Any]:
        """
        Check current drawdown level.
        
        Args:
            equity_curve: Portfolio value time series
            
        Returns:
            Dictionary with drawdown information
        """
        try:
            if len(equity_curve) < 2:
                return {'current_drawdown': 0, 'breach': False}
            
            # Calculate drawdown
            cumulative = equity_curve / equity_curve.iloc[0]
            rolling_max = cumulative.expanding().max()
            drawdown = (cumulative - rolling_max) / rolling_max
            
            current_drawdown = drawdown.iloc[-1]
            max_drawdown_period = drawdown.min()
            
            # Check for breach
            breach = current_drawdown < -self.max_drawdown
            
            if breach:
                logger.warning(f"Drawdown limit breach: {current_drawdown:.2%} < -{self.max_drawdown:.2%}")
                
                self.risk_breaches.append({
                    'date': datetime.now(),
                    'breach_type': 'DRAWDOWN',
                    'value': current_drawdown,
                    'limit': -self.max_drawdown
                })
            
            return {
                'current_drawdown': current_drawdown,
                'max_drawdown_period': max_drawdown_period,
                'breach': breach,
                'drawdown_series': drawdown
            }
            
        except Exception as e:
            logger.error(f"Error checking drawdown: {e}")
            return {'current_drawdown': 0, 'breach': False}
    
    def check_concentration_risk(self, positions: Dict[str, Any],
                               sector_mapping: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Check for concentration risk by sector or individual positions.
        
        Args:
            positions: Dictionary of current positions
            sector_mapping: Mapping of symbols to sectors
            
        Returns:
            Dictionary with concentration analysis
        """
        try:
            if not positions:
                return {'sector_concentration': {}, 'max_sector_exposure': 0}
            
            # Calculate sector exposures
            sector_exposures = {}
            total_exposure = 0
            
            for pair_id, position in positions.items():
                capital = position.get('capital_allocated', 0)
                total_exposure += capital
                
                if sector_mapping:
                    symbol1 = position.get('symbol1', '')
                    symbol2 = position.get('symbol2', '')
                    
                    sector1 = sector_mapping.get(symbol1, 'Unknown')
                    sector2 = sector_mapping.get(symbol2, 'Unknown')
                    
                    # Add exposure to both sectors
                    sector_exposures[sector1] = sector_exposures.get(sector1, 0) + capital / 2
                    sector_exposures[sector2] = sector_exposures.get(sector2, 0) + capital / 2
            
            # Calculate concentration percentages
            sector_concentrations = {}
            if total_exposure > 0:
                for sector, exposure in sector_exposures.items():
                    sector_concentrations[sector] = exposure / total_exposure
            
            max_sector_exposure = max(sector_concentrations.values()) if sector_concentrations else 0
            
            # Check for excessive concentration (>50% in one sector)
            concentration_breach = max_sector_exposure > 0.5
            
            if concentration_breach:
                logger.warning(f"Sector concentration risk: {max_sector_exposure:.2%} > 50%")
            
            return {
                'sector_concentration': sector_concentrations,
                'max_sector_exposure': max_sector_exposure,
                'concentration_breach': concentration_breach,
                'total_exposure': total_exposure
            }
            
        except Exception as e:
            logger.error(f"Error checking concentration risk: {e}")
            return {'sector_concentration': {}, 'max_sector_exposure': 0}
    
    def apply_stop_loss(self, positions: Dict[str, Any], 
                       z_scores: Dict[str, float]) -> List[str]:
        """
        Apply stop-loss rules to positions.
        
        Args:
            positions: Dictionary of current positions
            z_scores: Current z-scores for each pair
            
        Returns:
            List of pair IDs that should be closed
        """
        try:
            stop_loss_threshold = self.trading_config.get('stop_loss_z_score', 3.0)
            pairs_to_close = []
            
            for pair_id, position in positions.items():
                if pair_id not in z_scores:
                    continue
                
                current_z_score = z_scores[pair_id]
                position_side = position.get('side')
                
                # Check stop-loss conditions
                should_stop = False
                
                if position_side == 'LONG' and current_z_score > stop_loss_threshold:
                    should_stop = True
                elif position_side == 'SHORT' and current_z_score < -stop_loss_threshold:
                    should_stop = True
                
                if should_stop:
                    pairs_to_close.append(pair_id)
                    logger.info(f"Stop-loss triggered for {pair_id}: z-score={current_z_score:.2f}")
            
            return pairs_to_close
            
        except Exception as e:
            logger.error(f"Error applying stop-loss: {e}")
            return []
    
    def calculate_portfolio_exposure(self, positions: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate various portfolio exposure metrics.
        
        Args:
            positions: Dictionary of current positions
            
        Returns:
            Dictionary with exposure metrics
        """
        try:
            if not positions:
                return {
                    'total_long_exposure': 0,
                    'total_short_exposure': 0,
                    'net_exposure': 0,
                    'gross_exposure': 0,
                    'num_positions': 0
                }
            
            total_long_exposure = 0
            total_short_exposure = 0
            
            for position in positions.values():
                capital = position.get('capital_allocated', 0)
                side = position.get('side', 'FLAT')
                
                if side == 'LONG':
                    total_long_exposure += capital
                elif side == 'SHORT':
                    total_short_exposure += capital
            
            net_exposure = total_long_exposure - total_short_exposure
            gross_exposure = total_long_exposure + total_short_exposure
            
            return {
                'total_long_exposure': total_long_exposure,
                'total_short_exposure': total_short_exposure,
                'net_exposure': net_exposure,
                'gross_exposure': gross_exposure,
                'num_positions': len(positions)
            }
            
        except Exception as e:
            logger.error(f"Error calculating portfolio exposure: {e}")
            return {}
    
    def check_position_correlation(self, positions: Dict[str, Any],
                                 correlation_matrix: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Check correlation between positions to avoid concentration.
        
        Args:
            positions: Dictionary of current positions
            correlation_matrix: Correlation matrix between stocks
            
        Returns:
            Dictionary with correlation analysis
        """
        try:
            if not positions or correlation_matrix is None:
                return {'avg_correlation': 0, 'max_correlation': 0}
            
            # Get all symbols in positions
            symbols = set()
            for position in positions.values():
                symbols.add(position.get('symbol1', ''))
                symbols.add(position.get('symbol2', ''))
            
            symbols = [s for s in symbols if s in correlation_matrix.index]
            
            if len(symbols) < 2:
                return {'avg_correlation': 0, 'max_correlation': 0}
            
            # Calculate average correlation between positions
            correlations = []
            
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols[i+1:], i+1):
                    if sym1 in correlation_matrix.index and sym2 in correlation_matrix.columns:
                        corr = correlation_matrix.loc[sym1, sym2]
                        correlations.append(abs(corr))
            
            avg_correlation = np.mean(correlations) if correlations else 0
            max_correlation = np.max(correlations) if correlations else 0
            
            # Check for high correlation risk (>0.8)
            high_correlation_breach = max_correlation > 0.8
            
            if high_correlation_breach:
                logger.warning(f"High correlation risk: max correlation = {max_correlation:.3f}")
            
            return {
                'avg_correlation': avg_correlation,
                'max_correlation': max_correlation,
                'high_correlation_breach': high_correlation_breach,
                'num_correlations': len(correlations)
            }
            
        except Exception as e:
            logger.error(f"Error checking position correlation: {e}")
            return {'avg_correlation': 0, 'max_correlation': 0}
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.05) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) and Expected Shortfall (ES).
        
        Args:
            returns: Series of portfolio returns
            confidence_level: Confidence level for VaR calculation
            
        Returns:
            Dictionary with VaR and ES metrics
        """
        try:
            if len(returns) < 30:
                return {'var': 0, 'expected_shortfall': 0}
            
            # Historical VaR
            var = np.percentile(returns, confidence_level * 100)
            
            # Expected Shortfall (Conditional VaR)
            es_returns = returns[returns <= var]
            expected_shortfall = es_returns.mean() if len(es_returns) > 0 else var
            
            # Parametric VaR (assuming normal distribution)
            from scipy import stats
            parametric_var = stats.norm.ppf(confidence_level, returns.mean(), returns.std())
            
            return {
                'var': var,
                'expected_shortfall': expected_shortfall,
                'parametric_var': parametric_var,
                'confidence_level': confidence_level
            }
            
        except Exception as e:
            logger.error(f"Error calculating VaR: {e}")
            return {'var': 0, 'expected_shortfall': 0}
    
    def generate_risk_report(self, portfolio_value: float, positions: Dict[str, Any],
                           equity_curve: pd.Series = None,
                           returns: pd.Series = None) -> Dict[str, Any]:
        """
        Generate comprehensive risk report.
        
        Args:
            portfolio_value: Current portfolio value
            positions: Current positions
            equity_curve: Portfolio value time series
            returns: Portfolio returns series
            
        Returns:
            Comprehensive risk report
        """
        try:
            report = {
                'timestamp': datetime.now(),
                'portfolio_value': portfolio_value,
                'risk_breaches': self.risk_breaches[-10:],  # Last 10 breaches
            }
            
            # Position exposure analysis
            exposure_metrics = self.calculate_portfolio_exposure(positions)
            report['exposure'] = exposure_metrics
            
            # Drawdown analysis
            if equity_curve is not None and len(equity_curve) > 1:
                drawdown_analysis = self.check_drawdown(equity_curve)
                report['drawdown'] = drawdown_analysis
            
            # VaR analysis
            if returns is not None and len(returns) > 30:
                var_analysis = self.calculate_var(returns)
                report['var'] = var_analysis
            
            # Position limits check
            position_limit_usage = len(positions) / self.max_active_pairs * 100
            report['position_limits'] = {
                'current_positions': len(positions),
                'max_positions': self.max_active_pairs,
                'utilization_pct': position_limit_usage
            }
            
            # Overall risk score (0-100, higher = more risky)
            risk_factors = []
            
            # Drawdown factor
            if 'drawdown' in report:
                drawdown_factor = min(100, abs(report['drawdown']['current_drawdown']) * 1000)
                risk_factors.append(drawdown_factor)
            
            # Leverage factor
            if exposure_metrics['gross_exposure'] > 0:
                leverage = exposure_metrics['gross_exposure'] / portfolio_value
                leverage_factor = min(100, leverage * 50)
                risk_factors.append(leverage_factor)
            
            # Position concentration factor
            concentration_factor = position_limit_usage
            risk_factors.append(concentration_factor)
            
            overall_risk_score = np.mean(risk_factors) if risk_factors else 0
            report['overall_risk_score'] = overall_risk_score
            
            # Risk level classification
            if overall_risk_score < 30:
                risk_level = 'LOW'
            elif overall_risk_score < 60:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'HIGH'
            
            report['risk_level'] = risk_level
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating risk report: {e}")
            return {}
    
    def should_reduce_exposure(self, risk_report: Dict[str, Any]) -> bool:
        """
        Determine if exposure should be reduced based on risk metrics.
        
        Args:
            risk_report: Risk report from generate_risk_report
            
        Returns:
            True if exposure should be reduced
        """
        try:
            # Check multiple risk factors
            reduce_exposure = False
            
            # High overall risk score
            if risk_report.get('overall_risk_score', 0) > 70:
                reduce_exposure = True
                logger.warning("Reducing exposure due to high overall risk score")
            
            # Drawdown breach
            if risk_report.get('drawdown', {}).get('breach', False):
                reduce_exposure = True
                logger.warning("Reducing exposure due to drawdown breach")
            
            # Recent risk breaches
            recent_breaches = len([b for b in self.risk_breaches[-5:] 
                                 if (datetime.now() - b['date']).days < 7])
            if recent_breaches >= 3:
                reduce_exposure = True
                logger.warning("Reducing exposure due to recent risk breaches")
            
            return reduce_exposure
            
        except Exception as e:
            logger.error(f"Error checking exposure reduction: {e}")
            return False
    
    def reset_risk_tracking(self):
        """Reset risk tracking variables."""
        self.daily_pnl_history.clear()
        self.position_history.clear()
        self.risk_breaches.clear()