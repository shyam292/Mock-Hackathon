"""
Risk Manager
Implements volatility targeting, drawdown control, and position scaling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from utils.constants import (
    TARGET_VOLATILITY, 
    MAX_DRAWDOWN_THRESHOLD, 
    STOP_LOSS_THRESHOLD,
    ROLLING_WINDOW,
    TRADING_DAYS
)
from utils.helpers import calculate_drawdown, normalize_weights


class RiskManager:
    """
    Portfolio risk management engine.
    
    Implements:
    - Volatility targeting (target 10-15% annualized)
    - Drawdown-based exposure reduction
    - Position scaling
    - Stop-loss logic
    """
    
    def __init__(
        self, 
        target_vol: float = TARGET_VOLATILITY,
        max_drawdown: float = MAX_DRAWDOWN_THRESHOLD,
        stop_loss: float = STOP_LOSS_THRESHOLD
    ):
        """
        Initialize RiskManager.
        
        Args:
            target_vol: Target annualized volatility
            max_drawdown: Maximum drawdown threshold for exposure reduction
            stop_loss: Stop-loss threshold
        """
        self.target_vol = target_vol
        self.max_drawdown = max_drawdown
        self.stop_loss = stop_loss
        self.risk_adjustments = None
        
    def calculate_volatility_scalar(
        self, 
        portfolio_returns: pd.Series, 
        window: int = ROLLING_WINDOW
    ) -> float:
        """
        Calculate volatility scaling factor to target specific volatility.
        
        Args:
            portfolio_returns: Series of portfolio returns
            window: Rolling window for volatility calculation
            
        Returns:
            Scaling factor (1.0 = no adjustment)
        """
        if len(portfolio_returns) < window:
            return 1.0
        
        # Calculate current realized volatility (annualized)
        current_vol = portfolio_returns.iloc[-window:].std() * np.sqrt(TRADING_DAYS)
        
        if current_vol == 0:
            return 1.0
        
        # Scaling factor to achieve target volatility
        scalar = self.target_vol / current_vol
        
        # Cap scaling to prevent extreme leverage
        scalar = np.clip(scalar, 0.2, 2.0)
        
        return scalar
    
    def calculate_drawdown_scalar(self, portfolio_value: pd.Series) -> float:
        """
        Calculate exposure scalar based on current drawdown.
        Reduces exposure as drawdown increases.
        
        Args:
            portfolio_value: Series of portfolio values
            
        Returns:
            Scaling factor (0-1, where 1 = no reduction)
        """
        drawdown = calculate_drawdown(portfolio_value)
        current_dd = abs(drawdown.iloc[-1])
        
        if current_dd < self.max_drawdown:
            # No reduction if below threshold
            return 1.0
        elif current_dd > self.stop_loss:
            # Severe reduction if exceeding stop-loss
            return 0.2
        else:
            # Linear reduction between thresholds
            reduction_factor = 1.0 - ((current_dd - self.max_drawdown) / (self.stop_loss - self.max_drawdown))
            return np.clip(reduction_factor, 0.2, 1.0)
    
    def apply_risk_controls(
        self,
        allocations: pd.DataFrame,
        returns: pd.DataFrame,
        apply_vol_targeting: bool = True,
        apply_drawdown_control: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Apply risk controls to allocations.
        
        Args:
            allocations: DataFrame of asset allocations
            returns: DataFrame of asset returns
            apply_vol_targeting: Whether to apply volatility targeting
            apply_drawdown_control: Whether to apply drawdown control
            
        Returns:
            Tuple of (risk-adjusted allocations, exposure scalars)
        """
        adjusted_allocations = allocations.copy()
        exposure_scalars = pd.Series(1.0, index=allocations.index)
        
        # Calculate portfolio returns (without risk management)
        portfolio_returns = (allocations.shift(1) * returns).sum(axis=1)
        portfolio_value = (1 + portfolio_returns).cumprod()
        
        for i, date in enumerate(allocations.index):
            if i < ROLLING_WINDOW:
                # Not enough history for risk calculations
                continue
            
            scalar = 1.0
            
            # Apply volatility targeting
            if apply_vol_targeting:
                vol_scalar = self.calculate_volatility_scalar(
                    portfolio_returns.iloc[:i+1]
                )
                scalar *= vol_scalar
            
            # Apply drawdown control
            if apply_drawdown_control:
                dd_scalar = self.calculate_drawdown_scalar(
                    portfolio_value.iloc[:i+1]
                )
                scalar *= dd_scalar
            
            # Apply scalar to allocations
            adjusted_allocations.loc[date] = allocations.loc[date] * scalar
            
            # Normalize if scalar > 1 (to prevent leverage beyond 100%)
            if scalar > 1.0:
                adjusted_allocations.loc[date] = normalize_weights(
                    adjusted_allocations.loc[date].to_dict()
                )
            
            exposure_scalars.loc[date] = scalar
        
        self.risk_adjustments = exposure_scalars
        
        return adjusted_allocations, exposure_scalars
    
    def calculate_position_sizes(
        self,
        allocations: Dict[str, float],
        portfolio_value: float,
        prices: pd.Series
    ) -> Dict[str, int]:
        """
        Calculate number of shares to buy for each asset.
        
        Args:
            allocations: Dictionary of target weights
            portfolio_value: Current portfolio value
            prices: Series of current asset prices
            
        Returns:
            Dictionary of share quantities
        """
        positions = {}
        
        for ticker, weight in allocations.items():
            # Dollar allocation for this asset
            dollar_allocation = portfolio_value * weight
            
            # Number of shares (rounded down)
            shares = int(dollar_allocation / prices[ticker])
            
            positions[ticker] = shares
        
        return positions
    
    def calculate_risk_metrics(
        self,
        portfolio_returns: pd.Series,
        portfolio_value: pd.Series
    ) -> Dict[str, float]:
        """
        Calculate current risk metrics for monitoring.
        
        Args:
            portfolio_returns: Series of portfolio returns
            portfolio_value: Series of portfolio values
            
        Returns:
            Dictionary of risk metrics
        """
        # Current volatility (annualized)
        current_vol = portfolio_returns.iloc[-ROLLING_WINDOW:].std() * np.sqrt(TRADING_DAYS)
        
        # Current drawdown
        drawdown = calculate_drawdown(portfolio_value)
        current_dd = abs(drawdown.iloc[-1])
        max_dd = abs(drawdown.min())
        
        # Value at Risk (95% confidence)
        var_95 = np.percentile(portfolio_returns.iloc[-ROLLING_WINDOW:], 5)
        
        # Expected Shortfall (CVaR)
        returns_below_var = portfolio_returns[portfolio_returns < var_95]
        cvar_95 = returns_below_var.mean() if len(returns_below_var) > 0 else 0
        
        metrics = {
            'Current Volatility': current_vol,
            'Current Drawdown': current_dd,
            'Max Drawdown': max_dd,
            'VaR (95%)': var_95,
            'CVaR (95%)': cvar_95,
            'Vol Target': self.target_vol,
            'Vol Scalar': self.calculate_volatility_scalar(portfolio_returns),
            'DD Scalar': self.calculate_drawdown_scalar(portfolio_value)
        }
        
        return metrics
    
    def get_risk_signal(
        self,
        portfolio_returns: pd.Series,
        portfolio_value: pd.Series
    ) -> Tuple[str, str]:
        """
        Get current risk signal and recommendation.
        
        Args:
            portfolio_returns: Series of portfolio returns
            portfolio_value: Series of portfolio values
            
        Returns:
            Tuple of (signal, message)
        """
        metrics = self.calculate_risk_metrics(portfolio_returns, portfolio_value)
        
        current_vol = metrics['Current Volatility']
        current_dd = metrics['Current Drawdown']
        
        # Determine risk level
        if current_dd > self.stop_loss:
            signal = 'CRITICAL'
            message = f'Stop-loss triggered! Drawdown {current_dd:.1%} exceeds threshold {self.stop_loss:.1%}'
        elif current_dd > self.max_drawdown:
            signal = 'HIGH'
            message = f'High drawdown {current_dd:.1%}. Reducing exposure.'
        elif current_vol > self.target_vol * 1.5:
            signal = 'ELEVATED'
            message = f'Volatility {current_vol:.1%} above target. Scaling positions.'
        else:
            signal = 'NORMAL'
            message = 'Risk metrics within acceptable ranges.'
        
        return signal, message
