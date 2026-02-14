"""
Allocator
Implements dynamic allocation strategies based on regime detection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import warnings
warnings.filterwarnings('ignore')

from utils.constants import TICKERS, ROLLING_WINDOW
from utils.helpers import normalize_weights


class Allocator:
    """
    Dynamic portfolio allocation engine.
    
    Implements multiple allocation strategies:
    - Risk Parity (inverse volatility weighting)
    - Momentum weighting (based on rolling returns)
    - Correlation-aware diversification
    
    Allocation adapts based on detected market regime.
    """
    
    def __init__(self, tickers: List[str] = None):
        """
        Initialize Allocator.
        
        Args:
            tickers: List of asset tickers
        """
        self.tickers = tickers or TICKERS
        self.allocations = None
        
    def risk_parity_allocation(self, volatilities: pd.Series) -> Dict[str, float]:
        """
        Calculate Risk Parity allocation (inverse volatility weighting).
        
        Args:
            volatilities: Series of asset volatilities
            
        Returns:
            Dictionary of asset weights
        """
        # Inverse volatility
        inv_vol = 1.0 / volatilities
        
        # Normalize to sum to 1
        weights = {ticker: inv_vol[ticker] / inv_vol.sum() for ticker in self.tickers}
        
        return weights
    
    def momentum_allocation(self, returns: pd.Series) -> Dict[str, float]:
        """
        Calculate Momentum-based allocation.
        Positive momentum assets get higher weight.
        
        Args:
            returns: Series of asset rolling returns
            
        Returns:
            Dictionary of asset weights
        """
        # Only consider positive momentum assets
        positive_returns = returns.copy()
        positive_returns[positive_returns < 0] = 0
        
        if positive_returns.sum() == 0:
            # If all negative, use equal weights
            return {ticker: 1.0 / len(self.tickers) for ticker in self.tickers}
        
        # Weight by momentum magnitude
        weights = {ticker: positive_returns[ticker] / positive_returns.sum() for ticker in self.tickers}
        
        return weights
    
    def correlation_aware_allocation(
        self, 
        returns: pd.DataFrame, 
        volatilities: pd.Series,
        lookback: int = ROLLING_WINDOW
    ) -> Dict[str, float]:
        """
        Calculate correlation-aware allocation.
        Penalizes highly correlated assets.
        
        Args:
            returns: DataFrame of asset returns
            volatilities: Series of asset volatilities
            lookback: Lookback period for correlation
            
        Returns:
            Dictionary of asset weights
        """
        # Calculate correlation matrix
        corr_matrix = returns.iloc[-lookback:].corr()
        
        # Calculate diversification score (lower correlation = higher score)
        div_scores = {}
        for ticker in self.tickers:
            # Average correlation with other assets
            avg_corr = corr_matrix[ticker].drop(ticker).mean()
            # Diversification score (inverse of correlation)
            div_scores[ticker] = 1.0 - avg_corr
        
        # Combine with inverse volatility
        inv_vol = 1.0 / volatilities
        
        # Combined score
        combined_scores = {}
        for ticker in self.tickers:
            combined_scores[ticker] = div_scores[ticker] * inv_vol[ticker]
        
        # Normalize
        weights = normalize_weights(combined_scores)
        
        return weights
    
    def regime_adaptive_allocation(
        self,
        regime: int,
        returns: pd.DataFrame,
        volatilities: pd.Series,
        base_method: str = 'risk_parity'
    ) -> Dict[str, float]:
        """
        Calculate allocation adapted to current regime.
        
        Args:
            regime: Current regime ID (0-3)
            returns: DataFrame of asset returns
            volatilities: Series of asset volatilities
            base_method: Base allocation method
            
        Returns:
            Dictionary of asset weights
        """
        # Get base allocation
        if base_method == 'risk_parity':
            base_weights = self.risk_parity_allocation(volatilities)
        elif base_method == 'momentum':
            rolling_returns = returns.iloc[-ROLLING_WINDOW:].mean()
            base_weights = self.momentum_allocation(rolling_returns)
        elif base_method == 'correlation_aware':
            base_weights = self.correlation_aware_allocation(returns, volatilities)
        else:
            # Equal weight
            base_weights = {ticker: 1.0 / len(self.tickers) for ticker in self.tickers}
        
        # Adjust weights based on regime
        adjusted_weights = self._adjust_for_regime(base_weights, regime)
        
        return adjusted_weights
    
    def _adjust_for_regime(self, weights: Dict[str, float], regime: int) -> Dict[str, float]:
        """
        Adjust allocation based on market regime.
        
        Regime-specific adjustments:
        - Trending Up (0): Favor equities (SPY, QQQ)
        - Trending Down (1): Favor bonds and gold (TLT, GLD)
        - High Volatility (2): Increase bonds, reduce equities
        - Crash (3): Maximum defensive (bonds and gold)
        
        Args:
            weights: Base weights
            regime: Current regime ID
            
        Returns:
            Adjusted weights
        """
        adjusted = weights.copy()
        
        if regime == 0:  # Trending Up
            # Favor equities
            if 'SPY' in adjusted:
                adjusted['SPY'] *= 1.3
            if 'QQQ' in adjusted:
                adjusted['QQQ'] *= 1.3
            if 'TLT' in adjusted:
                adjusted['TLT'] *= 0.7
            if 'GLD' in adjusted:
                adjusted['GLD'] *= 0.7
                
        elif regime == 1:  # Trending Down
            # Balanced defensive shift
            if 'SPY' in adjusted:
                adjusted['SPY'] *= 0.9
            if 'QQQ' in adjusted:
                adjusted['QQQ'] *= 0.9
            if 'TLT' in adjusted:
                adjusted['TLT'] *= 1.2
            if 'GLD' in adjusted:
                adjusted['GLD'] *= 1.2
                
        elif regime == 2:  # High Volatility
            # Reduce equities, increase defensive assets
            if 'SPY' in adjusted:
                adjusted['SPY'] *= 0.7
            if 'QQQ' in adjusted:
                adjusted['QQQ'] *= 0.7
            if 'TLT' in adjusted:
                adjusted['TLT'] *= 1.4
            if 'GLD' in adjusted:
                adjusted['GLD'] *= 1.4
                
        elif regime == 3:  # Crash
            # Maximum defensive: prioritize safe havens
            if 'SPY' in adjusted:
                adjusted['SPY'] *= 0.4
            if 'QQQ' in adjusted:
                adjusted['QQQ'] *= 0.4
            if 'TLT' in adjusted:
                adjusted['TLT'] *= 1.8
            if 'GLD' in adjusted:
                adjusted['GLD'] *= 1.8
        
        # Normalize to sum to 1
        adjusted = normalize_weights(adjusted)
        
        return adjusted
    
    def calculate_allocations(
        self,
        regimes: pd.Series,
        returns: pd.DataFrame,
        features: pd.DataFrame,
        method: str = 'risk_parity'
    ) -> pd.DataFrame:
        """
        Calculate daily allocations for entire period.
        
        Args:
            regimes: Series of regime labels
            returns: DataFrame of asset returns
            features: DataFrame of features
            method: Allocation method
            
        Returns:
            DataFrame of daily allocations
        """
        allocations = []
        
        for date in regimes.index:
            # Get regime for this date
            regime = int(regimes.loc[date])
            
            # Get volatilities for this date
            volatilities = pd.Series({
                ticker: features.loc[date, f'{ticker}_volatility'] 
                for ticker in self.tickers
            })
            
            # Get returns up to this date (avoid lookahead)
            returns_upto_date = returns.loc[:date]
            
            # Calculate allocation
            weights = self.regime_adaptive_allocation(
                regime=regime,
                returns=returns_upto_date,
                volatilities=volatilities,
                base_method=method
            )
            
            allocations.append(weights)
        
        # Convert to DataFrame
        allocations_df = pd.DataFrame(allocations, index=regimes.index)
        
        self.allocations = allocations_df
        
        return allocations_df
    
    def get_allocation(self, date: pd.Timestamp = None) -> Dict[str, float]:
        """
        Get allocation for a specific date.
        
        Args:
            date: Date to query (default: last date)
            
        Returns:
            Dictionary of asset weights
        """
        if self.allocations is None:
            raise ValueError("Allocations not yet calculated. Call calculate_allocations() first.")
        
        if date is None:
            date = self.allocations.index[-1]
        
        return self.allocations.loc[date].to_dict()
