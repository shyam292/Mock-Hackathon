"""
Helper Functions
Common utility functions for portfolio calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .constants import TRADING_DAYS


def validate_weights(weights: Dict[str, float]) -> bool:
    """
    Validate that weights sum to 1.0 (allowing small numerical errors).
    
    Args:
        weights: Dictionary of asset weights
        
    Returns:
        True if weights are valid, False otherwise
    """
    total = sum(weights.values())
    return abs(total - 1.0) < 1e-6


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """
    Normalize weights to sum to 1.0.
    
    Args:
        weights: Dictionary of asset weights
        
    Returns:
        Normalized weights
    """
    total = sum(weights.values())
    if total == 0:
        # Equal weights if all zero
        return {k: 1.0 / len(weights) for k in weights.keys()}
    return {k: v / total for k, v in weights.items()}


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate daily returns from prices.
    
    Args:
        prices: DataFrame of asset prices
        
    Returns:
        DataFrame of daily returns
    """
    return prices.pct_change()


def calculate_volatility(returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate rolling annualized volatility.
    
    Args:
        returns: Series of returns
        window: Rolling window size
        
    Returns:
        Series of annualized volatility
    """
    return returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS)


def calculate_drawdown(prices: pd.Series) -> pd.Series:
    """
    Calculate drawdown from peak.
    
    Args:
        prices: Series of prices or portfolio values
        
    Returns:
        Series of drawdown percentages
    """
    cummax = prices.cummax()
    drawdown = (prices - cummax) / cummax
    return drawdown


def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sharpe ratio
    """
    if returns.std() == 0:
        return 0.0
    
    mean_return = returns.mean() * TRADING_DAYS
    volatility = returns.std() * np.sqrt(TRADING_DAYS)
    
    return (mean_return - risk_free_rate) / volatility


def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02) -> float:
    """
    Calculate annualized Sortino ratio (using downside deviation).
    
    Args:
        returns: Series of returns
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Sortino ratio
    """
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return 0.0
    
    mean_return = returns.mean() * TRADING_DAYS
    downside_std = downside_returns.std() * np.sqrt(TRADING_DAYS)
    
    return (mean_return - risk_free_rate) / downside_std


def calculate_max_drawdown(returns: pd.Series) -> float:
    """
    Calculate maximum drawdown.
    
    Args:
        returns: Series of returns
        
    Returns:
        Maximum drawdown (as positive number)
    """
    cumulative = (1 + returns).cumprod()
    drawdown = calculate_drawdown(cumulative)
    return abs(drawdown.min())


def calculate_cagr(returns: pd.Series) -> float:
    """
    Calculate Compound Annual Growth Rate.
    
    Args:
        returns: Series of returns
        
    Returns:
        CAGR
    """
    cumulative = (1 + returns).cumprod()
    n_years = len(returns) / TRADING_DAYS
    
    if n_years == 0:
        return 0.0
    
    return (cumulative.iloc[-1] ** (1 / n_years)) - 1


def calculate_calmar_ratio(returns: pd.Series) -> float:
    """
    Calculate Calmar ratio (CAGR / Max Drawdown).
    
    Args:
        returns: Series of returns
        
    Returns:
        Calmar ratio
    """
    cagr = calculate_cagr(returns)
    max_dd = calculate_max_drawdown(returns)
    
    if max_dd == 0:
        return 0.0
    
    return cagr / max_dd


def calculate_metrics(returns: pd.Series) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        returns: Series of returns
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'CAGR': calculate_cagr(returns),
        'Sharpe Ratio': calculate_sharpe_ratio(returns),
        'Sortino Ratio': calculate_sortino_ratio(returns),
        'Max Drawdown': calculate_max_drawdown(returns),
        'Calmar Ratio': calculate_calmar_ratio(returns),
        'Annualized Volatility': returns.std() * np.sqrt(TRADING_DAYS),
        'Total Return': (1 + returns).cumprod().iloc[-1] - 1
    }
    
    return metrics


def calculate_rolling_sharpe(returns: pd.Series, window: int = 60) -> pd.Series:
    """
    Calculate rolling Sharpe ratio.
    
    Args:
        returns: Series of returns
        window: Rolling window size
        
    Returns:
        Series of rolling Sharpe ratios
    """
    rolling_mean = returns.rolling(window=window).mean() * TRADING_DAYS
    rolling_std = returns.rolling(window=window).std() * np.sqrt(TRADING_DAYS)
    
    sharpe = rolling_mean / rolling_std
    return sharpe
