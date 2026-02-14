"""
Sample Data Generator
Generates synthetic financial data for testing/demo when yfinance fails.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List


def generate_sample_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generate realistic synthetic financial data for testing.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame of synthetic prices
    """
    print("\n⚠️  Using synthetic sample data for demonstration purposes.")
    print("   (yfinance API unavailable - this is normal for demo/offline mode)\n")
    
    # Parse dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    
    # Generate date range (business days only)
    dates = pd.bdate_range(start=start, end=end)
    
    # Define realistic parameters for each asset type
    asset_params = {
        'SPY': {'initial': 200, 'drift': 0.10, 'volatility': 0.15, 'regime_sensitive': True},
        'QQQ': {'initial': 150, 'drift': 0.12, 'volatility': 0.18, 'regime_sensitive': True},
        'TLT': {'initial': 120, 'drift': 0.03, 'volatility': 0.08, 'regime_sensitive': False},
        'GLD': {'initial': 110, 'drift': 0.04, 'volatility': 0.12, 'regime_sensitive': False}
    }
    
    # Generate prices using geometric Brownian motion with regime switches
    prices_dict = {}
    
    np.random.seed(42)  # For reproducibility
    
    for ticker in tickers:
        if ticker not in asset_params:
            # Default parameters for unknown tickers
            params = {'initial': 100, 'drift': 0.08, 'volatility': 0.15, 'regime_sensitive': True}
        else:
            params = asset_params[ticker]
        
        n_days = len(dates)
        prices = [params['initial']]
        
        # Simulate regime changes
        regime_changes = [0, n_days//4, n_days//2, 3*n_days//4]  # Quarterly regime changes
        
        for i in range(1, n_days):
            # Determine current regime
            if params['regime_sensitive']:
                # Add some volatility spikes and drawdowns
                if i in regime_changes:
                    # Regime change: add shock
                    shock = np.random.choice([-0.15, -0.05, 0.05, 0.10], p=[0.1, 0.3, 0.4, 0.2])
                    drift = params['drift'] + shock
                    vol = params['volatility'] * np.random.choice([1.0, 1.5, 2.0], p=[0.6, 0.3, 0.1])
                else:
                    drift = params['drift']
                    vol = params['volatility']
            else:
                drift = params['drift']
                vol = params['volatility']
            
            # Daily return using geometric Brownian motion
            daily_return = (drift / 252) + (vol / np.sqrt(252)) * np.random.randn()
            
            # Update price
            prices.append(prices[-1] * (1 + daily_return))
        
        prices_dict[ticker] = prices
    
    # Create DataFrame
    df = pd.DataFrame(prices_dict, index=dates)
    
    return df


def add_realistic_features(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Add realistic market features to synthetic data.
    
    Args:
        prices: DataFrame of prices
        
    Returns:
        DataFrame with added realistic features
    """
    # Add occasional gaps (like real market data)
    # Add correlation structure
    # Add trend periods
    
    return prices
