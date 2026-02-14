"""
Data Loader
Fetches and processes financial data with technical indicators.
Ensures no lookahead bias - all features use only past data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from utils.constants import TICKERS, ROLLING_WINDOW, MA_SHORT, MA_LONG

# Try to import sample data generator (fallback)
try:
    from data.sample_data import generate_sample_data
    HAS_SAMPLE_DATA = True
except ImportError:
    HAS_SAMPLE_DATA = False


class DataLoader:
    """
    Handles data fetching and feature engineering for financial assets.
    All computations avoid lookahead bias.
    """
    
    def __init__(self, tickers: List[str] = None, start_date: str = None, end_date: str = None):
        """
        Initialize DataLoader.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
        """
        self.tickers = tickers or TICKERS
        self.start_date = start_date or '2015-01-01'
        # Ensure end_date is not in the future
        if end_date:
            self.end_date = end_date
        else:
            # Use yesterday to avoid potential issues with today's data
            self.end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        self.prices = None
        self.returns = None
        self.features = None
        
    def fetch_data(self) -> pd.DataFrame:
        """
        Fetch historical price data using yfinance with retry logic.
        
        Returns:
            DataFrame of adjusted close prices
        """
        print(f"Fetching data for {self.tickers} from {self.start_date} to {self.end_date}...")
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                data = yf.download(
                    self.tickers,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    auto_adjust=True,
                    threads=True
                )
                
                # Check if data was actually fetched
                if data.empty:
                    if attempt < max_retries - 1:
                        print(f"  Attempt {attempt + 1} failed: No data returned. Retrying...")
                        continue
                    else:
                        raise ValueError("Failed to fetch data after multiple attempts. Please check:"
                                       "\n  1. Internet connection"
                                       "\n  2. Ticker symbols are valid"
                                       "\n  3. Date range is reasonable"
                                       "\n  4. yfinance package is up to date (pip install --upgrade yfinance)")
                
                # Extract adjusted close prices
                if len(self.tickers) == 1:
                    if 'Adj Close' in data.columns:
                        prices = data['Adj Close'].to_frame()
                    elif 'Close' in data.columns:
                        prices = data['Close'].to_frame()
                    else:
                        prices = data
                    prices.columns = self.tickers
                else:
                    if 'Adj Close' in data.columns:
                        prices = data['Adj Close'][self.tickers]
                    elif 'Close' in data.columns:
                        prices = data['Close'][self.tickers]
                    else:
                        prices = data[self.tickers]
                
                # Remove any NaN values
                prices = prices.dropna()
                
                # Validate we have sufficient data
                if len(prices) < 100:
                    raise ValueError(f"Insufficient data: Only {len(prices)} days fetched. Need at least 100 days.")
                
                self.prices = prices
                print(f"✓ Successfully fetched {len(prices)} days of data")
                
                return prices
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"  Attempt {attempt + 1} failed: {str(e)}. Retrying...")
                    import time
                    time.sleep(2)  # Wait before retry
                else:
                    print(f"\n❌ ERROR: Failed to fetch data from yfinance after {max_retries} attempts.")
                    
                    # Try to use sample data as fallback
                    if HAS_SAMPLE_DATA:
                        print("\n🔄 Attempting to use synthetic sample data for demonstration...")
                        try:
                            from data.sample_data import generate_sample_data
                            prices = generate_sample_data(self.tickers, self.start_date, self.end_date)
                            
                            if len(prices) < 100:
                                raise ValueError(f"Insufficient sample data generated: {len(prices)} days")
                            
                            self.prices = prices
                            print(f"✓ Successfully generated {len(prices)} days of sample data")
                            return prices
                        except Exception as sample_error:
                            print(f"❌ Sample data generation also failed: {sample_error}")
                            raise
                    else:
                        print("\n💡 TROUBLESHOOTING TIPS:")
                        print("   1. Check your internet connection")
                        print("   2. Update yfinance: pip install --upgrade yfinance")
                        print("   3. Try a different date range")
                        print("   4. Verify ticker symbols are correct")
                        raise
    
    def calculate_returns(self) -> pd.DataFrame:
        """
        Calculate daily returns.
        
        Returns:
            DataFrame of daily returns
        """
        if self.prices is None or self.prices.empty:
            self.fetch_data()
        
        if self.prices.empty:
            raise ValueError("Cannot calculate returns: price data is empty")
        
        self.returns = self.prices.pct_change().dropna()
        return self.returns
    
    def calculate_rolling_returns(self, window: int = ROLLING_WINDOW) -> pd.DataFrame:
        """
        Calculate rolling returns (past N days).
        Uses shift to avoid lookahead bias.
        
        Args:
            window: Rolling window size
            
        Returns:
            DataFrame of rolling returns
        """
        if self.prices is None:
            self.fetch_data()
        
        # Calculate rolling returns using past data only
        rolling_returns = (self.prices / self.prices.shift(window) - 1)
        
        return rolling_returns
    
    def calculate_rolling_volatility(self, window: int = ROLLING_WINDOW) -> pd.DataFrame:
        """
        Calculate rolling annualized volatility.
        Uses only past data to avoid lookahead bias.
        
        Args:
            window: Rolling window size
            
        Returns:
            DataFrame of annualized volatility
        """
        if self.returns is None:
            self.calculate_returns()
        
        # Calculate rolling std using past data only
        rolling_vol = self.returns.rolling(window=window).std() * np.sqrt(252)
        
        return rolling_vol
    
    def calculate_moving_averages(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Calculate short and long moving averages.
        
        Returns:
            Tuple of (MA_short, MA_long) DataFrames
        """
        if self.prices is None:
            self.fetch_data()
        
        ma_short = self.prices.rolling(window=MA_SHORT).mean()
        ma_long = self.prices.rolling(window=MA_LONG).mean()
        
        return ma_short, ma_long
    
    def calculate_correlation_matrix(self, window: int = ROLLING_WINDOW) -> pd.DataFrame:
        """
        Calculate rolling correlation matrix.
        
        Args:
            window: Rolling window size
            
        Returns:
            DataFrame with rolling correlation (averaged across all pairs)
        """
        if self.returns is None:
            self.calculate_returns()
        
        # Calculate rolling correlation for each pair
        correlations = []
        
        for i in range(len(self.returns)):
            if i < window:
                correlations.append(np.nan)
            else:
                # Get returns for the window
                window_returns = self.returns.iloc[i-window:i]
                corr_matrix = window_returns.corr()
                
                # Average absolute correlation (excluding diagonal)
                avg_corr = (corr_matrix.abs().sum().sum() - len(self.tickers)) / (len(self.tickers) * (len(self.tickers) - 1))
                correlations.append(avg_corr)
        
        corr_series = pd.Series(correlations, index=self.returns.index)
        
        return corr_series
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Create comprehensive feature set for analysis.
        All features use only past data to avoid lookahead bias.
        
        Returns:
            DataFrame with all features
        """
        print("Engineering features...")
        
        if self.prices is None:
            self.fetch_data()
        
        if self.returns is None:
            self.calculate_returns()
        
        # Calculate all features
        rolling_returns = self.calculate_rolling_returns()
        rolling_vol = self.calculate_rolling_volatility()
        ma_short, ma_long = self.calculate_moving_averages()
        avg_correlation = self.calculate_correlation_matrix()
        
        # Create feature DataFrame
        features = pd.DataFrame(index=self.prices.index)
        
        # Add rolling returns for each asset
        for ticker in self.tickers:
            features[f'{ticker}_rolling_return'] = rolling_returns[ticker]
            features[f'{ticker}_volatility'] = rolling_vol[ticker]
            features[f'{ticker}_ma_short'] = ma_short[ticker]
            features[f'{ticker}_ma_long'] = ma_long[ticker]
            features[f'{ticker}_price'] = self.prices[ticker]
        
        # Add portfolio-level features
        features['avg_correlation'] = avg_correlation
        features['portfolio_volatility'] = rolling_vol.mean(axis=1)
        
        # Calculate drawdown for SPY (market proxy)
        cummax = self.prices['SPY'].cummax()
        features['market_drawdown'] = (self.prices['SPY'] - cummax) / cummax
        
        # Drop NaN values
        features = features.dropna()
        
        self.features = features
        print(f"Engineered {len(features.columns)} features for {len(features)} days")
        
        return features
    
    def get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Get all data (prices, returns, features).
        
        Returns:
            Tuple of (prices, returns, features)
        """
        if self.features is None:
            self.engineer_features()
        
        # Align all DataFrames to the same index
        common_index = self.features.index
        
        prices_aligned = self.prices.loc[common_index]
        returns_aligned = self.returns.loc[common_index]
        
        return prices_aligned, returns_aligned, self.features
