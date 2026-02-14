"""
Regime Detector
Identifies market regimes using volatility, drawdown, and clustering.
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

from utils.constants import REGIME_LABELS, ROLLING_WINDOW


class RegimeDetector:
    """
    Detects market regimes using multi-factor analysis.
    
    Regimes:
    0: Trending Up - Low volatility, positive returns, uptrend
    1: Trending Down - Moderate volatility, negative returns, downtrend
    2: High Volatility - High volatility, mixed returns
    3: Crash - Extreme volatility, severe drawdown
    """
    
    def __init__(self, vol_threshold: float = 0.20, crash_threshold: float = -0.10):
        """
        Initialize RegimeDetector.
        
        Args:
            vol_threshold: Volatility threshold for high-vol regime
            crash_threshold: Drawdown threshold for crash regime
        """
        self.vol_threshold = vol_threshold
        self.crash_threshold = crash_threshold
        self.regimes = None
        
    def detect_regimes_rule_based(
        self, 
        returns: pd.DataFrame, 
        features: pd.DataFrame
    ) -> pd.Series:
        """
        Detect regimes using rule-based thresholds.
        
        Args:
            returns: DataFrame of asset returns
            features: DataFrame of engineered features
            
        Returns:
            Series of regime labels
        """
        regimes = pd.Series(index=features.index, dtype=int)
        
        for date in features.index:
            # Extract features for this date
            vol = features.loc[date, 'portfolio_volatility']
            drawdown = features.loc[date, 'market_drawdown']
            
            # Get SPY rolling return
            spy_return = features.loc[date, 'SPY_rolling_return']
            
            # Rule-based classification
            if drawdown < self.crash_threshold:
                # Crash regime: severe drawdown
                regime = 3
            elif vol > self.vol_threshold:
                # High volatility regime
                regime = 2
            elif spy_return > 0:
                # Trending up: positive momentum
                regime = 0
            else:
                # Trending down: negative momentum
                regime = 1
            
            regimes.loc[date] = regime
        
        return regimes
    
    def detect_regimes_clustering(
        self, 
        returns: pd.DataFrame, 
        features: pd.DataFrame,
        n_clusters: int = 4
    ) -> pd.Series:
        """
        Detect regimes using K-Means clustering on returns and volatility.
        
        Args:
            returns: DataFrame of asset returns
            features: DataFrame of engineered features
            n_clusters: Number of clusters
            
        Returns:
            Series of regime labels
        """
        # Prepare features for clustering
        clustering_features = pd.DataFrame({
            'avg_return': returns.mean(axis=1).rolling(ROLLING_WINDOW).mean(),
            'avg_volatility': features['portfolio_volatility'],
            'drawdown': features['market_drawdown'],
        }).dropna()
        
        # Normalize features
        normalized = (clustering_features - clustering_features.mean()) / clustering_features.std()
        
        # Apply K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(normalized)
        
        # Map clusters to regime labels based on characteristics
        cluster_labels = pd.Series(clusters, index=clustering_features.index)
        regimes = self._map_clusters_to_regimes(cluster_labels, clustering_features)
        
        return regimes
    
    def _map_clusters_to_regimes(
        self, 
        clusters: pd.Series, 
        features: pd.DataFrame
    ) -> pd.Series:
        """
        Map cluster IDs to regime labels based on characteristics.
        
        Args:
            clusters: Series of cluster IDs
            features: DataFrame of clustering features
            
        Returns:
            Series of regime labels
        """
        regimes = pd.Series(index=clusters.index, dtype=int)
        
        # Calculate average characteristics per cluster
        cluster_stats = {}
        for cluster_id in clusters.unique():
            mask = clusters == cluster_id
            cluster_stats[cluster_id] = {
                'avg_return': features.loc[mask, 'avg_return'].mean(),
                'avg_vol': features.loc[mask, 'avg_volatility'].mean(),
                'avg_drawdown': features.loc[mask, 'drawdown'].mean()
            }
        
        # Assign regime labels based on characteristics
        for cluster_id, stats in cluster_stats.items():
            mask = clusters == cluster_id
            
            if stats['avg_drawdown'] < -0.08:
                # Crash: severe drawdown
                regimes.loc[mask] = 3
            elif stats['avg_vol'] > 0.18:
                # High volatility
                regimes.loc[mask] = 2
            elif stats['avg_return'] > 0:
                # Trending up
                regimes.loc[mask] = 0
            else:
                # Trending down
                regimes.loc[mask] = 1
        
        return regimes
    
    def detect_regimes(
        self, 
        returns: pd.DataFrame, 
        features: pd.DataFrame,
        method: str = 'rule_based'
    ) -> pd.Series:
        """
        Detect market regimes using specified method.
        
        Args:
            returns: DataFrame of asset returns
            features: DataFrame of engineered features
            method: 'rule_based' or 'clustering'
            
        Returns:
            Series of regime labels (0-3)
        """
        if method == 'rule_based':
            regimes = self.detect_regimes_rule_based(returns, features)
        elif method == 'clustering':
            regimes = self.detect_regimes_clustering(returns, features)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.regimes = regimes
        
        # Print regime distribution
        print("\nRegime Distribution:")
        for regime_id, regime_name in REGIME_LABELS.items():
            count = (regimes == regime_id).sum()
            pct = count / len(regimes) * 100
            print(f"  {regime_name}: {count} days ({pct:.1f}%)")
        
        return regimes
    
    def get_regime_label(self, regime_id: int) -> str:
        """
        Get human-readable regime label.
        
        Args:
            regime_id: Regime ID (0-3)
            
        Returns:
            Regime name
        """
        return REGIME_LABELS.get(regime_id, 'Unknown')
    
    def get_current_regime(self, date: pd.Timestamp = None) -> Tuple[int, str]:
        """
        Get regime for a specific date.
        
        Args:
            date: Date to query (default: last date)
            
        Returns:
            Tuple of (regime_id, regime_name)
        """
        if self.regimes is None:
            raise ValueError("Regimes not yet detected. Call detect_regimes() first.")
        
        if date is None:
            date = self.regimes.index[-1]
        
        regime_id = int(self.regimes.loc[date])
        regime_name = self.get_regime_label(regime_id)
        
        return regime_id, regime_name
