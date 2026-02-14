"""
Backtester
Rolling window backtesting framework with comprehensive performance metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from utils.constants import INITIAL_CAPITAL, TRADING_DAYS, REBALANCE_FREQUENCY
from utils.helpers import (
    calculate_metrics,
    calculate_drawdown,
    calculate_rolling_sharpe
)


class Backtester:
    """
    Backtesting engine for portfolio strategies.
    
    Implements:
    - Rolling window approach (no data leakage)
    - Performance metrics calculation
    - Comparison of strategies
    - Visualization
    """
    
    def __init__(self, initial_capital: float = INITIAL_CAPITAL):
        """
        Initialize Backtester.
        
        Args:
            initial_capital: Starting portfolio value
        """
        self.initial_capital = initial_capital
        self.results = {}
        
    def run_backtest(
        self,
        allocations: pd.DataFrame,
        returns: pd.DataFrame,
        strategy_name: str = 'Strategy'
    ) -> Dict:
        """
        Run backtest for a given allocation strategy.
        
        Args:
            allocations: DataFrame of daily allocations
            returns: DataFrame of asset returns
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary of backtest results
        """
        print(f"\nRunning backtest for: {strategy_name}")
        
        # Align data
        common_index = allocations.index.intersection(returns.index)
        allocations = allocations.loc[common_index]
        returns = returns.loc[common_index]
        
        # Calculate portfolio returns
        # Use previous day's allocation with today's returns (realistic)
        portfolio_returns = (allocations.shift(1) * returns).sum(axis=1).dropna()
        
        # Calculate portfolio value over time
        portfolio_value = self.initial_capital * (1 + portfolio_returns).cumprod()
        
        # Calculate drawdown
        drawdown = calculate_drawdown(portfolio_value)
        
        # Calculate metrics
        metrics = calculate_metrics(portfolio_returns)
        
        # Warning if Sharpe ratio is unrealistically high
        if metrics['Sharpe Ratio'] > 3.0:
            print(f"⚠️  WARNING: Sharpe Ratio = {metrics['Sharpe Ratio']:.2f} is very high!")
            print("   This may indicate overfitting or data issues.")
        
        # Store results
        results = {
            'strategy_name': strategy_name,
            'portfolio_returns': portfolio_returns,
            'portfolio_value': portfolio_value,
            'drawdown': drawdown,
            'metrics': metrics,
            'allocations': allocations
        }
        
        self.results[strategy_name] = results
        
        # Print summary
        self._print_metrics(metrics)
        
        return results
    
    def _print_metrics(self, metrics: Dict[str, float]):
        """Print performance metrics in formatted table."""
        print("\nPerformance Metrics:")
        print("-" * 50)
        for metric, value in metrics.items():
            if 'Ratio' in metric or 'CAGR' in metric or 'Return' in metric or 'Volatility' in metric or 'Drawdown' in metric:
                print(f"  {metric:<25}: {value:>10.2%}")
            else:
                print(f"  {metric:<25}: {value:>10.2f}")
        print("-" * 50)
    
    def compare_strategies(self, strategy_names: list = None) -> pd.DataFrame:
        """
        Compare multiple strategies side by side.
        
        Args:
            strategy_names: List of strategy names to compare (default: all)
            
        Returns:
            DataFrame of comparative metrics
        """
        if strategy_names is None:
            strategy_names = list(self.results.keys())
        
        comparison = {}
        
        for name in strategy_names:
            if name in self.results:
                comparison[name] = self.results[name]['metrics']
        
        comparison_df = pd.DataFrame(comparison).T
        
        print("\n" + "="*70)
        print("STRATEGY COMPARISON")
        print("="*70)
        print(comparison_df.to_string())
        print("="*70)
        
        return comparison_df
    
    def plot_equity_curves(self, strategy_names: list = None, save_path: str = None):
        """
        Plot equity curves for strategies.
        
        Args:
            strategy_names: List of strategy names to plot (default: all)
            save_path: Path to save the figure (optional)
        """
        if strategy_names is None:
            strategy_names = list(self.results.keys())
        
        plt.figure(figsize=(14, 6))
        
        for name in strategy_names:
            if name in self.results:
                portfolio_value = self.results[name]['portfolio_value']
                plt.plot(portfolio_value.index, portfolio_value.values, label=name, linewidth=2)
        
        plt.title('Portfolio Equity Curves', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Equity curve saved to: {save_path}")
        
        return plt.gcf()
    
    def plot_drawdowns(self, strategy_names: list = None, save_path: str = None):
        """
        Plot drawdown curves for strategies.
        
        Args:
            strategy_names: List of strategy names to plot (default: all)
            save_path: Path to save the figure (optional)
        """
        if strategy_names is None:
            strategy_names = list(self.results.keys())
        
        plt.figure(figsize=(14, 6))
        
        for name in strategy_names:
            if name in self.results:
                drawdown = self.results[name]['drawdown']
                plt.plot(drawdown.index, drawdown.values * 100, label=name, linewidth=2)
        
        plt.title('Portfolio Drawdowns', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Drawdown (%)', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Drawdown curve saved to: {save_path}")
        
        return plt.gcf()
    
    def plot_rolling_sharpe(self, strategy_names: list = None, window: int = 60, save_path: str = None):
        """
        Plot rolling Sharpe ratios for strategies.
        
        Args:
            strategy_names: List of strategy names to plot (default: all)
            window: Rolling window size
            save_path: Path to save the figure (optional)
        """
        if strategy_names is None:
            strategy_names = list(self.results.keys())
        
        plt.figure(figsize=(14, 6))
        
        for name in strategy_names:
            if name in self.results:
                returns = self.results[name]['portfolio_returns']
                rolling_sharpe = calculate_rolling_sharpe(returns, window=window)
                plt.plot(rolling_sharpe.index, rolling_sharpe.values, label=name, linewidth=2)
        
        plt.title(f'Rolling Sharpe Ratio ({window}-day window)', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Sharpe Ratio', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.axhline(y=1, color='green', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Rolling Sharpe saved to: {save_path}")
        
        return plt.gcf()
    
    def plot_allocations(self, strategy_name: str, save_path: str = None):
        """
        Plot allocation weights over time (stacked area chart).
        
        Args:
            strategy_name: Name of the strategy
            save_path: Path to save the figure (optional)
        """
        if strategy_name not in self.results:
            raise ValueError(f"Strategy '{strategy_name}' not found in results")
        
        allocations = self.results[strategy_name]['allocations']
        
        plt.figure(figsize=(14, 6))
        
        plt.stackplot(
            allocations.index,
            *[allocations[col].values for col in allocations.columns],
            labels=allocations.columns,
            alpha=0.8
        )
        
        plt.title(f'Portfolio Allocation Over Time - {strategy_name}', fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Weight', fontsize=12)
        plt.legend(loc='upper left', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Allocation chart saved to: {save_path}")
        
        return plt.gcf()
    
    def generate_report(self, strategy_names: list = None, output_dir: str = 'reports'):
        """
        Generate comprehensive backtest report with all charts.
        
        Args:
            strategy_names: List of strategy names (default: all)
            output_dir: Directory to save reports
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if strategy_names is None:
            strategy_names = list(self.results.keys())
        
        print(f"\nGenerating backtest report for {len(strategy_names)} strategies...")
        
        # Comparison table
        comparison_df = self.compare_strategies(strategy_names)
        
        # Save charts
        self.plot_equity_curves(strategy_names, f'{output_dir}/equity_curves.png')
        plt.close()
        
        self.plot_drawdowns(strategy_names, f'{output_dir}/drawdowns.png')
        plt.close()
        
        self.plot_rolling_sharpe(strategy_names, save_path=f'{output_dir}/rolling_sharpe.png')
        plt.close()
        
        # Individual allocation charts
        for name in strategy_names:
            self.plot_allocations(name, f'{output_dir}/allocation_{name.replace(" ", "_")}.png')
            plt.close()
        
        print(f"\nReport generated in: {output_dir}/")
        
        return comparison_df
