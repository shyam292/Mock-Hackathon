"""
Stress Tester
Simulates crisis scenarios and tests portfolio resilience.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from utils.constants import (
    CRASH_SHOCK,
    CRASH_DURATION,
    VOL_SPIKE_MULTIPLIER,
    TRADING_DAYS
)
from utils.helpers import calculate_drawdown, calculate_metrics


class StressTester:
    """
    Stress testing engine for portfolio strategies.
    
    Tests portfolio resilience under:
    - Market crash scenarios
    - Volatility spikes
    - Correlation spikes
    """
    
    def __init__(self):
        """Initialize StressTester."""
        self.stress_results = {}
        
    def inject_crash_scenario(
        self,
        returns: pd.DataFrame,
        start_date: pd.Timestamp,
        shock: float = CRASH_SHOCK,
        duration: int = CRASH_DURATION
    ) -> pd.DataFrame:
        """
        Inject a market crash scenario into returns.
        
        Args:
            returns: DataFrame of asset returns
            start_date: Start date of crash
            shock: Daily shock magnitude (negative)
            duration: Number of days
            
        Returns:
            Modified returns DataFrame
        """
        stressed_returns = returns.copy()
        
        # Find index position of start date
        if start_date not in returns.index:
            # Find closest date
            start_date = returns.index[returns.index >= start_date][0]
        
        start_idx = returns.index.get_loc(start_date)
        
        # Apply shock to equity assets (SPY, QQQ)
        equity_assets = ['SPY', 'QQQ']
        
        for i in range(duration):
            if start_idx + i < len(returns):
                for asset in equity_assets:
                    if asset in stressed_returns.columns:
                        stressed_returns.iloc[start_idx + i, stressed_returns.columns.get_loc(asset)] = shock
        
        return stressed_returns
    
    def inject_volatility_spike(
        self,
        returns: pd.DataFrame,
        start_date: pd.Timestamp,
        multiplier: float = VOL_SPIKE_MULTIPLIER,
        duration: int = 10
    ) -> pd.DataFrame:
        """
        Inject a volatility spike scenario.
        
        Args:
            returns: DataFrame of asset returns
            start_date: Start date of volatility spike
            multiplier: Volatility multiplication factor
            duration: Number of days
            
        Returns:
            Modified returns DataFrame
        """
        stressed_returns = returns.copy()
        
        # Find index position
        if start_date not in returns.index:
            start_date = returns.index[returns.index >= start_date][0]
        
        start_idx = returns.index.get_loc(start_date)
        
        # Amplify returns (keeping sign but increasing magnitude)
        for i in range(duration):
            if start_idx + i < len(returns):
                stressed_returns.iloc[start_idx + i] = returns.iloc[start_idx + i] * multiplier
        
        return stressed_returns
    
    def inject_correlation_spike(
        self,
        returns: pd.DataFrame,
        start_date: pd.Timestamp,
        direction: str = 'down',
        duration: int = 5
    ) -> pd.DataFrame:
        """
        Inject a correlation spike (all assets move together).
        
        Args:
            returns: DataFrame of asset returns
            start_date: Start date of correlation spike
            direction: 'down' or 'up'
            duration: Number of days
            
        Returns:
            Modified returns DataFrame
        """
        stressed_returns = returns.copy()
        
        # Find index position
        if start_date not in returns.index:
            start_date = returns.index[returns.index >= start_date][0]
        
        start_idx = returns.index.get_loc(start_date)
        
        # Apply uniform shock
        shock = -0.03 if direction == 'down' else 0.03
        
        for i in range(duration):
            if start_idx + i < len(returns):
                stressed_returns.iloc[start_idx + i] = shock
        
        return stressed_returns
    
    def run_stress_test(
        self,
        allocations: pd.DataFrame,
        returns: pd.DataFrame,
        scenario_name: str,
        stressed_returns: pd.DataFrame
    ) -> Dict:
        """
        Run stress test with specific scenario.
        
        Args:
            allocations: DataFrame of allocations
            returns: Original returns
            scenario_name: Name of stress scenario
            stressed_returns: Returns with injected stress
            
        Returns:
            Dictionary of stress test results
        """
        print(f"\nRunning stress test: {scenario_name}")
        
        # Calculate portfolio returns under stress
        portfolio_returns_stress = (allocations.shift(1) * stressed_returns).sum(axis=1).dropna()
        portfolio_value_stress = (1 + portfolio_returns_stress).cumprod()
        
        # Calculate metrics
        metrics_stress = calculate_metrics(portfolio_returns_stress)
        
        # Calculate drawdown
        drawdown_stress = calculate_drawdown(portfolio_value_stress)
        max_drawdown_stress = abs(drawdown_stress.min())
        
        # Compare to original
        portfolio_returns_original = (allocations.shift(1) * returns).sum(axis=1).dropna()
        metrics_original = calculate_metrics(portfolio_returns_original)
        
        results = {
            'scenario_name': scenario_name,
            'portfolio_returns': portfolio_returns_stress,
            'portfolio_value': portfolio_value_stress,
            'drawdown': drawdown_stress,
            'max_drawdown': max_drawdown_stress,
            'metrics': metrics_stress,
            'metrics_original': metrics_original,
            'stressed_returns': stressed_returns
        }
        
        self.stress_results[scenario_name] = results
        
        # Print comparison
        print(f"\nStress Test Results: {scenario_name}")
        print("-" * 60)
        print(f"{'Metric':<25} {'Original':>15} {'Stressed':>15}")
        print("-" * 60)
        
        for metric in ['CAGR', 'Sharpe Ratio', 'Max Drawdown', 'Annualized Volatility']:
            orig = metrics_original[metric]
            stress = metrics_stress[metric]
            print(f"{metric:<25} {orig:>14.2%} {stress:>14.2%}")
        
        print("-" * 60)
        
        return results
    
    def run_comprehensive_stress_tests(
        self,
        allocations_with_risk: pd.DataFrame,
        allocations_without_risk: pd.DataFrame,
        returns: pd.DataFrame
    ) -> Dict:
        """
        Run comprehensive stress tests comparing portfolios with/without risk management.
        
        Args:
            allocations_with_risk: Allocations with risk management
            allocations_without_risk: Allocations without risk management
            returns: Original returns
            
        Returns:
            Dictionary of all stress test results
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE STRESS TESTING")
        print("="*70)
        
        # Select test dates (use middle of dataset)
        test_date = returns.index[len(returns) // 2]
        
        scenarios = {}
        
        # Scenario 1: Market Crash
        print("\n--- SCENARIO 1: MARKET CRASH ---")
        crash_returns = self.inject_crash_scenario(returns, test_date)
        
        scenarios['Crash - With Risk Mgmt'] = self.run_stress_test(
            allocations_with_risk,
            returns,
            'Crash - With Risk Mgmt',
            crash_returns
        )
        
        scenarios['Crash - Without Risk Mgmt'] = self.run_stress_test(
            allocations_without_risk,
            returns,
            'Crash - Without Risk Mgmt',
            crash_returns
        )
        
        # Scenario 2: Volatility Spike
        print("\n--- SCENARIO 2: VOLATILITY SPIKE ---")
        vol_spike_returns = self.inject_volatility_spike(returns, test_date)
        
        scenarios['Vol Spike - With Risk Mgmt'] = self.run_stress_test(
            allocations_with_risk,
            returns,
            'Vol Spike - With Risk Mgmt',
            vol_spike_returns
        )
        
        scenarios['Vol Spike - Without Risk Mgmt'] = self.run_stress_test(
            allocations_without_risk,
            returns,
            'Vol Spike - Without Risk Mgmt',
            vol_spike_returns
        )
        
        # Scenario 3: Correlation Spike
        print("\n--- SCENARIO 3: CORRELATION SPIKE ---")
        corr_spike_returns = self.inject_correlation_spike(returns, test_date, direction='down')
        
        scenarios['Corr Spike - With Risk Mgmt'] = self.run_stress_test(
            allocations_with_risk,
            returns,
            'Corr Spike - With Risk Mgmt',
            corr_spike_returns
        )
        
        scenarios['Corr Spike - Without Risk Mgmt'] = self.run_stress_test(
            allocations_without_risk,
            returns,
            'Corr Spike - Without Risk Mgmt',
            corr_spike_returns
        )
        
        self.stress_results = scenarios
        
        return scenarios
    
    def plot_stress_comparison(self, scenario_type: str = 'Crash', save_path: str = None):
        """
        Plot comparison of portfolio performance under stress.
        
        Args:
            scenario_type: Type of scenario ('Crash', 'Vol Spike', 'Corr Spike')
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Find matching scenarios
        scenarios = [k for k in self.stress_results.keys() if scenario_type in k]
        
        if len(scenarios) < 2:
            print(f"Not enough scenarios found for {scenario_type}")
            return
        
        # Plot portfolio values
        ax1 = axes[0]
        for scenario_name in scenarios:
            pv = self.stress_results[scenario_name]['portfolio_value']
            label = 'With Risk Mgmt' if 'With' in scenario_name else 'Without Risk Mgmt'
            ax1.plot(pv.index, pv.values, label=label, linewidth=2)
        
        ax1.set_title(f'Portfolio Value - {scenario_type} Scenario', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot drawdowns
        ax2 = axes[1]
        for scenario_name in scenarios:
            dd = self.stress_results[scenario_name]['drawdown']
            label = 'With Risk Mgmt' if 'With' in scenario_name else 'Without Risk Mgmt'
            ax2.plot(dd.index, dd.values * 100, label=label, linewidth=2)
        
        ax2.set_title(f'Drawdown - {scenario_type} Scenario', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Stress test comparison saved to: {save_path}")
        
        return fig
    
    def generate_stress_report(self) -> pd.DataFrame:
        """
        Generate summary report of all stress tests.
        
        Returns:
            DataFrame of stress test results
        """
        report_data = {}
        
        for scenario_name, results in self.stress_results.items():
            report_data[scenario_name] = {
                'Max Drawdown': results['max_drawdown'],
                'Final Value': results['portfolio_value'].iloc[-1],
                'Sharpe Ratio': results['metrics']['Sharpe Ratio'],
                'Total Return': results['metrics']['Total Return']
            }
        
        report_df = pd.DataFrame(report_data).T
        
        print("\n" + "="*70)
        print("STRESS TEST SUMMARY REPORT")
        print("="*70)
        print(report_df.to_string())
        print("="*70)
        
        return report_df
