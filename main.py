"""
Main Execution Script
Runs the complete Autonomous Adaptive Portfolio & Risk Management Engine.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from datetime import datetime
import os

# Import all modules
from data.data_loader import DataLoader
from regime.regime_detector import RegimeDetector
from allocation.allocator import Allocator
from risk.risk_manager import RiskManager
from backtest.backtester import Backtester
from stress_test.stress_tester import StressTester
from explainability.explainer import Explainer
from utils.constants import TICKERS
from utils.helpers import calculate_metrics


def main():
    """
    Main execution pipeline.
    """
    print("="*80)
    print("AUTONOMOUS ADAPTIVE PORTFOLIO & RISK MANAGEMENT ENGINE")
    print("="*80)
    print()
    
    # =========================================================================
    # STEP 1: DATA INGESTION
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: DATA INGESTION")
    print("="*80)
    
    loader = DataLoader(
        tickers=TICKERS,
        start_date='2015-01-01',
        end_date=None  # Let DataLoader determine safe end date
    )
    
    prices, returns, features = loader.get_data()
    
    print(f"\nData Summary:")
    print(f"  Date Range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Number of Days: {len(prices)}")
    print(f"  Assets: {', '.join(TICKERS)}")
    
    # =========================================================================
    # STEP 2: REGIME DETECTION
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: REGIME DETECTION")
    print("="*80)
    
    regime_detector = RegimeDetector()
    regimes = regime_detector.detect_regimes(returns, features, method='rule_based')
    
    # =========================================================================
    # STEP 3: ALLOCATION ENGINE
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: ALLOCATION ENGINE")
    print("="*80)
    
    allocator = Allocator(tickers=TICKERS)
    
    print("\nCalculating allocations WITHOUT risk management...")
    allocations_no_risk = allocator.calculate_allocations(
        regimes=regimes,
        returns=returns,
        features=features,
        method='risk_parity'
    )
    
    # =========================================================================
    # STEP 4: RISK MANAGEMENT
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 4: RISK MANAGEMENT ENGINE")
    print("="*80)
    
    risk_manager = RiskManager()
    
    print("\nApplying risk controls...")
    allocations_with_risk, exposure_scalars = risk_manager.apply_risk_controls(
        allocations=allocations_no_risk,
        returns=returns,
        apply_vol_targeting=True,
        apply_drawdown_control=True
    )
    
    print(f"\nRisk Management Summary:")
    print(f"  Average Exposure Scalar: {exposure_scalars.mean():.2f}")
    print(f"  Min Exposure Scalar: {exposure_scalars.min():.2f}")
    print(f"  Max Exposure Scalar: {exposure_scalars.max():.2f}")
    
    # =========================================================================
    # STEP 5: BACKTESTING
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 5: BACKTESTING")
    print("="*80)
    
    backtester = Backtester(initial_capital=100000)
    
    # Backtest without risk management
    results_no_risk = backtester.run_backtest(
        allocations=allocations_no_risk,
        returns=returns,
        strategy_name='Without Risk Management'
    )
    
    # Backtest with risk management
    results_with_risk = backtester.run_backtest(
        allocations=allocations_with_risk,
        returns=returns,
        strategy_name='With Risk Management'
    )
    
    # Compare strategies
    print("\n")
    comparison_df = backtester.compare_strategies()
    
    # =========================================================================
    # STEP 6: STRESS TESTING
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 6: STRESS TESTING")
    print("="*80)
    
    stress_tester = StressTester()
    
    stress_results = stress_tester.run_comprehensive_stress_tests(
        allocations_with_risk=allocations_with_risk,
        allocations_without_risk=allocations_no_risk,
        returns=returns
    )
    
    print("\n")
    stress_summary = stress_tester.generate_stress_report()
    
    # =========================================================================
    # STEP 7: EXPLAINABILITY
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 7: EXPLAINABILITY")
    print("="*80)
    
    explainer = Explainer()
    
    # Generate sample explanations
    sample_explanations = explainer.generate_summary_explanation(
        regimes=regimes,
        allocations=allocations_with_risk,
        risk_adjustments=exposure_scalars,
        n_samples=3
    )
    
    # =========================================================================
    # STEP 8: GENERATE REPORTS
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 8: GENERATING REPORTS")
    print("="*80)
    
    # Create reports directory
    os.makedirs('reports', exist_ok=True)
    
    # Generate backtest report
    print("\nGenerating backtest visualizations...")
    backtester.generate_report(
        strategy_names=['With Risk Management', 'Without Risk Management'],
        output_dir='reports'
    )
    
    # Generate stress test visualizations
    print("\nGenerating stress test visualizations...")
    for scenario_type in ['Crash', 'Vol Spike', 'Corr Spike']:
        stress_tester.plot_stress_comparison(
            scenario_type=scenario_type,
            save_path=f'reports/stress_{scenario_type.replace(" ", "_")}.png'
        )
    
    # Save explanations
    explainer.save_explanations('reports/explanations.txt')
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*80)
    print("EXECUTION SUMMARY")
    print("="*80)
    
    print("\n✅ All modules executed successfully!")
    print("\nKey Findings:")
    
    metrics_with = results_with_risk['metrics']
    metrics_without = results_no_risk['metrics']
    
    print(f"\n1. Risk Management Impact:")
    print(f"   Sharpe Ratio: {metrics_without['Sharpe Ratio']:.2f} → {metrics_with['Sharpe Ratio']:.2f}")
    print(f"   Max Drawdown: {metrics_without['Max Drawdown']:.2%} → {metrics_with['Max Drawdown']:.2%}")
    
    print(f"\n2. Returns:")
    print(f"   Without Risk Mgmt: {metrics_without['CAGR']:.2%} CAGR")
    print(f"   With Risk Mgmt: {metrics_with['CAGR']:.2%} CAGR")
    
    print(f"\n3. Risk-Adjusted Performance:")
    print(f"   Calmar Ratio (With Risk): {metrics_with['Calmar Ratio']:.2f}")
    print(f"   Sortino Ratio (With Risk): {metrics_with['Sortino Ratio']:.2f}")
    
    print("\n4. Reports Generated:")
    print("   - Equity curves: reports/equity_curves.png")
    print("   - Drawdowns: reports/drawdowns.png")
    print("   - Rolling Sharpe: reports/rolling_sharpe.png")
    print("   - Stress tests: reports/stress_*.png")
    print("   - Explanations: reports/explanations.txt")
    
    print("\n" + "="*80)
    print("To launch the interactive dashboard, run:")
    print("  streamlit run app/dashboard.py")
    print("="*80)
    print()


if __name__ == "__main__":
    main()
