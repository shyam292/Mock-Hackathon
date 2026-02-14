"""
Quick Test Script
Verifies the system is working correctly.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        import pandas
        import numpy
        import matplotlib
        import sklearn
        import yfinance
        print("  ✓ All core packages imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        print("  → Run: pip install -r requirements.txt")
        return False


def test_data_loading():
    """Test data loading with both real and sample data."""
    print("\nTesting data loading...")
    try:
        from data.data_loader import DataLoader
        
        # Test with a short date range
        loader = DataLoader(
            tickers=['SPY'],
            start_date='2024-01-01',
            end_date='2024-12-31'
        )
        
        prices = loader.fetch_data()
        
        if len(prices) > 0:
            print(f"  ✓ Data loaded successfully ({len(prices)} days)")
            return True
        else:
            print("  ✗ No data loaded")
            return False
            
    except Exception as e:
        print(f"  ✗ Data loading failed: {e}")
        return False


def test_sample_data():
    """Test sample data generation."""
    print("\nTesting sample data generation...")
    try:
        from data.sample_data import generate_sample_data
        
        prices = generate_sample_data(
            tickers=['SPY', 'QQQ', 'TLT', 'GLD'],
            start_date='2020-01-01',
            end_date='2024-01-01'
        )
        
        if len(prices) > 100:
            print(f"  ✓ Sample data generated successfully ({len(prices)} days)")
            return True
        else:
            print("  ✗ Insufficient sample data generated")
            return False
            
    except Exception as e:
        print(f"  ✗ Sample data generation failed: {e}")
        return False


def test_modules():
    """Test that all custom modules can be imported."""
    print("\nTesting custom modules...")
    try:
        from data.data_loader import DataLoader
        from regime.regime_detector import RegimeDetector
        from allocation.allocator import Allocator
        from risk.risk_manager import RiskManager
        from backtest.backtester import Backtester
        from stress_test.stress_tester import StressTester
        from explainability.explainer import Explainer
        from utils.helpers import calculate_metrics
        from utils.constants import TICKERS
        
        print("  ✓ All custom modules imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Module import failed: {e}")
        return False


def main():
    """Run all tests."""
    print("="*70)
    print("PORTFOLIO ENGINE - SYSTEM TEST")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Package Imports", test_imports()))
    results.append(("Custom Modules", test_modules()))
    results.append(("Sample Data", test_sample_data()))
    results.append(("Data Loading", test_data_loading()))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:<20}: {status}")
    
    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! System is ready to run.")
        print("\nNext steps:")
        print("  1. Run full analysis: python main.py")
        print("  2. Launch dashboard: streamlit run app/dashboard.py")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        print("\nTroubleshooting:")
        print("  1. Install missing packages: pip install -r requirements.txt")
        print("  2. See TROUBLESHOOTING.md for detailed help")
    
    print("="*70)


if __name__ == "__main__":
    main()
