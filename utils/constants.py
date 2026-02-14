"""
Constants and Configuration
"""

# Asset Universe
TICKERS = ['SPY', 'QQQ', 'TLT', 'GLD']

# Asset Names
ASSET_NAMES = {
    'SPY': 'S&P 500',
    'QQQ': 'NASDAQ 100',
    'TLT': 'Long-Term Treasury',
    'GLD': 'Gold'
}

# Regime Labels
REGIME_LABELS = {
    0: 'Trending Up',
    1: 'Trending Down',
    2: 'High Volatility',
    3: 'Crash'
}

# Technical Parameters
ROLLING_WINDOW = 20
MA_SHORT = 50
MA_LONG = 200

# Risk Parameters
TARGET_VOLATILITY = 0.12  # 12% annualized
MAX_DRAWDOWN_THRESHOLD = 0.15  # 15%
STOP_LOSS_THRESHOLD = 0.10  # 10%

# Backtesting Parameters
INITIAL_CAPITAL = 100000
TRAINING_WINDOW = 252  # 1 year
REBALANCE_FREQUENCY = 5  # days

# Stress Test Parameters
CRASH_SHOCK = -0.05  # -5% daily
CRASH_DURATION = 5  # days
VOL_SPIKE_MULTIPLIER = 3.0

# Trading Days per Year
TRADING_DAYS = 252
