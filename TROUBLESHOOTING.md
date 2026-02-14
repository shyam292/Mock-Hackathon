# 🔧 TROUBLESHOOTING GUIDE

## Common Issues and Solutions

### Issue 1: yfinance Data Download Failed

**Error Message:**
```
Failed to get ticker 'SPY' reason: Expecting value: line 1 column 1 (char 0)
Exception('%ticker%: No timezone found, symbol may be delisted')
```

**Cause:** 
- yfinance API issues (temporary outages)
- Yahoo Finance server problems
- Network connectivity issues
- Outdated yfinance package

**Solutions:**

#### Solution 1: Update yfinance (RECOMMENDED)
```bash
pip install --upgrade yfinance
```

#### Solution 2: Use Alternative Date Range
Try using a date range that ends a few days ago:
```python
loader = DataLoader(
    tickers=TICKERS,
    start_date='2015-01-01',
    end_date='2024-12-31'  # Use a past date
)
```

#### Solution 3: Use Sample Data (Automatic Fallback)
The system now automatically generates synthetic data if yfinance fails. You'll see:
```
⚠️  Using synthetic sample data for demonstration purposes.
```

This allows you to test and demonstrate the system even without internet access.

#### Solution 4: Manual Data Installation
If you have historical data files (CSV), you can modify `data_loader.py`:

```python
# In fetch_data method, add:
if os.path.exists('historical_data.csv'):
    prices = pd.read_csv('historical_data.csv', index_col=0, parse_dates=True)
    return prices
```

---

### Issue 2: Empty DataFrame Error

**Error Message:**
```
ValueError: attempt to get argmax of an empty sequence
```

**Cause:** No data was fetched (previous error cascaded)

**Solution:** Fix the data fetching issue first (see Issue 1)

---

### Issue 3: Insufficient Data

**Error Message:**
```
Insufficient data: Only X days fetched. Need at least 100 days.
```

**Solutions:**
1. Extend the date range:
   ```python
   start_date='2010-01-01'  # Earlier start date
   ```

2. Reduce minimum data requirement in `data_loader.py`:
   ```python
   if len(prices) < 50:  # Reduce from 100
   ```

---

### Issue 4: Network/Proxy Issues

**Symptoms:**
- Timeouts
- Connection errors
- SSL certificate errors

**Solutions:**

#### For Corporate Networks:
```bash
# Set proxy environment variables
set HTTP_PROXY=http://proxy.company.com:port
set HTTPS_PROXY=http://proxy.company.com:port

# Then run
python main.py
```

#### For SSL Issues:
```python
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
```

---

### Issue 5: ImportError or ModuleNotFoundError

**Error Message:**
```
ModuleNotFoundError: No module named 'streamlit'
```

**Solution:**
```bash
pip install -r requirements.txt
```

If that fails:
```bash
pip install pandas numpy matplotlib scikit-learn yfinance streamlit plotly
```

---

## Quick Diagnosis Commands

### Check Python Version
```bash
python --version  # Should be 3.8 or higher
```

### Check Package Installations
```bash
pip list | findstr "yfinance pandas numpy"
```

### Test yfinance Directly
```python
import yfinance as yf
data = yf.download('SPY', start='2024-01-01', end='2024-12-31')
print(data.head())
```

If this fails, the issue is with yfinance/Yahoo Finance, not our code.

---

## Alternative Data Sources

If yfinance continues to fail, consider:

### Option 1: Alpha Vantage
```bash
pip install alpha_vantage
```

Modify `data_loader.py` to use Alpha Vantage API (requires free API key)

### Option 2: CSV Files
Download historical data manually:
- [Yahoo Finance](https://finance.yahoo.com)
- [Google Finance](https://www.google.com/finance)
- [Investing.com](https://www.investing.com)

Place CSV files in `data/` folder and modify the loader.

### Option 3: Use Sample Data
The system includes a sample data generator that creates realistic synthetic data. This is automatically used if yfinance fails.

---

## System Requirements Check

### Minimum Requirements:
- Python 3.8+
- 4GB RAM
- Internet connection (or sample data mode)
- 100MB free disk space

### Check Your Setup:
```bash
python -c "import sys; print(f'Python {sys.version}')"
python -c "import pandas; print(f'pandas {pandas.__version__}')"
python -c "import yfinance; print(f'yfinance {yfinance.__version__}')"
```

---

## Still Having Issues?

### Check the Logs
The system provides detailed error messages. Look for:
- "Failed to fetch data" - yfinance issue
- "No data returned" - API issue
- "Insufficient data" - date range too short

### Verify Date Format
Ensure dates are in format: `YYYY-MM-DD`
```python
# Correct
start_date='2015-01-01'

# Incorrect
start_date='01/01/2015'
start_date='2015-1-1'
```

### Try Minimal Test
```python
from data.data_loader import DataLoader

loader = DataLoader(
    tickers=['SPY'],  # Just one ticker
    start_date='2023-01-01',
    end_date='2024-01-01'
)

prices = loader.fetch_data()
print(prices.head())
```

---

## Contact & Support

If none of these solutions work:

1. Check if Yahoo Finance is down: https://downdetector.com/status/yahoo/
2. Try again in a few hours (API issues are often temporary)
3. Use sample data mode for demonstration
4. Check GitHub issues for yfinance: https://github.com/ranaroussi/yfinance/issues

---

## Emergency Workaround: Always Use Sample Data

If you want to skip yfinance entirely and always use sample data:

Edit `data_loader.py`, add at the top of `fetch_data()`:
```python
def fetch_data(self):
    # Force sample data mode
    from data.sample_data import generate_sample_data
    self.prices = generate_sample_data(self.tickers, self.start_date, self.end_date)
    return self.prices
```

This ensures the system always works for demonstrations.
