"""
YFinance Data Extraction Diagnostic
Tests different methods to extract data from yfinance downloads
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*80)
print("YFINANCE DATA EXTRACTION DIAGNOSTIC")
print("="*80)

# Test parameters
test_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

print(f"\nTest Date Range: {start_date.date()} to {end_date.date()}")
print(f"Test Tickers: {test_tickers}\n")

# ============================================================================
# TEST 1: Default download
# ============================================================================
print("-"*80)
print("TEST 1: Default yfinance.download()")
print("-"*80)

try:
    data1 = yf.download(test_tickers, start=start_date, end=end_date, progress=False)

    print(f"✓ Download successful")
    print(f"  Data type: {type(data1)}")
    print(f"  Shape: {data1.shape}")
    print(f"  Columns type: {type(data1.columns)}")

    if isinstance(data1.columns, pd.MultiIndex):
        print(f"  MultiIndex levels: {data1.columns.nlevels}")
        print(f"  Level 0 values: {data1.columns.get_level_values(0).unique().tolist()}")
        print(f"  Level 1 values (first 5): {data1.columns.get_level_values(1).unique().tolist()[:5]}")
    else:
        print(f"  Single-level columns: {data1.columns.tolist()}")

    print(f"\n  First few rows:")
    print(data1.head(3))

except Exception as e:
    print(f"✗ Error: {e}")

# ============================================================================
# TEST 2: group_by='column'
# ============================================================================
print("\n" + "-"*80)
print("TEST 2: yfinance.download() with group_by='column'")
print("-"*80)

try:
    data2 = yf.download(test_tickers, start=start_date, end=end_date,
                        progress=False, group_by='column')

    print(f"✓ Download successful")
    print(f"  Data type: {type(data2)}")
    print(f"  Shape: {data2.shape}")
    print(f"  Columns type: {type(data2.columns)}")

    if isinstance(data2.columns, pd.MultiIndex):
        print(f"  MultiIndex levels: {data2.columns.nlevels}")
        print(f"  Level 0 values: {data2.columns.get_level_values(0).unique().tolist()}")
        print(f"  Level 1 values (first 5): {data2.columns.get_level_values(1).unique().tolist()[:5]}")

        # Try extracting Adj Close
        if 'Adj Close' in data2.columns.get_level_values(0):
            adj_close = data2['Adj Close']
            print(f"\n  Extracted 'Adj Close':")
            print(f"    Shape: {adj_close.shape}")
            print(f"    Columns: {adj_close.columns.tolist()}")
            print(f"    Sample:")
            print(adj_close.head(3))
    else:
        print(f"  Single-level columns: {data2.columns.tolist()}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 3: group_by='ticker'
# ============================================================================
print("\n" + "-"*80)
print("TEST 3: yfinance.download() with group_by='ticker'")
print("-"*80)

try:
    data3 = yf.download(test_tickers, start=start_date, end=end_date,
                        progress=False, group_by='ticker')

    print(f"✓ Download successful")
    print(f"  Data type: {type(data3)}")
    print(f"  Shape: {data3.shape}")
    print(f"  Columns type: {type(data3.columns)}")

    if isinstance(data3.columns, pd.MultiIndex):
        print(f"  MultiIndex levels: {data3.columns.nlevels}")
        print(f"  Level 0 values (first 5): {data3.columns.get_level_values(0).unique().tolist()[:5]}")
        print(f"  Level 1 values: {data3.columns.get_level_values(1).unique().tolist()}")

        # Try extracting Adj Close
        adj_close_dict = {}
        for ticker in test_tickers:
            if (ticker, 'Adj Close') in data3.columns:
                adj_close_dict[ticker] = data3[(ticker, 'Adj Close')]

        if adj_close_dict:
            adj_close = pd.DataFrame(adj_close_dict)
            print(f"\n  Extracted 'Adj Close' per ticker:")
            print(f"    Shape: {adj_close.shape}")
            print(f"    Columns: {adj_close.columns.tolist()}")
            print(f"    Sample:")
            print(adj_close.head(3))
    else:
        print(f"  Single-level columns: {data3.columns.tolist()}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# ============================================================================
# TEST 4: Individual ticker downloads
# ============================================================================
print("\n" + "-"*80)
print("TEST 4: Individual ticker downloads")
print("-"*80)

results = {}
for ticker in test_tickers[:3]:  # Test first 3
    try:
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        results[ticker] = {
            'success': True,
            'shape': data.shape,
            'columns': data.columns.tolist(),
            'has_adj_close': 'Adj Close' in data.columns
        }
        print(f"✓ {ticker}: {data.shape}, has Adj Close: {'Adj Close' in data.columns}")
    except Exception as e:
        results[ticker] = {'success': False, 'error': str(e)}
        print(f"✗ {ticker}: {e}")

# ============================================================================
# TEST 5: Extraction function test
# ============================================================================
print("\n" + "-"*80)
print("TEST 5: Testing extraction functions")
print("-"*80)

def extract_method_1(raw_data, tickers):
    """Original method from your code"""
    if isinstance(raw_data.columns, pd.MultiIndex):
        if 'Adj Close' in raw_data.columns.get_level_values(0):
            return raw_data['Adj Close'].copy()
    return None

def extract_method_2(raw_data, tickers):
    """Alternative: iterate and build DataFrame"""
    if not isinstance(raw_data.columns, pd.MultiIndex):
        return None

    price_data = {}
    for ticker in tickers:
        try:
            # Try (ticker, 'Adj Close') if group_by='ticker'
            if (ticker, 'Adj Close') in raw_data.columns:
                price_data[ticker] = raw_data[(ticker, 'Adj Close')]
            # Try ('Adj Close', ticker) if group_by='column'
            elif ('Adj Close', ticker) in raw_data.columns:
                price_data[ticker] = raw_data[('Adj Close', ticker)]
        except:
            pass

    return pd.DataFrame(price_data) if price_data else None

# Test on data2 (group_by='column')
print("\nTesting on group_by='column' data:")
result1 = extract_method_1(data2, test_tickers)
if result1 is not None:
    print(f"  Method 1: ✓ Extracted {result1.shape[1]} tickers")
else:
    print(f"  Method 1: ✗ Failed")

result2 = extract_method_2(data2, test_tickers)
if result2 is not None:
    print(f"  Method 2: ✓ Extracted {result2.shape[1]} tickers")
else:
    print(f"  Method 2: ✗ Failed")

# ============================================================================
# SUMMARY & RECOMMENDATION
# ============================================================================
print("\n" + "="*80)
print("SUMMARY & RECOMMENDED APPROACH")
print("="*80)

print("\n" + "="*80)
print("Run this diagnostic to understand the data structure before fixing the main code")
print("="*80)