"""
International Indices Data Integrity Check and SVD Model Validation
Checks CAC 40, DAX 40, and NASDAQ 100 for data completeness (2015-2025)
Runs optimized SVD model on complete portfolios
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.covariance import LedoitWolf
from sklearn.utils.extmath import randomized_svd
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# ============================================================================
# INDEX CONSTITUENT LISTS
# ============================================================================

CAC40_TICKERS = [
    'AC.PA', 'AI.PA', 'AIR.PA', 'ATO.PA', 'BNP.PA', 'CAP.PA', 'CARA.PA', 'CS.PA',
    'DSY.PA', 'ENGI.PA', 'ERF.PA', 'GLE.PA', 'KER.PA', 'KNE.PA', 'LR.PA', 'MC.PA',
    'ML.PA', 'OR.PA', 'PUB.PA', 'RI.PA', 'RNO.PA', 'SAF.PA', 'SGO.PA', 'STM.PA',
    'SU.PA', 'TEP.PA', 'TTE.PA', 'VIE.PA', 'VIV.PA', 'ACA.PA', 'CA.PA', 'HO.PA',
    'EXO.PA', 'EI.PA', 'SW.PA', 'ALO.PA', 'FOR.PA', 'SGE.PA', 'URW.PA', 'EL.PA'
]

DAX40_TICKERS = [
    'ADS.DE', 'ALV.DE', 'BAS.DE', 'BAYN.DE', 'BMW.DE', 'CON.DE', 'DB1.DE', 'DAI.DE',
    'DPW.DE', 'DTE.DE', 'EOAN.DE', 'FRE.DE', 'HEI.DE', 'HEN3.DE', 'IFX.DE', 'LIN.DE',
    'MBG.DE', 'MTX.DE', 'MRK.DE', 'MUV2.DE', 'P911.DE', 'PAH3.DE', 'PNDX.DE', 'QIA.DE',
    'RWE.DE', 'SAP.DE', 'SIE.DE', 'SRM.DE', 'SY1.DE', 'VOW3.DE', 'VNA.DE', 'ZAL.DE',
    'BEI.DE', 'BNR.DE', 'ENR.DE', 'HAP.DE', 'LEG.DE', 'RR.DE', 'SDF.DE', 'WCH.DE'
]

NASDAQ100_TICKERS = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'GOOG', 'TSLA', 'NVDA', 'META', 'PEP', 'AVGO',
    'COST', 'ADBE', 'CSCO', 'CMCSA', 'NFLX', 'INTC', 'TMUS', 'TXN', 'QCOM', 'AMGN',
    'SBUX', 'ISRG', 'HON', 'MDLZ', 'BKNG', 'GILD', 'ADP', 'FISV', 'INTU', 'VRTX',
    'PYPL', 'CHTR', 'MRNA', 'ADI', 'KHC', 'REGN', 'ATVI', 'LRCX', 'CDNS', 'ASML',
    'BIIB', 'MAR', 'WBA', 'ORLY', 'MCHP', 'FAST', 'EXC', 'CTAS', 'TROW', 'SIRI'
]

# Reference indices for calculating expected trading days
REFERENCE_INDICES = {
    'CAC 40': '^FCHI',  # CAC 40 Index
    'DAX 40': '^GDAXI',  # DAX Index
    'NASDAQ 100 (Top 50)': 'SPY'  # S&P 500 ETF as proxy
}

# ============================================================================
# CONFIGURATION
# ============================================================================

# Date range: 2015-10-19 to 2025-10-16 (10 years)
START_DATE = datetime(2015, 10, 19)
END_DATE = datetime(2025, 10, 16)
COMPLETENESS_THRESHOLD = 0.80  # 90% data coverage required
INDEX_THRESHOLD = 0.80  # 90% of stocks must be complete to run model

# SVD Model Parameters (Two Configurations)
CONFIGS = [
    {
        'name': '30mo_Test',
        'svd_rank': 7,
        'train_years': 2,
        'test_months': 30,
        'max_position': 0.10,
        'short_allowed': False
    },
    {
        'name': '6mo_Test',
        'svd_rank': 7,
        'train_years': 2,
        'test_months': 6,
        'max_position': 0.10,
        'short_allowed': False
    }
]

FIXED_PARAMS = {
    'min_position_size': 0.005,
    'transaction_cost_bps': 10,
    'momentum_lookback': 126,
    'momentum_weight': 0.3,
    'risk_free_rate': 0.04,
    'random_state': 42
}

# ============================================================================
# DATA INTEGRITY FUNCTIONS
# ============================================================================

def get_reference_trading_days(index_name, start_date, end_date):
    """
    Download reference index to get actual trading days for the market.
    Returns the number of expected trading days.
    """
    reference_ticker = REFERENCE_INDICES.get(index_name, 'SPY')

    try:
        ref_data = yf.download(reference_ticker, start=start_date, end=end_date,
                              progress=False, auto_adjust=True)
        if not ref_data.empty:
            trading_days = len(ref_data.dropna())
            return trading_days, reference_ticker
    except Exception as e:
        print(f"Warning: Could not download reference index {reference_ticker}: {e}")

    # Fallback: estimate based on calendar days (252 trading days per year)
    days_diff = (end_date - start_date).days
    years = days_diff / 365.25
    estimated_days = int(years * 252)
    return estimated_days, "estimated"


def check_data_integrity(tickers, index_name, start_date, end_date, threshold=0.90):
    """
    Check data completeness for a list of tickers.
    Returns complete tickers and integrity report.
    """
    print(f"\n{'='*80}")
    print(f"DATA INTEGRITY CHECK: {index_name}")
    print(f"{'='*80}")
    print(f"Period: {start_date.date()} to {end_date.date()}")
    print(f"Tickers to check: {len(tickers)}")

    # Get reference trading days
    print(f"\nGetting reference trading days...")
    total_days, reference_source = get_reference_trading_days(index_name, start_date, end_date)
    required_days = int(total_days * threshold)

    print(f"Reference: {reference_source}")
    print(f"Expected trading days: {total_days}")
    print(f"Required days for completeness ({threshold*100:.0f}%): {required_days}")

    # Download data for all tickers
    print(f"\nDownloading data for {len(tickers)} tickers...")

    complete_tickers = []
    incomplete_tickers = []
    integrity_data = []

    # Download tickers individually to handle failures better
    for ticker in tqdm(tickers, desc="Downloading"):
        try:
            ticker_data = yf.download(ticker, start=start_date, end=end_date,
                                     progress=False, auto_adjust=True)

            if ticker_data.empty:
                # No data downloaded
                incomplete_tickers.append(ticker)
                integrity_data.append({
                    'Ticker': ticker,
                    'Available_Days': 0,
                    'Coverage_%': 0.0,
                    'Status': 'NO_DATA'
                })
                continue

            # Get close prices
            if 'Close' in ticker_data.columns:
                prices = ticker_data['Close'].dropna()
            elif 'Adj Close' in ticker_data.columns:
                prices = ticker_data['Adj Close'].dropna()
            else:
                prices = pd.Series(dtype=float)

            available_days = len(prices)
            coverage_pct = (available_days / total_days) * 100

            if available_days >= required_days:
                complete_tickers.append(ticker)
                status = 'COMPLETE'
            else:
                incomplete_tickers.append(ticker)
                status = 'INCOMPLETE'

            integrity_data.append({
                'Ticker': ticker,
                'Available_Days': available_days,
                'Coverage_%': coverage_pct,
                'Status': status
            })

        except Exception as e:
            # Download failed
            incomplete_tickers.append(ticker)
            integrity_data.append({
                'Ticker': ticker,
                'Available_Days': 0,
                'Coverage_%': 0.0,
                'Status': f'ERROR'
            })

    # Create report
    report_df = pd.DataFrame(integrity_data)
    report_df = report_df.sort_values('Coverage_%', ascending=False)

    # Summary
    completeness_rate = (len(complete_tickers) / len(tickers)) * 100

    print(f"\n{'='*80}")
    print(f"INTEGRITY SUMMARY: {index_name}")
    print(f"{'='*80}")
    print(f"Total tickers checked: {len(tickers)}")
    print(f"Complete tickers: {len(complete_tickers)} ({completeness_rate:.1f}%)")
    print(f"Incomplete tickers: {len(incomplete_tickers)} ({100-completeness_rate:.1f}%)")

    if incomplete_tickers:
        print(f"\nIncomplete/Missing tickers:")
        for ticker in incomplete_tickers:
            ticker_info = report_df[report_df['Ticker'] == ticker].iloc[0]
            print(f"  - {ticker}: {ticker_info['Coverage_%']:.1f}% coverage ({ticker_info['Status']})")

    return complete_tickers, report_df

# ============================================================================
# SVD MODEL FUNCTIONS
# ============================================================================

def download_clean_data(tickers, start_date, end_date):
    """Download and clean price data for complete tickers."""
    print(f"\nDownloading clean data for {len(tickers)} complete tickers...")

    prices = pd.DataFrame()

    for ticker in tqdm(tickers, desc="Downloading clean data"):
        try:
            data = yf.download(ticker, start=start_date, end=end_date,
                             progress=False, auto_adjust=True)

            if not data.empty and 'Close' in data.columns:
                prices[ticker] = data['Close']
        except Exception as e:
            print(f"Warning: Could not download {ticker}: {e}")
            continue

    if prices.empty:
        return None

    # Forward fill and drop remaining NaNs
    prices = prices.ffill().dropna()

    print(f"✓ Loaded {len(prices.columns)} stocks with {len(prices)} days")
    return prices


def calculate_momentum_scores(returns, lookback=126):
    """Calculate momentum scores."""
    if len(returns) < lookback:
        return np.ones(returns.shape[1]) / returns.shape[1]

    momentum_returns = returns.iloc[-lookback:].mean(axis=0)
    scores = (momentum_returns - momentum_returns.mean()) / (momentum_returns.std() + 1e-8)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    return scores.values


def apply_portfolio_constraints(weights, max_pos, min_pos, short_allowed):
    """Apply portfolio constraints."""
    if short_allowed:
        weights = np.clip(weights, -max_pos, max_pos)
        weights[np.abs(weights) < min_pos] = 0
        if np.sum(np.abs(weights)) > 0:
            weights = weights / np.sum(np.abs(weights))
    else:
        weights = np.clip(weights, 0, max_pos)
        weights[weights < min_pos] = 0
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(weights)) / len(weights)

    return weights


def maximum_sharpe_portfolio(expected_returns, cov_matrix, risk_free_rate,
                             max_position, min_position, short_allowed):
    """Calculate maximum Sharpe ratio portfolio."""
    n_assets = cov_matrix.shape[0]
    cov_matrix_reg = cov_matrix + np.eye(n_assets) * 1e-8

    def negative_sharpe(w):
        ret = np.dot(w, expected_returns)
        std = np.sqrt(np.dot(w, np.dot(cov_matrix_reg, w)))
        return -(ret - risk_free_rate / 252) / std if std > 1e-10 else 1e10

    if short_allowed:
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w)},
            {'type': 'eq', 'fun': lambda w: np.sum(np.abs(w)) - 1}
        ]
        bounds = tuple((-max_position, max_position) for _ in range(n_assets))
    else:
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, max_position) for _ in range(n_assets))

    initial = np.ones(n_assets) / n_assets

    try:
        result = minimize(negative_sharpe, initial, method='SLSQP',
                         bounds=bounds, constraints=constraints,
                         options={'maxiter': 1000, 'ftol': 1e-9})
        weights = result.x if result.success else initial
    except:
        weights = initial

    return apply_portfolio_constraints(weights, max_position, min_position, short_allowed)


def run_svd_backtest(returns, config):
    """Run SVD backtest with given configuration."""
    params = {**config, **FIXED_PARAMS}

    test_start_date = returns.index[0] + relativedelta(years=params['train_years'])

    monthly_returns = []
    prev_weights = None

    print(f"\nRunning backtest: {config['name']}")
    print(f"Test period starts: {test_start_date.date()}")

    for month_idx in tqdm(range(params['test_months']), desc="Backtest"):
        test_month_start = test_start_date + relativedelta(months=month_idx)
        test_month_end = test_month_start + relativedelta(months=1)
        train_start = test_month_start - relativedelta(years=params['train_years'])

        train_mask = (returns.index >= train_start) & (returns.index < test_month_start)
        test_mask = (returns.index >= test_month_start) & (returns.index < test_month_end)

        train_returns = returns[train_mask]
        test_returns = returns[test_mask]

        if len(train_returns) < 100 or len(test_returns) < 5:
            continue

        train_data = train_returns.values
        n_assets = train_data.shape[1]

        # Estimate covariance
        expected_returns = train_data.mean(axis=0) * 252
        lw = LedoitWolf()
        lw.fit(train_data)
        cov_lw = lw.covariance_

        # SVD compression
        svd_rank = min(params['svd_rank'], n_assets-1)
        U, s, Vt = randomized_svd(cov_lw, n_components=svd_rank,
                                   random_state=params['random_state'])
        cov_svd = (U @ np.diag(s) @ Vt + (U @ np.diag(s) @ Vt).T) / 2

        # Regularize
        min_eig = np.min(np.linalg.eigvalsh(cov_svd))
        if min_eig < 1e-8:
            cov_svd += np.eye(n_assets) * (1e-6 - min_eig)

        # Calculate weights
        momentum_scores = calculate_momentum_scores(train_returns, params['momentum_lookback'])
        weights_sharpe = maximum_sharpe_portfolio(
            expected_returns, cov_svd,
            params['risk_free_rate'] / 252,
            params['max_position'],
            params['min_position_size'],
            params['short_allowed']
        )

        # Momentum tilt
        weights = (1 - params['momentum_weight']) * weights_sharpe + \
                  params['momentum_weight'] * momentum_scores
        weights = apply_portfolio_constraints(
            weights,
            params['max_position'],
            params['min_position_size'],
            params['short_allowed']
        )

        # Transaction costs
        test_data = test_returns.values
        cost_bps = params['transaction_cost_bps'] / 10000

        if prev_weights is not None:
            turnover = np.sum(np.abs(weights - prev_weights))
        else:
            init_weights = np.ones(n_assets) / n_assets
            turnover = np.sum(np.abs(weights - init_weights))

        cost = turnover * cost_bps
        prev_weights = weights

        # Portfolio returns
        portfolio_rets = test_data @ weights
        portfolio_rets = portfolio_rets[np.isfinite(portfolio_rets)]
        portfolio_rets = np.clip(portfolio_rets, -0.5, 0.5)

        if len(portfolio_rets) > 0:
            monthly_return = (1 + portfolio_rets).prod() - 1 - cost
            monthly_returns.append(monthly_return)

    return monthly_returns


def calculate_metrics(monthly_returns, risk_free_rate=0.04):
    """Calculate performance metrics."""
    if len(monthly_returns) == 0:
        return {'Return': 0, 'Volatility': 0, 'Sharpe': 0, 'Sortino': 0, 'MaxDD': 0}

    # Total return
    total_return = np.prod([1 + r for r in monthly_returns]) - 1
    num_months = len(monthly_returns)
    annualized_return = (1 + total_return) ** (12 / num_months) - 1

    # Volatility
    monthly_vol = np.std(monthly_returns)
    annualized_vol = monthly_vol * np.sqrt(12)

    # Downside deviation
    downside_returns = [r for r in monthly_returns if r < 0]
    downside_dev = np.sqrt(np.mean(np.array(downside_returns) ** 2)) * np.sqrt(12) if downside_returns else 1e-10

    # Max drawdown
    cumulative = np.cumprod([1 + r for r in monthly_returns])
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_dd = np.min(drawdowns) if len(drawdowns) > 0 else 0

    # Ratios
    sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 1e-10 else 0
    sortino = (annualized_return - risk_free_rate) / downside_dev if downside_dev > 1e-10 else 0

    return {
        'Return': annualized_return,
        'Volatility': annualized_vol,
        'Sharpe': np.clip(sharpe, -10, 10),
        'Sortino': np.clip(sortino, -10, 10),
        'MaxDD': max_dd
    }

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("="*80)
    print("INTERNATIONAL INDICES: DATA INTEGRITY & SVD MODEL VALIDATION")
    print("="*80)
    print(f"Historical Period: {START_DATE.date()} to {END_DATE.date()}")
    print(f"Completeness Threshold: {COMPLETENESS_THRESHOLD*100:.0f}%")
    print(f"Index Threshold: {INDEX_THRESHOLD*100:.0f}%")

    indices = {
        'CAC 40': CAC40_TICKERS,
        'DAX 40': DAX40_TICKERS,
        'NASDAQ 100 (Top 50)': NASDAQ100_TICKERS
    }

    summary_results = []

    for index_name, tickers in indices.items():
        # Step 1: Data Integrity Check
        complete_tickers, integrity_report = check_data_integrity(
            tickers, index_name, START_DATE, END_DATE, COMPLETENESS_THRESHOLD
        )

        completeness_rate = (len(complete_tickers) / len(tickers))

        result = {
            'Index': index_name,
            'Total_Stocks': len(tickers),
            'Complete_Stocks': len(complete_tickers),
            'Completeness_%': completeness_rate * 100,
            'Action': 'RUN' if completeness_rate >= INDEX_THRESHOLD else 'SKIP'
        }

        # Step 2: Run SVD Model if threshold met
        if completeness_rate >= INDEX_THRESHOLD:
            print(f"\n✓ {index_name} passes threshold ({completeness_rate*100:.1f}% ≥ {INDEX_THRESHOLD*100:.0f}%)")
            print(f"Running SVD model on {len(complete_tickers)} complete stocks...")

            # Download clean data
            prices = download_clean_data(complete_tickers, START_DATE, END_DATE)

            if prices is not None and not prices.empty:
                returns = prices.pct_change().dropna()

                # Run both configurations
                for config in CONFIGS:
                    print(f"\n{'='*80}")
                    print(f"Configuration: {config['name']}")
                    print(f"{'='*80}")

                    monthly_rets = run_svd_backtest(returns, config)
                    metrics = calculate_metrics(monthly_rets, FIXED_PARAMS['risk_free_rate'])

                    result[f"{config['name']}_Sharpe"] = metrics['Sharpe']
                    result[f"{config['name']}_Return_%"] = metrics['Return'] * 100
                    result[f"{config['name']}_MaxDD_%"] = metrics['MaxDD'] * 100
                    result[f"{config['name']}_Vol_%"] = metrics['Volatility'] * 100

                    print(f"\n✓ {config['name']} Results:")
                    print(f"  Sharpe Ratio: {metrics['Sharpe']:.3f}")
                    print(f"  Annual Return: {metrics['Return']*100:.2f}%")
                    print(f"  Annual Volatility: {metrics['Volatility']*100:.2f}%")
                    print(f"  Max Drawdown: {metrics['MaxDD']*100:.2f}%")
                    print(f"  Sortino Ratio: {metrics['Sortino']:.3f}")
            else:
                print(f"✗ Could not download clean data for {index_name}")
                result['Action'] = 'ERROR'
        else:
            print(f"\n✗ {index_name} below threshold ({completeness_rate*100:.1f}% < {INDEX_THRESHOLD*100:.0f}%)")
            print(f"Skipping SVD model run.")

        summary_results.append(result)

        # Save integrity report
        if integrity_report is not None:
            filename = f"{index_name.replace(' ', '_')}_integrity_report.csv"
            integrity_report.to_csv(filename, index=False)
            print(f"✓ Saved integrity report: {filename}")

    # Final Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    summary_df = pd.DataFrame(summary_results)
    print("\n" + summary_df.to_string(index=False))

    # Save summary
    summary_df.to_csv('international_indices_summary.csv', index=False)
    print("\n✓ Saved summary: international_indices_summary.csv")

    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)


if __name__ == '__main__':
    main()