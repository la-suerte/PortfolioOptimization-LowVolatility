"""
Comprehensive Parameter Testing for Portfolio Strategy
WITH CACHING, RESUME CAPABILITY, AND VISUALIZATIONS
"""

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.covariance import LedoitWolf
from sklearn.utils.extmath import randomized_svd
from dateutil.relativedelta import relativedelta
from scipy.optimize import minimize
import warnings
import itertools
from tqdm import tqdm
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

# ============================================================================
# CACHING CONFIGURATION
# ============================================================================
CACHE_DIR = 'parameter_test_cache'
CACHE_FILE = os.path.join(CACHE_DIR, 'progress.jsonl')
RESULTS_CSV = 'parameter_test_results.csv'
CHECKPOINT_INTERVAL = 5  # Save checkpoint every N combinations

# ============================================================================
# PARAMETER GRID
# ============================================================================
PARAM_GRID = {
    'svd_rank': [3, 7, 14],
    'train_years': [1, 2, 3],
    'test_months': [30, 60, 90],
    'max_position': [0.05, 0.10],
    'short_allowed': [False, True]
}

# Fixed parameters
FIXED_PARAMS = {
    'min_position_size': 0.005,
    'transaction_cost_bps': 10,
    'momentum_lookback': 126,
    'momentum_weight': 0.3,
    'risk_free_rate': 0.04,
    'random_state': 42
}


def setup_cache():
    """Initialize cache directory."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        print(f"✓ Created cache directory: {CACHE_DIR}")


def load_cached_results():
    """Load previously computed results from cache."""
    if not os.path.exists(CACHE_FILE):
        return pd.DataFrame()

    try:
        results = []
        with open(CACHE_FILE, 'r') as f:
            for line in f:
                results.append(json.loads(line.strip()))

        df = pd.DataFrame(results)
        print(f"✓ Loaded {len(df)} cached results from previous runs")
        return df
    except Exception as e:
        print(f"⚠ Warning: Could not load cache ({e}), starting fresh")
        return pd.DataFrame()


def save_result_to_cache(result):
    """Append single result to cache file."""
    with open(CACHE_FILE, 'a') as f:
        f.write(json.dumps(result) + '\n')


def get_completed_combinations(cached_df):
    """Return set of already-completed parameter combinations."""
    if cached_df.empty:
        return set()

    completed = set()
    for _, row in cached_df.iterrows():
        # Convert all to native Python types to ensure consistent tuple comparison
        combo = (
            int(row['SVD_Rank']),           # Convert numpy int64 to Python int
            int(row['Train_Years']),        # Convert numpy int64 to Python int
            int(row['Test_Months']),        # Convert numpy int64 to Python int
            float(row['Max_Position']),     # Ensure Python float
            bool(row['Short_Allowed'])      # Ensure Python bool
        )
        completed.add(combo)

    return completed


def get_sp500_tickers():
    """Get S&P 500 ticker list with fallback."""
    try:
        tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
        tickers = [t.replace('.', '-') for t in tables[0]['Symbol'].tolist()]
        print(f"✓ Retrieved {len(tickers)} S&P 500 tickers")
        return tickers
    except:
        print("Using fallback ticker list...")
        return ['APD', 'AFL', 'MMM', 'ACN', 'ABBV', 'ARE', 'ALB', 'ALL', 'GOOGL', 'A',
            'AKAM', 'ABNB', 'AES', 'LNT', 'ALLE', 'AOS', 'AMD', 'ABT', 'ADBE', 'ALGN',
            'MO', 'AON', 'AMT', 'AEP', 'AMGN', 'AMCR', 'AME', 'GOOG', 'APH', 'AIG',
            'AXP', 'AMZN', 'ATO', 'ACGL', 'T', 'APTV', 'ADSK', 'AMP', 'AIZ', 'AWK',
            'AJG', 'ANET', 'AAPL', 'APO', 'ADM', 'APP', 'AMAT', 'ADI', 'APA', 'BALL',
            'XYZ', 'AXON', 'BRK-B', 'BLK', 'BA', 'BKR', 'BBY', 'ADP', 'TECH', 'BIIB',
            'BAC', 'BDX', 'BAX', 'AVY', 'AVB', 'BX', 'BKNG', 'AZO', 'BK', 'BR',
            'COF', 'BSX', 'BRO', 'BF-B', 'CAT', 'CCL', 'BG', 'AVGO', 'CBOE', 'CAH',
            'CHRW', 'CPB', 'CDNS', 'C', 'CMG', 'CTAS', 'CINF', 'CB', 'CVX', 'CF',
            'COR', 'CNC', 'CHTR', 'CHD', 'CLX', 'CNP', 'SCHW', 'CDW', 'CBRE', 'CI',
            'CRL', 'CSCO', 'CFG', 'CME', 'COIN', 'CTVA', 'GLW', 'STZ', 'COO', 'CTRA',
            'KO', 'COP', 'ED', 'COST', 'CMS', 'CSGP', 'CPAY', 'CL', 'CAG', 'CTSH',
            'CPRT', 'CEG', 'CMCSA', 'DRI', 'CSX', 'DAL', 'DDOG', 'FANG', 'DVN', 'DECK',
            'DHR', 'DVA', 'DXCM', 'DLTR', 'CCI', 'DG', 'DELL', 'DE', 'CVS', 'DLR',
            'CMI', 'CRWD', 'DAY', 'DPZ', 'D', 'EIX', 'EMR', 'ELV', 'DOV', 'DUK',
            'EW', 'DTE', 'DD', 'ETN', 'ECL', 'ETR', 'EA', 'EME', 'DHI', 'EPAM',
            'EMN', 'EOG', 'EBAY', 'DASH', 'DOW', 'FFIV', 'EQIX', 'EG', 'ERIE', 'FICO',
            'EXR', 'EQT', 'EXPE', 'EXC', 'EFX', 'FAST', 'EXPD', 'EXE', 'EQR', 'ES',
            'XOM', 'EVRG', 'ESS', 'EL', 'FDS', 'FTNT', 'F', 'FITB', 'IT', 'BEN',
            'FE', 'FCX', 'FI', 'FIS', 'FOX', 'GEHC', 'FRT', 'GRMN', 'GEN', 'GEV',
            'FDX', 'FSLR', 'FTV', 'GE', 'HAS', 'HPE', 'GPC', 'GD', 'DOC', 'HSIC',
            'GILD', 'HCA', 'GPN', 'FOXA', 'HAL', 'HLT', 'HSY', 'GS', 'HIG', 'GL',
            'GDDY', 'GM', 'GIS', 'GNRC', 'IBM', 'HST', 'INTC', 'HRL', 'ITW', 'INCY',
            'HUM', 'IDXX', 'HBAN', 'IR', 'HOLX', 'IBKR', 'HD', 'IEX', 'HPQ', 'HON',
            'HII', 'PODD', 'HWM', 'ICE', 'INVH', 'ISRG', 'J', 'JBHT', 'INTU', 'IVZ',
            'IP', 'IQV', 'JPM', 'IRM', 'JNJ', 'JBL', 'IFF', 'IPG', 'HUBB', 'JCI',
            'K', 'KLAC', 'KVUE', 'KMB', 'KR', 'LII', 'KEYS', 'LIN', 'KKR', 'LHX',
            'KHC', 'JKHY', 'LLY', 'KMI', 'LDOS', 'KIM', 'LRCX', 'LVS', 'KDP', 'LEN',
            'LH', 'KEY', 'LW', 'LOW', 'MAS', 'MTCH', 'LYV', 'MA', 'LYB', 'MPC',
            'L', 'MLM', 'MMC', 'LULU', 'MKC', 'MTB', 'MAR', 'LKQ', 'LMT', 'MHK',
            'MCD', 'MGM', 'MSFT', 'MTD', 'MNST', 'MDT', 'MDLZ', 'MU', 'TAP', 'MCHP',
            'MET', 'MRK', 'MPWR', 'MCO', 'MCK', 'MOH', 'META', 'MRNA', 'MAA', 'NFLX',
            'MOS', 'MSCI', 'NKE', 'NWSA', 'NEM', 'MSI', 'NWS', 'MS', 'NEE', 'NUE',
            'NSC', 'NTAP', 'NTRS', 'NCLH', 'NRG', 'NDAQ', 'NOC', 'NDSN', 'NI', 'PLTR',
            'NXPI', 'ON', 'PH', 'PCAR', 'ORCL', 'OTIS', 'NVDA', 'PKG', 'PYPL', 'PANW',
            'OKE', 'PSKY', 'OXY', 'ORLY', 'ODFL', 'PAYC', 'NVR', 'PAYX', 'OMC', 'PPL',
            'PEG', 'PGR', 'PNC', 'PFG', 'PCG', 'PRU', 'PFE', 'PG', 'PNR', 'PSX',
            'PNW', 'PSA', 'PEP', 'PTC', 'POOL', 'PPG', 'PLD', 'PM', 'PHM', 'ROST',
            'REG', 'RVTY', 'REGN', 'HOOD', 'RMD', 'QCOM', 'RL', 'DGX', 'RJF', 'ROP',
            'RSG', 'PWR', 'ROK', 'RCL', 'SPGI', 'O', 'RF', 'RTX', 'ROL', 'CRM',
            'SWK', 'LUV', 'SHW', 'SRE', 'STT', 'SJM', 'STX', 'SPG', 'SNA', 'SOLV',
            'SBAC', 'SW', 'STLD', 'STE', 'SO', 'SBUX', 'SLB', 'SWKS', 'NOW', 'TPL',
            'TMUS', 'TMO', 'SYF', 'TXT', 'TEL', 'TTWO', 'TXN', 'TJX', 'SNPS', 'TER',
            'TDY', 'TGT', 'TSLA', 'SMCI', 'TRGP', 'TROW', 'TPR', 'TDG', 'TT', 'TSN',
            'ULTA', 'TTD', 'TRV', 'TFC', 'TYL', 'UNH', 'USB', 'TRMB', 'URI', 'SYY',
            'UAL', 'TKO', 'TSCO', 'UPS', 'UDR', 'UNP', 'WBD', 'VLTO', 'DIS', 'VTRS',
            'VRTX', 'VTR', 'UBER', 'VRSN', 'VZ', 'VMC', 'WRB', 'WMT', 'VICI', 'VRSK',
            'V', 'VLO', 'GWW', 'UHS', 'WDC', 'WMB', 'WEC', 'WELL', 'WTW', 'ZBH'
        ]


def download_data(tickers, start_date, end_date):
    """Download and clean data."""
    print(f"Downloading {len(tickers)} stocks from {start_date.date()} to {end_date.date()}...")

    raw_data = yf.download(tickers, start=start_date, end=end_date,
                           progress=False, threads=True, group_by='column')

    if raw_data.empty:
        return None

    # Extract adjusted close
    if isinstance(raw_data.columns, pd.MultiIndex):
        data = raw_data['Adj Close'] if 'Adj Close' in raw_data.columns.get_level_values(0) else raw_data['Close']
    else:
        data = raw_data[['Adj Close']] if 'Adj Close' in raw_data.columns else raw_data[['Close']]

    # Clean data - more lenient threshold
    data = data.ffill().dropna(axis=1, thresh=int(0.80 * len(data)))
    data = data.dropna()

    print(f"✓ Loaded {len(data.columns)} stocks with {len(data)} days")
    return data


def calculate_momentum_scores(returns, lookback=126):
    """Calculate momentum scores for each stock."""
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
        if weights.sum() != 0:
            weights = weights / np.abs(weights).sum()
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


def run_single_backtest(returns, params):
    """Run backtest with specific parameters."""
    test_start_date = returns.index[0] + relativedelta(years=params['train_years'])

    models = ['Max-Sharpe-Momentum']
    monthly_metrics = {m: [] for m in models}
    prev_weights = {m: None for m in models}

    for month_idx in range(params['test_months']):
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

        # Estimate expected returns and covariance
        expected_returns = train_data.mean(axis=0) * 252
        lw = LedoitWolf()
        lw.fit(train_data)
        cov_lw = lw.covariance_

        # Apply SVD compression
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

        # Apply momentum tilt
        weights_momentum = (1 - params['momentum_weight']) * weights_sharpe + \
                          params['momentum_weight'] * momentum_scores
        weights_momentum = apply_portfolio_constraints(
            weights_momentum,
            params['max_position'],
            params['min_position_size'],
            params['short_allowed']
        )

        # Calculate transaction costs
        test_data = test_returns.values
        cost_bps = params['transaction_cost_bps'] / 10000

        model_name = 'Max-Sharpe-Momentum'
        weights = weights_momentum

        if prev_weights[model_name] is not None:
            turnover = np.sum(np.abs(weights - prev_weights[model_name]))
        else:
            init_weights = np.zeros(n_assets) if params['short_allowed'] else np.ones(n_assets) / n_assets
            turnover = np.sum(np.abs(weights - init_weights))

        cost = turnover * cost_bps
        prev_weights[model_name] = weights

        # Calculate portfolio returns
        portfolio_rets = test_data @ weights
        portfolio_rets = portfolio_rets[np.isfinite(portfolio_rets)]
        portfolio_rets = np.clip(portfolio_rets, -0.5, 0.5)

        if len(portfolio_rets) > 0:
            monthly_return = (1 + portfolio_rets).prod() - 1 - cost
            monthly_metrics[model_name].append(monthly_return)

    return monthly_metrics


def calculate_metrics(monthly_returns, risk_free_rate=0.04):
    """Calculate annualized metrics from monthly returns."""
    if len(monthly_returns) == 0:
        return {'Return': 0, 'Volatility': 0, 'Sharpe': 0, 'Sortino': 0, 'MaxDD': 0}

    # Compound returns
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


def run_parameter_sweep(returns):
    """Run comprehensive parameter sweep with caching."""
    # Setup cache
    setup_cache()

    # Load previous results
    cached_df = load_cached_results()
    completed = get_completed_combinations(cached_df)

    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        PARAM_GRID['svd_rank'],
        PARAM_GRID['train_years'],
        PARAM_GRID['test_months'],
        PARAM_GRID['max_position'],
        PARAM_GRID['short_allowed']
    ))

    total_tests = len(param_combinations)
    already_done = len(completed)
    remaining = total_tests - already_done

    print(f"\n{'='*80}")
    print(f"PARAMETER SWEEP STATUS")
    print(f"{'='*80}")
    print(f"Total combinations: {total_tests}")
    print(f"Already completed: {already_done}")
    print(f"Remaining: {remaining}")
    print(f"{'='*80}\n")

    if remaining == 0:
        print("✓ All combinations already completed!")
        return cached_df

    # Process remaining combinations
    for idx, combo in enumerate(tqdm(param_combinations, desc="Testing parameters")):
        svd_rank, train_years, test_months, max_position, short_allowed = combo

        # Skip if already completed
        if combo in completed:
            continue

        params = {
            'svd_rank': svd_rank,
            'train_years': train_years,
            'test_months': test_months,
            'max_position': max_position,
            'short_allowed': short_allowed,
            **FIXED_PARAMS
        }

        try:
            monthly_metrics = run_single_backtest(returns, params)
            metrics = calculate_metrics(monthly_metrics['Max-Sharpe-Momentum'], params['risk_free_rate'])

            result = {
                'SVD_Rank': svd_rank,
                'Train_Years': train_years,
                'Test_Months': test_months,
                'Max_Position': max_position * 100,
                'Short_Allowed': short_allowed,
                'Return_%': metrics['Return'] * 100,
                'Vol_%': metrics['Volatility'] * 100,
                'Sharpe': metrics['Sharpe'],
                'Sortino': metrics['Sortino'],
                'MaxDD_%': metrics['MaxDD'] * 100
            }

            # Save to cache immediately
            save_result_to_cache(result)
            completed.add(combo)

            # Progress update
            done = len(completed)
            pct = (done / total_tests) * 100
            print(f"\n✓ [{done}/{total_tests}] ({pct:.1f}%) - Sharpe: {metrics['Sharpe']:.3f} | "
                  f"Params: SVD={svd_rank}, Train={train_years}Y, Test={test_months}mo, "
                  f"MaxPos={max_position*100:.0f}%, Short={short_allowed}")

        except Exception as e:
            print(f"\n✗ Error in combination {combo}: {e}")
            continue

    # Reload all results
    return load_cached_results()


def create_visualizations(results_df):
    """Create comprehensive visualizations."""
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 12)

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

    # 1. Sharpe Ratio Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    results_df['Sharpe'].hist(bins=30, ax=ax1, color='steelblue', edgecolor='black')
    ax1.axvline(results_df['Sharpe'].mean(), color='red', linestyle='--', label=f'Mean: {results_df["Sharpe"].mean():.3f}')
    ax1.set_xlabel('Sharpe Ratio')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Sharpe Ratios')
    ax1.legend()

    # 2. Return vs Volatility Scatter
    ax2 = fig.add_subplot(gs[0, 1])
    scatter = ax2.scatter(results_df['Vol_%'], results_df['Return_%'],
                         c=results_df['Sharpe'], cmap='RdYlGn', s=100, alpha=0.6)
    ax2.set_xlabel('Volatility (%)')
    ax2.set_ylabel('Return (%)')
    ax2.set_title('Risk-Return Profile (colored by Sharpe)')
    plt.colorbar(scatter, ax=ax2, label='Sharpe Ratio')

    # 3. Sharpe by SVD Rank
    ax3 = fig.add_subplot(gs[0, 2])
    results_df.boxplot(column='Sharpe', by='SVD_Rank', ax=ax3)
    ax3.set_xlabel('SVD Rank')
    ax3.set_ylabel('Sharpe Ratio')
    ax3.set_title('Sharpe Ratio by SVD Rank')
    plt.sca(ax3)
    plt.xticks(rotation=0)

    # 4. Sharpe by Training Years
    ax4 = fig.add_subplot(gs[1, 0])
    results_df.boxplot(column='Sharpe', by='Train_Years', ax=ax4)
    ax4.set_xlabel('Training Years')
    ax4.set_ylabel('Sharpe Ratio')
    ax4.set_title('Sharpe Ratio by Training Period')

    # 5. Sharpe by Test Months
    ax5 = fig.add_subplot(gs[1, 1])
    results_df.boxplot(column='Sharpe', by='Test_Months', ax=ax5)
    ax5.set_xlabel('Test Months')
    ax5.set_ylabel('Sharpe Ratio')
    ax5.set_title('Sharpe Ratio by Test Period')

    # 6. Sharpe by Max Position
    ax6 = fig.add_subplot(gs[1, 2])
    results_df.boxplot(column='Sharpe', by='Max_Position', ax=ax6)
    ax6.set_xlabel('Max Position (%)')
    ax6.set_ylabel('Sharpe Ratio')
    ax6.set_title('Sharpe Ratio by Max Position Size')

    # 7. Sharpe by Short Allowed
    ax7 = fig.add_subplot(gs[2, 0])
    results_df.boxplot(column='Sharpe', by='Short_Allowed', ax=ax7)
    ax7.set_xlabel('Short Selling Allowed')
    ax7.set_ylabel('Sharpe Ratio')
    ax7.set_title('Sharpe Ratio: Long-Only vs Long-Short')

    # 8. Heatmap: SVD Rank vs Train Years (avg Sharpe)
    ax8 = fig.add_subplot(gs[2, 1])
    pivot1 = results_df.pivot_table(values='Sharpe', index='SVD_Rank', columns='Train_Years', aggfunc='mean')
    sns.heatmap(pivot1, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax8)
    ax8.set_title('Avg Sharpe: SVD Rank vs Training Years')

    # 9. Heatmap: Max Position vs Short Allowed (avg Sharpe)
    ax9 = fig.add_subplot(gs[2, 2])
    pivot2 = results_df.pivot_table(values='Sharpe', index='Max_Position', columns='Short_Allowed', aggfunc='mean')
    sns.heatmap(pivot2, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax9)
    ax9.set_title('Avg Sharpe: Max Position vs Short Selling')

    # 10. Top 10 Configurations
    ax10 = fig.add_subplot(gs[3, :])
    top_10 = results_df.nlargest(10, 'Sharpe').copy()
    top_10['Config'] = top_10.apply(
        lambda x: f"SVD{int(x['SVD_Rank'])}_T{int(x['Train_Years'])}Y_M{int(x['Test_Months'])}_P{x['Max_Position']:.0f}%_S{int(x['Short_Allowed'])}",
        axis=1
    )

    x_pos = np.arange(len(top_10))
    bars = ax10.bar(x_pos, top_10['Sharpe'], color='steelblue', alpha=0.7)

    # Color best bar differently
    bars[0].set_color('gold')
    bars[0].set_edgecolor('black')
    bars[0].set_linewidth(2)

    ax10.set_xticks(x_pos)
    ax10.set_xticklabels(top_10['Config'], rotation=45, ha='right')
    ax10.set_ylabel('Sharpe Ratio')
    ax10.set_title('Top 10 Parameter Configurations by Sharpe Ratio', fontsize=14, fontweight='bold')
    ax10.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, top_10['Sharpe'])):
        height = bar.get_height()
        ax10.text(bar.get_x() + bar.get_width()/2., height,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Parameter Testing Analysis Dashboard', fontsize=16, fontweight='bold', y=0.995)

    # Save figure
    plt.savefig('parameter_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved visualization to 'parameter_analysis.png'")

    # Additional detailed plots
    create_detailed_plots(results_df)

    plt.show()


def create_detailed_plots(results_df):
    """Create additional detailed analysis plots."""

    # Plot 1: Parameter Interaction Effects
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # SVD Rank vs Max Position
    ax1 = axes[0, 0]
    pivot = results_df.pivot_table(values='Sharpe', index='SVD_Rank', columns='Max_Position', aggfunc='mean')
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax1, cbar_kws={'label': 'Sharpe'})
    ax1.set_title('Interaction: SVD Rank vs Max Position')

    # Train Years vs Test Months
    ax2 = axes[0, 1]
    pivot2 = results_df.pivot_table(values='Sharpe', index='Train_Years', columns='Test_Months', aggfunc='mean')
    sns.heatmap(pivot2, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax2, cbar_kws={'label': 'Sharpe'})
    ax2.set_title('Interaction: Train Years vs Test Months')

    # Return distribution by Short Allowed
    ax3 = axes[1, 0]
    results_df.boxplot(column='Return_%', by='Short_Allowed', ax=ax3)
    ax3.set_xlabel('Short Selling Allowed')
    ax3.set_ylabel('Return (%)')
    ax3.set_title('Return Distribution: Long-Only vs Long-Short')

    # Max Drawdown by parameters
    ax4 = axes[1, 1]
    results_df.boxplot(column='MaxDD_%', by='SVD_Rank', ax=ax4)
    ax4.set_xlabel('SVD Rank')
    ax4.set_ylabel('Max Drawdown (%)')
    ax4.set_title('Max Drawdown by SVD Rank')

    plt.tight_layout()
    plt.savefig('parameter_interactions.png', dpi=300, bbox_inches='tight')
    print("✓ Saved detailed analysis to 'parameter_interactions.png'")
    plt.close()

    # Plot 2: Sortino vs Sharpe comparison
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    scatter = ax.scatter(results_df['Sharpe'], results_df['Sortino'],
                        c=results_df['Return_%'], s=100, alpha=0.6, cmap='viridis')
    ax.plot([results_df['Sharpe'].min(), results_df['Sharpe'].max()],
            [results_df['Sharpe'].min(), results_df['Sharpe'].max()],
            'r--', alpha=0.5, label='Sharpe = Sortino')
    ax.set_xlabel('Sharpe Ratio')
    ax.set_ylabel('Sortino Ratio')
    ax.set_title('Sharpe vs Sortino Ratio (colored by Return)')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Return (%)')
    plt.savefig('sharpe_sortino_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved Sharpe-Sortino comparison to 'sharpe_sortino_comparison.png'")
    plt.close()


def analyze_results(results_df):
    """Analyze and display parameter testing results."""
    print("\n" + "="*80)
    print("PARAMETER TESTING RESULTS")
    print("="*80)

    # Summary statistics
    print("\n📊 OVERALL STATISTICS:")
    print("-" * 80)
    summary_stats = results_df[['Sharpe', 'Return_%', 'Vol_%', 'Sortino', 'MaxDD_%']].describe()
    print(summary_stats.round(3))

    # Top 10 configurations by Sharpe
    print("\n🏆 TOP 10 CONFIGURATIONS (by Sharpe Ratio):")
    print("-" * 80)
    top_10 = results_df.nlargest(10, 'Sharpe')
    print(top_10.to_string(index=False))

    # Parameter impact analysis
    print("\n" + "="*80)
    print("PARAMETER IMPACT ANALYSIS")
    print("="*80)

    for param in ['SVD_Rank', 'Train_Years', 'Test_Months', 'Max_Position', 'Short_Allowed']:
        print(f"\n📈 Impact of {param}:")
        print("-" * 80)
        grouped = results_df.groupby(param).agg({
            'Sharpe': ['mean', 'std', 'min', 'max'],
            'Return_%': 'mean',
            'Vol_%': 'mean'
        }).round(3)
        print(grouped)

    # Best vs Worst
    print("\n" + "="*80)
    print("BEST vs WORST CONFIGURATION")
    print("="*80)

    best_idx = results_df['Sharpe'].idxmax()
    worst_idx = results_df['Sharpe'].idxmin()

    print("\n🥇 BEST Configuration:")
    print("-" * 80)
    print(results_df.loc[best_idx].to_string())

    print("\n❌ WORST Configuration:")
    print("-" * 80)
    print(results_df.loc[worst_idx].to_string())

    improvement = ((results_df.loc[best_idx, 'Sharpe'] - results_df.loc[worst_idx, 'Sharpe']) /
                   abs(results_df.loc[worst_idx, 'Sharpe'])) * 100
    print(f"\n💡 Performance Range: {improvement:.1f}% improvement from worst to best")

    # Quartile analysis
    print("\n" + "="*80)
    print("QUARTILE ANALYSIS")
    print("="*80)

    quartiles = results_df['Sharpe'].quantile([0.25, 0.5, 0.75])
    print(f"\n25th percentile: {quartiles[0.25]:.3f}")
    print(f"50th percentile (median): {quartiles[0.5]:.3f}")
    print(f"75th percentile: {quartiles[0.75]:.3f}")

    # Top quartile characteristics
    top_quartile = results_df[results_df['Sharpe'] >= quartiles[0.75]]
    print(f"\n🎯 TOP QUARTILE CHARACTERISTICS (n={len(top_quartile)}):")
    print("-" * 80)
    for param in ['SVD_Rank', 'Train_Years', 'Test_Months', 'Max_Position', 'Short_Allowed']:
        mode_val = top_quartile[param].mode()
        if len(mode_val) > 0:
            pct = (top_quartile[param] == mode_val.iloc[0]).sum() / len(top_quartile) * 100
            print(f"{param}: {mode_val.iloc[0]} ({pct:.1f}% of top quartile)")


def print_progress_summary():
    """Print summary of current progress."""
    if not os.path.exists(CACHE_FILE):
        return

    cached_df = load_cached_results()
    if cached_df.empty:
        return

    print("\n" + "="*80)
    print("CURRENT PROGRESS SUMMARY")
    print("="*80)

    total_combinations = np.prod([len(v) for v in PARAM_GRID.values()])
    completed = len(cached_df)
    pct = (completed / total_combinations) * 100

    print(f"\nCompleted: {completed}/{total_combinations} ({pct:.1f}%)")
    print(f"Best Sharpe so far: {cached_df['Sharpe'].max():.3f}")
    print(f"Mean Sharpe so far: {cached_df['Sharpe'].mean():.3f}")
    print(f"Worst Sharpe so far: {cached_df['Sharpe'].min():.3f}")

    if completed >= 10:
        print("\n🏆 Current Top 5:")
        top_5 = cached_df.nlargest(5, 'Sharpe')[['SVD_Rank', 'Train_Years', 'Test_Months', 'Max_Position', 'Short_Allowed', 'Sharpe']]
        print(top_5.to_string(index=False))


def main():
    """Main execution."""
    print("=" * 80)
    print("COMPREHENSIVE PARAMETER TESTING SESSION WITH CACHING")
    print("=" * 80)

    # Check if we should resume
    print_progress_summary()

    # Download data
    tickers = get_sp500_tickers()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=10 * 365)

    data = download_data(tickers, start_date, end_date)
    if data is None or data.empty:
        print("ERROR: No data downloaded")
        return

    returns = data.pct_change().dropna()
    print(f"Universe: {len(returns.columns)} stocks, {len(returns)} trading days\n")

    # Run parameter sweep (with caching)
    results_df = run_parameter_sweep(returns)

    # Save final results
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\n✓ Final results saved to '{RESULTS_CSV}'")

    # Analyze results
    analyze_results(results_df)

    # Create visualizations
    create_visualizations(results_df)

    print("\n" + "="*80)
    print("✅ PARAMETER TESTING COMPLETE!")
    print("="*80)
    print(f"\nOutputs generated:")
    print(f"  - {RESULTS_CSV} (full results)")
    print(f"  - {CACHE_FILE} (progress cache)")
    print(f"  - parameter_analysis.png (main dashboard)")
    print(f"  - parameter_interactions.png (detailed analysis)")
    print(f"  - sharpe_sortino_comparison.png (ratio comparison)")


if __name__ == '__main__':
    main()