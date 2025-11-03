import numpy as np
import pandas as pd
import cvxpy as cp
import logging
from numpy.lib.stride_tricks import sliding_window_view
from typing import Tuple

TOLERANCE = 1e-3


def _below_vs_chord_sliding(df: pd.DataFrame, window: int, price_col='price') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Implementation using numpy's sliding_window_view for maximum performance.
    
    Args:
        df: DataFrame with at least columns: date, <price_col>
        window: Window size for the chord
        price_col: Column name for the price series
    
    Returns:
        Tuple of (summary_df, mask_df):
        - summary_df: DataFrame with share_below and count_below per end date
        - mask_df: Boolean DataFrame where True means price was below the chord
    """
    # Ensure sorted and extract price series
    s = df.sort_values('date').set_index('date')[price_col].astype(float)
    p = s.to_numpy()
    N = window
    
    if len(p) <= N:
        raise ValueError(f"Need > {N} rows of price data")

    # Create 2D view of rolling windows: shape (n-N, N+1)
    w = sliding_window_view(p, window_shape=N+1)

    # Build the chord for each window via broadcasting
    y0 = w[:, [0]]           # (n-N, 1) - price at t-window
    yN = w[:, [-1]]          # (n-N, 1) - price at t
    k = np.arange(N+1)[None, :]  # (1, N+1) - position in window
    y_line = y0 + (k / N) * (yN - y0)  # chord equation

    # Compare interior points to chord (exclude endpoints which lie on the chord)
    interior_cmp = w[:, 1:-1] + TOLERANCE < y_line[:, 1:-1]  # (n-N, N-1) - True if below

    # Aggregates per end date
    count_below = interior_cmp.sum(axis=1)
    share_below = count_below / (N - 1)

    # Return as DataFrames
    end_idx = s.index[N:]  # window ends at these dates
    cols = pd.Index(range(N-1, 0, -1), name='days_ago')  # (N-1)..1
    mask_df = pd.DataFrame(interior_cmp, index=end_idx, columns=cols)
    out = pd.DataFrame({
        'count_below': count_below, 
        'share_below': share_below
    }, index=end_idx)
    
    return out, mask_df

def below_chord(df: pd.DataFrame, window: int, price_col: str) -> pd.DataFrame:
    """
    Calculate the percentage of days below the chord, grouped by symbol.
    
    Args:
        df: DataFrame with columns: date, symbol, <price_col>
        window: Window size for the chord
        price_col: Column name for the price series
    
    Returns:
        Original DataFrame with additional columns:
        - below_chord_count: number of interior days where price was below the chord
        - below_chord_share: fraction of interior days where price was below the chord
    """
    # Ensure we have required columns
    if 'symbol' not in df.columns:
        raise ValueError("DataFrame must contain 'symbol' column")
    if 'date' not in df.columns:
        raise ValueError("DataFrame must contain 'date' column")
    
    # Create a copy of the input DataFrame
    result_df = df.copy()
    
    # Calculate metrics for each symbol group and collect in a list
    all_metrics = []
    
    for symbol, group in df.groupby('symbol'):
        # Get the chord metrics for this symbol
        metrics_df, _ = _below_vs_chord_sliding(group, window=window, price_col=price_col)
        
        # Add symbol column to metrics
        metrics_df = metrics_df.reset_index()
        metrics_df['symbol'] = symbol
        
        # Rename columns to match requested names
        metrics_df = metrics_df.rename(columns={
            'count_below': 'below_chord_count',
            'share_below': 'below_chord_share'
        })
        
        all_metrics.append(metrics_df)
    
    # Combine all metrics
    if not all_metrics:
        # No metrics calculated, return original DataFrame
        return result_df
    
    combined_metrics = pd.concat(all_metrics)
    
    # Merge metrics back to original DataFrame
    result_df = pd.merge(
        result_df, 
        combined_metrics[['symbol', 'date', 'below_chord_count', 'below_chord_share']], 
        on=['symbol', 'date'], 
        how='left'
    )
    
    return result_df

def risk_adjusted_momentum(df: pd.DataFrame, return_days: int, delay_days: int, price_col: str) -> pd.DataFrame:
    """
    Compute risk-adjusted momentum per symbol and merge back to the input DataFrame.

    For each symbol and at each date t, using prices up to t-D only:
    - raw_momentum(t) = P(t-D) / P(t-return_days) - 1
    - daily_return_std(t) = std of daily returns over the previous (return_days - delay_days) days,
      i.e., returns from (t-return_days+1) .. (t-D), aligned to date t
    - risk_adjusted_momentum(t) = raw_momentum(t) / daily_return_std(t)

    Args:
        df: DataFrame with columns: date, symbol, <price_col>
        return_days: N in the above formula (e.g., 13 for 12-1 momentum)
        delay_days: D in the above formula (e.g., 1 for 12-1 momentum)
        price_col: Column name for the price series

    Returns:
        Original DataFrame with two additional columns:
        - raw_momentum
        - risk_adjusted_momentum
    """
    if 'symbol' not in df.columns:
        raise ValueError("DataFrame must contain 'symbol' column")
    if 'date' not in df.columns:
        raise ValueError("DataFrame must contain 'date' column")
    if price_col not in df.columns:
        raise ValueError(f"DataFrame must contain '{price_col}' column")
    if delay_days < 1:
        raise ValueError("delay_days must be >= 1")
    if return_days <= delay_days:
        raise ValueError("return_days must be > delay_days")

    # Work on a copy to avoid mutating the original
    result_df = df.copy()

    all_metrics = []
    for symbol, group in df.groupby('symbol'):
        g = group.sort_values('date').copy()
        s = g.set_index('date')[price_col].astype(float)

        # Raw momentum aligned to date t: uses prices at t-D and t-return_days
        raw = s.shift(delay_days) / s.shift(return_days + delay_days) - 1.0

        # Daily returns and rolling std over previous (return_days - delay_days) returns
        daily_ret = s.pct_change(fill_method=None)
        daily_std = daily_ret.rolling(window=return_days).std().shift(delay_days)

        out = pd.DataFrame({
            'date': raw.index,
            'raw_momentum': raw.values,
            'risk_adjusted_momentum': (raw / daily_std).values,
            'symbol': symbol,
        })

        all_metrics.append(out)

    if not all_metrics:
        return result_df

    metrics = pd.concat(all_metrics, ignore_index=True)

    # Merge metrics back to original DataFrame
    result_df = pd.merge(
        result_df,
        metrics[['symbol', 'date', 'raw_momentum', 'risk_adjusted_momentum']],
        on=['symbol', 'date'],
        how='left'
    )

    return result_df


def spmo_weights(data: pd.DataFrame, n_top: int = 100, stock_cap=0.05, sector_cap=0.30):
    """
    Compute the SPMO portfolio weights following actual SPMO methodology.
    
    Args:
        data: DataFrame with columns: symbol, risk_adjusted_momentum, market_cap, sector
        cap: Individual stock cap (default 0.05)
        sector_cap: Sector cap (default 0.30)
    
    Returns:
        DataFrame with original data plus 'spmo_weight' column
    """
    # Validate required columns
    required_cols = ['symbol', 'risk_adjusted_momentum', 'market_cap', 'sector']
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Calculate momentum-weighted market cap score
    data = data.copy()
    data['mom_cap'] = data['risk_adjusted_momentum'] * data['market_cap']
    g = data.nlargest(n_top, "mom_cap")
    n = len(g)

    # target weights (mom_cap proportional)
    mom_cap = g['mom_cap'].values.astype(float)
    w_target = mom_cap / mom_cap.sum()

    # index mapping for sectors
    sectors = g['sector'].values
    sector_names, sector_idx = np.unique(sectors, return_inverse=True)

    # cvxpy variable
    w = cp.Variable(n)

    # objective: minimize ||w - w_target||^2 (Euclidean)
    obj = cp.Minimize(cp.sum_squares(w - w_target))

    # constraints
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= stock_cap
    ]

    # sector constraints
    for s in range(len(sector_names)):
        mask = (sector_idx == s)
        if mask.sum() > 0:
            constraints.append(cp.sum(w[mask]) <= sector_cap)

    prob = cp.Problem(obj, constraints)
    
    '''
    Solver choices:

    Solver	    | Best For	        | Pros	                                | Cons
    OSQP	    | General QP	    | Fast, robust, handles large problems	| Can be less accurate
    ECOS	    | Small-medium QP	| High accuracy, reliable	            | Slower on large problems
    CLARABEL	| Modern QP	        | Fast + accurate, good scaling	        | Newer, less tested
    SCS	        | Large-scale	    | Handles very large problems	        | Less accurate
    '''
    prob.solve(solver=cp.CLARABEL, verbose=False)

    # 1) check solver status
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"QP solve failed or not optimal. status={prob.status}")

    # 2) retrieve and validate solution
    if w.value is None:
        raise RuntimeError("Solver returned no solution (w.value is None)")    

    w_res = np.array(w.value).reshape(-1)
    w_res = np.maximum(w_res, 0)
    assert w_res.min() >= -1e-6, f"Negative weight: min={w_res.min()}"
    assert abs(w_res.sum() - 1.0) <= 1e-6, f"Weights do not sum to 1: sum={w_res.sum()}"


    g['final_wt'] = w_res        

    result = data.copy()
    result['spmo_weight'] = 0.0  # Initialize all weights to 0
    
    # Set weights for selected stocks
    for _, row in g.iterrows():
        symbol_mask = result['symbol'] == row['symbol']
        result.loc[symbol_mask, 'spmo_weight'] = row['final_wt']

    return result