import numpy as np
import pandas as pd
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

def below_chord(df: pd.DataFrame, window: int, price_col='price') -> pd.DataFrame:
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