import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from utils.measure import _below_vs_chord_sliding, below_chord, spmo_weights, TOLERANCE
from typing import Tuple


def _below_vs_chord_sliding_reference(df: pd.DataFrame, window: int, price_col='price') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Reference implementation using for-loop for cross-verification.
    Should produce identical results to the vectorized version.
    """
    # Ensure sorted and extract price series
    s = df.sort_values('date').set_index('date')[price_col].astype(float)
    prices = s.to_numpy()
    N = window
    
    if len(prices) <= N:
        raise ValueError(f"Need > {N} rows of price data")
    
    # Initialize result arrays
    n_windows = len(prices) - N
    count_below = np.zeros(n_windows, dtype=int)
    mask_data = np.zeros((n_windows, N-1), dtype=bool)
    
    # Process each window
    for i in range(n_windows):
        # Get window prices
        window_prices = prices[i:i+N+1]
        
        # Calculate chord line for this window
        y0 = window_prices[0]    # Start price
        yN = window_prices[-1]   # End price
        
        # Check each interior point
        below_count = 0
        for j in range(1, N):  # Interior points (exclude endpoints)
            # Position in window (0 to N)
            k = j
            # Expected price on chord at this position
            chord_price = y0 + (k / N) * (yN - y0)
            
            # Check if actual price is below chord (with tolerance)
            actual_price = window_prices[j]
            is_below = actual_price + TOLERANCE < chord_price
            
            if is_below:
                below_count += 1
            
            # Store in mask - vectorized version uses interior_cmp directly
            # interior_cmp has shape (n_windows, N-1) where columns 0 to N-2 correspond to positions 1 to N-1
            # So position j (1 to N-1) maps to column j-1 (0 to N-2)
            mask_data[i, j-1] = is_below
        
        count_below[i] = below_count
    
    # Create output DataFrames
    share_below = count_below / (N - 1)
    end_idx = s.index[N:]  # window ends at these dates
    
    out = pd.DataFrame({
        'count_below': count_below, 
        'share_below': share_below
    }, index=end_idx)
    
    cols = pd.Index(range(N-1, 0, -1), name='days_ago')  # (N-1)..1
    mask_df = pd.DataFrame(mask_data, index=end_idx, columns=cols)
    
    return out, mask_df


def test_below_chord_with_linear_price():
    """Test with a perfectly linear price series - all points should be on the chord."""
    # Create a linear price series
    dates = pd.date_range(start='2020-01-01', periods=100)
    prices = np.linspace(100, 200, 100)  # Linear from 100 to 200
    
    df = pd.DataFrame({
        'date': dates,
        'price': prices
    })
    
    # With a linear price series and tolerance, all points should be exactly on the chord
    out, mask = _below_vs_chord_sliding(df, window=30)
    
    # All counts should be 0 (no points below chord with tolerance)
    assert np.all(out['count_below'] == 0)
    assert np.all(out['share_below'] == 0)
    assert not mask.any().any()  # No True values in mask


def test_below_chord_with_sine_wave():
    """Test with a sine wave where we can predict the pattern of below/above."""
    # Create 200 days with a sine wave pattern added to linear trend
    dates = pd.date_range(start='2020-01-01', periods=200)
    
    # Linear component (slope=1)
    linear = np.arange(200)
    
    # Sine component (period=60 days, amplitude=10)
    sine = 10 * np.sin(2 * np.pi * np.arange(200) / 60)
    
    # Combined price
    prices = 100 + linear + sine
    
    df = pd.DataFrame({
        'date': dates,
        'price': prices
    })
    
    # Compare vectorized and reference implementations
    vectorized_out, vectorized_mask = _below_vs_chord_sliding(df, window=60)
    reference_out, reference_mask = _below_vs_chord_sliding_reference(df, window=60)
    
    # Ensure both implementations produce identical results
    pd.testing.assert_frame_equal(vectorized_out, reference_out, check_dtype=False)
    pd.testing.assert_frame_equal(vectorized_mask, reference_mask, check_dtype=False)
    
    # With window=60 (exactly one period), the chord connects points with the same phase
    # So approximately half the points should be below the chord
    out, mask = vectorized_out, vectorized_mask
    
    # Check that the share_below is close to 0.5 for most windows
    # We allow some margin because of edge effects
    middle_indices = slice(30, 90)  # Examine windows away from edges
    assert 0.45 <= out.iloc[middle_indices]['share_below'].mean() <= 0.55
    
    # For a 60-day window, we expect the pattern to repeat
    # The share_below should be similar for windows that are 60 days apart
    for i in range(30, 60):
        if i + 60 < len(out):
            assert abs(out['share_below'].iloc[i] - out['share_below'].iloc[i + 60]) < 0.1


def test_below_chord_with_step_function():
    """Test with a step function where we know exactly what should be below/above."""
    # Create a price series with a step in the middle
    dates = pd.date_range(start='2020-01-01', periods=100)
    prices = np.ones(100) * 100
    prices[50:] = 200  # Step up at day 50
    
    df = pd.DataFrame({
        'date': dates,
        'price': prices
    })
    
    # Compare vectorized and reference implementations
    vectorized_out, vectorized_mask = _below_vs_chord_sliding(df, window=60)
    reference_out, reference_mask = _below_vs_chord_sliding_reference(df, window=60)
    
    # Ensure both implementations produce identical results
    pd.testing.assert_frame_equal(vectorized_out, reference_out, check_dtype=False)
    pd.testing.assert_frame_equal(vectorized_mask, reference_mask, check_dtype=False)
    
    # For a window that spans the step (e.g., day 20 to day 80)
    # All points before the step should be below the chord
    out, mask = vectorized_out, vectorized_mask
    
    # Check window that starts at day 20 (ends at day 80)
    # Days 21-49 should be below chord (29 days), days 50-79 should be above
    window_20_idx = 20
    expected_below = 29  # Days 21-49 are below (29 days)
    assert out['count_below'].iloc[window_20_idx] == expected_below
    assert out['share_below'].iloc[window_20_idx] == expected_below / 59  # 59 interior points
    
    # Days 21-49 should be below (True in mask), days 50-79 should not be below
    assert mask.iloc[window_20_idx, :29].all()  # First 29 days (days 21-49) below
    assert not mask.iloc[window_20_idx, 29:].any()  # Remaining days not below


def test_below_chord_with_real_pattern():
    """Test with a more realistic price pattern: uptrend with pullbacks."""
    # Create a price series with an uptrend but periodic pullbacks
    dates = pd.date_range(start='2020-01-01', periods=200)
    
    # Base uptrend
    trend = np.linspace(100, 200, 200)
    
    # Add pullbacks every 20 days (use fixed seed for reproducible results)
    np.random.seed(42)
    pullbacks = np.zeros(200)
    for i in range(20, 200, 20):
        pullback_size = np.random.uniform(5, 15)  # Random pullback between 5-15%
        pullback_length = np.random.randint(3, 8)  # Random length between 3-7 days
        pullbacks[i:i+pullback_length] = -pullback_size
    
    # Cumulative effect of pullbacks
    cumulative_effect = np.cumsum(pullbacks)
    
    # Final price
    prices = trend + cumulative_effect
    
    df = pd.DataFrame({
        'date': dates,
        'price': prices
    })
    
    # Compare vectorized and reference implementations
    vectorized_out, vectorized_mask = _below_vs_chord_sliding(df, window=60)
    reference_out, reference_mask = _below_vs_chord_sliding_reference(df, window=60)
    
    # Ensure both implementations produce identical results
    pd.testing.assert_frame_equal(vectorized_out, reference_out, check_dtype=False)
    pd.testing.assert_frame_equal(vectorized_mask, reference_mask, check_dtype=False)
    
    # With a 60-day window, pullbacks should create periods with points below the chord
    out, mask = vectorized_out, vectorized_mask
    
    # We expect some windows to have significant below-chord counts
    assert out['count_below'].max() > 10
    
    # Since we're using random pullbacks, the exact share below can vary
    # Just check that it's in a reasonable range
    assert 0.3 < out['share_below'].mean() < 0.7
    
    # Check that pullback periods correspond to below-chord points
    # For each major pullback, verify nearby windows show increased below-chord counts
    for i in range(20, 140, 20):  # Check pullbacks within our window range
        nearby_windows = out.iloc[max(0, i-10):min(len(out), i+10)]
        assert nearby_windows['count_below'].max() > 5  # At least some below-chord points


def test_below_chord_with_symbols():
    """Test the by-symbol function with multiple stocks."""
    # Create price data for 3 symbols with different patterns
    dates = pd.date_range(start='2020-01-01', periods=100)
    
    # Symbol 1: Linear uptrend
    prices1 = np.linspace(100, 200, 100)
    
    # Symbol 2: Downtrend
    prices2 = np.linspace(200, 100, 100)
    
    # Symbol 3: Flat with spike
    prices3 = np.ones(100) * 100
    prices3[40:60] = 150  # Spike in the middle
    
    # Combine into one DataFrame
    df = pd.DataFrame({
        'date': dates.tolist() * 3,
        'symbol': ['AAPL'] * 100 + ['MSFT'] * 100 + ['GOOG'] * 100,
        'price': np.concatenate([prices1, prices2, prices3])
    })
    
    # Apply the by-symbol function
    result = below_chord(df, window=30, price_col='price')
    
    # Check that all original rows are preserved
    assert len(result) == len(df)
    
    # Check that the new columns are added
    assert 'below_chord_count' in result.columns
    assert 'below_chord_share' in result.columns
    
    # Check symbol-specific patterns
    
    # Symbol 1 (uptrend): Linear trend should have no below-chord points with tolerance
    aapl_data = result[result['symbol'] == 'AAPL']
    assert aapl_data['below_chord_count'].max() == 0
    assert aapl_data['below_chord_share'].max() == 0
    
    # Symbol 2 (downtrend): Should have many below-chord points
    msft_data = result[result['symbol'] == 'MSFT']
    # For a linear downtrend, most interior points should be above the chord
    # (chord connects higher start to lower end, so middle points are above)
    assert msft_data['below_chord_count'].max() == 0
    assert msft_data['below_chord_share'].max() == 0
    
    # Symbol 3 (spike): Should have mixed pattern
    goog_data = result[result['symbol'] == 'GOOG']
    
    # For flat with spike pattern, we need to check windows that include the spike
    # Windows that include the spike should have some below points
    spike_windows = goog_data.iloc[40:70]  # Windows that include the spike
    assert spike_windows['below_chord_count'].max() > 0


def test_below_chord_with_missing_data():
    """Test with missing data points to ensure robust handling."""
    # Create a price series with some NaN values
    dates = pd.date_range(start='2020-01-01', periods=100)
    prices = np.linspace(100, 200, 100)
    
    # Insert some NaN values
    prices[10:15] = np.nan  # 5 consecutive NaNs
    prices[30] = np.nan     # Single NaN
    prices[50:52] = np.nan  # 2 consecutive NaNs
    
    df = pd.DataFrame({
        'date': dates,
        'price': prices
    })
    
    # Apply the function - it should handle NaNs gracefully
    out, mask = _below_vs_chord_sliding(df.dropna(), window=30)
    
    # Check that we get results (fewer than original due to NaNs and window)
    assert len(out) > 0
    assert not out['share_below'].isna().any()  # No NaNs in output
    
    # The windows containing NaNs should be excluded
    assert len(out) < len(df) - 30  # Fewer windows than expected without NaNs


def test_below_chord_with_extreme_values():
    """Test with extreme price changes to ensure numerical stability."""
    # Create a price series with extreme spikes and crashes
    dates = pd.date_range(start='2020-01-01', periods=100)
    prices = np.ones(100) * 100
    
    # Add extreme spike
    prices[30] = 10000  # 100x spike
    
    # Add extreme crash
    prices[60] = 1  # 99% crash
    
    df = pd.DataFrame({
        'date': dates,
        'price': prices
    })
    
    # The function should handle extreme values without crashing
    out, mask = _below_vs_chord_sliding(df, window=20)
    
    # Basic sanity checks
    assert len(out) > 0
    assert not out.isna().any().any()
    assert not mask.isna().any().any()
    
    # Windows containing extreme values should show significant below-chord activity
    extreme_windows = out.iloc[20:80]  # Windows that might include extreme values
    assert extreme_windows['count_below'].max() > 0


def test_vectorized_vs_reference_implementation():
    """Test that vectorized and for-loop implementations produce identical results."""
    # Test with multiple different data patterns
    test_cases = [
        # Linear uptrend
        {
            'name': 'linear_uptrend',
            'dates': pd.date_range(start='2020-01-01', periods=50),
            'prices': np.linspace(100, 200, 50)
        },
        # Linear downtrend
        {
            'name': 'linear_downtrend', 
            'dates': pd.date_range(start='2020-01-01', periods=50),
            'prices': np.linspace(200, 100, 50)
        },
        # Sine wave
        {
            'name': 'sine_wave',
            'dates': pd.date_range(start='2020-01-01', periods=100),
            'prices': 100 + 50 * np.sin(2 * np.pi * np.arange(100) / 20)
        },
        # Step function
        {
            'name': 'step_function',
            'dates': pd.date_range(start='2020-01-01', periods=60),
            'prices': np.concatenate([np.ones(30) * 100, np.ones(30) * 200])
        },
        # Random walk
        {
            'name': 'random_walk',
            'dates': pd.date_range(start='2020-01-01', periods=80),
            'prices': 100 + np.cumsum(np.random.randn(80) * 2)
        }
    ]
    
    for case in test_cases:
        df = pd.DataFrame({
            'date': case['dates'],
            'price': case['prices']
        })
        
        # Test with different window sizes
        for window in [10, 20, 30]:
            if len(df) > window:
                # Get results from both implementations
                vectorized_out, vectorized_mask = _below_vs_chord_sliding(df, window=window)
                reference_out, reference_mask = _below_vs_chord_sliding_reference(df, window=window)
                
                # Compare summary DataFrames
                pd.testing.assert_frame_equal(
                    vectorized_out, reference_out,
                    check_dtype=False,  # Allow int vs int64 differences
                    rtol=1e-10, atol=1e-10
                )
                
                # Compare mask DataFrames
                pd.testing.assert_frame_equal(
                    vectorized_mask, reference_mask,
                    check_dtype=False
                )


def test_implementations_with_tolerance_edge_cases():
    """Test both implementations handle tolerance edge cases identically."""
    # Create data where prices are exactly at tolerance boundary
    dates = pd.date_range(start='2020-01-01', periods=50)
    
    # Linear trend with small deviations at tolerance boundary
    base_prices = np.linspace(100, 200, 50)
    
    # Add small deviations that are exactly at the tolerance boundary
    deviations = np.zeros(50)
    deviations[10] = -TOLERANCE * 0.9  # Just within tolerance
    deviations[20] = -TOLERANCE * 1.1  # Just outside tolerance
    deviations[30] = TOLERANCE * 0.9   # Positive deviation within tolerance
    
    prices = base_prices + deviations
    
    df = pd.DataFrame({
        'date': dates,
        'price': prices
    })
    
    # Test with window that includes the tolerance edge cases
    vectorized_out, vectorized_mask = _below_vs_chord_sliding(df, window=30)
    reference_out, reference_mask = _below_vs_chord_sliding_reference(df, window=30)
    
    # Results should be identical
    pd.testing.assert_frame_equal(vectorized_out, reference_out, check_dtype=False)
    pd.testing.assert_frame_equal(vectorized_mask, reference_mask, check_dtype=False)


# SPMO Portfolio Tests

def test_spmo_portfolio_select_by_momentum_and_market_cap():
    """Test basic SPMO portfolio functionality with simple data."""
    # Create test data with 10 stocks
    data = pd.DataFrame({
        'symbol': [f'STOCK{i:02d}' for i in range(10)],
        'risk_adjusted_momentum':  [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1],
        'market_cap': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'sector': ['Tech', 'Tech', 'Finance', 'Finance', 'Healthcare', 
                  'Healthcare', 'Energy', 'Energy', 'Consumer', 'Consumer']
    })
    
    result = spmo_weights(data, n_top=10, stock_cap=1.0, sector_cap=1.0) # Ignore caps
    
    # Check that result is a DataFrame with same length as input
    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(data)
    
    # Check that spmo_weight column is added
    assert 'spmo_weight' in result.columns
    
    assert sum(result['spmo_weight'] > 0) == 10
    assert pytest.approx(result['spmo_weight'].sum()) == 1.0
    c = data['risk_adjusted_momentum'] * data['market_cap'] / sum(data['risk_adjusted_momentum'] * data['market_cap'])
    assert pytest.approx(result['spmo_weight'].values) == c.values
    


def test_spmo_portfolio_stock_cap():
    """Test individual stock cap functionality."""
    # Create data where one stock would dominate without cap
    data = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'T', 'JPM', 'BAC'],
        'risk_adjusted_momentum': [1.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001],
        'market_cap': [3000, 2000, 1500, 1200, 800, 700, 600, 500, 400, 300],  # AAPL much larger
        'sector': ['Tech', 'Tech', 'Tech', 'Consumer', 'Auto', 'Tech', 'Tech', 'Tech', 'Finance', 'Finance']
    })
    
    # With 20% selection, all 5 stocks would be selected
    # Set very low cap to test capping
    result = spmo_weights(data, n_top=4, stock_cap=1/4, sector_cap=1.0)
    
    # Check that no stock exceeds the cap
    max_weight = result['spmo_weight'].max()
    assert max_weight <= 1/4 + 1e-9  # Allow small floating point error
    
    # Check that weights still sum to 1
    total_weight = result['spmo_weight'].sum()
    assert abs(total_weight - 1.0) < 1e-9


def test_spmo_portfolio_sector_cap():
    """Test sector cap functionality."""
    # Create data with heavy concentration in one sector
    data = pd.DataFrame({
        'symbol': ['AAPL', 'BRK', 'GOOG', 'AMZN', 'TSLA', 'META', 'NVDA', 'T', 'JPM', 'BAC'],
        'risk_adjusted_momentum': [1.0, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.005, 0.001],
        'market_cap': [3000, 2000, 1500, 1200, 800, 700, 600, 500, 400, 300],  # AAPL much larger
        'sector': ['Tech', 'Finance', 'Tech', 'Consumer', 'Auto', 'Tech', 'Tech', 'Tech', 'Finance', 'Finance']
    })
    
    # With 20% selection, would select top ~1.4 stocks, so top 1 stock
    # But let's test with all stocks to see sector capping
    result = spmo_weights(data, n_top=10, stock_cap=1.0, sector_cap=1/4)  # 50% sector cap
    
    # Check sector weights don't exceed cap
    sector_weights = result.groupby('sector')['spmo_weight'].sum()
    max_sector_weight = sector_weights.max()
    assert max_sector_weight <= 1/4 + 1e-9
    assert pytest.approx(sector_weights.sum()) == 1.0
    
    # Check that total weights sum to 1
    assert pytest.approx(result['spmo_weight'].sum()) == 1.0
