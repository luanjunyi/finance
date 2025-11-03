import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from utils.measure import risk_adjusted_momentum


def _build_prices_from_returns(start_price: float, daily_returns: np.ndarray) -> np.ndarray:
    prices = [start_price]
    for r in daily_returns:
        prices.append(prices[-1] * (1 + r))
    return np.array(prices[1:])


def _expected_ram(df: pd.DataFrame, return_days: int, delay_days: int, price_col: str):
    """Reference implementation using explicit loops for independent verification."""
    s = df.sort_values('date').set_index('date')[price_col].astype(float)
    prices = s.values
    dates = s.index
    n = len(prices)
    
    raw_momentum = np.full(n, np.nan)
    daily_std = np.full(n, np.nan)
    
    # Compute daily returns first
    daily_returns = np.full(n, np.nan)
    for i in range(1, n):
        daily_returns[i] = prices[i] / prices[i-1] - 1.0
    
    for i in range(n):
        # Raw momentum at date i: P(i-delay_days) / P(i-return_days-delay_days) - 1
        price_recent_idx = i - delay_days
        price_old_idx = i - return_days - delay_days
        
        if price_recent_idx >= 0 and price_old_idx >= 0:
            raw_momentum[i] = prices[price_recent_idx] / prices[price_old_idx] - 1.0
        
        # Daily return std: std of returns over window ending at (i-delay_days)
        # Window spans from (i-return_days-delay_days+1) to (i-delay_days)
        end_idx = i - delay_days
        start_idx = end_idx - return_days + 1
        
        if start_idx >= 1 and end_idx >= 1:  # Need at least index 1 for daily returns
            window_returns = daily_returns[start_idx:end_idx+1]
            if len(window_returns) == return_days and not np.isnan(window_returns).any():
                daily_std[i] = np.std(window_returns, ddof=1)
    
    # Create result series
    raw_series = pd.Series(raw_momentum, index=dates)
    std_series = pd.Series(daily_std, index=dates)
    ram_series = raw_series / std_series
    
    return raw_series, std_series, ram_series


def test_risk_adjusted_momentum_basic_correctness():
    # Create deterministic daily returns that alternate to ensure non-zero std
    n_days = 60
    pattern = np.array([0.02, -0.01])  # +2%, -1%
    daily_returns = np.resize(pattern, n_days)
    prices = _build_prices_from_returns(100.0, daily_returns)
    dates = pd.date_range(start='2020-01-01', periods=n_days)

    df = pd.DataFrame({
        'date': dates,
        'symbol': ['AAA'] * n_days,
        'price': prices,
    })

    return_days = 13
    delay_days = 5

    result = risk_adjusted_momentum(df, return_days=return_days, delay_days=delay_days, price_col='price')

    # Compute expected using the same pandas operations to verify alignment
    raw, std, ram = _expected_ram(df[['date', 'price']], return_days, delay_days, 'price')

    # Join expected to compare a subset of non-NaN rows
    cmp = result.merge(
        pd.DataFrame({'date': raw.index, 'e_raw': raw.values, 'e_std': std.values, 'e_ram': ram.values}),
        on='date', how='left'
    )

    # Consider rows where expected is not NaN
    mask = ~cmp['e_ram'].isna()
    assert mask.sum() > 0

    # Compare with tolerances
    pd.testing.assert_series_equal(cmp.loc[mask, 'raw_momentum'].reset_index(drop=True),
                                   cmp.loc[mask, 'e_raw'].reset_index(drop=True),
                                   check_names=False, rtol=1e-12, atol=1e-12)
    pd.testing.assert_series_equal(cmp.loc[mask, 'risk_adjusted_momentum'].reset_index(drop=True),
                                   cmp.loc[mask, 'e_ram'].reset_index(drop=True),
                                   check_names=False, rtol=1e-12, atol=1e-12)


def test_risk_adjusted_momentum_multiple_symbols_and_merge_shape():
    # Two symbols with different patterns
    n_days = 70
    dates = pd.date_range('2020-01-01', periods=n_days)

    # Symbol AAA: alternating +2%/-1%
    pattern1 = np.array([0.02, -0.01])
    prices1 = _build_prices_from_returns(100.0, np.resize(pattern1, n_days))

    # Symbol BBB: small steady growth +0.5%, with periodic dip -2%
    pattern2 = np.array([0.005]*4 + [-0.02])  # 5-day pattern
    prices2 = _build_prices_from_returns(50.0, np.resize(pattern2, n_days))

    df = pd.DataFrame({
        'date': dates.tolist() * 2,
        'symbol': ['AAA'] * n_days + ['BBB'] * n_days,
        'price': np.concatenate([prices1, prices2])
    })

    return_days = 20
    delay_days = 5

    result = risk_adjusted_momentum(df, return_days=return_days, delay_days=delay_days, price_col='price')

    # Shape preserved
    assert len(result) == len(df)
    # Columns present
    assert {'raw_momentum', 'risk_adjusted_momentum'}.issubset(result.columns)

    # Validate a specific date per symbol matches expected
    for sym in ['AAA', 'BBB']:
        sub = df[df['symbol'] == sym][['date', 'price']]
        raw, std, ram = _expected_ram(sub, return_days, delay_days, 'price')
        merged = result[result['symbol'] == sym].merge(
            pd.DataFrame({'date': raw.index, 'e_ram': ram.values}), on='date', how='left'
        )
        mask = ~merged['e_ram'].isna()
        assert mask.sum() > 0
        # Check a middle point to avoid edge effects
        mid_idx = merged.index[mask][len(merged.index[mask]) // 2]
        got = merged.loc[mid_idx, 'risk_adjusted_momentum']
        exp = merged.loc[mid_idx, 'e_ram']
        assert np.isfinite(got)
        assert np.isfinite(exp)
        assert got == pytest.approx(exp, rel=1e-12, abs=1e-12)


def test_risk_adjusted_momentum_input_validation():
    dates = pd.date_range('2020-01-01', periods=10)
    df = pd.DataFrame({'date': dates, 'symbol': ['X']*10, 'price': np.linspace(100, 110, 10)})

    with pytest.raises(ValueError):
        risk_adjusted_momentum(df, return_days=10, delay_days=0, price_col='price')
    with pytest.raises(ValueError):
        risk_adjusted_momentum(df, return_days=5, delay_days=5, price_col='price')
    with pytest.raises(ValueError):
        risk_adjusted_momentum(df.drop(columns=['symbol']), return_days=10, delay_days=1, price_col='price')
    with pytest.raises(ValueError):
        risk_adjusted_momentum(df.drop(columns=['date']), return_days=10, delay_days=1, price_col='price')
    with pytest.raises(ValueError):
        risk_adjusted_momentum(df.drop(columns=['price']), return_days=10, delay_days=1, price_col='price')


def test_risk_adjusted_momentum_with_large_delay_window():
    # Ensure that with a large delay (e.g., 30 days) we still produce values when enough history exists
    n_days = 120
    dates = pd.date_range('2020-01-01', periods=n_days)
    # Mild noisy returns
    rng = np.random.default_rng(42)
    daily_returns = rng.normal(0.001, 0.01, size=n_days)
    prices = _build_prices_from_returns(100.0, daily_returns)

    df = pd.DataFrame({'date': dates, 'symbol': ['SPY']*n_days, 'price': prices})

    return_days = 60
    delay_days = 30

    result = risk_adjusted_momentum(df, return_days=return_days, delay_days=delay_days, price_col='price')

    # There should be some non-null values after sufficient warmup
    assert result['risk_adjusted_momentum'].notna().sum() > 0
    # Spot-check equivalence with expected computation
    raw, std, ram = _expected_ram(df[['date', 'price']], return_days, delay_days, 'price')
    merged = result.merge(pd.DataFrame({'date': raw.index, 'e_ram': ram.values}), on='date', how='left')
    mask = ~merged['e_ram'].isna()
    if mask.any():
        pd.testing.assert_series_equal(merged.loc[mask, 'risk_adjusted_momentum'].reset_index(drop=True),
                                       merged.loc[mask, 'e_ram'].reset_index(drop=True),
                                       check_names=False, rtol=1e-10, atol=1e-10)
