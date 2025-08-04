"""Tests for both historical_tradable_price methods in OfflineData class."""

import pytest
import pandas as pd
import sqlite3
from unittest.mock import patch, MagicMock

from fmp_data.offline_data import OfflineData


# Tests for historical_tradable_price_fmp (FMP version)
def test_historical_tradable_price_fmp_basic():
    """Test the basic functionality of historical_tradable_price_fmp method."""
    with patch('sqlite3.connect') as mock_connect:
        # Create mock connection
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        # Create test data with continuous dates (FMP format: daily_price table)
        test_data = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'AAPL'],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'high': [110, 120, 130],
            'low': [90, 100, 110]
        })
        
        # Mock read_sql_query to return our test data
        with patch('pandas.read_sql_query', return_value=test_data):
            # Call the FMP method
            result = OfflineData.historical_tradable_price_fmp(
                symbol='AAPL',
                start_date='2023-01-01',
                end_date='2023-01-03'
            )
            
            # Verify results
            assert len(result) == 3
            assert list(result['symbol'].unique()) == ['AAPL']
            assert list(result['date']) == ['2023-01-01', '2023-01-02', '2023-01-03']
            
            # Check tradable price calculation (high + low) / 2
            assert result.loc[0, 'tradable_price'] == 100  # (110 + 90) / 2
            assert result.loc[1, 'tradable_price'] == 110  # (120 + 100) / 2
            assert result.loc[2, 'tradable_price'] == 120  # (130 + 110) / 2


def test_historical_tradable_price_fmp_with_gaps():
    """Test that the FMP method fills gaps in price data correctly."""
    with patch('sqlite3.connect') as mock_connect:
        # Create mock connection
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        # Create test data with gaps (missing 2023-01-02)
        test_data = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL'],
            'date': ['2023-01-01', '2023-01-03'],
            'high': [110, 130],
            'low': [90, 110]
        })
        
        # Mock read_sql_query to return our test data
        with patch('pandas.read_sql_query', return_value=test_data):
            # Call the FMP method
            result = OfflineData.historical_tradable_price_fmp(
                symbol='AAPL',
                start_date='2023-01-01',
                end_date='2023-01-04'
            )
            
            # Verify results
            assert len(result) == 4  # Should have entries for all 4 days
            assert list(result['date']) == ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']
            
            # Check tradable price calculation and gap filling
            assert result.loc[0, 'tradable_price'] == 100  # (110 + 90) / 2
            assert result.loc[1, 'tradable_price'] == 100  # Should use previous day's price
            assert result.loc[2, 'tradable_price'] == 120  # (130 + 110) / 2
            assert result.loc[3, 'tradable_price'] == 120  # Should use previous day's price


def test_historical_tradable_price_fmp_empty_data():
    """Test that the FMP method handles empty data correctly."""
    with patch('sqlite3.connect') as mock_connect:
        # Create mock connection
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        # Create empty test data
        test_data = pd.DataFrame(columns=['symbol', 'date', 'high', 'low'])
        
        # Mock read_sql_query to return empty data
        with patch('pandas.read_sql_query', return_value=test_data):
            # Call the FMP method
            result = OfflineData.historical_tradable_price_fmp(
                symbol='AAPL',
                start_date='2023-01-01',
                end_date='2023-01-03'
            )
            
            # Verify results
            assert result.empty


# Tests for historical_tradable_price (EODHD version)
def test_historical_tradable_price_eodhd_basic():
    """Test the basic functionality of historical_tradable_price method (EODHD version)."""
    with patch('sqlite3.connect') as mock_connect:
        # Create mock connection
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        # Create test data with continuous dates (EODHD format: daily_price_eodhd table)
        test_data = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'AAPL'],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'high': [110.0, 120.0, 130.0],
            'low': [90.0, 100.0, 110.0],
            'close': [105.0, 115.0, 125.0],
            'adjusted_close': [100.0, 110.0, 120.0]  # Adjustment factor: 100/105, 110/115, 120/125
        })
        
        # Mock read_sql_query to return our test data
        with patch('pandas.read_sql_query', return_value=test_data):
            # Call the EODHD method
            result = OfflineData.historical_tradable_price(
                symbol='AAPL',
                start_date='2023-01-01',
                end_date='2023-01-03'
            )
            
            # Verify results
            assert len(result) == 3
            assert list(result['symbol'].unique()) == ['AAPL']
            assert list(result['date']) == ['2023-01-01', '2023-01-02', '2023-01-03']
            
            # Check tradable price calculation: (high + low) / 2 * (adjusted_close / close)
            # Day 1: (110 + 90) / 2 * (100 / 105)
            # Day 2: (120 + 100) / 2 * (110 / 115)
            # Day 3: (130 + 110) / 2 * (120 / 125)
            expected_prices = [(110 + 90) / 2 * (100 / 105), (120 + 100) / 2 * (110 / 115), (130 + 110) / 2 * (120 / 125)]
            for i, expected in enumerate(expected_prices):
                assert result.loc[i, 'tradable_price'] == pytest.approx(expected)

            expected_columns = ['symbol', 'date', 'tradable_price']
            assert set(result.columns) == set(expected_columns)


def test_historical_tradable_price_eodhd_with_gaps():
    """Test that the EODHD method fills gaps in price data correctly."""
    with patch('sqlite3.connect') as mock_connect:
        # Create mock connection
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        # Create test data with gaps (missing 2023-01-02)
        test_data = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL'],
            'date': ['2023-01-01', '2023-01-03'],
            'high': [110.0, 130.0],
            'low': [90.0, 110.0],
            'close': [105.0, 125.0],
            'adjusted_close': [100.0, 120.0]
        })
        
        # Mock read_sql_query to return our test data
        with patch('pandas.read_sql_query', return_value=test_data):
            # Call the EODHD method
            result = OfflineData.historical_tradable_price(
                symbol='AAPL',
                start_date='2023-01-01',
                end_date='2023-01-04'
            )
            
            # Verify results
            assert len(result) == 4  # Should have entries for all 4 days
            assert list(result['date']) == ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']
            
            # Check tradable price calculation and gap filling
            day1_price = (110 + 90) / 2 * (100 / 105)
            day3_price = (130 + 110) / 2 * (120 / 125)
            
            assert result.loc[0, 'tradable_price'] == pytest.approx(day1_price)
            assert result.loc[1, 'tradable_price'] == pytest.approx(day1_price)  # Forward filled
            assert result.loc[2, 'tradable_price'] == pytest.approx(day3_price)
            assert result.loc[3, 'tradable_price'] == pytest.approx(day3_price)  # Forward filled


def test_historical_tradable_price_eodhd_empty_data():
    """Test that the EODHD method handles empty data correctly."""
    with patch('sqlite3.connect') as mock_connect:
        # Create mock connection
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        # Create empty test data
        test_data = pd.DataFrame(columns=['symbol', 'date', 'high', 'low', 'close', 'adjusted_close'])
        
        # Mock read_sql_query to return empty data
        with patch('pandas.read_sql_query', return_value=test_data):
            # Call the EODHD method
            result = OfflineData.historical_tradable_price(
                symbol='NONEXISTENT',
                start_date='2023-01-01',
                end_date='2023-01-03'
            )
            
            # Verify results
            assert result.empty
            assert set(result.columns) == set(['symbol', 'date'])


@pytest.fixture(scope="module")
def price_test_db():
    """Create a temporary test database with sample price data for testing filtering logic."""
    import tempfile
    import sqlite3
    import os
    
    temp_db = tempfile.NamedTemporaryFile(delete=False)
    conn = sqlite3.connect(temp_db.name)
    
    # Create the daily_price_eodhd table
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS daily_price_eodhd (
            symbol TEXT,
            date TEXT,
            high REAL,
            low REAL,
            close REAL,
            adjusted_close REAL,
            volume INTEGER,
            PRIMARY KEY (symbol, date)
        );
    """)
    
    # Insert test data with mixed volume values and first trading day scenarios
    conn.executemany(
        """INSERT INTO daily_price_eodhd (
            symbol, date, high, low, close, adjusted_close, volume
        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            # AAPL data - test volume filtering
            ('AAPL', '2023-01-01', 100, 80, 90, 90, 5000),   # First trading day (will be excluded by date filter)
            ('AAPL', '2023-01-02', 110, 90, 100, 100, 0),    # volume = 0 (should be excluded)
            ('AAPL', '2023-01-03', 120, 100, 110, 110, -100), # volume < 0 (should be excluded)
            ('AAPL', '2023-01-04', 130, 110, 120, 120, 3000), # Valid record (should be included)
            ('AAPL', '2023-01-05', 140, 120, 130, 130, 4000), # Valid record (should be included)
            
            # MSFT data - test first day exclusion
            ('MSFT', '2023-01-01', 200, 180, 190, 190, 2000), # First trading day (will be excluded)
            ('MSFT', '2023-01-02', 210, 190, 200, 200, 2500), # Valid record (should be included)
            ('MSFT', '2023-01-03', 220, 200, 210, 210, 3000), # Valid record (should be included)
            
            # GOOGL data - test combined filtering
            ('GOOGL', '2023-01-01', 1000, 900, 950, 950, 1000), # First trading day (will be excluded)
            ('GOOGL', '2023-01-02', 1100, 1000, 1050, 1050, 0), # volume = 0 (should be excluded)
            ('GOOGL', '2023-01-03', 1200, 1100, 1150, 1150, 1500), # Valid record (should be included)
        ]
    )
    
    conn.commit()
    conn.close()
    
    yield temp_db.name
    
    # Cleanup
    os.unlink(temp_db.name)


def test_historical_tradable_price_filtering_logic(price_test_db):
    """Test that the filtering logic in lines 258-263 works correctly.
    
    Tests both filtering conditions applied by the SQL query:
    - volume > 0 (line 258)
    - date > (SELECT MIN(date) FROM daily_price_eodhd WHERE symbol = ?) (lines 259-263)
    """
    # Test with AAPL data that has multiple filtering scenarios:
    # - First day excluded by date filter
    # - Zero and negative volume records excluded by volume filter
    # - Valid records with positive volume included
    result = OfflineData.historical_tradable_price(
        symbol='AAPL',
        start_date='2023-01-01',
        end_date='2023-01-05',
        db_path=price_test_db
    )
    
    # Verify the filtering worked correctly
    assert len(result) == 5  # All dates in range due to forward filling
    
    # Based on our test data:
    # - 2023-01-01: excluded by first-day filter (volume=5000 but first trading day)
    # - 2023-01-02: excluded by volume = 0 filter
    # - 2023-01-03: excluded by volume < 0 filter (volume=-100)
    # - 2023-01-04: valid (volume=3000, not first day)
    # - 2023-01-05: valid (volume=4000, not first day)
    
    # Only valid records should have calculated prices
    valid_dates = ['2023-01-04', '2023-01-05']
    for date in valid_dates:
        row = result[result['date'] == date].iloc[0]
        assert pd.notna(row['tradable_price']), f"Expected valid price for {date}"
        assert row['tradable_price'] > 0, f"Expected positive price for {date}"
    
    # Excluded records should have NaN (no data to forward fill from)
    excluded_dates = ['2023-01-01', '2023-01-02', '2023-01-03']
    for date in excluded_dates:
        row = result[result['date'] == date].iloc[0]
        assert pd.isna(row['tradable_price']), f"Expected no data for excluded date {date}"
    
    # Verify specific tradable price calculations for valid records
    # Tradable price = (high + low) / 2 * (adjusted_close / close)
    jan_4_row = result[result['date'] == '2023-01-04'].iloc[0]
    expected_jan_4 = (130 + 110) / 2 * (120/120)  # = 120.0
    assert jan_4_row['tradable_price'] == expected_jan_4
    
    jan_5_row = result[result['date'] == '2023-01-05'].iloc[0]
    expected_jan_5 = (140 + 120) / 2 * (130/130)  # = 130.0
    assert jan_5_row['tradable_price'] == expected_jan_5
