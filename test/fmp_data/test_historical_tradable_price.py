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
