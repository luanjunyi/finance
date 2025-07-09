"""Tests for the historical_tradable_price method in OfflineData class."""

import pytest
import pandas as pd
import sqlite3
from unittest.mock import patch, MagicMock

from fmp_data.offline_data import OfflineData


def test_historical_tradable_price_basic():
    """Test the basic functionality of historical_tradable_price method."""
    with patch('sqlite3.connect') as mock_connect:
        # Create mock connection
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        # Create test data with continuous dates
        test_data = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'AAPL'],
            'date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'high': [110, 120, 130],
            'low': [90, 100, 110]
        })
        
        # Mock read_sql_query to return our test data
        with patch('pandas.read_sql_query', return_value=test_data):
            # Call the method
            result = OfflineData.historical_tradable_price(
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


def test_historical_tradable_price_with_gaps():
    """Test that the method fills gaps in price data correctly."""
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
            # Call the method
            result = OfflineData.historical_tradable_price(
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


def test_historical_tradable_price_empty_data():
    """Test that the method handles empty data correctly."""
    with patch('sqlite3.connect') as mock_connect:
        # Create mock connection
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        # Create empty test data
        test_data = pd.DataFrame(columns=['symbol', 'date', 'high', 'low'])
        
        # Mock read_sql_query to return empty data
        with patch('pandas.read_sql_query', return_value=test_data):
            # Call the method
            result = OfflineData.historical_tradable_price(
                symbol='AAPL',
                start_date='2023-01-01',
                end_date='2023-01-03'
            )
            
            # Verify results
            assert result.empty


def test_historical_tradable_price_extended_gaps():
    """Test that the method handles extended gaps correctly."""
    with patch('sqlite3.connect') as mock_connect:
        # Create mock connection
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        # Create test data with only first day data
        test_data = pd.DataFrame({
            'symbol': ['AAPL'],
            'date': ['2023-01-01'],
            'high': [110],
            'low': [90]
        })
        
        # Mock read_sql_query to return our test data
        with patch('pandas.read_sql_query', return_value=test_data):
            # Call the method
            result = OfflineData.historical_tradable_price(
                symbol='AAPL',
                start_date='2023-01-01',
                end_date='2023-01-05'
            )
            
            # Verify results
            assert len(result) == 5  # Should have entries for all 5 days
            assert list(result['date']) == ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']
            
            # All days should have the same tradable price (100) since we're forward filling
            for i in range(5):
                assert result.loc[i, 'tradable_price'] == 100
