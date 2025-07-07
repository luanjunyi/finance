"""
Tests for the OfflineData class.
"""

import pytest
import pandas as pd
import sqlite3
from unittest.mock import patch, MagicMock

from fmp_data.offline_data import OfflineData


@pytest.fixture
def mock_db_connection():
    """Create a mock database connection with test data."""
    conn = MagicMock()
    
    # Mock the read_sql_query function to return test data
    def mock_read_sql_query(query, conn, params=None):
        if 'income_statement' in query:
            return pd.DataFrame({
                'symbol': ['AAPL', 'AAPL', 'MSFT', 'MSFT'],
                'date': ['2023-03-31', '2022-12-31', '2023-03-31', '2022-12-31'],
                'revenue': [100, 120, 200, 220],
                'cost_of_revenue': [60, 70, 100, 110]
            })
        return pd.DataFrame()
    
    pd.read_sql_query = mock_read_sql_query
    
    # Mock the execute and fetchall methods
    cursor_mock = MagicMock()
    cursor_mock.fetchall.return_value = [
        (0, 'symbol', 'TEXT', 0, None, 0),
        (1, 'date', 'TEXT', 0, None, 0),
        (2, 'revenue', 'REAL', 0, None, 0),
        (3, 'cost_of_revenue', 'REAL', 0, None, 0)
    ]
    conn.execute.return_value = cursor_mock
    
    return conn


@pytest.fixture
def offline_data(mock_db_connection):
    """Create an OfflineData instance with mock database."""
    with patch('sqlite3.connect') as mock_connect:
        mock_connect.return_value.__enter__.return_value = mock_db_connection
        
        # Create OfflineData instance with test parameters
        return OfflineData(
            symbol=['AAPL', 'MSFT'],
            metrics={'revenue': '', 'cost_of_revenue': ''},
            for_date=['2023-03-31', '2022-12-31']
        )


def test_gen_method():
    """Test the gen method yields correct data for each symbol."""
    # Mock the database connection and cursor
    with patch('sqlite3.connect') as mock_connect:
        # Create mock connection and cursor
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        # Mock the _find_metric_locations method to return test data
        with patch('fmp_data.offline_data.OfflineData._find_metric_locations') as mock_find_locations:
            mock_find_locations.return_value = {
                'revenue': 'income_statement',
                'cost_of_revenue': 'income_statement'
            }
            
            # Mock read_sql_query to return different data for each symbol
            def mock_read_sql_query(query, conn, params=None):
                # Check which symbol we're querying for
                symbol = params[0] if params else None
                
                if symbol == 'AAPL':
                    return pd.DataFrame({
                        'filing_date': ['2023-03-31', '2022-12-31'],
                        'revenue': [100, 120],
                        'cost_of_revenue': [60, 70]
                    })
                elif symbol == 'MSFT':
                    return pd.DataFrame({
                        'filing_date': ['2023-03-31', '2022-12-31'],
                        'revenue': [200, 220],
                        'cost_of_revenue': [100, 110]
                    })
                return pd.DataFrame()
                
            with patch('pandas.read_sql_query', mock_read_sql_query):
                # Create OfflineData instance
                offline_data = OfflineData(
                    symbol=['AAPL', 'MSFT'],
                    metrics=['revenue', 'cost_of_revenue'],
                    for_date=['2023-03-31', '2022-12-31']
                )
                
                # Get the generator
                symbol_data_gen = offline_data.gen()
                
                # Get the first symbol and its data
                symbol1, data1 = next(symbol_data_gen)
                assert symbol1 == 'AAPL'
                assert len(data1) == 2  # Two dates for AAPL
                assert 'revenue' in data1.columns
                assert 'cost_of_revenue' in data1.columns
                
                # Get the second symbol and its data
                symbol2, data2 = next(symbol_data_gen)
                assert symbol2 == 'MSFT'
                assert len(data2) == 2  # Two dates for MSFT
                assert 'revenue' in data2.columns
                assert 'cost_of_revenue' in data2.columns
                
                # Verify we've exhausted the generator
                with pytest.raises(StopIteration):
                    next(symbol_data_gen)


def test_gen_method_empty_data():
    """Test the gen method with empty data."""
    # Mock the database connection and cursor
    with patch('sqlite3.connect') as mock_connect:
        # Create mock connection and cursor
        mock_conn = MagicMock()
        mock_connect.return_value.__enter__.return_value = mock_conn
        
        # Mock the _find_metric_locations method to return test data
        with patch('fmp_data.offline_data.OfflineData._find_metric_locations') as mock_find_locations:
            mock_find_locations.return_value = {
                'revenue': 'income_statement'
            }
            
            # Mock read_sql_query to return empty data
            def mock_read_sql_query(query, conn, params=None):
                return pd.DataFrame()
                
            with patch('pandas.read_sql_query', mock_read_sql_query):
                # Create OfflineData instance
                offline_data = OfflineData(
                    symbol=['AAPL'],
                    metrics=['revenue'],
                    for_date=['2023-03-31']
                )
                
                # The generator should not yield anything
                assert list(offline_data.gen()) == []
