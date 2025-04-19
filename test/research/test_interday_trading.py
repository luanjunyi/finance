import pytest
import pandas as pd
import numpy as np
from datetime import datetime, date, timedelta
from unittest.mock import patch, MagicMock
from research.interday_trading import InterdayTrading
import sqlite3


@pytest.fixture
def create_trader():
    """Factory fixture that creates an InterdayTrading instance with customizable mocks."""
    def _create_trader(fcf_data=None, price_data=None, stock_data=None, date_range=('2024-01-01', '2024-01-05')):
        # Default stock data if not provided
        if stock_data is None:
            stock_data = pd.DataFrame({
                'symbol': ['AAPL', 'MSFT', 'GOOGL'],
                'sector': ['Technology', 'Technology', 'Technology'],
                'industry': ['Consumer Electronics', 'Software', 'Internet Services']
            })
        
        # Default FCF data if not provided
        if fcf_data is None:
            # Create hardcoded FCF data with 5 quarters for 3 symbols
            fcf_data = pd.DataFrame([
                # AAPL data - 5.0 per quarter
                {'symbol': 'AAPL', 'date': date(2023, 1, 1), 'free_cash_flow_per_share': 5.0},
                {'symbol': 'AAPL', 'date': date(2023, 4, 1), 'free_cash_flow_per_share': -5.0},
                {'symbol': 'AAPL', 'date': date(2023, 7, 1), 'free_cash_flow_per_share': 5.0},
                {'symbol': 'AAPL', 'date': date(2023, 10, 1), 'free_cash_flow_per_share': 15.0},
                {'symbol': 'AAPL', 'date': date(2024, 1, 1), 'free_cash_flow_per_share': 32325.0},
                
                # MSFT data - 4.0 per quarter
                {'symbol': 'MSFT', 'date': date(2023, 1, 1), 'free_cash_flow_per_share': 4.0},
                {'symbol': 'MSFT', 'date': date(2023, 4, 1), 'free_cash_flow_per_share': 4.0},
                {'symbol': 'MSFT', 'date': date(2023, 7, 1), 'free_cash_flow_per_share': 4.0},
                {'symbol': 'MSFT', 'date': date(2023, 10, 1), 'free_cash_flow_per_share': 4.0},
                {'symbol': 'MSFT', 'date': date(2024, 1, 1), 'free_cash_flow_per_share': 4323.0},
                
                # GOOGL data - 3.0 per quarter
                {'symbol': 'GOOGL', 'date': date(2023, 1, 1), 'free_cash_flow_per_share': 3.0},
                {'symbol': 'GOOGL', 'date': date(2023, 4, 1), 'free_cash_flow_per_share': 3.0},
                {'symbol': 'GOOGL', 'date': date(2023, 7, 1), 'free_cash_flow_per_share': 3.0},
                {'symbol': 'GOOGL', 'date': date(2023, 10, 1), 'free_cash_flow_per_share': 3.0},
                {'symbol': 'GOOGL', 'date': date(2024, 1, 1), 'free_cash_flow_per_share': 233.0}
            ])
        
        # Default price data if not provided
        if price_data is None:
            price_data = pd.DataFrame({
                'symbol': ['AAPL', 'MSFT', 'GOOGL'],
                'date': [date(2024, 1, 1)] * 3,
                'price': [150.0, 300.0, 2000.0]
            })
        
        # Create a trader with method patches
        with patch('research.interday_trading.InterdayTrading._load_valid_stocks') as mock_load_stocks, \
             patch('research.interday_trading.InterdayTrading._load_fundamentals') as mock_load_fundamentals, \
             patch('research.interday_trading.InterdayTrading._get_price_data') as mock_get_price, \
             patch('research.interday_trading.InterdayTrading._is_trading_day') as mock_is_trading_day:
            
            # Configure the method stubs
            mock_load_stocks.return_value = stock_data
            mock_load_fundamentals.return_value = fcf_data
            mock_get_price.return_value = price_data
            mock_is_trading_day.return_value = True
            
            # Create the trader instance
            begin_date, end_date = date_range
            trader = InterdayTrading(begin_date, end_date)
            
            return trader
    
    return _create_trader


def test_get_price_to_fcf(create_trader):
    """Test that get_price_to_fcf correctly calculates all FCF metrics and price ratios."""
    # Create a trader with default test data
    trader = create_trader()
    
    # Define test price data
    price_data = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL'],
        'date': [date(2024, 1, 1)] * 3,
        'price': [200.0, 400.0, 3000.0]
    })
    
    # Call the method under test
    result = trader.get_price_to_fcf(price_data, date(2024, 1, 1))
    
    # Verify the results
    assert len(result) == 3  # All 3 symbols should be in the result
    assert set(result['symbol']) == {'AAPL', 'MSFT', 'GOOGL'}
    
    # Check that all expected columns are present
    expected_columns = {
        'symbol', 'free_cash_flow', 'min_fcf', 'last_fcf', 
        'price', 'price_to_fcf'
    }
    assert expected_columns.issubset(set(result.columns))
    
    # Check the values for each symbol
    aapl_row = result[result['symbol'] == 'AAPL'].iloc[0]
    msft_row = result[result['symbol'] == 'MSFT'].iloc[0]
    googl_row = result[result['symbol'] == 'GOOGL'].iloc[0]
    
    # AAPL: Sum=20.0, Min=-5.0, Last=15.0
    assert aapl_row['free_cash_flow'] == 20.0  # Sum of 4 quarters
    assert aapl_row['min_fcf'] == -5.0  # Minimum quarterly value
    assert aapl_row['last_fcf'] == 15.0  # Most recent quarter
    assert aapl_row['price'] == 200.0
    assert aapl_row['price_to_fcf'] == 10.0  # 200 / 20
    
    # MSFT: Sum=16.0, Min=4.0, Last=4.0
    assert msft_row['free_cash_flow'] == 16.0  # Sum of 4 quarters
    assert msft_row['min_fcf'] == 4.0  # Minimum quarterly value
    assert msft_row['last_fcf'] == 4.0  # Most recent quarter
    assert msft_row['price'] == 400.0
    assert msft_row['price_to_fcf'] == 25.0  # 400 / 16
    
    # GOOGL: Sum=12.0, Min=3.0, Last=3.0
    assert googl_row['free_cash_flow'] == 12.0  # Sum of 4 quarters
    assert googl_row['min_fcf'] == 3.0  # Minimum quarterly value
    assert googl_row['last_fcf'] == 3.0  # Most recent quarter
    assert googl_row['price'] == 3000.0
    assert googl_row['price_to_fcf'] == 250.0  # 3000 / 12


def test_generate_basic_functionality(create_trader):
    """Test the basic functionality of the generate method."""
    # Create a trader with default test data
    trader = create_trader(date_range=('2024-01-01', '2024-01-03'))
    
    # Patch the _determine_operations method to return known operations
    with patch.object(trader, '_determine_operations') as mock_determine_ops, \
         patch.object(trader, '_is_trading_day', return_value=True):
        # Configure the mock to return different operations for each date
        def side_effect(df, current_date):
            if current_date == date(2024, 1, 1):
                return [['AAPL', '2024-01-01', 'BUY', 0.1]]
            elif current_date == date(2024, 1, 2):
                return [['MSFT', '2024-01-02', 'BUY', 0.2]]
            elif current_date == date(2024, 1, 3):
                return [['GOOGL', '2024-01-03', 'SELL', 0.3]]
            return []
        
        mock_determine_ops.side_effect = side_effect
        
        # Call the method under test
        operations = trader.generate()
        
        # Verify the results
        assert len(operations) == 3  # One operation for each day
        assert operations[0] == ['AAPL', '2024-01-01', 'BUY', 0.1]
        assert operations[1] == ['MSFT', '2024-01-02', 'BUY', 0.2]
        assert operations[2] == ['GOOGL', '2024-01-03', 'SELL', 0.3]
        
        # Verify that _determine_operations was called once for each day
        assert mock_determine_ops.call_count == 3


def test_generate_skips_non_trading_days(create_trader):
    """Test that generate skips non-trading days."""
    # Create a trader with default test data
    trader = create_trader(date_range=('2024-01-01', '2024-01-03'))
    
    # Patch the _is_trading_day method to simulate market closures
    with patch.object(trader, '_is_trading_day') as mock_is_trading_day, \
         patch.object(trader, '_determine_operations') as mock_determine_ops:
        
        # Configure _is_trading_day to return False for Jan 2
        def is_trading_day_side_effect(check_date):
            if check_date == date(2024, 1, 2):  # Simulate market closed on Jan 2
                return False
            return True
        
        mock_is_trading_day.side_effect = is_trading_day_side_effect
        
        # Configure _determine_operations to return different operations for each date
        def determine_ops_side_effect(df, current_date):
            if current_date == date(2024, 1, 1):
                return [['AAPL', '2024-01-01', 'BUY', 0.1]]
            elif current_date == date(2024, 1, 3):
                return [['GOOGL', '2024-01-03', 'SELL', 0.3]]
            return []
        
        mock_determine_ops.side_effect = determine_ops_side_effect
        
        # Call the method under test
        operations = trader.generate()
        
        # Verify the results
        assert len(operations) == 2  # Only operations for Jan 1 and Jan 3
        assert operations[0] == ['AAPL', '2024-01-01', 'BUY', 0.1]
        assert operations[1] == ['GOOGL', '2024-01-03', 'SELL', 0.3]
        
        # Verify that _determine_operations was called only for trading days
        assert mock_determine_ops.call_count == 2
