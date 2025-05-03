import pytest
import pandas as pd
from datetime import date, timedelta
from unittest.mock import patch
from research.interday_trading import InterdayTrading


@pytest.fixture(scope="session")
def create_trader(trading_test_db):
    """Factory fixture that creates an InterdayTrading instance using the test database."""
    def _create_trader(date_range=('2024-01-01', '2024-01-05')):
        # Create the trader instance
        begin_date, end_date = date_range
        
        # Create the trader instance with string dates
        trader = InterdayTrading(begin_date, end_date, trading_test_db)
        
        return trader
    
    return _create_trader


@pytest.fixture(scope="module")
def trader(create_trader):
    """Shared trader instance that can be reused across multiple tests."""
    return create_trader()


def test_get_price_to_fcf(trader):
    """Test that get_price_to_fcf correctly calculates all FCF metrics and price ratios."""
    
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
 'price_to_fcf'
    }
    assert expected_columns.issubset(set(result.columns))
    
    # Check the values for each symbol
    aapl_row = result[result['symbol'] == 'AAPL'].iloc[0]
    msft_row = result[result['symbol'] == 'MSFT'].iloc[0]
    googl_row = result[result['symbol'] == 'GOOGL'].iloc[0]
    
    # AAPL: Free cash flow values from the test database
    assert pytest.approx(aapl_row['free_cash_flow'], rel=1e-6) == 3.6 + 3.45 + 3.3 + 3.15
    assert pytest.approx(aapl_row['min_fcf'], rel=1e-6) == 3.15  # Minimum quarterly value
    assert pytest.approx(aapl_row['last_fcf'], rel=1e-6) == 3.6  # Most recent quarter
    assert pytest.approx(aapl_row['price_to_fcf'], rel=1e-6) == 200 / (3.6 + 3.45 + 3.3 + 3.15)
    
    # MSFT: Free cash flow values from the test database
    assert pytest.approx(msft_row['free_cash_flow'], rel=1e-6) == 2.465 + 2.38 + 2.295 + 2.21
    assert pytest.approx(msft_row['min_fcf'], rel=1e-6) == 2.21  # Minimum quarterly value
    assert pytest.approx(msft_row['last_fcf'], rel=1e-6) == 2.465  # Most recent quarter
    assert pytest.approx(msft_row['price_to_fcf'], rel=1e-6) == 400 / (2.465 + 2.38 + 2.295 + 2.21)
    
    # GOOGL: Free cash flow values from the test database
    assert pytest.approx(googl_row['free_cash_flow'], rel=1e-6) == 3.93 + 3.82 + 3.72 + 3.62
    assert pytest.approx(googl_row['min_fcf'], rel=1e-6) == 3.62  # Minimum quarterly value
    assert pytest.approx(googl_row['last_fcf'], rel=1e-6) == 3.93  # Most recent quarter
    assert pytest.approx(googl_row['price_to_fcf'], rel=1e-6) == 3000 / (3.93 + 3.82 + 3.72 + 3.62)


def test_generate_basic_functionality(trader):
    """Test the basic functionality of the generate method."""
    
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
        assert mock_determine_ops.call_count == 5


def test_generate_skips_non_trading_days(trader):
    """Test that generate skips non-trading days."""
    
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
        assert mock_determine_ops.call_count == 4


def test_filter_fundamentals(trader):
    """Test that _filter_fundamentals correctly filters data based on date range and minimum records."""
    
    # Test case 1: Filter with min_records=4 (should include symbols with enough data)
    current_date = date(2024, 1, 1)  # Current date is Jan 1, 2024
    result1 = trader._filter_fundamentals(current_date, 400, 4)
    
    # Verify we have some results
    assert not result1.empty
    
    # Get unique symbols in the result
    symbols_in_result = set(result1['symbol'].unique())
    
    # Verify date filtering (last quarter date would be around Oct 1, 2023)
    # All dates should be before the current date
    # Ensure all dates are date objects
    assert all(isinstance(d, date) for d in result1['date'])
    assert all(d <= current_date for d in result1['date'])
    
    # Test case 2: Filter with a higher min_records requirement
    result2 = trader._filter_fundamentals(current_date, 400, 8)
    
    # If we require more records, we should get fewer symbols or none
    if not result2.empty:
        symbols_in_result2 = set(result2['symbol'].unique())
        assert len(symbols_in_result2) <= len(symbols_in_result)
        # Ensure all dates are date objects
        assert all(isinstance(d, date) for d in result2['date'])
    
    # Test case 3: Filter with a smaller window
    result3 = trader._filter_fundamentals(current_date, 90, 2)  # Just one quarter window
    
    # Verify date filtering for smaller window
    if not result3.empty:
        # All dates should be within ~90 days of the last quarter date
        last_quarter_date = current_date - timedelta(days=90)
        first_quarter_date = last_quarter_date - timedelta(days=90)
        # Ensure all dates are date objects
        assert all(isinstance(d, date) for d in result3['date'])
        assert all(d >= first_quarter_date for d in result3['date'])
    
    # Verify sorting (should be sorted by symbol asc, date desc)
    # The _filter_fundamentals method should return data sorted by symbol and date
    assert result1.equals(result1.sort_values(['symbol', 'date'], ascending=[True, False]))


def test_get_price_before(trader):
    """Test that _get_price_before correctly finds the most recent trading day and returns price data."""
    
    # Get the symbols from the cached data
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Test case 1: Get price for a specific date
    target_date = date(2024, 1, 1)  # This date exists in our test database
    result = trader._get_price_before(symbols, target_date)
    
    # Verify the results
    assert len(result) == 3  # Should have data for all 3 symbols
    assert set(result['symbol']) == set(symbols)  # All requested symbols should be present
    assert 'price' in result.columns  # Should have price data
    assert 'actual_date' in result.columns  # Should have the actual date used
    
    # All rows should have the same actual_date
    assert len(result['actual_date'].unique()) == 1
    
    # Ensure actual_date is a date object
    assert isinstance(result['actual_date'].iloc[0], date)
    
    # Test case 2: Get price for a date that might not be in the database
    # The method should find the most recent date before the target
    future_date = date(2023, 11, 5)  # A date that might not be in our test database
    result2 = trader._get_price_before(symbols, future_date)
    
    # Verify the results
    assert len(result2) == 3  # Should still have data for all 3 symbols
    assert set(result2['symbol']) == set(symbols)
    
    # The actual_date should be before or equal to the target date
    actual_date = result2['actual_date'].iloc[0]
    assert isinstance(actual_date, date)  # Ensure it's a date object
    assert actual_date <= future_date
    
    # Test case 3: Exception handling with patching
    # We'll patch _is_trading_day to always return False to force the error
    with patch.object(trader, '_is_trading_day', return_value=False):
        with pytest.raises(ValueError, match="Could not find trading day within"):
            trader._get_price_before(symbols, target_date)


def test_get_price_momentum(trader):
    """Test that get_price_momentum correctly calculates price momentum metrics."""
    
    # Use a date that exists in our test database
    current_date = date(2024, 1, 1)
    
    # Get the symbols from the cached data
    symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    # Test with a specific date range by patching _get_price_before
    # This allows us to test the calculation logic with controlled inputs
    with patch.object(trader, '_get_price_before') as mock_get_price_before:
        # Set up mock to return controlled price data
        def get_price_before_side_effect(symbols, date_param):
            if date_param == current_date:
                return pd.DataFrame({
                    'symbol': ['AAPL', 'MSFT', 'GOOGL'],
                    'price': [200.0, 400.0, 3000.0],
                    'actual_date': [current_date] * 3
                })
            elif date_param == current_date - timedelta(days=90):  # 3 months ago
                return pd.DataFrame({
                    'symbol': ['AAPL', 'MSFT', 'GOOGL'],
                    'price': [180.0, 380.0, 2800.0],
                    'actual_date': [current_date - timedelta(days=90)] * 3
                })
            elif date_param == current_date - timedelta(days=180):  # 6 months ago
                return pd.DataFrame({
                    'symbol': ['AAPL', 'MSFT', 'GOOGL'],
                    'price': [170.0, 360.0, 2600.0],
                    'actual_date': [current_date - timedelta(days=180)] * 3
                })
            elif date_param == current_date - timedelta(days=270):  # 9 months ago
                return pd.DataFrame({
                    'symbol': ['AAPL', 'MSFT', 'GOOGL'],
                    'price': [160.0, 340.0, 2400.0],
                    'actual_date': [current_date - timedelta(days=270)] * 3
                })
            elif date_param == current_date - timedelta(days=365):  # 12 months ago
                return pd.DataFrame({
                    'symbol': ['AAPL', 'MSFT', 'GOOGL'],
                    'price': [150.0, 320.0, 2200.0],
                    'actual_date': [current_date - timedelta(days=365)] * 3
                })
            return pd.DataFrame(columns=['symbol', 'price', 'actual_date'])
        
        mock_get_price_before.side_effect = get_price_before_side_effect
        
        # Call the method again with our controlled data
        controlled_result = trader.get_price_momentum(current_date)
        
        # Verify the results
        assert not controlled_result.empty  # Should have some results
        
        # Check that all expected columns are present
        expected_columns = {
            'symbol', 'm3', 'm6', 'm9', 'm12'
        }
        assert expected_columns.issubset(set(controlled_result.columns))
    
    # Since we're using real data from the database, we can't assert exact values
    # Instead, we'll verify that the calculations are reasonable
    for symbol in controlled_result['symbol'].unique():
        row = controlled_result[controlled_result['symbol'] == symbol].iloc[0]
        
        # Verify that all momentum metrics are floats
        assert isinstance(row['m3'], float)
        assert isinstance(row['m6'], float)
        assert isinstance(row['m9'], float)
        assert isinstance(row['m12'], float)        
        
        # Now we can verify exact calculations
        aapl_data = controlled_result[controlled_result['symbol'] == 'AAPL'].iloc[0]
        
        # Verify momentum calculations for AAPL
        assert pytest.approx(aapl_data['m3'], rel=1e-6) == (200.0/180.0) - 1  # 3-month momentum
        assert pytest.approx(aapl_data['m6'], rel=1e-6) == (200.0/170.0) - 1  # 6-month momentum
        assert pytest.approx(aapl_data['m9'], rel=1e-6) == (200.0/160.0) - 1  # 9-month momentum
        assert pytest.approx(aapl_data['m12'], rel=1e-6) == (200.0/150.0) - 1  # 12-month momentum
        
        # Verify momentum calculations for MSFT
        msft_data = controlled_result[controlled_result['symbol'] == 'MSFT'].iloc[0]
        assert pytest.approx(msft_data['m3'], rel=1e-6) == (400.0/380.0) - 1  # 3-month momentum
        assert pytest.approx(msft_data['m6'], rel=1e-6) == (400.0/360.0) - 1  # 6-month momentum
        assert pytest.approx(msft_data['m9'], rel=1e-6) == (400.0/340.0) - 1  # 9-month momentum
        assert pytest.approx(msft_data['m12'], rel=1e-6) == (400.0/320.0) - 1  # 12-month momentum
        
        # Verify momentum calculations for GOOGL
        googl_data = controlled_result[controlled_result['symbol'] == 'GOOGL'].iloc[0]
        assert pytest.approx(googl_data['m3'], rel=1e-6) == (3000.0/2800.0) - 1  # 3-month momentum
        assert pytest.approx(googl_data['m6'], rel=1e-6) == (3000.0/2600.0) - 1  # 6-month momentum
        assert pytest.approx(googl_data['m9'], rel=1e-6) == (3000.0/2400.0) - 1  # 9-month momentum
        assert pytest.approx(googl_data['m12'], rel=1e-6) == (3000.0/2200.0) - 1  # 12-month momentum
    


def test_get_revenue_growth(trader):
    """Test that get_revenue_growth correctly calculates YoY growth metrics."""
    
    # Use a date that exists in our test database
    current_date = date(2024, 1, 1)
    
    # Call the method under test
    result = trader.get_revenue_growth(current_date)
    
    # Since we're using real data from the database, we may or may not get results
    # depending on if there's enough historical data for YoY calculations
    if not result.empty:
        # Verify the structure of the results
        expected_columns = {
            'symbol', 'median_yoy', 'min_yoy', 'last_yoy'
        }
        assert expected_columns.issubset(set(result.columns))
        
        # Verify that the metrics are reasonable
        for symbol in result['symbol'].unique():
            row = result[result['symbol'] == symbol].iloc[0]
            
            # Verify that the metrics are floats
            assert isinstance(row['median_yoy'], float)
            assert isinstance(row['min_yoy'], float)
            assert isinstance(row['last_yoy'], float)
            
            # Verify that min_yoy <= median_yoy (by definition)
            assert row['min_yoy'] <= row['median_yoy']
    
    # Test with controlled data by patching _filter_fundamentals
    with patch.object(trader, '_filter_fundamentals') as mock_filter_fundamentals:
        # Create test data with clear growth patterns
        test_data = pd.DataFrame([
            # AAPL - 10% YoY growth each quarter
            {'symbol': 'AAPL', 'date': date(2022, 1, 1), 'free_cash_flow_per_share': 5.0, 'revenue': 100.0},
            {'symbol': 'AAPL', 'date': date(2022, 4, 1), 'free_cash_flow_per_share': 5.0, 'revenue': 110.0},
            {'symbol': 'AAPL', 'date': date(2022, 7, 1), 'free_cash_flow_per_share': 5.0, 'revenue': 120.0},
            {'symbol': 'AAPL', 'date': date(2022, 10, 1), 'free_cash_flow_per_share': 5.0, 'revenue': 130.0},
            {'symbol': 'AAPL', 'date': date(2023, 1, 1), 'free_cash_flow_per_share': 5.0, 'revenue': 110.0},  # +10% YoY
            {'symbol': 'AAPL', 'date': date(2023, 4, 1), 'free_cash_flow_per_share': 5.0, 'revenue': 121.0},  # +10% YoY
            {'symbol': 'AAPL', 'date': date(2023, 7, 1), 'free_cash_flow_per_share': 5.0, 'revenue': 132.0},  # +10% YoY
            {'symbol': 'AAPL', 'date': date(2023, 10, 1), 'free_cash_flow_per_share': 5.0, 'revenue': 143.0},  # +10% YoY
            
            # MSFT - varying growth rates: 20%, 15%, 10%, 5%
            {'symbol': 'MSFT', 'date': date(2022, 1, 1), 'free_cash_flow_per_share': 4.0, 'revenue': 200.0},
            {'symbol': 'MSFT', 'date': date(2022, 4, 1), 'free_cash_flow_per_share': 4.0, 'revenue': 220.0},
            {'symbol': 'MSFT', 'date': date(2022, 7, 1), 'free_cash_flow_per_share': 4.0, 'revenue': 240.0},
            {'symbol': 'MSFT', 'date': date(2022, 10, 1), 'free_cash_flow_per_share': 4.0, 'revenue': 260.0},
            {'symbol': 'MSFT', 'date': date(2023, 1, 1), 'free_cash_flow_per_share': 4.0, 'revenue': 240.0},  # +20% YoY
            {'symbol': 'MSFT', 'date': date(2023, 4, 1), 'free_cash_flow_per_share': 4.0, 'revenue': 253.0},  # +15% YoY
            {'symbol': 'MSFT', 'date': date(2023, 7, 1), 'free_cash_flow_per_share': 4.0, 'revenue': 264.0},  # +10% YoY
            {'symbol': 'MSFT', 'date': date(2023, 10, 1), 'free_cash_flow_per_share': 4.0, 'revenue': 273.0},  # +5% YoY
            
            # GOOGL - negative growth: -5%, -10%, -15%, -20%
            {'symbol': 'GOOGL', 'date': date(2022, 1, 1), 'free_cash_flow_per_share': 3.0, 'revenue': 300.0},
            {'symbol': 'GOOGL', 'date': date(2022, 4, 1), 'free_cash_flow_per_share': 3.0, 'revenue': 330.0},
            {'symbol': 'GOOGL', 'date': date(2022, 7, 1), 'free_cash_flow_per_share': 3.0, 'revenue': 360.0},
            {'symbol': 'GOOGL', 'date': date(2022, 10, 1), 'free_cash_flow_per_share': 3.0, 'revenue': 390.0},
            {'symbol': 'GOOGL', 'date': date(2023, 1, 1), 'free_cash_flow_per_share': 3.0, 'revenue': 285.0},  # -5% YoY
            {'symbol': 'GOOGL', 'date': date(2023, 4, 1), 'free_cash_flow_per_share': 3.0, 'revenue': 297.0},  # -10% YoY
            {'symbol': 'GOOGL', 'date': date(2023, 7, 1), 'free_cash_flow_per_share': 3.0, 'revenue': 306.0},  # -15% YoY
            {'symbol': 'GOOGL', 'date': date(2023, 10, 1), 'free_cash_flow_per_share': 3.0, 'revenue': 312.0},  # -20% YoY
        ])
        
        # Configure the mock to return our test data
        mock_filter_fundamentals.return_value = test_data
        
        # Call the method with our controlled data
        controlled_result = trader.get_revenue_growth(current_date)
        
        # Verify all symbols are included
        assert set(controlled_result['symbol']) == {'AAPL', 'MSFT', 'GOOGL'}
        
        # Verify the calculated metrics for each symbol
        aapl_row = controlled_result[controlled_result['symbol'] == 'AAPL'].iloc[0]
        msft_row = controlled_result[controlled_result['symbol'] == 'MSFT'].iloc[0]
        googl_row = controlled_result[controlled_result['symbol'] == 'GOOGL'].iloc[0]
        
        # AAPL: All quarters have 10% growth
        assert pytest.approx(aapl_row['median_yoy'], 0.01) == (110.0 / 100.0 - 1 + 121.0 / 110.0 - 1 + 132.0 / 120.0 - 1 + 143.0 / 130.0 - 1) / 4
        assert pytest.approx(aapl_row['min_yoy'], 0.01) == min(110.0 / 100.0 - 1, 121.0 / 110.0 - 1, 132.0 / 120.0 - 1, 143.0 / 130.0 - 1)
        assert pytest.approx(aapl_row['last_yoy'], 0.01) == 143.0 / 130.0 - 1
        
        # MSFT: 20%, 15%, 10%, 5% growth (median = 12.5%)
        assert pytest.approx(msft_row['median_yoy'], 0.01) == (240.0 / 200.0 - 1 + 253.0 / 220.0 - 1 + 264.0 / 240.0 - 1 + 273.0 / 260.0 - 1) / 4
        assert pytest.approx(msft_row['min_yoy'], 0.01) == 273.0 / 260.0 - 1  # 5% is the minimum
        assert pytest.approx(msft_row['last_yoy'], 0.01) == 273.0 / 260.0 - 1  # Last quarter has 5% growth
        
        # GOOGL: -5%, -10%, -15%, -20% growth (median = -12.5%)
        assert pytest.approx(googl_row['median_yoy'], 0.01) == (285.0 / 300.0 - 1 + 297.0 / 330.0 - 1 + 306.0 / 360.0 - 1 + 312.0 / 390.0 - 1) / 4
        assert pytest.approx(googl_row['min_yoy'], 0.01) == 312.0 / 390.0 - 1  # -20% is the minimum
        assert pytest.approx(googl_row['last_yoy'], 0.01) == 312.0 / 390.0 - 1  # Last quarter has -20% growth


def test_get_profit_margin(trader):
    """Test that get_profit_margin correctly retrieves operating profit margin metrics."""
    
    # Use a date that exists in our test database
    current_date = date(2024, 1, 1)
    
    # Call the method under test
    result = trader.get_profit_margin(current_date)
    
    # Since we're using real data from the database, we should get results
    # because we've added operating_margin to the test database
    assert not result.empty
    
    # Verify the structure of the results
    expected_columns = {
        'symbol', 'opm_3m', 'opm_6m', 'opm_9m', 'opm_12m'
    }
    assert expected_columns.issubset(set(result.columns))
    
    # Verify that the metrics are reasonable
    for symbol in result['symbol'].unique():
        row = result[result['symbol'] == symbol].iloc[0]
        
        # Verify that the metrics are floats
        assert isinstance(row['opm_3m'], float)
        assert isinstance(row['opm_6m'], float)
        assert isinstance(row['opm_9m'], float)
        assert isinstance(row['opm_12m'], float)
    
    # Test with controlled data by patching _filter_fundamentals
    with patch.object(trader, '_filter_fundamentals') as mock_filter_fundamentals:
        # Create test data with specific operating margins
        test_data = pd.DataFrame([
            # AAPL - operating margins for 4 quarters
            {'symbol': 'AAPL', 'date': date(2023, 10, 1), 'free_cash_flow_per_share': 3.6, 'revenue': 120.0, 'operating_margin': 0.30},
            {'symbol': 'AAPL', 'date': date(2023, 7, 1), 'free_cash_flow_per_share': 3.45, 'revenue': 115.0, 'operating_margin': 0.29},
            {'symbol': 'AAPL', 'date': date(2023, 4, 1), 'free_cash_flow_per_share': 3.3, 'revenue': 110.0, 'operating_margin': 0.28},
            {'symbol': 'AAPL', 'date': date(2023, 1, 1), 'free_cash_flow_per_share': 3.15, 'revenue': 105.0, 'operating_margin': 0.27},
            
            # MSFT - operating margins for 4 quarters
            {'symbol': 'MSFT', 'date': date(2023, 10, 1), 'free_cash_flow_per_share': 2.465, 'revenue': 58.0, 'operating_margin': 0.40},
            {'symbol': 'MSFT', 'date': date(2023, 7, 1), 'free_cash_flow_per_share': 2.38, 'revenue': 56.0, 'operating_margin': 0.39},
            {'symbol': 'MSFT', 'date': date(2023, 4, 1), 'free_cash_flow_per_share': 2.295, 'revenue': 54.0, 'operating_margin': 0.38},
            {'symbol': 'MSFT', 'date': date(2023, 1, 1), 'free_cash_flow_per_share': 2.21, 'revenue': 52.0, 'operating_margin': 0.37},
            
            # GOOGL - operating margins for 4 quarters
            {'symbol': 'GOOGL', 'date': date(2023, 10, 1), 'free_cash_flow_per_share': 3.93, 'revenue': 76.0, 'operating_margin': 0.35},
            {'symbol': 'GOOGL', 'date': date(2023, 7, 1), 'free_cash_flow_per_share': 3.82, 'revenue': 74.0, 'operating_margin': 0.34},
            {'symbol': 'GOOGL', 'date': date(2023, 4, 1), 'free_cash_flow_per_share': 3.72, 'revenue': 72.0, 'operating_margin': 0.33},
            {'symbol': 'GOOGL', 'date': date(2023, 1, 1), 'free_cash_flow_per_share': 3.62, 'revenue': 70.0, 'operating_margin': 0.32},
        ])
        
        # Configure the mock to return our test data
        mock_filter_fundamentals.return_value = test_data
        
        # Call the method with our controlled data
        controlled_result = trader.get_profit_margin(current_date)
        
        # Verify all symbols are included
        assert set(controlled_result['symbol']) == {'AAPL', 'MSFT', 'GOOGL'}
        
        # Verify the operating margins for each symbol
        aapl_row = controlled_result[controlled_result['symbol'] == 'AAPL'].iloc[0]
        msft_row = controlled_result[controlled_result['symbol'] == 'MSFT'].iloc[0]
        googl_row = controlled_result[controlled_result['symbol'] == 'GOOGL'].iloc[0]
        
        # AAPL: Check operating margins for each quarter
        assert pytest.approx(aapl_row['opm_3m'], 0.01) == 0.30  # Most recent quarter
        assert pytest.approx(aapl_row['opm_6m'], 0.01) == 0.29  # 2 quarters ago
        assert pytest.approx(aapl_row['opm_9m'], 0.01) == 0.28  # 3 quarters ago
        assert pytest.approx(aapl_row['opm_12m'], 0.01) == 0.27  # 4 quarters ago
        
        # MSFT: Check operating margins for each quarter
        assert pytest.approx(msft_row['opm_3m'], 0.01) == 0.40  # Most recent quarter
        assert pytest.approx(msft_row['opm_6m'], 0.01) == 0.39  # 2 quarters ago
        assert pytest.approx(msft_row['opm_9m'], 0.01) == 0.38  # 3 quarters ago
        assert pytest.approx(msft_row['opm_12m'], 0.01) == 0.37  # 4 quarters ago
        
        # GOOGL: Check operating margins for each quarter
        assert pytest.approx(googl_row['opm_3m'], 0.01) == 0.35  # Most recent quarter
        assert pytest.approx(googl_row['opm_6m'], 0.01) == 0.34  # 2 quarters ago
        assert pytest.approx(googl_row['opm_9m'], 0.01) == 0.33  # 3 quarters ago
        assert pytest.approx(googl_row['opm_12m'], 0.01) == 0.32  # 4 quarters ago
