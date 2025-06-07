# pytest: --cov=fmp_fetch --no-cov-on-fail
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from fmp_fetch.online_dataset import Dataset, INCOME_STATEMENT, CASHFLOW_STATEMENT, BALANCE_SHEET


@pytest.fixture
def mock_dataset():
    """Create a Dataset instance with mocked FMPAPI."""
    with patch('fmp_fetch.online_dataset.FMPAPI') as mock_fmp_api_class:
        # Setup mock FMPAPI instance
        mock_api = MagicMock()
        mock_fmp_api_class.return_value = mock_api
        
        # Initialize the Dataset instance
        dataset = Dataset(
            symbols=['AAPL', 'MSFT'],
            metrics=['revenue', 'netIncome'],
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        # With lazy loading, build() won't be called until data is accessed
        
        # Return both the instance and the mock for assertions
        yield dataset, mock_api


def test_initialization():
    """Test Dataset initialization."""
    with patch('fmp_fetch.online_dataset.FMPAPI') as mock_fmp_api_class:
        
        # Test with a single symbol
        dataset = Dataset(
            symbols='AAPL',
            metrics=['revenue', 'netIncome'],
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        assert dataset.symbols == ['AAPL']
        assert dataset.metrics == ['revenue', 'netIncome']
        assert dataset.start_date == '2024-01-01'
        assert dataset.end_date == '2024-01-31'
        assert dataset.api is mock_fmp_api_class.return_value
        assert dataset._data is None  # Data should not be built yet
        
        # Test with multiple symbols
        dataset = Dataset(
            symbols=['AAPL', 'MSFT'],
            metrics=['revenue', 'netIncome'],
            start_date='2024-01-01',
            end_date='2024-01-31'
        )
        
        assert dataset.symbols == ['AAPL', 'MSFT']
        assert dataset._data is None  # Data should not be built yet


def test_categorize_metrics(mock_dataset):
    """Test the _categorize_metrics method."""
    dataset, _ = mock_dataset
    
    # Test with income statement metrics
    dataset.metrics = ['revenue', 'netIncome']
    categorized = dataset._categorize_metrics()
    
    assert 'revenue' in categorized[INCOME_STATEMENT]
    assert 'netIncome' in categorized[INCOME_STATEMENT]
    assert len(categorized[CASHFLOW_STATEMENT]) == 0
    assert len(categorized[BALANCE_SHEET]) == 0
    
    # Test with cash flow statement metrics
    dataset.metrics = ['freeCashFlow', 'operatingCashFlow']
    categorized = dataset._categorize_metrics()
    
    assert 'freeCashFlow' in categorized[CASHFLOW_STATEMENT]
    assert 'operatingCashFlow' in categorized[CASHFLOW_STATEMENT]
    assert len(categorized[INCOME_STATEMENT]) == 0
    assert len(categorized[BALANCE_SHEET]) == 0
    
    # Test with balance sheet metrics
    dataset.metrics = ['totalAssets', 'totalLiabilities']
    categorized = dataset._categorize_metrics()
    
    assert 'totalAssets' in categorized[BALANCE_SHEET]
    assert 'totalLiabilities' in categorized[BALANCE_SHEET]
    assert len(categorized[INCOME_STATEMENT]) == 0
    assert len(categorized[CASHFLOW_STATEMENT]) == 0
    
    # Test with mixed metrics
    dataset.metrics = ['revenue', 'freeCashFlow', 'totalAssets', 'close_price']
    categorized = dataset._categorize_metrics()
    
    assert 'revenue' in categorized[INCOME_STATEMENT]
    assert 'freeCashFlow' in categorized[CASHFLOW_STATEMENT]
    assert 'totalAssets' in categorized[BALANCE_SHEET]
    assert 'close_price' in categorized['price']


def test_fetch_financial_statements(mock_dataset):
    """Test the _fetch_financial_statements method."""
    dataset, mock_api = mock_dataset
    
    # Setup mock income statement data
    mock_income_statements_aapl = [
        {
            'date': '2024-01-31',
            'symbol': 'AAPL',
            'filingDate': '2024-02-15',
            'revenue': 1,
            'netIncome': 2
        },
        {
            'date': '2023-10-31',
            'symbol': 'AAPL',
            'filingDate': '2023-11-15',
            'revenue': 3,
            'netIncome': 4
        },
        {
            'date': '2023-07-31',
            'symbol': 'AAPL',
            'filingDate': '2023-08-15',
            'revenue': 5,
            'netIncome': 6
        }
    ]
    
    mock_income_statements_msft = [
        {
            'date': '2024-01-31',
            'symbol': 'MSFT',
            'filingDate': '2024-02-10',
            'revenue': 10,
            'netIncome': 20
        },
        {
            'date': '2023-10-31',
            'symbol': 'MSFT',
            'filingDate': '2023-11-10',
            'revenue': 30,
            'netIncome': 40
        },
        {
            'date': '2023-07-31',
            'symbol': 'MSFT',
            'filingDate': '2023-08-10',
            'revenue': 50,
            'netIncome': 60
        }
    ]
    
    # Configure the mock to return different data for each symbol
    def get_income_statement_side_effect(symbol, period, limit):
        if symbol == 'AAPL':
            return mock_income_statements_aapl
        elif symbol == 'MSFT':
            return mock_income_statements_msft
        return []
    
    mock_api.get_income_statement.side_effect = get_income_statement_side_effect
    
    # Call the method
    result = dataset._fetch_financial_statements(INCOME_STATEMENT, {'revenue', 'netIncome'})
    
    # Verify the result
    assert not result.empty
    assert len(result) == 4  # 2 statements for each of the 2 symbols
    assert 'revenue' in result.columns
    assert 'netIncome' in result.columns
    
    # Test that the data is correctly processed
    assert 'symbol' in result.index.names
    assert 'filing_date' in result.columns
    
    # Test that the data for each symbol is present
    aapl_data = result.xs('AAPL')
    msft_data = result.xs('MSFT')
    
    assert len(aapl_data) == 2
    assert len(msft_data) == 2
    
    # Verify specific values
    assert aapl_data[aapl_data['date'] == pd.to_datetime('2023-10-31')]['revenue'].values[0] == 3
    assert aapl_data[aapl_data['date'] == pd.to_datetime('2023-10-31')]['netIncome'].values[0] == 4
    assert aapl_data[aapl_data['date'] == pd.to_datetime('2023-07-31')]['revenue'].values[0] == 5
    assert aapl_data[aapl_data['date'] == pd.to_datetime('2023-07-31')]['netIncome'].values[0] == 6

    assert msft_data[msft_data['date'] == pd.to_datetime('2023-10-31')]['revenue'].values[0] == 30
    assert msft_data[msft_data['date'] == pd.to_datetime('2023-10-31')]['netIncome'].values[0] == 40
    assert msft_data[msft_data['date'] == pd.to_datetime('2023-07-31')]['revenue'].values[0] == 50
    assert msft_data[msft_data['date'] == pd.to_datetime('2023-07-31')]['netIncome'].values[0] == 60


def test_fetch_price_data(mock_dataset):
    """Test the _fetch_price_data method."""
    import numpy as np
    dataset, mock_api = mock_dataset
    
    # Setup mock price data with incomplete date coverage
    mock_prices_aapl = [
        {'date': '2024-01-01', 'adjClose': 190.0, 'symbol': 'AAPL'},
        # Missing 2024-01-02
        {'date': '2024-01-03', 'adjClose': 195.0, 'symbol': 'AAPL'}
    ]
    
    mock_prices_msft = [
        # Missing 2024-01-01
        {'date': '2024-01-02', 'adjClose': 385.0, 'symbol': 'MSFT'},
        {'date': '2024-01-03', 'adjClose': 390.0, 'symbol': 'MSFT'}
    ]
    
    # Configure the mock to return different data for each symbol
    def get_prices_side_effect(symbol, start_date, end_date):
        if symbol == 'AAPL':
            return mock_prices_aapl
        elif symbol == 'MSFT':
            return mock_prices_msft
        return []
    
    mock_api.get_prices.side_effect = get_prices_side_effect
    
    # Call the method
    result = dataset._fetch_price_data()
    
    # Test the core logic - verify that the price data is correctly processed
    assert not result.empty
    
    # We should have entries for all dates in the range (Jan 1-31) for both symbols
    expected_date_count = 31  # Days in January
    expected_total_entries = expected_date_count * 2  # 2 symbols
    assert len(result) == expected_total_entries
    
    # Verify the structure of the result
    assert 'symbol' in result.index.names
    assert 'close_price' in result.columns
    
    # Verify that the adjClose values are correctly mapped to close_price
    aapl_data = result.xs('AAPL')
    msft_data = result.xs('MSFT')
    
    # Verify specific values to ensure correct data transformation
    # Since date is no longer in the index, we need to filter by the date column
    assert aapl_data[aapl_data['date'] == pd.Timestamp('2024-01-01')]['close_price'].values[0] == 190.0
    # 2024-01-02 should use the previous day's price (190.0) for AAPL
    assert aapl_data[aapl_data['date'] == pd.Timestamp('2024-01-02')]['close_price'].values[0] == 190.0
    assert aapl_data[aapl_data['date'] == pd.Timestamp('2024-01-03')]['close_price'].values[0] == 195.0
    
    # 2024-01-01 should be np.nan for MSFT since there's no previous price
    assert np.isnan(msft_data[msft_data['date'] == pd.Timestamp('2024-01-01')]['close_price'].values[0])
    assert msft_data[msft_data['date'] == pd.Timestamp('2024-01-02')]['close_price'].values[0] == 385.0
    assert msft_data[msft_data['date'] == pd.Timestamp('2024-01-03')]['close_price'].values[0] == 390.0
    
    # Verify that all dates in the range have entries
    date_range = pd.date_range(start=dataset.start_date, end=dataset.end_date)
    for date in date_range:
        assert not aapl_data[aapl_data['date'] == date].empty, f"Missing date {date} for AAPL"
        assert not msft_data[msft_data['date'] == date].empty, f"Missing date {date} for MSFT"


def test_get_latest_values_for_dates(mock_dataset):
    """Test the _get_latest_values_for_dates method."""
    dataset, _ = mock_dataset
    
    # Create test data with different dates and filing dates
    test_data = pd.DataFrame([
        {'symbol': 'AAPL', 'date': '2024-01-15', 'filing_date': '2024-01-15', 'revenue': 100},
        {'symbol': 'AAPL', 'date': '2023-10-15', 'filing_date': '2023-10-15', 'revenue': 95},
        {'symbol': 'AAPL', 'date': '2023-07-15', 'filing_date': '2023-07-15', 'revenue': 90},
        {'symbol': 'MSFT', 'date': '2024-01-20', 'filing_date': '2024-01-20', 'revenue': 50},
        {'symbol': 'MSFT', 'date': '2023-10-20', 'filing_date': '2023-10-20', 'revenue': 48},
        {'symbol': 'MSFT', 'date': '2023-07-20', 'filing_date': '2023-07-20', 'revenue': 46}
    ])
    
    test_data['date'] = pd.to_datetime(test_data['date'])
    test_data['filing_date'] = pd.to_datetime(test_data['filing_date'])
    test_data.set_index('symbol', inplace=True)
    
    # Create date range
    date_range = pd.date_range(start='2024-01-01', end='2024-01-10', freq='D')
    
    # Call the method
    result = dataset._get_latest_values_for_dates(test_data, date_range)
    
    # Verify the core logic - test that the latest values are correctly assigned to each date
    assert not result.empty
    
    # The result DataFrame should have 'symbol' and 'date' as regular columns, not as index
    assert 'revenue' in result.columns
    
    # Filter data for each symbol
    aapl_data = result.xs('AAPL', level='symbol')
    msft_data = result.xs('MSFT', level='symbol')
    
    # Check that we have data for both symbols
    assert len(aapl_data) == 10
    assert len(msft_data) == 10
    
    # For dates on or after Jan 15, we shouldn't have data (filing date is in the future)
    # For dates before Jan 15 but after Oct 15 - 90 days, we should use Oct 15 data
    
    # Check a specific date (Jan 5) for AAPL
    jan_05_aapl = aapl_data.xs(pd.Timestamp('2024-01-05'))
    if not jan_05_aapl.empty:
        assert jan_05_aapl['revenue'] == 95  # Oct 15 data
    
    # Check a specific date (Jan 5) for MSFT
    jan_05_msft = msft_data.xs(pd.Timestamp('2024-01-05'))
    if not jan_05_msft.empty:
        assert jan_05_msft['revenue'] == 48  # Oct 20 data


def test_build_with_basic_metrics(mock_dataset):
    """Test the build method with basic metrics."""
    dataset, mock_api = mock_dataset
    
    # Setup dataset with basic metrics
    dataset.metrics = ['revenue', 'netIncome', 'close_price']
    
    # Setup mock financial statement data
    mock_income_statements_aapl = [
        {
            'date': '2024-01-15',
            'symbol': 'AAPL',
            'filingDate': '2024-01-15',
            'revenue': 100,
            'netIncome': 20
        },
        {
            'date': '2023-10-15',
            'symbol': 'AAPL',
            'filingDate': '2023-10-15',
            'revenue': 95,
            'netIncome': 19
        }
    ]
    
    mock_income_statements_msft = [
        {
            'date': '2024-01-20',
            'symbol': 'MSFT',
            'filingDate': '2024-01-20',
            'revenue': 50,
            'netIncome': 15
        },
        {
            'date': '2023-10-20',
            'symbol': 'MSFT',
            'filingDate': '2023-10-20',
            'revenue': 48,
            'netIncome': 14
        }
    ]
    
    # Setup mock price data
    mock_prices_aapl = [
        {'date': '2024-01-01', 'adjClose': 190.0, 'symbol': 'AAPL'},
        {'date': '2024-01-02', 'adjClose': 192.0, 'symbol': 'AAPL'}
    ]
    
    mock_prices_msft = [
        {'date': '2024-01-01', 'adjClose': 380.0, 'symbol': 'MSFT'},
        {'date': '2024-01-02', 'adjClose': 385.0, 'symbol': 'MSFT'}
    ]
    
    # Configure the mocks to return different data for each symbol
    def get_income_statement_side_effect(symbol, period, limit):
        if symbol == 'AAPL':
            return mock_income_statements_aapl
        elif symbol == 'MSFT':
            return mock_income_statements_msft
        return []
    
    def get_prices_side_effect(symbol, start_date, end_date):
        if symbol == 'AAPL':
            return mock_prices_aapl
        elif symbol == 'MSFT':
            return mock_prices_msft
        return []
    
    mock_api.get_income_statement.side_effect = get_income_statement_side_effect
    mock_api.get_prices.side_effect = get_prices_side_effect
    
    # We'll test the actual implementation without mocking internal methods
    # This tests the integration of the components
    
    # Call the build method
    result = dataset.build()
    
    # Verify the core logic - test that the build method correctly assembles the dataset
    assert not result.empty
    
    # Verify the structure of the result
    assert 'symbol' in result.index.names
    assert 'date' in result.index.names
    assert 'revenue' in result.columns
    assert 'netIncome' in result.columns
    assert 'close_price' in result.columns
    
    # Check specific values
    aapl_jan01 = result.loc[('AAPL', pd.Timestamp('2024-01-01'))]
    assert aapl_jan01['close_price'] == 190.0
    # The revenue should come from the Oct 15 statement (since Jan 15 is in the future)
    assert aapl_jan01['revenue'] == 95
    assert aapl_jan01['netIncome'] == 19
    
    msft_jan02 = result.loc[(('MSFT', pd.Timestamp('2024-01-02')))]
    assert msft_jan02['close_price'] == 385.0
    # The revenue should come from the Oct 20 statement (since Jan 20 is in the future)
    assert msft_jan02['revenue'] == 48
    assert msft_jan02['netIncome'] == 14

##########################################
# Derive metrics
##########################################


@pytest.fixture
def mock_dataset_with_statements():
    """Create a Dataset instance with mocked financial statements for PE and FCF tests."""
    with patch('fmp_fetch.online_dataset.FMPAPI') as mock_fmp_api_class:
        # Setup mock FMPAPI instance
        mock_api = MagicMock()
        mock_fmp_api_class.return_value = mock_api
        
        # Initialize the Dataset instance
        dataset = Dataset(
            symbols=['AAPL', 'MSFT'],
            metrics=['pe', 'price_to_fcf', 'price_to_owner_earning', 'close_price'],
            start_date='2023-01-01',
            end_date='2024-01-31'
        )
        
        # Income statements
        income_data = pd.DataFrame([
            # AAPL data
            {'symbol': 'AAPL', 'date': '2022-09-30', 'filing_date': '2022-10-15', 'netIncome': 48, 'weightedAverageShsOutDil': 15},
            {'symbol': 'AAPL', 'date': '2022-12-31', 'filing_date': '2023-01-15', 'netIncome': 50, 'weightedAverageShsOutDil': 15},
            {'symbol': 'AAPL', 'date': '2023-03-31', 'filing_date': '2023-04-15', 'netIncome': 22, 'weightedAverageShsOutDil': 15},
            {'symbol': 'AAPL', 'date': '2023-06-30', 'filing_date': '2023-07-15', 'netIncome': 20, 'weightedAverageShsOutDil': 15},
            {'symbol': 'AAPL', 'date': '2023-09-30', 'filing_date': '2023-10-15', 'netIncome': 25, 'weightedAverageShsOutDil': 15},
            {'symbol': 'AAPL', 'date': '2023-12-31', 'filing_date': '2024-01-15', 'netIncome': 30, 'weightedAverageShsOutDil': 15},
            
            # MSFT data
            {'symbol': 'MSFT', 'date': '2022-09-30', 'filing_date': '2022-10-20', 'netIncome': 17, 'weightedAverageShsOutDil': 10},
            {'symbol': 'MSFT', 'date': '2022-12-31', 'filing_date': '2023-01-20', 'netIncome': 19, 'weightedAverageShsOutDil': 10},
            {'symbol': 'MSFT', 'date': '2023-03-31', 'filing_date': '2023-04-20', 'netIncome': 15, 'weightedAverageShsOutDil': 10},
            {'symbol': 'MSFT', 'date': '2023-06-30', 'filing_date': '2023-07-20', 'netIncome': 16, 'weightedAverageShsOutDil': 10},
            {'symbol': 'MSFT', 'date': '2023-09-30', 'filing_date': '2023-10-20', 'netIncome': 18, 'weightedAverageShsOutDil': 10},
            {'symbol': 'MSFT', 'date': '2023-12-31', 'filing_date': '2024-01-20', 'netIncome': 20, 'weightedAverageShsOutDil': 10}
        ])
        
        # Cash flow statements
        cashflow_data = pd.DataFrame([
            # AAPL data - ordered by filing_date (newest to oldest)
            {'symbol': 'AAPL', 'date': '2023-12-31', 'filing_date': '2024-01-15', 'freeCashFlow': 40, 'depreciationAndAmortization': 10, 'capitalExpenditure': -8},
            {'symbol': 'AAPL', 'date': '2023-09-30', 'filing_date': '2023-10-15', 'freeCashFlow': 35, 'depreciationAndAmortization': 9, 'capitalExpenditure': -7},
            {'symbol': 'AAPL', 'date': '2023-06-30', 'filing_date': '2023-07-15', 'freeCashFlow': 30, 'depreciationAndAmortization': 8, 'capitalExpenditure': -6},
            {'symbol': 'AAPL', 'date': '2023-03-31', 'filing_date': '2023-04-15', 'freeCashFlow': 32, 'depreciationAndAmortization': 10, 'capitalExpenditure': -5},
            {'symbol': 'AAPL', 'date': '2022-12-31', 'filing_date': '2023-01-15', 'freeCashFlow': 60, 'depreciationAndAmortization': 12, 'capitalExpenditure': -4},
            {'symbol': 'AAPL', 'date': '2022-09-30', 'filing_date': '2022-10-15', 'freeCashFlow': 55, 'depreciationAndAmortization': 11, 'capitalExpenditure': -3},
            
            # MSFT data - ordered by filing_date (newest to oldest)
            {'symbol': 'MSFT', 'date': '2023-12-31', 'filing_date': '2024-01-20', 'freeCashFlow': 25, 'depreciationAndAmortization': 4, 'capitalExpenditure': -6},
            {'symbol': 'MSFT', 'date': '2023-09-30', 'filing_date': '2023-10-20', 'freeCashFlow': 22, 'depreciationAndAmortization': 5, 'capitalExpenditure': -5},
            {'symbol': 'MSFT', 'date': '2023-06-30', 'filing_date': '2023-07-20', 'freeCashFlow': 20, 'depreciationAndAmortization': 6, 'capitalExpenditure': -4},
            {'symbol': 'MSFT', 'date': '2023-03-31', 'filing_date': '2023-04-20', 'freeCashFlow': 18, 'depreciationAndAmortization': 7, 'capitalExpenditure': -3},
            {'symbol': 'MSFT', 'date': '2022-12-31', 'filing_date': '2023-01-20', 'freeCashFlow': 24, 'depreciationAndAmortization': 8, 'capitalExpenditure': -2},
            {'symbol': 'MSFT', 'date': '2022-09-30', 'filing_date': '2022-10-20', 'freeCashFlow': 21, 'depreciationAndAmortization': 9, 'capitalExpenditure': -1}
        ])
        
        # Convert dates to datetime and set index
        income_data['date'] = pd.to_datetime(income_data['date'])
        income_data['filing_date'] = pd.to_datetime(income_data['filing_date'])
        cashflow_data['date'] = pd.to_datetime(cashflow_data['date'])
        cashflow_data['filing_date'] = pd.to_datetime(cashflow_data['filing_date'])
        
        income_data = income_data.set_index('symbol')
        cashflow_data = cashflow_data.set_index('symbol')
        
        # Store the financial statements in the dataset
        dataset.financial_statements = {
            INCOME_STATEMENT: income_data,
            CASHFLOW_STATEMENT: cashflow_data
        }
        
    # Return the dataset and the mock API
    yield dataset, mock_api


def test_compute_pe(mock_dataset_with_statements):
    """Test the _compute_pe method."""
    dataset, _ = mock_dataset_with_statements
    
    # Create mock price data with dates to test different scenarios
    price_data = pd.DataFrame([
        # This date should have exactly 2 valid quarters (2022-09-30 and 2022-12-31)
        # But that's less than 4, so it should be skipped
        {'symbol': 'AAPL', 'date': '2023-01-20', 'close_price': 190},
        
        # This date should have exactly 3 valid quarters (2023-03-31, 2023-06-30, 2023-09-30)
        # Dec 31 data is too new (filing date 2024-01-15 is after 2024-01-05)
        # But that's less than 4, so it should be skipped
        {'symbol': 'AAPL', 'date': '2024-01-05', 'close_price': 200},
        
        # This date will have all 4 quarters from 2023
        {'symbol': 'AAPL', 'date': '2024-01-20', 'close_price': 210},

        # This date should have exactly 4 valid quarters for AAPL
        # (2022-09-30, 2022-12-31, 2023-03-31, 2023-06-30)
        {'symbol': 'AAPL', 'date': '2023-07-20', 'close_price': 195},        
        
        # This date should have exactly 3 valid quarters for MSFT
        # But that's less than 4, so it should be skipped
        {'symbol': 'MSFT', 'date': '2024-01-05', 'close_price': 400},
        
        # This date will have all 4 quarters from 2023
        {'symbol': 'MSFT', 'date': '2024-01-25', 'close_price': 420},
    ])
    
    price_data['date'] = pd.to_datetime(price_data['date'])
    price_data = price_data.set_index('symbol')
    
    # Call the method
    result = dataset._compute_pe(price_data)
    
    # Verify the result has the correct structure
    assert not result.empty
    assert set(result.columns) == {'symbol', 'date', 'pe'}
    
    # Convert to a more easily queryable format for testing
    result_df = result.set_index(['symbol', 'date'])
    
    # Check that dates with insufficient data (less than 4 quarters) are not in the results
    assert ('AAPL', pd.Timestamp('2023-01-20')) not in result_df.index, "Date with insufficient quarters should be skipped"
    
    # For AAPL on Jan 5, 2024 - should use 4 quarters
    # (2022-12-31, 2023-03-31, 2023-06-30, 2023-09-30)
    # Total netIncome = 50 + 22 + 20 + 25 = 117
    # Shares = 15
    aapl_jan05_2024_pe = result_df.loc[('AAPL', pd.Timestamp('2024-01-05')), 'pe']
    expected_pe = 200 / (117 / 15)
    assert aapl_jan05_2024_pe == pytest.approx(expected_pe, rel=1e-6)
    
    # For AAPL on Jan 20, 2024 - should use all 4 quarters from 2023
    # Total netIncome = 30 + 25 + 20 + 22 = 97
    # Shares = 15
    aapl_jan20_2024_pe = result_df.loc[('AAPL', pd.Timestamp('2024-01-20')), 'pe']
    expected_pe = 210 / (97 / 15)
    assert aapl_jan20_2024_pe == pytest.approx(expected_pe, rel=1e-6)

    # For AAPL on Jul 20, 2023 - should use 4 quarters
    # (2022-09-30, 2022-12-31, 2023-03-31, 2023-06-30)
    # Total netIncome = 48 + 50 + 22 + 20 = 140
    # Shares = 15
    # EPS = 140 / 15 = 9.33
    # PE = 195 / 9.33 = 20.9
    aapl_jul20_2023_pe = result_df.loc[('AAPL', pd.Timestamp('2023-07-20')), 'pe']
    expected_pe = 195 / (140 / 15)
    assert aapl_jul20_2023_pe == pytest.approx(expected_pe, rel=1e-6)    
    
    # For MSFT on Jan 5, 2024 - should use 4 quarters
    # (2022-12-31, 2023-03-31, 2023-06-30, 2023-09-30)
    # Total netIncome = 19 + 15 + 16 + 18 = 68
    # Shares = 10
    # EPS = 68 / 10 = 6.8
    # PE = 400 / 6.8 = 58.82
    msft_jan05_2024_pe = result_df.loc[('MSFT', pd.Timestamp('2024-01-05')), 'pe']
    expected_pe = 400 / (68 / 10)
    assert msft_jan05_2024_pe == pytest.approx(expected_pe, rel=1e-6)
    
    # For MSFT on Jan 25, 2024 - should use all 4 quarters from 2023
    # Total netIncome = 20 + 18 + 16 + 15 = 69
    # Shares = 10
    # EPS = 69 / 10 = 6.9
    # PE = 420 / 6.9 = 60.87
    msft_jan25_2024_pe = result_df.loc[('MSFT', pd.Timestamp('2024-01-25')), 'pe']
    expected_pe = 420 / (69 / 10)
    assert msft_jan25_2024_pe == pytest.approx(expected_pe, rel=1e-6)



def test_compute_price_to_fcf(mock_dataset_with_statements):
    """Test the _compute_price_to_fcf method."""
    dataset, _ = mock_dataset_with_statements
    
    # Create mock price data with dates to test different scenarios
    price_data = pd.DataFrame([
        # This date should have exactly 2 valid quarters (2022-09-30 and 2022-12-31)
        # But that's less than 4, so it should be skipped
        {'symbol': 'AAPL', 'date': '2023-01-20', 'close_price': 190},
        
        # This date should have exactly 3 valid quarters (2023-03-31, 2023-06-30, 2023-09-30)
        # Dec 31 data is too new (filing date 2024-01-15 is after 2024-01-05)
        # But that's less than 4, so it should be skipped
        {'symbol': 'AAPL', 'date': '2024-01-05', 'close_price': 200},
        
        # This date will have all 4 quarters from 2023
        {'symbol': 'AAPL', 'date': '2024-01-20', 'close_price': 210},

        # This date should have exactly 4 valid quarters for AAPL
        # (2022-09-30, 2022-12-31, 2023-03-31, 2023-06-30)
        {'symbol': 'AAPL', 'date': '2023-07-20', 'close_price': 195},        
        
        # This date should have exactly 3 valid quarters for MSFT
        # But that's less than 4, so it should be skipped
        {'symbol': 'MSFT', 'date': '2024-01-05', 'close_price': 400},
        
        # This date will have all 4 quarters from 2023
        {'symbol': 'MSFT', 'date': '2024-01-25', 'close_price': 420},
    ])
    
    price_data['date'] = pd.to_datetime(price_data['date'])
    price_data = price_data.set_index('symbol')
    
    # Call the method
    result = dataset._compute_price_to_fcf(price_data)
    
    # Verify the result has the correct structure
    assert not result.empty
    assert set(result.columns) == {'symbol', 'date', 'price_to_fcf'}
    
    # Convert to a more easily queryable format for testing
    result_df = result.set_index(['symbol', 'date'])
    
    # Check that dates with insufficient data (less than 4 quarters) are not in the results
    assert ('AAPL', pd.Timestamp('2023-01-20')) not in result_df.index, "Date with insufficient quarters should be skipped"
    
    # For AAPL on Jan 5, 2024 - should use 4 quarters
    # (2022-12-31, 2023-03-31, 2023-06-30, 2023-09-30)
    # Total FCF = 60 + 32 + 30 + 35 = 157
    # Shares = 15
    # FCF per share = 157 / 15 = 10.47
    # Price to FCF = 200 / 10.47 = 19.10
    aapl_jan05_2024_ptf = result_df.loc[('AAPL', pd.Timestamp('2024-01-05')), 'price_to_fcf']
    expected_ptf = 200 / (157 / 15)
    assert aapl_jan05_2024_ptf == pytest.approx(expected_ptf, rel=1e-6)
    
    # For AAPL on Jan 20, 2024 - should use all 4 quarters from 2023
    # Total FCF = 40 + 35 + 30 + 32 = 137
    # Shares = 15
    # FCF per share = 137 / 15 = 9.13
    # Price to FCF = 210 / 9.13 = 23.00
    aapl_jan20_2024_ptf = result_df.loc[('AAPL', pd.Timestamp('2024-01-20')), 'price_to_fcf']
    expected_ptf = 210 / (137 / 15)
    assert aapl_jan20_2024_ptf == pytest.approx(expected_ptf, rel=1e-6)

    # For AAPL on Jul 20, 2023 - should use 4 quarters
    # (2022-09-30, 2022-12-31, 2023-03-31, 2023-06-30)
    # Total FCF = 55 + 60 + 32 + 30 = 177
    # Shares = 15
    # FCF per share = 177 / 15 = 11.8
    # Price to FCF = 195 / 11.8 = 16.53
    aapl_jul20_2023_ptf = result_df.loc[('AAPL', pd.Timestamp('2023-07-20')), 'price_to_fcf']
    expected_ptf = 195 / (177 / 15)
    assert aapl_jul20_2023_ptf == pytest.approx(expected_ptf, rel=1e-6)    
    
    # For MSFT on Jan 5, 2024 - should use 4 quarters
    # (2022-12-31, 2023-03-31, 2023-06-30, 2023-09-30)
    # Total FCF = 24 + 18 + 20 + 22 = 84
    # Shares = 10
    # FCF per share = 84 / 10 = 8.4
    # Price to FCF = 400 / 8.4 = 47.62
    msft_jan05_2024_ptf = result_df.loc[('MSFT', pd.Timestamp('2024-01-05')), 'price_to_fcf']
    expected_ptf = 400 / (84 / 10)
    assert msft_jan05_2024_ptf == pytest.approx(expected_ptf, rel=1e-6)
    
    # For MSFT on Jan 25, 2024 - should use all 4 quarters from 2023
    # Total FCF = 25 + 22 + 20 + 18 = 85
    # Shares = 10
    # FCF per share = 85 / 10 = 8.5
    # Price to FCF = 420 / 8.5 = 49.41
    msft_jan25_2024_ptf = result_df.loc[('MSFT', pd.Timestamp('2024-01-25')), 'price_to_fcf']
    expected_ptf = 420 / (85 / 10)
    assert msft_jan25_2024_ptf == pytest.approx(expected_ptf, rel=1e-6)



def test_compute_price_to_owner_earning(mock_dataset_with_statements):
    """Test the _compute_price_to_owner_earning method."""
    dataset, _ = mock_dataset_with_statements
    
    # Create mock price data with dates to test different scenarios
    price_data = pd.DataFrame([
        # This date should have exactly 2 valid quarters (2022-09-30 and 2022-12-31)
        # But that's less than 4, so it should be skipped
        {'symbol': 'AAPL', 'date': '2023-01-20', 'close_price': 190},
        
        {'symbol': 'AAPL', 'date': '2024-01-05', 'close_price': 200},
        
        # This date will have all 4 quarters from 2023
        {'symbol': 'AAPL', 'date': '2024-01-20', 'close_price': 210},

        # This date should have exactly 4 valid quarters for AAPL
        # (2022-09-30, 2022-12-31, 2023-03-31, 2023-06-30)
        {'symbol': 'AAPL', 'date': '2023-07-20', 'close_price': 195},        
        
        # This date should have exactly 3 valid quarters for MSFT
        # But that's less than 4, so it should be skipped
        {'symbol': 'MSFT', 'date': '2024-01-05', 'close_price': 400},
        
        # This date will have all 4 quarters from 2023
        {'symbol': 'MSFT', 'date': '2024-01-25', 'close_price': 420},
    ])
    
    price_data['date'] = pd.to_datetime(price_data['date'])
    price_data = price_data.set_index('symbol')
    
    # Call the method
    result = dataset._compute_price_to_owner_earning(price_data)
    
    # Verify the result has the correct structure
    assert not result.empty
    assert set(result.columns) == {'symbol', 'date', 'price_to_owner_earning'}
    
    # Convert to a more easily queryable format for testing
    result_df = result.set_index(['symbol', 'date'])
    
    # Check that dates with insufficient data (less than 4 quarters) are not in the results
    assert ('AAPL', pd.Timestamp('2023-01-20')) not in result_df.index, "Date with insufficient quarters should be skipped"
    
    # Verify AAPL on 2024-01-05 has the correct calculation
    # Net Income: 50 + 22 + 20 + 25 = 117
    # D&A: 12 + 10 + 8 + 9 = 39
    # CapEx: 4 + 5 + 6 + 7 = 22
    # Owner's Earnings = 117 + 39 - 22 = 134
    # Shares = 15
    # OE per share = 134 / 15 = 8.93
    # Price to OE = 200 / 8.93 = 22.39
    aapl_jan05_2024_ptoe = result_df.loc[('AAPL', pd.Timestamp('2024-01-05')), 'price_to_owner_earning']
    expected_ptoe = 200 / ((117 + 39 - 22) / 15)
    assert aapl_jan05_2024_ptoe == pytest.approx(expected_ptoe, rel=1e-6)

    # For AAPL on Jan 20, 2024 - should use all 4 quarters from 2023
    # Net Income: 22 + 20 + 25 + 30 = 97
    # D&A: 10 + 8 + 9 + 10 = 37
    # CapEx: 5 + 6 + 7 + 8 = 26
    # Owner's Earnings = 97 + 37 - 26 = 108
    # Shares = 15
    # OE per share = 108 / 15 = 7.2
    # Price to OE = 210 / 7.2 = 29.1667
    aapl_jan20_2024_ptoe = result_df.loc[('AAPL', pd.Timestamp('2024-01-20')), 'price_to_owner_earning']
    expected_ptoe = 210 / ((97 + 37 - 26) / 15)
    assert aapl_jan20_2024_ptoe == pytest.approx(expected_ptoe, rel=1e-6)        
    
    # MSFT on 2024-01-05 should have 4 valid quarters:
    # (2022-12-31, 2023-03-31, 2023-06-30, 2023-09-30)
    # The 2023-12-31 quarter is excluded because filing date (2024-01-20) is after test date (2024-01-05)
    assert ('MSFT', pd.Timestamp('2024-01-05')) in result_df.index, "Date with 4 valid quarters should be included"
    
    # Verify MSFT on 2024-01-05 has the correct calculation
    # Net Income: 19 + 15 + 16 + 18 = 68
    # D&A: 8 + 7 + 6 + 5 = 26
    # CapEx: 2 + 3 + 4 + 5 = 14
    # Owner's Earnings = 68 + 26 - 14 = 80
    # Shares = 10
    # OE per share = 80 / 10 = 8
    # Price to OE = 400 / 8 = 50
    msft_jan05_2024_ptoe = result_df.loc[('MSFT', pd.Timestamp('2024-01-05')), 'price_to_owner_earning']
    expected_ptoe = 400 / ((68 + 26 - 14) / 10)
    assert msft_jan05_2024_ptoe == pytest.approx(expected_ptoe, rel=1e-6)
    
    # For AAPL on Jul 20, 2023 - should use 4 valid quarters
    # (2022-09-30, 2022-12-31, 2023-03-31, 2023-06-30)
    # Net Income: 48 + 50 + 22 + 20 = 140
    # D&A: 11 + 12 + 10 + 8 = 41
    # CapEx: 3 + 4 + 5 + 6 = 18
    # Owner's Earnings = 140 + 41 - 18 = 163
    # Shares = 15
    # OE per share = 163 / 15 = 10.87
    # Price to OE = 195 / 10.87 = 17.94
    aapl_jul20_2023_ptoe = result_df.loc[('AAPL', pd.Timestamp('2023-07-20')), 'price_to_owner_earning']
    expected_ptoe = 195 / ((140 + 41 - 18) / 15)
    assert aapl_jul20_2023_ptoe == pytest.approx(expected_ptoe, rel=1e-6)
    
    # For MSFT on Jan 25, 2024 - should use all 4 quarters from 2023
    # Net Income: 15 + 16 + 18 + 20 = 69
    # D&A: 7 + 6 + 5 + 4 = 22
    # CapEx: 3 + 4 + 5 + 6 = 18
    # Owner's Earnings = 69 + 22 - 18 = 73
    # Shares = 10
    # OE per share = 73 / 10 = 7.3
    # Price to OE = 420 / 7.3 = 57.53
    msft_jan25_2024_ptoe = result_df.loc[('MSFT', pd.Timestamp('2024-01-25')), 'price_to_owner_earning']
    expected_ptoe = 420 / ((69 + 22 - 18) / 10)
    assert msft_jan25_2024_ptoe == pytest.approx(expected_ptoe, rel=1e-6)


def test_compute_price_to_fcf(mock_dataset_with_statements):
    """Test the _compute_price_to_fcf method."""
    dataset, _ = mock_dataset_with_statements
    
    # Create mock price data with dates to test different scenarios
    price_data = pd.DataFrame([
        # This date should have exactly 2 valid quarters (2022-09-30 and 2022-12-31)
        # But that's less than 4, so it should be skipped
        {'symbol': 'AAPL', 'date': '2023-01-20', 'close_price': 190},
        
        # This date should have exactly 3 valid quarters (2023-03-31, 2023-06-30, 2023-09-30)
        # Dec 31 data is too new (filing date 2024-01-15 is after 2024-01-05)
        # But that's less than 4, so it should be skipped
        {'symbol': 'AAPL', 'date': '2024-01-05', 'close_price': 200},
        
        # This date will have all 4 quarters from 2023
        {'symbol': 'AAPL', 'date': '2024-01-20', 'close_price': 210},

        # This date should have exactly 4 valid quarters for AAPL
        # (2022-09-30, 2022-12-31, 2023-03-31, 2023-06-30)
        {'symbol': 'AAPL', 'date': '2023-07-20', 'close_price': 195},        
        
        # This date should have exactly 3 valid quarters for MSFT
        # But that's less than 4, so it should be skipped
        {'symbol': 'MSFT', 'date': '2024-01-05', 'close_price': 400},
        
        # This date will have all 4 quarters from 2023
        {'symbol': 'MSFT', 'date': '2024-01-25', 'close_price': 420},
    ])
    
    price_data['date'] = pd.to_datetime(price_data['date'])
    price_data = price_data.set_index('symbol')
    
    # Call the method
    result = dataset._compute_price_to_fcf(price_data)
    
    # Verify the result has the correct structure
    assert not result.empty
    assert set(result.columns) == {'symbol', 'date', 'price_to_fcf'}
    
    # Convert to a more easily queryable format for testing
    result_df = result.set_index(['symbol', 'date'])
    
    # Check that dates with insufficient data (less than 4 quarters) are not in the results
    assert ('AAPL', pd.Timestamp('2023-01-20')) not in result_df.index, "Date with insufficient quarters should be skipped"
    
    # For AAPL on Jan 5, 2024 - should use 4 quarters
    # (2022-12-31, 2023-03-31, 2023-06-30, 2023-09-30)
    # Total FCF = 60 + 32 + 30 + 35 = 157
    # Shares = 15
    # FCF per share = 157 / 15 = 10.47
    # Price to FCF = 200 / 10.47 = 19.10
    aapl_jan05_2024_ptf = result_df.loc[('AAPL', pd.Timestamp('2024-01-05')), 'price_to_fcf']
    expected_ptf = 200 / (157 / 15)
    assert aapl_jan05_2024_ptf == pytest.approx(expected_ptf, rel=1e-6)
    
    # For AAPL on Jan 20, 2024 - should use all 4 quarters from 2023
    # Total FCF = 40 + 35 + 30 + 32 = 137
    # Shares = 15
    # FCF per share = 137 / 15 = 9.13
    # Price to FCF = 210 / 9.13 = 23.00
    aapl_jan20_2024_ptf = result_df.loc[('AAPL', pd.Timestamp('2024-01-20')), 'price_to_fcf']
    expected_ptf = 210 / (137 / 15)
    assert aapl_jan20_2024_ptf == pytest.approx(expected_ptf, rel=1e-6)

    # For AAPL on Jul 20, 2023 - should use 4 quarters
    # (2022-09-30, 2022-12-31, 2023-03-31, 2023-06-30)
    # Total FCF = 55 + 60 + 32 + 30 = 177
    # Shares = 15
    # FCF per share = 177 / 15 = 11.8
    # Price to FCF = 195 / 11.8 = 16.53
    aapl_jul20_2023_ptf = result_df.loc[('AAPL', pd.Timestamp('2023-07-20')), 'price_to_fcf']
    expected_ptf = 195 / (177 / 15)
    assert aapl_jul20_2023_ptf == pytest.approx(expected_ptf, rel=1e-6)    
    
    # For MSFT on Jan 5, 2024 - should use 4 quarters
    # (2022-12-31, 2023-03-31, 2023-06-30, 2023-09-30)
    # Total FCF = 24 + 18 + 20 + 22 = 84
    # Shares = 10
    # FCF per share = 84 / 10 = 8.4
    # Price to FCF = 400 / 8.4 = 47.62
    msft_jan05_2024_ptf = result_df.loc[('MSFT', pd.Timestamp('2024-01-05')), 'price_to_fcf']
    expected_ptf = 400 / (84 / 10)
    assert msft_jan05_2024_ptf == pytest.approx(expected_ptf, rel=1e-6)
    
    # For MSFT on Jan 25, 2024 - should use all 4 quarters from 2023
    # Total FCF = 25 + 22 + 20 + 18 = 85
    # Shares = 10
    # FCF per share = 85 / 10 = 8.5
    # Price to FCF = 420 / 8.5 = 49.41
    msft_jan25_2024_ptf = result_df.loc[('MSFT', pd.Timestamp('2024-01-25')), 'price_to_fcf']
    expected_ptf = 420 / (85 / 10)
    assert msft_jan25_2024_ptf == pytest.approx(expected_ptf, rel=1e-6)


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
