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
    dataset, mock_api = mock_dataset
    
    # Setup mock price data
    mock_prices_aapl = [
        {'date': '2024-01-01', 'adjClose': 190.0, 'symbol': 'AAPL'},
        {'date': '2024-01-02', 'adjClose': 192.0, 'symbol': 'AAPL'}
    ]
    
    mock_prices_msft = [
        {'date': '2024-01-01', 'adjClose': 380.0, 'symbol': 'MSFT'},
        {'date': '2024-01-02', 'adjClose': 385.0, 'symbol': 'MSFT'}
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
    assert len(result) == 4  # 2 dates × 2 symbols
    
    # Verify the structure of the result
    assert 'symbol' in result.index.names
    assert 'close_price' in result.columns
    
    # Verify that the adjClose values are correctly mapped to close_price
    aapl_data = result.xs('AAPL')
    msft_data = result.xs('MSFT')
    
    # Verify specific values to ensure correct data transformation
    # Since date is no longer in the index, we need to filter by the date column
    assert aapl_data[aapl_data['date'] == pd.Timestamp('2024-01-01')]['close_price'].values[0] == 190.0
    assert aapl_data[aapl_data['date'] == pd.Timestamp('2024-01-02')]['close_price'].values[0] == 192.0
    assert msft_data[msft_data['date'] == pd.Timestamp('2024-01-01')]['close_price'].values[0] == 380.0
    assert msft_data[msft_data['date'] == pd.Timestamp('2024-01-02')]['close_price'].values[0] == 385.0


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
            metrics=['pe', 'price_to_fcf', 'close_price'],
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
            {'symbol': 'AAPL', 'date': '2023-12-31', 'filing_date': '2024-01-15', 'freeCashFlow': 40},
            {'symbol': 'AAPL', 'date': '2023-09-30', 'filing_date': '2023-10-15', 'freeCashFlow': 35},
            {'symbol': 'AAPL', 'date': '2023-06-30', 'filing_date': '2023-07-15', 'freeCashFlow': 30},
            {'symbol': 'AAPL', 'date': '2023-03-31', 'filing_date': '2023-04-15', 'freeCashFlow': 32},
            {'symbol': 'AAPL', 'date': '2022-12-31', 'filing_date': '2023-01-15', 'freeCashFlow': 60},
            {'symbol': 'AAPL', 'date': '2022-09-30', 'filing_date': '2022-10-15', 'freeCashFlow': 55},
            
            # MSFT data - ordered by filing_date (newest to oldest)
            {'symbol': 'MSFT', 'date': '2023-12-31', 'filing_date': '2024-01-20', 'freeCashFlow': 25},
            {'symbol': 'MSFT', 'date': '2023-09-30', 'filing_date': '2023-10-20', 'freeCashFlow': 22},
            {'symbol': 'MSFT', 'date': '2023-06-30', 'filing_date': '2023-07-20', 'freeCashFlow': 20},
            {'symbol': 'MSFT', 'date': '2023-03-31', 'filing_date': '2023-04-20', 'freeCashFlow': 18},
            {'symbol': 'MSFT', 'date': '2022-12-31', 'filing_date': '2023-01-20', 'freeCashFlow': 24},
            {'symbol': 'MSFT', 'date': '2022-09-30', 'filing_date': '2022-10-20', 'freeCashFlow': 21}
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



def test_build_with_derived_metrics(mock_dataset_with_statements):
    """Test the build method with derived metrics (PE and price-to-FCF).
    
    This test focuses on the integration aspects of the build method:
    1. Every date in the date range is included in the result
    2. The returned dataframe has correct columns
    3. The return result has correct index structure
    4. The dataframes are joined/merged correctly
    """
    dataset, mock_api = mock_dataset_with_statements
    
    # Define test data
    test_date_range = pd.date_range(start='2023-01-01', end='2023-01-10')
    test_symbols = ['AAPL', 'MSFT']
    
    # Create mock price data with one entry for each symbol and date
    mock_prices = []
    for symbol in test_symbols:
        for date in test_date_range:
            mock_prices.append({
                'date': date.strftime('%Y-%m-%d'),
                'adjClose': 100.0,
                'symbol': symbol
            })
    
    # Setup mock return values
    mock_api.get_prices.return_value = mock_prices
    
    # Create mock dataframes that will be returned by the compute methods
    mock_pe_df = pd.DataFrame([
        {'symbol': 'AAPL', 'date': pd.Timestamp('2023-01-01'), 'pe': 15.0},
        {'symbol': 'AAPL', 'date': pd.Timestamp('2023-01-02'), 'pe': 16.0},
        {'symbol': 'MSFT', 'date': pd.Timestamp('2023-01-01'), 'pe': 25.0},
        {'symbol': 'MSFT', 'date': pd.Timestamp('2023-01-02'), 'pe': 26.0}
    ])
    
    mock_ptf_df = pd.DataFrame([
        {'symbol': 'AAPL', 'date': pd.Timestamp('2023-01-01'), 'price_to_fcf': 10.0},
        {'symbol': 'AAPL', 'date': pd.Timestamp('2023-01-02'), 'price_to_fcf': 11.0},
        {'symbol': 'MSFT', 'date': pd.Timestamp('2023-01-01'), 'price_to_fcf': 20.0},
        {'symbol': 'MSFT', 'date': pd.Timestamp('2023-01-02'), 'price_to_fcf': 21.0}
    ])
    
    # Create a mock price dataframe that would be created by _fetch_price_data
    mock_price_df = pd.DataFrame([
        {'symbol': 'AAPL', 'date': pd.Timestamp('2023-01-01'), 'close_price': 150.0},
        {'symbol': 'AAPL', 'date': pd.Timestamp('2023-01-02'), 'close_price': 155.0},
        {'symbol': 'MSFT', 'date': pd.Timestamp('2023-01-01'), 'close_price': 250.0},
        {'symbol': 'MSFT', 'date': pd.Timestamp('2023-01-02'), 'close_price': 255.0}
    ])
    
    # Create mock financial statements dataframe with proper structure
    mock_income_df = pd.DataFrame([
        {'symbol': 'AAPL', 'date': pd.Timestamp('2022-12-01'), 'filing_date': pd.Timestamp('2022-12-15'), 'netIncome': 100.0, 'weightedAverageShsOutDil': 15.0},
        {'symbol': 'MSFT', 'date': pd.Timestamp('2022-12-01'), 'filing_date': pd.Timestamp('2022-12-15'), 'netIncome': 200.0, 'weightedAverageShsOutDil': 10.0}
    ]).set_index('symbol')
    
    mock_cashflow_df = pd.DataFrame([
        {'symbol': 'AAPL', 'date': pd.Timestamp('2022-12-01'), 'filing_date': pd.Timestamp('2022-12-15'), 'freeCashFlow': 120.0},
        {'symbol': 'MSFT', 'date': pd.Timestamp('2022-12-01'), 'filing_date': pd.Timestamp('2022-12-15'), 'freeCashFlow': 220.0}
    ]).set_index('symbol')
    
    # Create mock financial metrics dataframe that would be returned by _get_latest_values_for_dates
    mock_income_latest_df = pd.DataFrame([
        {'symbol': 'AAPL', 'date': pd.Timestamp('2023-01-01'), 'netIncome': 100.0, 'weightedAverageShsOutDil': 15.0},
        {'symbol': 'AAPL', 'date': pd.Timestamp('2023-01-02'), 'netIncome': 105.0, 'weightedAverageShsOutDil': 15.0},
        {'symbol': 'MSFT', 'date': pd.Timestamp('2023-01-01'), 'netIncome': 200.0, 'weightedAverageShsOutDil': 10.0},
        {'symbol': 'MSFT', 'date': pd.Timestamp('2023-01-02'), 'netIncome': 210.0, 'weightedAverageShsOutDil': 10.0}
    ]).set_index(['symbol', 'date'])
    
    mock_cashflow_latest_df = pd.DataFrame([
        {'symbol': 'AAPL', 'date': pd.Timestamp('2023-01-01'), 'freeCashFlow': 120.0},
        {'symbol': 'AAPL', 'date': pd.Timestamp('2023-01-02'), 'freeCashFlow': 125.0},
        {'symbol': 'MSFT', 'date': pd.Timestamp('2023-01-01'), 'freeCashFlow': 220.0},
        {'symbol': 'MSFT', 'date': pd.Timestamp('2023-01-02'), 'freeCashFlow': 225.0}
    ]).set_index(['symbol', 'date'])
    
    # Mock dictionary to store financial statements
    mock_financial_statements = {
        'income_statement': mock_income_df,
        'cashflow_statement': mock_cashflow_df
    }
    
    # Function to return appropriate mock data based on statement type
    def mock_fetch_financial_statements(statement_type, required_metrics):
        return mock_financial_statements.get(statement_type, pd.DataFrame())
    
    # Function to return appropriate mock data based on df
    def mock_get_latest_values_for_dates(df, date_range):
        if 'netIncome' in df.columns:
            return mock_income_latest_df
        elif 'freeCashFlow' in df.columns:
            return mock_cashflow_latest_df
        return pd.DataFrame(index=pd.MultiIndex.from_product([test_symbols, date_range], names=['symbol', 'date']))
    
    # Mock all the component methods to isolate testing of the build method's integration logic
    with patch.object(dataset, '_fetch_price_data', return_value=mock_price_df), \
         patch.object(dataset, '_compute_pe', return_value=mock_pe_df), \
         patch.object(dataset, '_compute_price_to_fcf', return_value=mock_ptf_df), \
         patch.object(dataset, '_fetch_financial_statements', side_effect=mock_fetch_financial_statements), \
         patch.object(dataset, '_get_latest_values_for_dates', side_effect=mock_get_latest_values_for_dates):
        
        # Override date range for testing
        dataset.start_date = '2023-01-01'
        dataset.end_date = '2023-01-02'
        
        # Call the build method
        result = dataset.build()
        
        # 1. Verify all dates in the range are included
        expected_dates = pd.date_range(start='2023-01-01', end='2023-01-02')
        for symbol in test_symbols:
            for date in expected_dates:
                assert (symbol, date) in result.index, f"Missing expected date {date} for {symbol}"
        
        # 2. Verify the returned dataframe has correct columns
        expected_columns = ['close_price', 'pe', 'price_to_fcf']
        for col in expected_columns:
            assert col in result.columns, f"Missing expected column {col}"
        
        # 3. Verify the return result has correct index structure
        assert result.index.names == ['symbol', 'date'], "Index should be a MultiIndex with symbol and date"
        
        # 4. Verify the dataframes are joined/merged correctly
        # Check that values from each source dataframe are correctly merged
        for symbol in test_symbols:
            for date in expected_dates:
                row = result.loc[(symbol, date)]
                # Find corresponding rows in source dataframes
                price_row = mock_price_df[(mock_price_df['symbol'] == symbol) & (mock_price_df['date'] == date)]
                pe_row = mock_pe_df[(mock_pe_df['symbol'] == symbol) & (mock_pe_df['date'] == date)]
                ptf_row = mock_ptf_df[(mock_ptf_df['symbol'] == symbol) & (mock_ptf_df['date'] == date)]
                
                # Verify values match
                assert row['close_price'] == price_row['close_price'].values[0]
                assert row['pe'] == pe_row['pe'].values[0]
                assert row['price_to_fcf'] == ptf_row['price_to_fcf'].values[0]


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])
