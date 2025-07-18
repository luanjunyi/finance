# pytest: --cov=fmp_fetch --no-cov-on-fail
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from fmp_fetch.fmp_online import FMPOnline
from fmp_fetch.fmp_api import FMPAPI


@pytest.fixture
def mock_fmp_online():
    """Create a FMPOnline instance with mocked FMPAPI."""
    with patch('fmp_fetch.fmp_online.FMPAPI') as mock_fmp_api_class:
        # Setup mock FMPAPI instance
        mock_api = MagicMock()
        mock_fmp_api_class.return_value = mock_api
        
        # Initialize the FMPOnline instance
        fmp = FMPOnline()
        
        # Return both the instance and the mock for assertions
        yield fmp, mock_api


def test_initialization():
    """Test FMPOnline initialization creates an FMPAPI instance."""
    with patch('fmp_fetch.fmp_online.FMPAPI') as mock_fmp_api_class:
        fmp = FMPOnline()
        mock_fmp_api_class.assert_called_once()
        assert fmp.api is mock_fmp_api_class.return_value


def test_get_price(mock_fmp_online):
    """Test the get_price method."""
    fmp, mock_api = mock_fmp_online
    
    # Setup mock price data for a single day
    mock_price = {"date": "2024-01-01", "open": 100.0, "high": 105.0, "low": 99.0, "close": 102.0, "adjClose": 102.0, "volume": 1000000}
    mock_api.get_prices.return_value = [mock_price]
    
    # Call the method
    result = fmp.get_price('AAPL', '2024-01-01')
    
    # Verify the result
    assert result == mock_price
    


def test_get_price_not_found(mock_fmp_online):
    """Test the get_price method when price is not found."""
    fmp, mock_api = mock_fmp_online
    
    # Setup empty response
    mock_api.get_prices.return_value = []
    
    # Call the method and expect an exception
    with pytest.raises(ValueError, match="Can't find price for AAPL on 2024-01-01"):
        fmp.get_price('AAPL', '2024-01-01')


def test_get_pe_ratio(mock_fmp_online):
    """Test the get_pe_ratio method."""
    fmp, mock_api = mock_fmp_online
    
    # Setup mock responses
    mock_ratios = [
        {"date": "2023-12-31", "netIncomePerShare": 2.5},
        {"date": "2023-09-30", "netIncomePerShare": 2.3},
        {"date": "2023-06-30", "netIncomePerShare": 2.2},
        {"date": "2023-03-31", "netIncomePerShare": 2.1},
        {"date": "2022-12-31", "netIncomePerShare": 2.0},
        {"date": "2022-09-30", "netIncomePerShare": 1.0}
    ]
    mock_api.get_ratios.return_value = mock_ratios
    
    # Setup mock for get_close_price
    with patch.object(fmp, 'get_close_price', return_value=100.0):
        # Call the method
        result = fmp.get_pe_ratio('AAPL', '2024-01-01')
    
    # Expected PE ratio calculation
    expected_pe = 100.0 / (2.3 + 2.2 + 2.1 + 2.0)
    
    # Verify the result
    assert pytest.approx(result, 0.0001) == expected_pe
    
    # Verify the API call was made correctly
    mock_api.get_ratios.assert_called_once_with('AAPL', 'quarter', 120)


def test_get_pe_ratio_insufficient_data(mock_fmp_online):
    """Test the get_pe_ratio method with insufficient EPS data."""
    fmp, mock_api = mock_fmp_online
    
    # Setup mock response with fewer than 4 quarters of data
    mock_ratios = [
        {"date": "2023-12-31", "netIncomePerShare": 2.5},
        {"date": "2023-09-30", "netIncomePerShare": 2.3}
    ]
    mock_api.get_ratios.return_value = mock_ratios
    
    # Call the method and expect an exception
    with pytest.raises(ValueError, match="AAPL has only 1 EPS data points before 2024-01-01"):
        fmp.get_pe_ratio('AAPL', '2024-01-01')

def test_get_pe_ratio_data_too_old(mock_fmp_online):
    """Test the get_pe_ratio method with data too old."""
    fmp, mock_api = mock_fmp_online
    
    # Setup mock response with data too old
    mock_ratios = [
        {"date": "2022-12-31", "netIncomePerShare": 2.5},
        {"date": "2022-09-30", "netIncomePerShare": 2.3}
    ]
    mock_api.get_ratios.return_value = mock_ratios
    
    # Call the method and expect an exception
    with pytest.raises(ValueError, match="AAPL has only 0 EPS data points before 2025-01-01"):
        fmp.get_pe_ratio('AAPL', '2025-01-01')


def test_get_close_price(mock_fmp_online):
    """Test the get_close_price method."""
    fmp, mock_api = mock_fmp_online
    
    # Setup mock price data
    mock_price = {"date": "2024-01-01", "open": 100.0, "high": 105.0, "low": 99.0, "close": 102.0, "adjClose": 102.0, "volume": 1000000}
    
    # Mock the get_price method to return our mock price
    with patch.object(fmp, 'get_price', return_value=mock_price):
        # Call the method
        result = fmp.get_close_price('AAPL', '2024-01-01')
    
    # Verify the result
    assert result == 102.0


def test_get_close_prices_during(mock_fmp_online):
    """Test the get_close_prices_during method."""
    fmp, mock_api = mock_fmp_online
    
    # Setup mock price data for multiple symbols and dates
    mock_prices_aapl = [
        {"symbol": "AAPL", "date": "2024-01-03", "adjOpen": 100.0, "adjHigh": 105.0, "adjLow": 99.0, "adjClose": 102.0, "volume": 1000000},
        {"symbol": "AAPL", "date": "2024-01-02", "adjOpen": 98.0, "adjHigh": 103.0, "adjLow": 97.0, "adjClose": 101.0, "volume": 900000},
        {"symbol": "AAPL", "date": "2024-01-01", "adjOpen": 95.0, "adjHigh": 100.0, "adjLow": 94.0, "adjClose": 99.0, "volume": 800000}
    ]
    
    mock_prices_msft = [
        {"symbol": "MSFT", "date": "2024-01-03", "adjOpen": 200.0, "adjHigh": 205.0, "adjLow": 199.0, "adjClose": 202.0, "volume": 500000},
        {"symbol": "MSFT", "date": "2024-01-02", "adjOpen": 198.0, "adjHigh": 203.0, "adjLow": 197.0, "adjClose": 201.0, "volume": 450000},
        {"symbol": "MSFT", "date": "2024-01-01", "adjOpen": 195.0, "adjHigh": 200.0, "adjLow": 194.0, "adjClose": 199.0, "volume": 400000}
    ]
    
    # Configure mock to return different data for different symbols
    def mock_get_prices(symbol, start_date, end_date):
        if symbol == 'AAPL':
            return mock_prices_aapl
        elif symbol == 'MSFT':
            return mock_prices_msft
        return []
    
    mock_api.get_prices.side_effect = mock_get_prices
    
    # Call the method
    result = fmp.get_close_prices_during(['AAPL', 'MSFT'], '2024-01-01', '2024-01-03')
    
    # Verify the API calls were made correctly
    assert mock_api.get_prices.call_count == 2
    mock_api.get_prices.assert_any_call('AAPL', '2024-01-01', '2024-01-03')
    mock_api.get_prices.assert_any_call('MSFT', '2024-01-01', '2024-01-03')
    
    # Verify the result is a DataFrame with the expected structure
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'symbol', 'date', 'close_price'}
    
    # Verify the DataFrame contains the expected data
    assert len(result) == 6  # 3 days * 2 symbols
    
    # Check that the data is sorted by symbol and date
    assert result.iloc[0]['symbol'] == 'AAPL'
    assert result.iloc[0]['date'] == pd.Timestamp('2024-01-01')
    assert result.iloc[0]['close_price'] == 99.0
    
    assert result.iloc[3]['symbol'] == 'MSFT'
    assert result.iloc[3]['date'] == pd.Timestamp('2024-01-01')
    assert result.iloc[3]['close_price'] == 199.0


def test_realtime_batch_price_quote_small_batch(mock_fmp_online):
    """Test the realtime_batch_price_quote method with a small batch of symbols."""
    fmp, mock_api = mock_fmp_online
    
    # Setup mock price quote data
    mock_quotes = [
        {"symbol": "AAPL", "price": 200.0, "change": 1.5, "volume": 1000000},
        {"symbol": "MSFT", "price": 300.0, "change": 2.0, "volume": 500000}
    ]
    
    # Configure mock to return our mock data
    mock_api.batch_price_quote.return_value = mock_quotes
    
    # Call the method with a small batch that doesn't need splitting
    result = fmp.realtime_batch_price_quote(['AAPL', 'MSFT'])
    
    # Verify the API call was made correctly
    mock_api.batch_price_quote.assert_called_once_with(['AAPL', 'MSFT'])
    
    # Verify the result is a DataFrame with the expected structure
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'symbol', 'current_price'}
    
    # Verify the DataFrame contains the expected data
    assert len(result) == 2
    
    # Check the values
    assert result.loc[result['symbol'] == 'AAPL', 'current_price'].values[0] == 200.0
    assert result.loc[result['symbol'] == 'MSFT', 'current_price'].values[0] == 300.0


def test_realtime_batch_price_quote_large_batch(mock_fmp_online):
    """Test the realtime_batch_price_quote method with a large batch of symbols that needs splitting."""
    fmp, mock_api = mock_fmp_online
    
    # Create a large list of symbols that exceeds the batch size
    symbols = [f"SYMBOL{i}" for i in range(1, 1202)]  # 1201 symbols
    
    # Setup mock price quote data for first batch
    mock_quotes_batch1 = [{"symbol": f"SYMBOL{i}", "price": float(i), "change": 0.1, "volume": 1000} for i in range(1, 1001)]
    
    # Setup mock price quote data for second batch
    mock_quotes_batch2 = [{"symbol": f"SYMBOL{i}", "price": float(i), "change": 0.1, "volume": 1000} for i in range(1001, 1202)]
    
    # Configure mock to return different data for different batches
    mock_api.batch_price_quote.side_effect = [mock_quotes_batch1, mock_quotes_batch2]
    
    # Mock tqdm to return the original iterable
    with patch('fmp_fetch.fmp_online.tqdm', lambda x, **kwargs: x):
        result = fmp.realtime_batch_price_quote(symbols)
    
    # Verify the API calls were made correctly
    assert mock_api.batch_price_quote.call_count == 2
    mock_api.batch_price_quote.assert_any_call(symbols[:1000])
    mock_api.batch_price_quote.assert_any_call(symbols[1000:])
    
    # Verify the result is a DataFrame with the expected structure
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'symbol', 'current_price'}
    
    # Verify the DataFrame contains the expected data
    assert len(result) == 1201
    
    # Check some values
    assert result.loc[result['symbol'] == 'SYMBOL1', 'current_price'].values[0] == 1.0
    assert result.loc[result['symbol'] == 'SYMBOL1000', 'current_price'].values[0] == 1000.0
    assert result.loc[result['symbol'] == 'SYMBOL1201', 'current_price'].values[0] == 1201.0


def test_realtime_batch_price_quote_api_failure(mock_fmp_online):
    """Test the realtime_batch_price_quote method when the API call fails."""
    fmp, mock_api = mock_fmp_online
    
    # Configure mock to return None (API failure)
    mock_api.batch_price_quote.return_value = None
    
    # Call the method
    with patch('fmp_fetch.fmp_online.tqdm', lambda x, **kwargs: x):
        result = fmp.realtime_batch_price_quote(['AAPL', 'MSFT'])
    
    # Verify the API call was made
    mock_api.batch_price_quote.assert_called_once_with(['AAPL', 'MSFT'])
    
    # Verify the result is an empty DataFrame with the expected columns
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'symbol', 'current_price'}
    assert len(result) == 0


def test_realtime_batch_market_cap_small_batch(mock_fmp_online):
    """Test the realtime_batch_market_cap method with a small batch of symbols."""
    fmp, mock_api = mock_fmp_online
    
    # Setup mock market cap data
    mock_market_caps = [
        {"symbol": "AAPL", "marketCap": 2000000000000.0},
        {"symbol": "MSFT", "marketCap": 1800000000000.0}
    ]
    
    # Configure mock to return our mock data
    mock_api.batch_market_cap.return_value = mock_market_caps
    
    # Call the method with a small batch that doesn't need splitting
    result = fmp.realtime_batch_market_cap(['AAPL', 'MSFT'])
    
    # Verify the API call was made correctly
    mock_api.batch_market_cap.assert_called_once_with(['AAPL', 'MSFT'])
    
    # Verify the result is a DataFrame with the expected structure
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'symbol', 'market_cap'}
    
    # Verify the DataFrame contains the expected data
    assert len(result) == 2
    
    # Check the values
    assert result.loc[result['symbol'] == 'AAPL', 'market_cap'].values[0] == 2000000000000.0
    assert result.loc[result['symbol'] == 'MSFT', 'market_cap'].values[0] == 1800000000000.0


def test_realtime_batch_market_cap_large_batch(mock_fmp_online):
    """Test the realtime_batch_market_cap method with a large batch of symbols that needs splitting."""
    fmp, mock_api = mock_fmp_online
    
    # Create a large list of symbols that exceeds the batch size
    symbols = [f"SYMBOL{i}" for i in range(1, 1202)]  # 1201 symbols
    
    # Setup mock market cap data for first batch
    mock_market_caps_batch1 = [{"symbol": f"SYMBOL{i}", "marketCap": float(i) * 1000000000} for i in range(1, 1001)]
    
    # Setup mock market cap data for second batch
    mock_market_caps_batch2 = [{"symbol": f"SYMBOL{i}", "marketCap": float(i) * 1000000000} for i in range(1001, 1202)]
    
    # Configure mock to return different data for different batches
    mock_api.batch_market_cap.side_effect = [mock_market_caps_batch1, mock_market_caps_batch2]
    
    # Call the method
    with patch('fmp_fetch.fmp_online.tqdm', lambda x, **kwargs: x):
        result = fmp.realtime_batch_market_cap(symbols)
    
    # Verify the API calls were made correctly
    assert mock_api.batch_market_cap.call_count == 2
    mock_api.batch_market_cap.assert_any_call(symbols[:1000])
    mock_api.batch_market_cap.assert_any_call(symbols[1000:])
    
    # Verify the result is a DataFrame with the expected structure
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'symbol', 'market_cap'}
    
    # Verify the DataFrame contains the expected data
    assert len(result) == 1201
    
    # Check some values
    assert result.loc[result['symbol'] == 'SYMBOL1', 'market_cap'].values[0] == 1000000000.0
    assert result.loc[result['symbol'] == 'SYMBOL1000', 'market_cap'].values[0] == 1000000000000.0
    assert result.loc[result['symbol'] == 'SYMBOL1201', 'market_cap'].values[0] == 1201000000000.0


def test_realtime_batch_market_cap_api_failure(mock_fmp_online):
    """Test the realtime_batch_market_cap method when the API call fails."""
    fmp, mock_api = mock_fmp_online
    
    # Configure mock to return None (API failure)
    mock_api.batch_market_cap.return_value = None
    
    # Call the method
    with patch('fmp_fetch.fmp_online.tqdm', lambda x, **kwargs: x):
        result = fmp.realtime_batch_market_cap(['AAPL', 'MSFT'])
    
    # Verify the API call was made
    mock_api.batch_market_cap.assert_called_once_with(['AAPL', 'MSFT'])
    
    # Verify the result is an empty DataFrame with the expected columns
    assert isinstance(result, pd.DataFrame)
    assert set(result.columns) == {'symbol', 'market_cap'}
    assert len(result) == 0
