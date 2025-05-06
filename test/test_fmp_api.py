
import pytest
from unittest.mock import patch, MagicMock

from fmp_fetch.fmp_api import FMPAPI


@pytest.fixture
def mock_fmp_api():
    """Create a FMPAPI instance with mocked API responses."""
    with patch.dict('os.environ', {'FMP_API_KEY': 'dummy_key'}), \
         patch('fmp_fetch.fmp_api.requests.get') as mock_get:
        
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        # Initialize the FMPAPI instance
        fmp = FMPAPI()
        
        # Return both the instance and the mock for assertions
        yield fmp, mock_get


def test_initialization():
    """Test FMPAPI initialization with environment variable."""
    with patch.dict('os.environ', {'FMP_API_KEY': 'test_key'}):
        fmp = FMPAPI()
        assert fmp.api_key == 'test_key'
        assert fmp.base_url == "https://financialmodelingprep.com/stable"


def test_initialization_missing_api_key():
    """Test FMPAPI initialization with missing API key."""
    with patch.dict('os.environ', {'FMP_API_KEY': ''}, clear=True):
        with pytest.raises(ValueError, match="FMP_API_KEY environment variable not set"):
            FMPAPI()


def test_get_prices(mock_fmp_api):
    """Test the get_prices method."""
    fmp, mock_get = mock_fmp_api
    
    # Setup mock price data
    mock_prices = [
        {"date": "2024-01-01", "adjOpen": 100.0, "adjHigh": 105.0, "adjLow": 99.0, "adjClose": 102.0, "volume": 1000000},
        {"date": "2024-01-02", "adjOpen": 102.0, "adjHigh": 106.0, "adjLow": 101.0, "adjClose": 105.0, "volume": 1200000}
    ]
    mock_get.return_value.json.return_value = mock_prices
    
    # Call the method
    result = fmp.get_prices('AAPL', '2024-01-01', '2024-01-02')
    
    # Verify the result
    assert result == mock_prices
    
    # Verify the request was made correctly
    mock_get.assert_called_once()
    args, kwargs = mock_get.call_args
    assert args[0] == "https://financialmodelingprep.com/stable/historical-price-eod/dividend-adjusted"
    assert kwargs['params'] == {'symbol': 'AAPL', 'from': '2024-01-01', 'to': '2024-01-02', 'apikey': 'dummy_key'}


def test_get_ratios(mock_fmp_api):
    """Test the get_ratios method."""
    fmp, mock_get = mock_fmp_api
    
    # Setup mock ratios data
    mock_ratios = [
        {"date": "2023-12-31", "netIncomePerShare": 2.5, "peRatio": 25.0},
        {"date": "2023-09-30", "netIncomePerShare": 2.3, "peRatio": 24.0}
    ]
    mock_get.return_value.json.return_value = mock_ratios
    
    # Call the method with default parameters
    result = fmp.get_ratios('AAPL')
    
    # Verify the result
    assert result == mock_ratios
    
    # Verify the request was made correctly
    mock_get.assert_called_once()
    args, kwargs = mock_get.call_args
    assert args[0] == "https://financialmodelingprep.com/stable/ratios"
    assert kwargs['params'] == {'symbol': 'AAPL', 'period': 'quarter', 'limit': 120, 'apikey': 'dummy_key'}


def test_get_ratios_with_custom_params(mock_fmp_api):
    """Test the get_ratios method with custom parameters."""
    fmp, mock_get = mock_fmp_api
    
    # Setup mock ratios data
    mock_ratios = [
        {"date": "2023-12-31", "netIncomePerShare": 2.5, "peRatio": 25.0},
        {"date": "2022-12-31", "netIncomePerShare": 2.0, "peRatio": 20.0}
    ]
    mock_get.return_value.json.return_value = mock_ratios
    
    # Call the method with custom parameters
    result = fmp.get_ratios('AAPL', period='annual', limit=10)
    
    # Verify the result
    assert result == mock_ratios
    
    # Verify the request was made correctly
    mock_get.assert_called_once()
    args, kwargs = mock_get.call_args
    assert args[0] == "https://financialmodelingprep.com/stable/ratios"
    assert kwargs['params'] == {'symbol': 'AAPL', 'period': 'annual', 'limit': 10, 'apikey': 'dummy_key'}
