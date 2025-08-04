import pytest
import sys
from unittest.mock import patch, MagicMock, ANY

from fmp_fetch.fmp_api import FMPAPI


# We'll use monkeypatch to set environment variables before importing modules


@pytest.fixture(autouse=True)
def mock_config(monkeypatch):
    """Mock the config module to avoid environment variable dependencies."""
    # Set environment variables for testing
    monkeypatch.setenv('FMP_DB_PATH', '/test/path/to/db.sqlite')
    monkeypatch.setenv('FMP_API_KEY', 'test_api_key')

    # If the modules are already imported, remove them to ensure fresh imports
    for module in ['utils.config', 'fmp_fetch.fmp_api']:
        if module in sys.modules:
            del sys.modules[module]

    # Now we can safely import
    import utils.config
    import fmp_fetch.fmp_api
    from fmp_fetch.fmp_api import FMPAPI

    # Make these available to the tests
    return {
        'config': utils.config,
        'fmp_api': fmp_fetch.fmp_api,
        'FMPAPI': FMPAPI
    }


@pytest.fixture
def mock_fmp_api(mock_config):
    """Create a FMPAPI instance with mocked API responses."""
    FMPAPI = mock_config['FMPAPI']

    with patch('fmp_fetch.fmp_api.requests.get') as mock_get:
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        # Initialize the FMPAPI instance
        fmp = FMPAPI()

        # Return both the instance and the mock for assertions
        yield fmp, mock_get


def test_initialization(mock_config):
    """Test FMPAPI initialization with config module."""
    FMPAPI = mock_config['FMPAPI']
    fmp = FMPAPI()
    assert fmp.api_key == 'test_api_key'
    assert fmp.base_url == "https://financialmodelingprep.com/stable"


def test_initialization_missing_api_key(monkeypatch):
    """Test FMPAPI initialization with missing API key."""
    # Clear the API_KEY but keep the DB_PATH
    monkeypatch.delenv('FMP_API_KEY', raising=False)

    # Clear the modules to force reimport
    for module in ['utils.config', 'fmp_fetch.fmp_api']:
        if module in sys.modules:
            del sys.modules[module]

    # Now try to import, which should fail
    with pytest.raises(ValueError, match="Mandatory environment variable FMP_API_KEY is not set"):
        import utils.config


@patch("fmp_fetch.fmp_api.requests.get")
def test_fetch_data(mock_get):
    # Setup mock response
    mock_response = mock_get.return_value
    fake_json = [{"key": "value"}]
    mock_response.json.return_value = fake_json
    mock_response.status_code = 200
    fetcher = FMPAPI()
    # Call the function
    result = fetcher.get_insider_trading()

    # Assertions
    url = 'https://financialmodelingprep.com/stable/insider-trading/search'
    params = {'transactionType': 'P-Purchase', 'page': 0, 'limit': 1000, 'apikey': ANY}
    mock_get.assert_called_once_with(url, params=params)
    assert result == fake_json


@patch("fmp_fetch.fmp_api.requests.get")
def test_fetch_data_with_symbol(mock_get):
    # Setup mock response
    mock_response = mock_get.return_value
    fake_json = [{"key": "value"}]
    mock_response.json.return_value = fake_json
    mock_response.status_code = 200
    fetcher = FMPAPI()
    # Call the function
    result = fetcher.get_insider_trading(symbol='BTCUSD')

    # Assertions
    url = 'https://financialmodelingprep.com/stable/insider-trading/search'
    params = {'transactionType': ANY, 'page': ANY, 'limit': ANY, 'apikey': ANY, 'symbol': 'BTCUSD'}
    mock_get.assert_called_once_with(url, params=params)
    assert result == fake_json


@patch("fmp_fetch.fmp_api.requests.get")
def test_fetch_data_with_page(mock_get):
    # Setup mock response
    mock_response = mock_get.return_value
    fake_json = [{"key": "value"}]
    mock_response.json.return_value = fake_json
    mock_response.status_code = 200
    fetcher = FMPAPI()
    # Call the function
    result = fetcher.get_insider_trading(page=17)

    # Assertions
    url = 'https://financialmodelingprep.com/stable/insider-trading/search'
    params = {'transactionType': ANY, 'page': 17, 'limit': ANY, 'apikey': ANY}
    mock_get.assert_called_once_with(url, params=params)
    assert result == fake_json


@patch("fmp_fetch.fmp_api.requests.get")
def test_fetch_data_with_limit(mock_get):
    # Setup mock response
    mock_response = mock_get.return_value
    fake_json = [{"key": "value"}]
    mock_response.json.return_value = fake_json
    mock_response.status_code = 200
    fetcher = FMPAPI()
    # Call the function
    result = fetcher.get_insider_trading(limit=17)

    # Assertions
    url = 'https://financialmodelingprep.com/stable/insider-trading/search'
    params = {'transactionType': ANY, 'page': ANY, 'limit': 17, 'apikey': ANY}
    mock_get.assert_called_once_with(url, params=params)
    assert result == fake_json


@patch("fmp_fetch.fmp_api.requests.get")
@pytest.mark.parametrize("limit", [-1, 1001])
def test_fetch_data_with_invalid_limit(mock_get, limit):
    # Setup mock response
    mock_response = mock_get.return_value
    fake_json = [{"key": "value"}]
    mock_response.json.return_value = fake_json
    mock_response.status_code = 200
    fetcher = FMPAPI()
    # Execute
    with pytest.raises(ValueError):
        fetcher.get_insider_trading(limit=limit)


@patch("fmp_fetch.fmp_api.requests.get")
@pytest.mark.parametrize("page", [-1, 101])
def test_fetch_data_with_invalid_page(mock_get, page):
    # Setup mock response
    mock_response = mock_get.return_value
    fake_json = [{"key": "value"}]
    mock_response.json.return_value = fake_json
    mock_response.status_code = 200
    fetcher = FMPAPI()
    # Execute
    with pytest.raises(ValueError):
        fetcher.get_insider_trading(page=page)


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
    params = kwargs['params']
    assert params['symbol'] == 'AAPL'
    assert params['from'] == '2024-01-01'
    assert params['to'] == '2024-01-02'
    assert params['apikey'] == 'test_api_key'


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
    params = kwargs['params']
    assert params['symbol'] == 'AAPL'
    assert params['period'] == 'quarter'
    assert params['limit'] == 120
    assert params['apikey'] == 'test_api_key'


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
    params = kwargs['params']
    assert params['symbol'] == 'AAPL'
    assert params['period'] == 'annual'
    assert params['limit'] == 10
    assert params['apikey'] == 'test_api_key'
