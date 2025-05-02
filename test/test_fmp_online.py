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
