import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import sqlite3

from feature_gen.revenue_growth import RevenueGrowth


@pytest.fixture
def mock_revenue_growth():
    """Create a RevenueGrowth instance with mocked dependencies."""
    # Create a mock for the database connection
    with patch('sqlite3.connect') as mock_db_connect, \
         patch('feature_gen.revenue_growth.Dataset') as mock_dataset:
        
        # Setup mock connection
        mock_connection = MagicMock()
        mock_db_connect.return_value.__enter__.return_value = mock_connection
        
        # Initialize the RevenueGrowth instance
        revenue_growth = RevenueGrowth(db_path='mock_db_path')
        
        yield revenue_growth, mock_dataset


@pytest.fixture
def sample_tech_symbols():
    """Return a list of sample technology stock symbols."""
    return ['AAPL', 'MSFT', 'GOOGL']


@pytest.fixture
def sample_sectors():
    """Return a dictionary of sectors with their symbols."""
    return {
        'Technology': ['AAPL', 'MSFT', 'GOOGL'],
        'Energy': ['XOM', 'CVX'],
        'Healthcare': ['JNJ', 'PFE']
    }
