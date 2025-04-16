import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from datetime import datetime

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


def test_get_year_quarter(mock_revenue_growth):
    """Test the _get_year_quarter method."""
    revenue_growth, _ = mock_revenue_growth
    
    test_cases = [
        ('2024-01-15', '2024Q1'),
        ('2024-03-31', '2024Q1'),
        ('2024-04-01', '2024Q2'),
        ('2024-06-30', '2024Q2'),
        ('2024-07-01', '2024Q3'),
        ('2024-09-30', '2024Q3'),
        ('2024-10-01', '2024Q4'),
        ('2024-12-31', '2024Q4'),
    ]
    
    for date_str, expected in test_cases:
        result = revenue_growth._get_year_quarter(date_str)
        assert result == expected


def test_calculate_yoy_growth(mock_revenue_growth):
    """Test the calculation of YoY growth rates."""
    revenue_growth, mock_dataset_class = mock_revenue_growth
    
    # Mock the _get_valid_symbols method
    mock_symbols_df = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL'],
        'sector': ['Technology', 'Technology', 'Technology']
    })
    revenue_growth._get_valid_symbols = MagicMock(return_value=mock_symbols_df)
        
    # Create mock revenue data with enough quarters for YoY calculation
    # AAPL: 20% growth from 2023Q1 to 2024Q1, 25% growth from 2023Q2 to 2024Q2
    # MSFT: 15% growth from 2023Q1 to 2024Q1, 10% growth from 2023Q2 to 2024Q2
    # GOOGL: 5% growth from 2023Q1 to 2024Q1, -5% growth from 2023Q2 to 2024Q2
    mock_revenue_data = pd.DataFrame({
        'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL',
                  'MSFT', 'MSFT', 'MSFT', 'MSFT', 'MSFT', 'MSFT',
                  'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL'],
        'date': ['2022-03-31', '2022-06-30', '2022-09-30', '2023-03-31', '2023-06-30', '2024-03-31',
                '2022-03-31', '2022-06-30', '2022-09-30', '2023-03-31', '2023-06-30', '2024-03-31',
                '2022-03-31', '2022-06-30', '2022-09-30', '2023-03-31', '2023-06-30', '2024-03-31'],
        'revenue': [80, 80, 90, 100, 100, 120,
                   180, 180, 190, 200, 200, 230,
                   280, 380, 290, 300, 300, 315]
    })
    
    # Mock the Dataset.get_data method
    mock_dataset_instance = MagicMock()
    mock_dataset_instance.get_data.return_value = mock_revenue_data
    mock_dataset_class.return_value = mock_dataset_instance
    
    # Call the calculate method
    result_df = revenue_growth.calculate()
    
    # Verify the results
    assert not result_df.empty
    
    # Check AAPL growth rates
    aapl_results = result_df[result_df['symbol'] == 'AAPL'].sort_values('date', ascending=False)
    assert len(aapl_results) == 2
    assert pytest.approx(aapl_results.iloc[0]['yoy'], 0.0001) == 0.20  # 20% growth
    assert pytest.approx(aapl_results.iloc[1]['yoy'], 0.0001) == 0.25  # 25% growth
    
    # Check MSFT growth rates
    msft_results = result_df[result_df['symbol'] == 'MSFT'].sort_values('date', ascending=False)
    assert len(msft_results) == 2
    assert pytest.approx(msft_results.iloc[0]['yoy'], 0.01) == 0.15  # 15% growth
    assert pytest.approx(msft_results.iloc[1]['yoy'], 0.01) == 2/18  # 10% growth
    
    # Check GOOGL growth rates
    googl_results = result_df[result_df['symbol'] == 'GOOGL'].sort_values('date', ascending=False)
    assert len(googl_results) == 2
    assert pytest.approx(googl_results.iloc[0]['yoy'], 0.01) == 0.05  # 5% growth
    assert pytest.approx(googl_results.iloc[1]['yoy'], 0.01) == -80/380  

def test_sector_quantiles(mock_revenue_growth):
    """Test the calculation of sector quantiles."""
    revenue_growth, mock_dataset_class = mock_revenue_growth
    
    # Mock the _get_valid_symbols method
    mock_symbols_df = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'],
        'sector': ['Technology', 'Technology', 'Technology', 'Technology', 'Technology']
    })
    revenue_growth._get_valid_symbols = MagicMock(return_value=mock_symbols_df)
    
    # Create mock revenue data with different growth rates for the same quarter
    # Add enough quarters for each stock to meet the 5-quarter minimum requirement
    mock_revenue_data = pd.DataFrame({
        'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL',
                 'MSFT', 'MSFT', 'MSFT', 'MSFT', 'MSFT', 'MSFT',
                 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL', 'GOOGL',
                 'AMZN', 'AMZN', 'AMZN', 'AMZN', 'AMZN', 'AMZN',
                 'META', 'META', 'META', 'META', 'META', 'META'],
        'date': ['2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31', '2023-03-31', '2024-06-30',
                '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31', '2023-03-31', '2024-06-30',
                '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31', '2023-03-31', '2024-06-30',
                '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31', '2023-03-31', '2024-06-30',
                '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31', '2023-03-31', '2024-06-30'],
        'revenue': [80, 85, 90, 95, 100, 120,
                   170, 175, 180, 190, 200, 230,
                   280, 285, 290, 295, 300, 315,
                   360, 370, 380, 390, 400, 440,
                   450, 460, 470, 480, 500, 600]
    })
    
    # Expected growth rates:
    # AAPL: 20% (middle)
    # MSFT: 15% (second lowest)
    # GOOGL: 5% (lowest)
    # AMZN: 10% (second highest)
    # META: 20% (highest)
    
    # Mock the Dataset.get_data method
    mock_dataset_instance = MagicMock()
    mock_dataset_instance.get_data.return_value = mock_revenue_data
    mock_dataset_class.return_value = mock_dataset_instance
    
    # Call the calculate method
    result_df = revenue_growth.calculate()
    
    # Verify the results
    assert not result_df.empty
    
    result_df = result_df[result_df['year_quarter'] == '2023Q1']
    
    # Check sector quantiles
    # GOOGL should have the lowest quantile (0.2 or 1/5)
    googl_quantile = result_df[result_df['symbol'] == 'GOOGL']['sector_quantile'].iloc[0]
    assert pytest.approx(googl_quantile, 0.1) == 0.2
    
    # META should have the highest quantile (1.0 or 5/5)
    meta_quantile = result_df[result_df['symbol'] == 'META']['sector_quantile'].iloc[0]
    assert pytest.approx(meta_quantile, 0.1) == 0.5
    
    # AAPL should be in the middle (0.6 or 3/5)
    aapl_quantile = result_df[result_df['symbol'] == 'AAPL']['sector_quantile'].iloc[0]
    assert pytest.approx(aapl_quantile, 0.1) == 1.0


def test_multiple_sectors(mock_revenue_growth):
    """Test the calculation of sector quantiles across multiple sectors."""
    revenue_growth, mock_dataset_class = mock_revenue_growth
    
    # Mock the _get_valid_symbols method with stocks from different sectors
    mock_symbols_df = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'XOM', 'CVX'],
        'sector': ['Technology', 'Technology', 'Energy', 'Energy']
    })
    revenue_growth._get_valid_symbols = MagicMock(return_value=mock_symbols_df)
    
    # Create mock revenue data with enough quarters for YoY calculation
    # Technology sector: AAPL (20% growth), MSFT (10% growth)
    # Energy sector: XOM (5% growth), CVX (15% growth)
    mock_revenue_data = pd.DataFrame({
        'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL',
                 'MSFT', 'MSFT', 'MSFT', 'MSFT', 'MSFT', 'MSFT',
                 'XOM', 'XOM', 'XOM', 'XOM', 'XOM', 'XOM',
                 'CVX', 'CVX', 'CVX', 'CVX', 'CVX', 'CVX'],
        'date': ['2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31', '2023-03-31', '2023-06-30',
                '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31', '2023-03-31', '2023-06-30',
                '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31', '2023-03-31', '2023-06-30',
                '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31', '2023-03-31', '2023-06-30'],
        'revenue': [80, 85, 90, 95, 100, 120,
                   180, 185, 190, 195, 200, 220,
                   280, 285, 290, 295, 300, 315,
                   380, 385, 390, 395, 400, 460]
    })
    
    # Mock the Dataset.get_data method
    mock_dataset_instance = MagicMock()
    mock_dataset_instance.get_data.return_value = mock_revenue_data
    mock_dataset_class.return_value = mock_dataset_instance
    
    # Call the calculate method
    result_df = revenue_growth.calculate()
    
    # Verify the results
    assert not result_df.empty

    result_df = result_df[result_df['year_quarter'] == '2023Q2']
    
    # Check sector quantiles for Technology sector
    tech_results = result_df[result_df['sector'] == 'Technology'].sort_values('yoy', ascending=False)
    assert len(tech_results) == 2
    
    # AAPL should have higher quantile than MSFT within Technology sector
    aapl_quantile = tech_results[tech_results['symbol'] == 'AAPL']['sector_quantile'].iloc[0]
    msft_quantile = tech_results[tech_results['symbol'] == 'MSFT']['sector_quantile'].iloc[0]
    assert aapl_quantile > msft_quantile
    
    # Check sector quantiles for Energy sector
    energy_results = result_df[result_df['sector'] == 'Energy'].sort_values('yoy', ascending=False)
    assert len(energy_results) == 2
    
    # CVX should have higher quantile than XOM within Energy sector
    cvx_quantile = energy_results[energy_results['symbol'] == 'CVX']['sector_quantile'].iloc[0]
    xom_quantile = energy_results[energy_results['symbol'] == 'XOM']['sector_quantile'].iloc[0]
    assert cvx_quantile > xom_quantile
    
    # The quantiles should be calculated separately for each sector
    # So both AAPL and CVX should have high quantiles in their respective sectors
    assert pytest.approx(aapl_quantile, 0.1) == 1.0  # Highest in Tech
    assert pytest.approx(cvx_quantile, 0.1) == 1.0   # Highest in Energy
