import pytest
import pandas as pd
from fmp_data import Dataset


def test_dataset_date_range_basic(test_db):
    """Test Dataset with start_date and end_date parameters."""
    # Test with a date range that includes all dates in the test database
    dataset = Dataset(
        symbol='AAPL',
        metrics={'close': 'price'},
        start_date='2024-01-01',
        end_date='2024-01-02',
        db_path=test_db
    )
    
    # Check if we have data for both dates
    assert len(dataset.data) == 2
    assert all(date in dataset.data['date'].values for date in ['2024-01-01', '2024-01-02'])
    
    # Verify price values for each date
    jan1_data = dataset.data[dataset.data['date'] == '2024-01-01'].iloc[0]
    jan2_data = dataset.data[dataset.data['date'] == '2024-01-02'].iloc[0]
    
    assert jan1_data['price'] == 102.0
    assert jan2_data['price'] == 105.0


def test_dataset_date_range_with_holes(test_db):
    """Test Dataset with a date range that has holes in the data."""
    dataset = Dataset(
        symbol='AAPL',
        metrics={'close': 'price'},
        start_date='2024-01-01',
        end_date='2024-01-04',
        db_path=test_db
    )
    
    # Check if we have data for all dates in the range
    assert len(dataset.data) == 4  # 4 days from Jan 1 to Jan 4
    
    # Check that all dates are present
    expected_dates = ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']
    assert all(date in dataset.data['date'].values for date in expected_dates)
    
    # Get data for each date
    data_by_date = {date: dataset.data[dataset.data['date'] == date].iloc[0]['price'] 
                   for date in expected_dates}
    
    # Verify price values for each date
    assert data_by_date['2024-01-01'] == 102.0
    assert data_by_date['2024-01-02'] == 105.0
    assert data_by_date['2024-01-03'] == 105.0  # Should be filled with previous day's value
    assert data_by_date['2024-01-04'] == 105.0


def test_dataset_date_range_multiple_symbols(test_db):
    """Test Dataset with multiple symbols and a date range."""
    # Test with multiple symbols
    dataset = Dataset(
        symbol=['AAPL', 'GOOGL'],
        metrics={'close': 'price'},
        start_date='2024-01-01',
        end_date='2024-01-02',
        db_path=test_db
    )
    
    # Check if we have data for both symbols
    assert len(dataset.data) == 4  # 2 symbols * 2 days
    
    # Check that all symbols and dates are present
    assert all(sym in dataset.data['symbol'].values for sym in ['AAPL', 'GOOGL'])
    assert all(date in dataset.data['date'].values for date in ['2024-01-01', '2024-01-02'])
    
    # Get data for each symbol and date
    aapl_jan1 = dataset.data[(dataset.data['symbol'] == 'AAPL') & 
                             (dataset.data['date'] == '2024-01-01')].iloc[0]['price']
    aapl_jan2 = dataset.data[(dataset.data['symbol'] == 'AAPL') & 
                             (dataset.data['date'] == '2024-01-02')].iloc[0]['price']
    googl_jan1 = dataset.data[(dataset.data['symbol'] == 'GOOGL') & 
                              (dataset.data['date'] == '2024-01-01')].iloc[0]['price']
    googl_jan2 = dataset.data[(dataset.data['symbol'] == 'GOOGL') & 
                              (dataset.data['date'] == '2024-01-02')].iloc[0]['price']
    
    # Verify price values
    assert aapl_jan1 == 102.0
    assert aapl_jan2 == 105.0
    assert googl_jan1 == 152.0
    assert pytest.approx(googl_jan2) == 152.0  # Should be filled with previous day's value


def test_dataset_date_range_with_holes_multiple_symbols(test_db):
    """Test Dataset with multiple symbols and holes in the data."""
    
    # Test with a date range that includes dates with no data for some symbols
    dataset = Dataset(
        symbol=['AAPL', 'GOOGL'],
        metrics={'close': 'price'},
        start_date='2024-01-01',
        end_date='2024-01-03',
        db_path=test_db
    )
    
    # Check if we have data for all dates and symbols
    assert len(dataset.data) == 6  # 2 symbols * 3 days
    
    # Get data for each symbol and date
    aapl_jan1 = dataset.data[(dataset.data['symbol'] == 'AAPL') & 
                             (dataset.data['date'] == '2024-01-01')].iloc[0]['price']
    aapl_jan2 = dataset.data[(dataset.data['symbol'] == 'AAPL') & 
                             (dataset.data['date'] == '2024-01-02')].iloc[0]['price']
    aapl_jan3 = dataset.data[(dataset.data['symbol'] == 'AAPL') & 
                             (dataset.data['date'] == '2024-01-03')].iloc[0]['price']
    googl_jan1 = dataset.data[(dataset.data['symbol'] == 'GOOGL') & 
                              (dataset.data['date'] == '2024-01-01')].iloc[0]['price']
    googl_jan2 = dataset.data[(dataset.data['symbol'] == 'GOOGL') & 
                              (dataset.data['date'] == '2024-01-02')].iloc[0]['price']
    googl_jan3 = dataset.data[(dataset.data['symbol'] == 'GOOGL') & 
                              (dataset.data['date'] == '2024-01-03')].iloc[0]['price']
    
    # Verify price values
    assert aapl_jan1 == 102.0
    assert aapl_jan2 == 105.0
    assert aapl_jan3 == 105.0  # Should be filled with previous day's value
    assert googl_jan1 == 152.0
    assert googl_jan2 == 152.0  # Should be filled with previous day's value
    assert googl_jan3 == 152.0


def test_dataset_date_range_empty_result(test_db):
    """Test Dataset with a date range that has no data."""
    # Test with a date range that has no data in the database
    dataset = Dataset(
        symbol='AAPL',
        metrics={'close': 'price'},
        start_date='2023-01-01',
        end_date='2023-01-05',
        db_path=test_db
    )
    
    # Check that we get data for all dates in the range, but with NaN values
    assert len(dataset.data) == 5  # 5 days from Jan 1 to Jan 5, 2023
    
    # Check that all dates are present but values are NaN
    for date in pd.date_range(start='2023-01-01', end='2023-01-05'):
        date_str = date.strftime('%Y-%m-%d')
        row = dataset.data[dataset.data['date'] == date_str]
        assert not row.empty
        assert pd.isna(row.iloc[0]['price'])
