import pytest
import pandas as pd
from fmp_data import Dataset, FMPPriceLoader, BEFORE_PRICE, AFTER_PRICE


def test_dataset_initialization(test_db):
    """Test basic Dataset initialization with metrics from different tables."""
    dataset = Dataset(
        symbol='AAPL',
        metrics={
            'revenue': 'rev',
            'operating_income': 'op_income',
            'net_income': 'net_inc',
            'total_assets': 'assets',
            'total_equity': 'equity',
            'operating_cash_flow': 'op_cf',
            'free_cash_flow': 'fcf',
            'roe': 'return_on_equity'
        },
        db_path=test_db
    )
    assert dataset is not None
    assert isinstance(dataset.data, pd.DataFrame)
    
    # Check if all metrics are present
    expected_columns = {'rev', 'op_income', 'net_inc', 'assets', 'equity', 
                       'op_cf', 'fcf', 'return_on_equity'}
    assert all(col in dataset.data.columns for col in expected_columns)
    
    # Verify values from different tables
    assert dataset.data.iloc[0]['rev'] == 100000000
    assert dataset.data.iloc[0]['op_income'] == 30000000
    assert dataset.data.iloc[0]['net_inc'] == 25000000
    assert dataset.data.iloc[0]['assets'] == 500000000
    assert dataset.data.iloc[0]['equity'] == 300000000
    assert dataset.data.iloc[0]['op_cf'] == 35000000
    assert dataset.data.iloc[0]['fcf'] == 30000000
    assert dataset.data.iloc[0]['return_on_equity'] == 0.15

def test_dataset_multiple_symbols_with_fundamentals(test_db, sample_symbols):
    """Test Dataset with multiple symbols."""
    dataset = Dataset(
        symbol=sample_symbols,
        metrics={'revenue': 'rev', 'total_assets': 'assets'},
        db_path=test_db
    )
    assert len(dataset.data) == len(sample_symbols)
    assert 'rev' in dataset.data.columns
    assert 'assets' in dataset.data.columns
    
    # Verify values for both symbols
    aapl_data = dataset.data[dataset.data['symbol'] == 'AAPL'].iloc[0]
    googl_data = dataset.data[dataset.data['symbol'] == 'GOOGL'].iloc[0]
    
    assert aapl_data['rev'] == 100000000
    assert aapl_data['assets'] == 500000000
    assert googl_data['rev'] == 80000000
    assert googl_data['assets'] == 400000000


def test_dataset_price_metrics(test_db):
    """Test Dataset with price metrics."""
    # Test with regular price data first
    dataset = Dataset(
        symbol='AAPL',
        metrics={'close': 'price'},
        db_path=test_db
    )
    assert 'price' in dataset.data.columns
    assert dataset.data['price'].tolist() == [102.0, 105.0]
    assert dataset.data['date'].tolist() == ['2024-01-01', '2024-01-02']

def test_dataset_after_price_metrics(test_db):
    """Test Dataset with before and after price metrics."""
    dataset = Dataset(
        symbol='AAPL',
        metrics={
            AFTER_PRICE: 'price',
            'revenue': 'rev'  # Include a regular metric
        },
        for_date='2024-01-01',  # Changed from 2024-01-02 to match expected behavior
        db_path=test_db
    )
    
    # Check if all metrics are present
    assert 'price' in dataset.data.columns
    assert 'rev' in dataset.data.columns
    
    # Verify price values
    assert dataset.data.iloc[0]['price'] == 102.0
    assert dataset.data.iloc[0]['rev'] == 100000000


def test_dataset_per_share_metrics(test_db):
    """Test Dataset with per-share metrics from metrics table."""
    dataset = Dataset(
        symbol='AAPL',
        metrics={
            'revenue_per_share': 'rev_ps',
            'net_income_per_share': 'eps',
            'operating_cash_flow_per_share': 'ocf_ps',
            'free_cash_flow_per_share': 'fcf_ps',
            'book_value_per_share': 'bvps'
        },
        db_path=test_db
    )
    
    # Check if all metrics are present
    assert 'rev_ps' in dataset.data.columns
    assert 'eps' in dataset.data.columns
    assert 'ocf_ps' in dataset.data.columns
    assert 'fcf_ps' in dataset.data.columns
    assert 'bvps' in dataset.data.columns
    
    # Verify values
    assert dataset.data.iloc[0]['rev_ps'] == 10.0
    assert dataset.data.iloc[0]['eps'] == 2.5
    assert dataset.data.iloc[0]['ocf_ps'] == 3.5
    assert dataset.data.iloc[0]['fcf_ps'] == 3.0
    assert dataset.data.iloc[0]['bvps'] == 30.0


def test_dataset_multiple_symbols(test_db, sample_symbols):
    """Test Dataset with multiple symbols."""
    dataset = Dataset(
        symbol=sample_symbols,
        metrics={
            'revenue': 'rev',
            'net_income': 'net_inc',
            'roe': 'return_on_equity'
        },
        db_path=test_db
    )
    
    # Check if we have data for both symbols
    assert len(dataset.data) == len(sample_symbols)
    assert all(sym in dataset.data['symbol'].values for sym in sample_symbols)
    
    # Verify values for both symbols
    aapl_data = dataset.data[dataset.data['symbol'] == 'AAPL'].iloc[0]
    googl_data = dataset.data[dataset.data['symbol'] == 'GOOGL'].iloc[0]
    
    assert aapl_data['rev'] == 100000000
    assert aapl_data['net_inc'] == 25000000
    assert aapl_data['return_on_equity'] == 0.15
    
    assert googl_data['rev'] == 80000000
    assert googl_data['net_inc'] == 20000000
    assert googl_data['return_on_equity'] == 0.12



def test_price_loader_get_price(test_db):
    """Test FMPPriceLoader.get_price method."""
    loader = FMPPriceLoader(db_path=test_db)
    price = loader.get_price('AAPL', '2024-01-01', 'close')
    assert price == 102.0

    # Test different price types
    assert loader.get_price('AAPL', '2024-01-01', 'open') == 100.0
    assert loader.get_price('AAPL', '2024-01-01', 'high') == 105.0
    assert loader.get_price('AAPL', '2024-01-01', 'low') == 99.0


def test_price_loader_last_available_price(test_db):
    """Test FMPPriceLoader.get_last_available_price method."""
    loader = FMPPriceLoader(db_path=test_db)
    price, date = loader.get_last_available_price('AAPL', '2024-01-02')
    assert price == 105.0
    assert date == '2024-01-02'

    price, date = loader.get_last_available_price('AAPL', '2024-01-05')    
    assert price == 105.0
    assert date == '2024-01-02'    


def test_price_loader_next_available_price(test_db):
    """Test FMPPriceLoader.get_next_available_price method."""
    loader = FMPPriceLoader(db_path=test_db)
    price, date = loader.get_next_available_price('AAPL', '2024-01-01')
    assert price == 102.0
    assert date == '2024-01-01'

    price, date = loader.get_next_available_price('AAPL', '2023-12-09')
    assert price == 102.0
    assert date == '2024-01-01'    


def test_price_loader_price_range(test_db):
    """Test FMPPriceLoader.get_close_price_during method."""
    loader = FMPPriceLoader(db_path=test_db)
    prices = loader.get_close_price_during(
        'AAPL', 
        '2024-01-01', 
        '2024-01-02'
    )
    assert len(prices) == 2
    assert prices['2024-01-01'] == 102.0
    assert prices['2024-01-02'] == 105.0


def test_price_loader_multiple_stocks(test_db, sample_symbols, sample_dates):
    """Test FMPPriceLoader with multiple stocks."""
    loader = FMPPriceLoader(db_path=test_db)
    prices = loader.get_price_for_stocks_during(
        sample_symbols,
        '2024-01-01',
        '2024-01-02'
    )
    assert len(prices) == len(sample_symbols)
    assert prices['AAPL'][0][1] == 105.0  # Latest price
    assert prices['GOOGL'][0][1] == 152.0


def test_error_handling(test_db):
    """Test error handling for invalid inputs."""
    loader = FMPPriceLoader(db_path=test_db)
    
    # Test invalid symbol
    with pytest.raises(TypeError):  
        loader.get_price('INVALID', '2024-01-01')
    
    # Test invalid date
    with pytest.raises(TypeError):  
        loader.get_price('AAPL', '2025-01-01')
    
    # Test invalid price type
    with pytest.raises(ValueError):
        loader.get_price('AAPL', '2024-01-01', 'invalid_type')
