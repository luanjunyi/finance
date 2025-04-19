import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock, call
import sqlite3
from datetime import datetime

from feature_gen.base_feature import FinancialFeatureBase


# Create a concrete implementation of FinancialFeatureBase for testing
class MockFeature(FinancialFeatureBase):
    """Concrete implementation of FinancialFeatureBase for testing.
    Note: This class is named 'MockFeature' instead of 'TestFeature' to avoid
    pytest trying to collect it as a test class.
    """
    
    def _create_features_table(self):
        """Create the test_features table if it doesn't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS test_features (
                symbol VARCHAR(10),
                sector VARCHAR(255),
                date DATE,
                test_value DECIMAL(10, 4),
                sector_quantile DECIMAL(10, 4),
                year_quarter VARCHAR(6),
                PRIMARY KEY (symbol, date)
            )
            """)
    
    def _get_table_name(self) -> str:
        """Get the name of the database table where test features are stored."""
        return 'test_features'
    
    def calculate(self) -> pd.DataFrame:
        """Calculate test values for all valid symbols."""
        # Return a simple DataFrame with test data
        return pd.DataFrame({
            'symbol': ['AAPL', 'MSFT', 'GOOGL'],
            'sector': ['Technology', 'Technology', 'Technology'],
            'date': ['2024-03-31', '2024-03-31', '2024-03-31'],
            'test_value': [0.5, 0.7, 0.3],
            'year_quarter': ['2024Q1', '2024Q1', '2024Q1']
        })


@pytest.fixture
def mock_test_feature():
    """Create a TestFeature instance with mocked dependencies."""
    # Create a mock for the database connection
    with patch('sqlite3.connect') as mock_db_connect:
        # Setup mock connection
        mock_connection = MagicMock()
        mock_db_connect.return_value.__enter__.return_value = mock_connection
        
        # Initialize the MockFeature instance
        test_feature = MockFeature(db_path='mock_db_path')
        
        yield test_feature, mock_connection


def test_get_year_quarter():
    """Test the _get_year_quarter method."""
    test_feature = MockFeature()
    
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
        result = test_feature._get_year_quarter(date_str)
        assert result == expected


def test_calculate_sector_quantiles():
    """Test the calculate_sector_quantiles method with complex scenarios."""
    test_feature = MockFeature()
    
    # Create a test DataFrame with multiple sectors and many symbols
    # Technology sector: 12 companies with different values
    # Healthcare sector: 8 companies with different values
    # Energy sector: 5 companies with different values
    # Financial sector: 3 companies with different values
    # Multiple quarters to test time-based isolation
    
    # Generate symbols
    tech_symbols = [f'TECH{i}' for i in range(1, 13)]
    health_symbols = [f'HLTH{i}' for i in range(1, 9)]
    energy_symbols = [f'ENRG{i}' for i in range(1, 6)]
    finance_symbols = [f'FIN{i}' for i in range(1, 4)]
    
    # Generate values - intentionally create ties and edge cases
    tech_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.5, 0.6, 0.7, 0.8, 0.9, 0.9, 1.0]  # Note: two pairs of ties
    health_values = [0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
    energy_values = [0.3, 0.3, 0.5, 0.7, 0.7]  # Note: two pairs of ties
    finance_values = [0.2, 0.5, 0.8]
    
    # Create quarters
    quarters = ['2023Q1', '2023Q2']
    
    # Build the DataFrame
    symbols = []
    sectors = []
    dates = []
    values = []
    year_quarters = []
    
    # Add Technology sector data
    for quarter in quarters:
        for i, symbol in enumerate(tech_symbols):
            symbols.append(symbol)
            sectors.append('Technology')
            dates.append('2023-03-31' if quarter == '2023Q1' else '2023-06-30')
            values.append(tech_values[i])
            year_quarters.append(quarter)
    
    # Add Healthcare sector data
    for quarter in quarters:
        for i, symbol in enumerate(health_symbols):
            symbols.append(symbol)
            sectors.append('Healthcare')
            dates.append('2023-03-31' if quarter == '2023Q1' else '2023-06-30')
            values.append(health_values[i])
            year_quarters.append(quarter)
    
    # Add Energy sector data
    for quarter in quarters:
        for i, symbol in enumerate(energy_symbols):
            symbols.append(symbol)
            sectors.append('Energy')
            dates.append('2023-03-31' if quarter == '2023Q1' else '2023-06-30')
            values.append(energy_values[i])
            year_quarters.append(quarter)
    
    # Add Financial sector data
    for quarter in quarters:
        for i, symbol in enumerate(finance_symbols):
            symbols.append(symbol)
            sectors.append('Financial')
            dates.append('2023-03-31' if quarter == '2023Q1' else '2023-06-30')
            values.append(finance_values[i])
            year_quarters.append(quarter)
    
    # Create the test DataFrame
    test_df = pd.DataFrame({
        'symbol': symbols,
        'sector': sectors,
        'date': dates,
        'test_value': values,
        'year_quarter': year_quarters
    })
    
    # Calculate sector quantiles
    result_df = test_feature.calculate_sector_quantiles(test_df, 'test_value')
    
    # Verify the results
    assert not result_df.empty
    
    # Test for each quarter separately
    for quarter in quarters:
        quarter_df = result_df[result_df['year_quarter'] == quarter]
        
        # Check Technology sector quantiles
        tech_df = quarter_df[quarter_df['sector'] == 'Technology']
        assert len(tech_df) == 12
        
        # Check specific ranks in Technology sector
        # TECH1 should have the lowest value (0.1)
        tech1_quantile = tech_df[tech_df['symbol'] == 'TECH1']['sector_quantile'].iloc[0]
        assert pytest.approx(tech1_quantile, 0.0001) == 1/12
        
        # TECH12 should have the highest value (1.0)
        tech12_quantile = tech_df[tech_df['symbol'] == 'TECH12']['sector_quantile'].iloc[0]
        assert pytest.approx(tech12_quantile, 0.0001) == 1.0
        
        # Check ties - TECH5 and TECH6 both have value 0.5
        tech5_quantile = tech_df[tech_df['symbol'] == 'TECH5']['sector_quantile'].iloc[0]
        tech6_quantile = tech_df[tech_df['symbol'] == 'TECH6']['sector_quantile'].iloc[0]
        assert pytest.approx(tech5_quantile, 0.0001) == 5.5/12
        assert pytest.approx(tech6_quantile, 0.0001) == 5.5/12
        
        # Check ties - TECH10 and TECH11 both have value 0.9
        tech10_quantile = tech_df[tech_df['symbol'] == 'TECH10']['sector_quantile'].iloc[0]
        tech11_quantile = tech_df[tech_df['symbol'] == 'TECH11']['sector_quantile'].iloc[0]
        assert pytest.approx(tech10_quantile, 0.0001) == 10.5/12
        assert pytest.approx(tech10_quantile, 0.0001) == 10.5/12
        
        # Check Healthcare sector quantiles
        health_df = quarter_df[quarter_df['sector'] == 'Healthcare']
        assert len(health_df) == 8
        
        # HLTH1 should have the lowest value (0.15)
        hlth1_quantile = health_df[health_df['symbol'] == 'HLTH1']['sector_quantile'].iloc[0]
        assert pytest.approx(hlth1_quantile, 0.0001) == 1/8
        
        # HLTH8 should have the highest value (0.85)
        hlth8_quantile = health_df[health_df['symbol'] == 'HLTH8']['sector_quantile'].iloc[0]
        assert pytest.approx(hlth8_quantile, 0.0001) == 1.0
        
        # Check Energy sector quantiles with ties
        energy_df = quarter_df[quarter_df['sector'] == 'Energy']
        assert len(energy_df) == 5
        
        # ENRG1 and ENRG2 both have value 0.3
        enrg1_quantile = energy_df[energy_df['symbol'] == 'ENRG1']['sector_quantile'].iloc[0]
        enrg2_quantile = energy_df[energy_df['symbol'] == 'ENRG2']['sector_quantile'].iloc[0]
        assert pytest.approx(enrg1_quantile, 0.0001) == enrg2_quantile
        assert pytest.approx(enrg1_quantile, 0.0001) == 1.5/5
        
        # ENRG4 and ENRG5 both have value 0.7
        enrg4_quantile = energy_df[energy_df['symbol'] == 'ENRG4']['sector_quantile'].iloc[0]
        enrg5_quantile = energy_df[energy_df['symbol'] == 'ENRG5']['sector_quantile'].iloc[0]
        assert pytest.approx(enrg4_quantile, 0.0001) == enrg5_quantile
        assert pytest.approx(enrg4_quantile, 0.0001) == 4.5/5
        
        # Check Financial sector quantiles
        finance_df = quarter_df[quarter_df['sector'] == 'Financial']
        assert len(finance_df) == 3
        
        # FIN1 should have the lowest value (0.2)
        fin1_quantile = finance_df[finance_df['symbol'] == 'FIN1']['sector_quantile'].iloc[0]
        assert pytest.approx(fin1_quantile, 0.0001) == 1/3
        
        # FIN3 should have the highest value (0.8)
        fin3_quantile = finance_df[finance_df['symbol'] == 'FIN3']['sector_quantile'].iloc[0]
        assert pytest.approx(fin3_quantile, 0.0001) == 1.0


def test_calculate_sector_quantiles_simple():
    """Test the calculate_sector_quantiles method with a simple case."""
    test_feature = MockFeature()
    
    # Create a simple test DataFrame with multiple sectors and values
    test_df = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL', 'XOM', 'CVX'],
        'sector': ['Technology', 'Technology', 'Technology', 'Energy', 'Energy'],
        'date': ['2024-03-31', '2024-03-31', '2024-03-31', '2024-03-31', '2024-03-31'],
        'test_value': [0.5, 0.7, 0.3, 0.6, 0.4],
        'year_quarter': ['2024Q1', '2024Q1', '2024Q1', '2024Q1', '2024Q1']
    })
    
    # Calculate sector quantiles
    result_df = test_feature.calculate_sector_quantiles(test_df, 'test_value')
    
    # Verify the results
    assert not result_df.empty
    
    # Check Technology sector quantiles
    tech_df = result_df[result_df['sector'] == 'Technology']
    assert len(tech_df) == 3
    
    # GOOGL should have the lowest quantile in Technology (0.33 or 1/3)
    googl_quantile = tech_df[tech_df['symbol'] == 'GOOGL']['sector_quantile'].iloc[0]
    assert pytest.approx(googl_quantile, 0.0001) == 1/3
    
    # MSFT should have the highest quantile in Technology (1.0 or 3/3)
    msft_quantile = tech_df[tech_df['symbol'] == 'MSFT']['sector_quantile'].iloc[0]
    assert pytest.approx(msft_quantile, 0.0001) == 1.0
    
    # AAPL should be in the middle (0.67 or 2/3)
    aapl_quantile = tech_df[tech_df['symbol'] == 'AAPL']['sector_quantile'].iloc[0]
    assert pytest.approx(aapl_quantile, 0.0001) == 2/3
    
    # Check Energy sector quantiles
    energy_df = result_df[result_df['sector'] == 'Energy']
    assert len(energy_df) == 2
    
    # CVX should have the lower quantile in Energy (0.5 or 1/2)
    cvx_quantile = energy_df[energy_df['symbol'] == 'CVX']['sector_quantile'].iloc[0]
    assert pytest.approx(cvx_quantile, 0.0001) == 0.5
    
    # XOM should have the higher quantile in Energy (1.0 or 2/2)
    xom_quantile = energy_df[energy_df['symbol'] == 'XOM']['sector_quantile'].iloc[0]
    assert pytest.approx(xom_quantile, 0.0001) == 1.0


def test_get_valid_symbols(mock_test_feature):
    """Test the _get_valid_symbols method."""
    test_feature, mock_connection = mock_test_feature
    
    # Mock the database query result
    mock_cursor = MagicMock()
    mock_connection.execute.return_value = mock_cursor
    mock_connection.cursor.return_value = mock_cursor
    
    # Mock pd.read_sql_query to return a DataFrame
    mock_symbols_df = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL'],
        'sector': ['Technology', 'Technology', 'Technology']
    })
    
    with patch('pandas.read_sql_query', return_value=mock_symbols_df) as mock_read_sql:
        result = test_feature._get_valid_symbols()
        
        # Verify the SQL query was executed correctly
        mock_read_sql.assert_called_once()
        
        # Verify the result
        pd.testing.assert_frame_equal(result, mock_symbols_df)


def test_store_in_database(mock_test_feature):
    """Test the store_in_database method."""
    test_feature, mock_connection = mock_test_feature
    
    # Mock the calculate method to return a test DataFrame
    test_df = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL'],
        'sector': ['Technology', 'Technology', 'Technology'],
        'date': ['2024-03-31', '2024-03-31', '2024-03-31'],
        'test_value': [0.5, 0.7, 0.3],
        'year_quarter': ['2024Q1', '2024Q1', '2024Q1']
    })
    test_feature.calculate = MagicMock(return_value=test_df)
    
    # Mock the to_sql method
    with patch.object(pd.DataFrame, 'to_sql') as mock_to_sql:
        # Call the method
        test_feature.store_in_database()
        
        # Verify the table was dropped and recreated
        assert mock_connection.execute.call_count >= 2
        mock_connection.execute.assert_any_call("DROP TABLE IF EXISTS test_features")
        
        # Verify to_sql was called with the correct parameters
        mock_to_sql.assert_called_once()
        args, kwargs = mock_to_sql.call_args
        assert args[0] == 'test_features'
        assert kwargs['if_exists'] == 'replace'
        assert kwargs['index'] == False


def test_get_feature_data(mock_test_feature):
    """Test the get_feature_data method."""
    test_feature, mock_connection = mock_test_feature
    
    # Mock the database query result
    mock_cursor = MagicMock()
    mock_connection.execute.return_value = mock_cursor
    mock_connection.cursor.return_value = mock_cursor
    
    # Mock pd.read_sql_query to return a DataFrame
    mock_data_df = pd.DataFrame({
        'symbol': ['AAPL', 'MSFT', 'GOOGL'],
        'sector': ['Technology', 'Technology', 'Technology'],
        'date': ['2024-03-31', '2024-03-31', '2024-03-31'],
        'test_value': [0.5, 0.7, 0.3],
        'sector_quantile': [0.67, 1.0, 0.33],
        'year_quarter': ['2024Q1', '2024Q1', '2024Q1']
    })
    
    with patch('pandas.read_sql_query', return_value=mock_data_df) as mock_read_sql:
        result = test_feature.get_feature_data()
        
        # Verify the SQL query was executed correctly
        mock_read_sql.assert_called_once_with(
            "SELECT * FROM test_features ORDER BY symbol, date DESC", 
            mock_connection
        )
        
        # Verify the result
        pd.testing.assert_frame_equal(result, mock_data_df)
