"""
Tests for the Dataset class.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from fmp_data.dataset import Dataset


class TestDataset:
    """Test cases for the Dataset class."""

    @pytest.fixture
    def mock_offline_data(self):
        """Create a mock for the OfflineData class."""
        with patch('fmp_data.dataset.OfflineData') as mock:
            # Configure the mock to return a predefined DataFrame
            instance = mock.return_value
            
            # Default behavior for build method
            def side_effect_build():
                # Return a simple DataFrame with test data
                return pd.DataFrame({
                    'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL'],
                    'date': pd.to_datetime(['2023-01-01', '2023-04-01', '2023-07-01', '2023-10-01']),
                    'filing_date': pd.to_datetime(['2023-01-15', '2023-04-15', '2023-07-15', '2023-10-15']),
                    'revenue': [100, 120, 110, 130],
                    'cost_of_revenue': [60, 70, 65, 75],
                    'operating_income': [20, 25, 22, 28]
                })
            
            instance.build.side_effect = side_effect_build
            yield mock

    def test_plan_query(self):
        """Test that metrics are correctly categorized as direct or derived."""
        # Test with only direct metrics
        dataset = Dataset(['AAPL'], ['revenue', 'cost_of_revenue'], '2023-01-01', '2023-12-31')
        assert set(dataset.direct_metrics) == {'revenue', 'cost_of_revenue'}
        assert dataset.derived_metrics == []
        
        # Test with only derived metrics
        dataset = Dataset(['AAPL'], ['gross_margin_ttm'], '2023-01-01', '2023-12-31')
        assert 'gross_margin_ttm' in dataset.derived_metrics
        assert set(dataset.dependencies) == {'revenue', 'cost_of_revenue'}
        
        # Test with mixed metrics
        dataset = Dataset(['AAPL'], ['revenue', 'gross_margin_ttm', 'operating_margin_ttm'], 
                         '2023-01-01', '2023-12-31')
        assert set(dataset.direct_metrics) == {'revenue'}
        assert set(dataset.derived_metrics) == {'gross_margin_ttm', 'operating_margin_ttm'}
        assert set(dataset.dependencies) == {'revenue', 'cost_of_revenue', 'operating_income'}

    @patch('fmp_data.dataset.OfflineData')
    def test_gen_direct_metrics_only(self, mock_offline_data_class):
        """Test generating dataset with only direct metrics."""
        # Configure the mock
        mock_instance = mock_offline_data_class.return_value
        mock_instance.build.return_value = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL'],
            'date': pd.to_datetime(['2023-01-01', '2023-04-01']),
            'filing_date': pd.to_datetime(['2023-01-15', '2023-04-15']),
            'revenue': [100, 120]
        })
        
        # Create dataset with only direct metrics
        dataset = Dataset(['AAPL'], ['revenue'], '2023-01-01', '2023-5-01')
        
        # Get the first (and only) result from the generator
        symbol, df = next(dataset.gen())
        
        # Verify the results
        assert symbol == 'AAPL'
        assert 'revenue' in df.columns
        assert len(df) == 121
        assert pd.isna(df[df['date'] == '2023-01-01']['revenue'].iloc[0])
        assert pd.isna(df[df['date'] == '2023-01-15']['revenue'].iloc[0])
        assert df[df['date'] == '2023-01-16']['revenue'].iloc[0] == 100
        assert df[df['date'] == '2023-04-15']['revenue'].iloc[0] == 100
        assert df[df['date'] == '2023-04-16']['revenue'].iloc[0] == 120
        assert df[df['date'] == '2023-04-30']['revenue'].iloc[0] == 120


    @patch('fmp_data.dataset.OfflineData')
    def test_gen_derived_metrics_only(self, mock_offline_data_class):
        """Test generating dataset with only derived metrics."""
        # Configure the mock for dependencies
        mock_instance = mock_offline_data_class.return_value
        mock_instance.build.return_value = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL'],
            'date': pd.to_datetime(['2023-01-01', '2023-04-01', '2023-07-01', '2023-10-01']),
            'filing_date': pd.to_datetime(['2023-01-15', '2023-04-15', '2023-07-15', '2023-10-15']),
            'revenue': [100, 120, 110, 130],
            'cost_of_revenue': [60, 70, 65, 75]
        }).sort_values('date', ascending=True)
        
        # Create dataset with only derived metrics
        dataset = Dataset(['AAPL'], ['gross_margin_ttm'], '2023-01-01', '2023-12-31')
        
        # Get the first (and only) result from the generator
        symbol, df = next(dataset.gen())
        
        # Verify the results
        assert symbol == 'AAPL'
        assert 'gross_margin_ttm' in df.columns
        assert len(df) == 365
        assert df[df.date <= '2023-10-15'].gross_margin_ttm.isna().all()
        assert df[df.date >= '2023-10-16'].gross_margin_ttm.nunique() == 1
        assert pytest.approx(df[df.date >= '2023-10-16'].gross_margin_ttm.iloc[0]) == 1 - (sum([60, 70, 65, 75]) / sum([100, 120, 110, 130]))
        

    @patch('fmp_data.dataset.OfflineData')
    def test_gen_mixed_metrics(self, mock_offline_data_class):
        """Test generating dataset with both direct and derived metrics."""
        # Configure the mocks for both direct metrics and dependencies
        mock_instance = MagicMock()
        mock_offline_data_class.return_value = mock_instance
        
        # First call for direct metrics
        direct_df = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL'],
            'date': pd.to_datetime(['2023-01-01', '2023-04-01']),
            'filing_date': pd.to_datetime(['2023-01-15', '2023-04-15']),
            'revenue': [100, 120]
        })
        
        # Second call for dependencies
        dependencies_df = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'AAPL', 'AAPL'],
            'date': pd.to_datetime(['2023-01-01', '2023-04-01', '2023-07-01', '2023-10-01']),
            'filing_date': pd.to_datetime(['2023-01-15', '2023-04-15', '2023-07-15', '2023-10-15']),
            'revenue': [100, 120, 110, 130],
            'cost_of_revenue': [60, 70, 65, 75]
        }).sort_values('date', ascending=True)
        
        # Set up the mock to return different DataFrames on consecutive calls
        mock_instance.build.side_effect = [direct_df, dependencies_df]
        
        # Create dataset with mixed metrics
        dataset = Dataset(['AAPL'], ['revenue', 'gross_margin_ttm'], '2023-01-01', '2023-12-31')
        
        # Get the first (and only) result from the generator
        symbol, df = next(dataset.gen())
        
        # Verify the results
        assert symbol == 'AAPL'
        assert 'revenue' in df.columns
        assert 'gross_margin_ttm' in df.columns
        assert len(df) == 365
        assert df[df.date <= '2023-10-15'].gross_margin_ttm.isna().all()
        assert df[df.date >= '2023-10-16'].gross_margin_ttm.nunique() == 1
        assert pytest.approx(df[df.date >= '2023-10-16'].gross_margin_ttm.iloc[0]) == 1 - (sum([60, 70, 65, 75]) / sum([100, 120, 110, 130]))
        assert df[df.date <= '2023-01-15'].revenue.isna().all()
        assert np.all(df[df.date.between('2023-01-16', '2023-04-15')].revenue == 100)
        assert np.all(df[df.date >= '2023-04-16'].revenue == 120)
        
        

    def test_fill_date_gaps(self):
        """Test filling date gaps in the data."""
        # Create a test dataset
        dataset = Dataset(['AAPL'], ['goodwill'], '2023-01-01', '2023-12-31')
        
        # Create a test DataFrame with gaps
        df = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL'],
            'date': pd.to_datetime(['2023-01-01', '2023-04-01']),
            'filing_date': pd.to_datetime(['2023-01-15', '2023-04-15']),
            'goodwill': [100, 120]
        })
        
        # Fill date gaps
        filled_df = dataset.fill_date_gaps(df)
        
        # Verify the results
        assert len(filled_df) == (pd.to_datetime('2023-12-31') - pd.to_datetime('2023-01-01')).days + 1
        assert filled_df['date'].min() == pd.to_datetime('2023-01-01')
        assert filled_df['date'].max() == pd.to_datetime('2023-12-31')
        
        # Check that values are forward-filled
        # The first value should be forward-filled until the next filing date
        assert np.all(filled_df[filled_df['date'] <= '2023-01-15']['goodwill'].isna())
        
        # The second value should be used for dates after its filing date
        assert np.all(filled_df[filled_df['date'].between('2023-01-16', '2023-04-15')]['goodwill'] == 100)
        # The third value should be used for dates after its filing date
        assert np.all(filled_df[filled_df['date'] >= '2023-04-16']['goodwill'] == 120)

    def test_fill_date_gaps_empty_df(self):
        """Test filling date gaps with an empty DataFrame."""
        # Create a test dataset
        dataset = Dataset(['AAPL'], ['revenue'], '2023-01-01', '2023-12-31')
        
        # Create an empty DataFrame
        df = pd.DataFrame(columns=['symbol', 'date', 'filing_date', 'revenue'])
        
        # Fill date gaps
        filled_df = dataset.fill_date_gaps(df)
        
        # Verify the results
        assert filled_df.empty  # Should still be empty
        
    @patch('fmp_data.dataset.OfflineData')
    def test_gen_with_price(self, mock_offline_data_class):
        """Test generating dataset with price data when with_price=True."""
        # Configure the mock for financial data
        mock_instance = MagicMock()
        mock_offline_data_class.return_value = mock_instance
        
        # Mock financial data
        financial_df = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL'],
            'date': pd.to_datetime(['2023-01-01', '2023-04-01']),
            'filing_date': pd.to_datetime(['2023-01-15', '2023-04-15']),
            'revenue': [100, 120]
        })
        
        # Mock price data
        price_df = pd.DataFrame({
            'symbol': ['AAPL'] * 5,
            'date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
            'tradable_price': [150.0, 152.5, 151.0, 153.0, 155.0]
        })
        
        # Set up the mock to return financial data for build
        mock_instance.build.return_value = financial_df
        
        # Set up the mock for historical_tradable_price
        mock_offline_data_class.historical_tradable_price = MagicMock(return_value=price_df)
        
        # Create dataset with with_price=True
        dataset = Dataset(['AAPL'], ['revenue'], '2023-01-01', '2023-01-05', with_price=True)
        
        # Get the first (and only) result from the generator
        symbol, df = next(dataset.gen())
        
        # Verify the results
        assert symbol == 'AAPL'
        assert 'revenue' in df.columns
        assert 'tradable_price' in df.columns
        assert len(df) == 5  # 5 days from Jan 1 to Jan 5
        
        # Check that price data is correctly merged
        assert df[df['date'] == '2023-01-01']['tradable_price'].iloc[0] == 150.0
        assert df[df['date'] == '2023-01-05']['tradable_price'].iloc[0] == 155.0
        
        # Check that financial data is correctly filled
        assert df[df['date'] <= '2023-01-15']['revenue'].isna().all()
        
        # Verify that historical_tradable_price was called with correct parameters
        mock_offline_data_class.historical_tradable_price.assert_called_once_with(
            'AAPL', '2023-01-01', '2023-01-05', db_path=dataset.db_path
        )
