"""
Tests for the base metrics framework.

This module tests the core functionality of the metrics framework,
including the base MetricCalculator class and utility functions.
"""

import pytest
import pandas as pd
import numpy as np
from typing import Dict, Set
from fmp_offline.metrics.metrics import (
    MetricCalculator, 
    QuarterlyMetricCalculator,
    DailyMetricCalculator,
    METRIC_REGISTRY,
    calculate_metrics,
    get_dependencies
)


class TestMetricCalculator:
    """Tests for the MetricCalculator base class."""
    
    def test_template_method_pattern(self):
        """Test that calculate() calls _validate_data_sources() and _perform_calculation()."""
        # Create a test calculator that tracks method calls
        class TestCalculator(MetricCalculator):
            def __init__(self):
                super().__init__(
                    name='test_metric',
                    dependencies={'source1': {'col1', 'col2'}}
                )
                self.validate_called = False
                self.perform_calculation_called = False
                
            def _validate_data_sources(self, data_sources):
                self.validate_called = True
                # Skip actual validation
                
            def _perform_calculation(self, data_sources):
                self.perform_calculation_called = True
                return pd.DataFrame({'result': [1, 2, 3]})
        
        # Create calculator and test data
        calculator = TestCalculator()
        data_sources = {'source1': pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})}
        
        # Call calculate and check that both methods were called
        result = calculator.calculate(data_sources)
        
        assert calculator.validate_called
        assert calculator.perform_calculation_called
        assert list(result['result']) == [1, 2, 3]
    
    def test_validate_data_sources_missing_source(self):
        """Test validation when a required data source is missing."""
        calculator = MetricCalculator(
            name='test_metric',
            dependencies={'source1': {'col1', 'col2'}}
        )
        data_sources = {}
        
        with pytest.raises(ValueError) as excinfo:
            calculator._validate_data_sources(data_sources)
        
        assert "Missing sources: source1" in str(excinfo.value)
    
    def test_validate_data_sources_missing_columns(self):
        """Test validation when required columns are missing."""
        calculator = MetricCalculator(
            name='test_metric',
            dependencies={'source1': {'col1', 'col2', 'col3'}}
        )
        data_sources = {
            'source1': pd.DataFrame({'col1': [1, 2, 3]})
        }
        
        with pytest.raises(ValueError) as excinfo:
            calculator._validate_data_sources(data_sources)
        
        assert "Missing columns in source1:" in str(excinfo.value)
        assert "col2" in str(excinfo.value)
        assert "col3" in str(excinfo.value)
    
    def test_validate_data_sources_success(self):
        """Test successful validation of data sources."""
        calculator = MetricCalculator(
            name='test_metric',
            dependencies={'source1': {'col1', 'col2'}}
        )
        data_sources = {
            'source1': pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        }
        
        # Should not raise any exception
        calculator._validate_data_sources(data_sources)
    
    def test_validate_data_sources_with_index(self):
        """Test validation when required columns are in the index."""
        calculator = MetricCalculator(
            name='test_metric',
            dependencies={'source1': {'symbol', 'date', 'col1'}}
        )
        
        # Create DataFrame with multi-index
        df = pd.DataFrame({'col1': [1, 2, 3]})
        df['symbol'] = ['A', 'B', 'C']
        df['date'] = pd.to_datetime(['2022-01-01', '2022-01-02', '2022-01-03'])
        df = df.set_index(['symbol', 'date'])
        
        data_sources = {'source1': df}
        
        # Should not raise any exception
        calculator._validate_data_sources(data_sources)
    
    def test_register(self):
        """Test registering a calculator in the global registry."""
        # Clear registry first
        METRIC_REGISTRY.clear()
        
        calculator = MetricCalculator(
            name='test_metric',
            dependencies={'source1': {'col1', 'col2'}}
        )
        
        # Register the calculator
        calculator.register()
        
        assert 'test_metric' in METRIC_REGISTRY
        assert METRIC_REGISTRY['test_metric'] is calculator


class TestUtilityFunctions:
    """Tests for utility functions in the metrics module."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Clear registry before each test
        METRIC_REGISTRY.clear()
        
        # Create test calculators
        class TestCalculator1(MetricCalculator):
            def __init__(self):
                super().__init__(
                    name='metric1',
                    dependencies={'source1': {'col1', 'col2'}}
                )
            
            def _perform_calculation(self, data_sources):
                return pd.DataFrame({'metric1': [1, 2, 3]})
        
        class TestCalculator2(MetricCalculator):
            def __init__(self):
                super().__init__(
                    name='metric2',
                    dependencies={'source1': {'col2', 'col3'}, 'source2': {'col4'}}
                )
            
            def _perform_calculation(self, data_sources):
                return pd.DataFrame({'metric2': [4, 5, 6]})
        
        # Register calculators
        self.calc1 = TestCalculator1().register()
        self.calc2 = TestCalculator2().register()
    
    def test_get_dependencies(self):
        """Test getting combined dependencies for multiple metrics."""
        deps = get_dependencies(['metric1', 'metric2'])
        
        assert 'source1' in deps
        assert 'source2' in deps
        assert deps['source1'] == {'col1', 'col2', 'col3'}
        assert deps['source2'] == {'col4'}
    
    def test_get_dependencies_unknown_metric(self):
        """Test getting dependencies for an unknown metric."""
        with pytest.raises(ValueError) as excinfo:
            get_dependencies(['metric1', 'unknown_metric'])
        
        assert "Metric 'unknown_metric' not found in registry" in str(excinfo.value)
    
    def test_calculate_metrics(self):
        """Test calculating multiple metrics."""
        # Create test data sources
        data_sources = {
            'source1': pd.DataFrame({
                'col1': [1, 2, 3],
                'col2': [4, 5, 6],
                'col3': [7, 8, 9]
            }),
            'source2': pd.DataFrame({
                'col4': [10, 11, 12]
            })
        }
        
        # Calculate metrics
        results = calculate_metrics(['metric1', 'metric2'], data_sources)
        
        assert 'metric1' in results
        assert 'metric2' in results
        assert list(results['metric1']['metric1']) == [1, 2, 3]
        assert list(results['metric2']['metric2']) == [4, 5, 6]
    
    def test_calculate_metrics_unknown_metric(self):
        """Test calculating an unknown metric."""
        with pytest.raises(ValueError) as excinfo:
            calculate_metrics(['metric1', 'unknown_metric'], {'source1': pd.DataFrame({
                'col1': [1, 2, 3],
                'col2': [4, 5, 6]
            })})
        
        assert "Unknown metric: unknown_metric" in str(excinfo.value)


class TestQuarterlyMetricCalculator:
    """Tests for the QuarterlyMetricCalculator class."""
    
    def test_prepare_data_for_query(self):
        """Test the _prepare_data_for_query method for efficient querying."""
        calculator = QuarterlyMetricCalculator('dummy_calculator', {})
        
        # Create test data
        statements = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'MSFT', 'MSFT'],
            'date': pd.to_datetime(['2022-12-31', '2022-09-30', '2022-12-31', '2022-09-30']),
            'filing_date': pd.to_datetime(['2023-01-15', '2022-10-15', '2023-01-20', '2022-10-20']),
            'value': [100, 200, 600, 700]
        })
        
        # Prepare data for querying
        prepared_data = calculator._prepare_data_for_query(statements)
        
        # Check that the data is properly indexed
        assert isinstance(prepared_data.index, pd.MultiIndex)
        assert prepared_data.index.names == ['symbol', 'filing_date']
        
        # Check that we can efficiently query by symbol and filing_date
        aapl_data = prepared_data.loc['AAPL']
        assert len(aapl_data) == 2
        assert aapl_data.loc[pd.Timestamp('2023-01-15')]['value'] == 100
        
        # Check that the index is sorted
        assert prepared_data.index.values.tolist() == sorted(prepared_data.index.values)
    
    def test_fill_date_gaps(self):
        """Test the _fill_date_gaps method for continuous daily data."""      
        calculator = QuarterlyMetricCalculator('dummy_calculator', {})
        
        # Create test data with effective dates
        df = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'MSFT', 'MSFT'],
            'first_effective_date': pd.to_datetime(['2023-01-02', '2023-01-11', '2023-01-06', '2023-01-16']),
            'value': [100, 200, 300, 400]
        })
        
        # Fill date gaps
        filled_df = calculator._fill_date_gaps(df)
        
        # Check that gaps are filled
        aapl_data = filled_df[filled_df['symbol'] == 'AAPL'].sort_values('date')
        assert len(aapl_data) == 10  # Should have more rows than original
        
        # Check that we have continuous daily data
        date_diffs = np.diff([d.timestamp() for d in aapl_data['date']])
        assert all(date_diffs == 86400)  # All differences should be 1 day (86400 seconds)
        
        # Check that data is only available from effective date onwards
        # For dates between Jan 2 and Jan 10, we should have value=100
        jan5_data = aapl_data[aapl_data['date'] == pd.Timestamp('2023-01-05')].iloc[0]
        assert jan5_data['value'] == 100
        assert jan5_data['first_effective_date'] == pd.Timestamp('2023-01-02')
        
        # For dates from Jan 11 onwards, we should have value=200
        jan12_data = aapl_data[aapl_data['date'] == pd.Timestamp('2023-01-11')].iloc[0]
        assert jan12_data['value'] == 200
        assert jan12_data['first_effective_date'] == pd.Timestamp('2023-01-11')
        
        # Check that we don't have data before the first effective date
        assert not any(aapl_data['date'] < pd.Timestamp('2023-01-02'))
        
        # Check MSFT data as well
        msft_data = filled_df[filled_df['symbol'] == 'MSFT'].sort_values('date')
        assert not any(msft_data['date'] < pd.Timestamp('2023-01-06'))
        assert all(msft_data[msft_data['date'] < pd.Timestamp('2023-01-16')]['value'] == 300)
        
    def test_fill_date_gaps_empty_df(self):
        calculator = QuarterlyMetricCalculator('dummy_calculator', {})
        
        empty_df = pd.DataFrame(columns=['symbol', 'first_effective_date', 'value'])
        filled_df = calculator._fill_date_gaps(empty_df)
        
        assert filled_df.empty
        # The result should have both the original columns plus the new 'date' column
        assert set(filled_df.columns) == {'symbol', 'first_effective_date', 'value'}


class TestDailyMetricCalculator:
    """Tests for the DailyMetricCalculator class."""
    pass