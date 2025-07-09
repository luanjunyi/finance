"""
Tests for the metric_calculator module.

This module tests the metric calculation functions in the metric_calculator module.
"""

import pytest
import pandas as pd
import numpy as np
from fmp_data.metric_calculator import gross_margin_ttm, operating_margin_ttm, eps_ttm


class TestGrossMarginTTM:
    """Tests for the gross_margin_ttm function."""
    
    def test_basic_calculation(self):
        """Test basic calculation of gross margin TTM."""
        # Create test data with 5 quarters in ascending order (oldest to newest)
        data = pd.DataFrame({
            'date': pd.to_datetime(['2021-12-31', '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31']),
            'filing_date': pd.to_datetime(['2022-01-15', '2022-04-15', '2022-07-15', '2022-10-15', '2023-01-15']),
            'revenue': [958.0, 70.0, 80.0, 90.0, 100.0],
            'cost_of_revenue': [57.0, 42.0, 48.0, 54.0, 60.0]
        })
        
        # Calculate gross margin TTM
        result = gross_margin_ttm(data)
        

        # Check results
        assert len(result) == 5
        assert pytest.approx(result.iloc[3]) == (958.0 + 70.0 + 80.0 + 90.0 - (57.0 + 42.0 + 48.0 + 54.0)) / (958.0 + 70.0 + 80.0 + 90.0)
        assert pytest.approx(result.iloc[4]) == (70.0 + 80.0 + 90.0 + 100.0 - (42.0 + 48.0 + 54.0 + 60.0)) / (70.0 + 80.0 + 90.0 + 100.0)
        
        # First 3 rows should be NaN (not enough data for TTM)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])
    
    def test_all_zero_revenue(self):
        """Test handling of all zero revenue quarters."""
        # Create test data with all zero revenue in ascending order
        data = pd.DataFrame({
            'date': pd.to_datetime(['2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31']),
            'filing_date': pd.to_datetime(['2022-04-15', '2022-07-15', '2022-10-15', '2023-01-15']),
            'revenue': [0.0, 0.0, 0.0, 0.0],  # All zero revenue
            'cost_of_revenue': [0.0, 0.0, 0.0, 0.0]  # All zero COGS
        })
        
        # Calculate gross margin TTM
        result = gross_margin_ttm(data)
        
        # Should be NaN due to division by zero
        assert np.isnan(result.iloc[3])
    
    def test_insufficient_data(self):
        """Test handling of insufficient data for TTM calculation."""
        # Create test data with only 3 quarters in ascending order
        data = pd.DataFrame({
            'date': pd.to_datetime(['2022-06-30', '2022-09-30', '2022-12-31']),
            'filing_date': pd.to_datetime(['2022-07-15', '2022-10-15', '2023-01-15']),
            'revenue': [80.0, 90.0, 100.0],
            'cost_of_revenue': [48.0, 54.0, 60.0]
        })
        
        # Calculate gross margin TTM
        result = gross_margin_ttm(data)
        
        # All rows should be NaN (not enough data for TTM)
        assert len(result) == 3
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])


class TestOperatingMarginTTM:
    """Tests for the operating_margin_ttm function."""
    
    def test_basic_calculation(self):
        """Test basic calculation of operating margin TTM."""
        # Create test data with 5 quarters in ascending order (oldest to newest)
        data = pd.DataFrame({
            'date': pd.to_datetime(['2021-12-31', '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31']),
            'filing_date': pd.to_datetime(['2022-01-15', '2022-04-15', '2022-07-15', '2022-10-15', '2023-01-15']),
            'operating_income': [35.0, 25.0, 30.0, 32.0, 38.0],
            'revenue': [95.0, 70.0, 80.0, 90.0, 100.0]
        })
        
        # Calculate operating margin TTM
        result = operating_margin_ttm(data)
        
        # Expected values for the last two rows (which have 4 quarters of data)
        expected_3 = (35.0 + 25.0 + 30.0 + 32.0) / (95.0 + 70.0 + 80.0 + 90.0)
        expected_4 = (25.0 + 30.0 + 32.0 + 38.0) / (70.0 + 80.0 + 90.0 + 100.0)
        
        # Check results
        assert len(result) == 5
        assert pytest.approx(result.iloc[3]) == expected_3
        assert pytest.approx(result.iloc[4]) == expected_4
        # First 3 rows should be NaN (not enough data for TTM)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])
    

class TestEpsTTM:
    """Tests for the eps_ttm function."""
    
    def test_basic_calculation(self):
        """Test basic calculation of EPS TTM."""
        # Create test data with 5 quarters in ascending order (oldest to newest)
        data = pd.DataFrame({
            'date': pd.to_datetime(['2021-12-31', '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31']),
            'filing_date': pd.to_datetime(['2022-01-15', '2022-04-15', '2022-07-15', '2022-10-15', '2023-01-15']),
            'net_income': [100.0, 120.0, 140.0, 160.0, 180.0],
            'weighted_average_shares_outstanding_diluted': [10.0, 20.0, 30.0, 40.0, 50.0]
        })
        
        # Calculate EPS TTM
        result = eps_ttm(data)
        
        # Expected values for the last two rows (which have 4 quarters of data)
        # Using the most recent shares outstanding (50.0)
        expected_3 = (100.0 + 120.0 + 140.0 + 160.0) / 40.0
        expected_4 = (120.0 + 140.0 + 160.0 + 180.0) / 50.0
        
        # Check results
        assert len(result) == 5
        assert pytest.approx(result.iloc[3]) == expected_3
        assert pytest.approx(result.iloc[4]) == expected_4
        
        # First 3 rows should be NaN (not enough data for TTM)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])
    
    def test_insufficient_data(self):
        """Test handling of insufficient data for TTM calculation."""
        # Create test data with only 3 quarters in ascending order
        data = pd.DataFrame({
            'date': pd.to_datetime(['2022-06-30', '2022-09-30', '2022-12-31']),
            'filing_date': pd.to_datetime(['2022-07-15', '2022-10-15', '2023-01-15']),
            'net_income': [140.0, 160.0, 180.0],
            'weighted_average_shares_outstanding_diluted': [50.0, 50.0, 50.0]
        })
        
        # Calculate EPS TTM
        result = eps_ttm(data)
        
        # All rows should be NaN (not enough data for TTM)
        assert len(result) == 3
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])
    
    def test_changing_shares_outstanding(self):
        """Test calculation with changing shares outstanding."""
        # Create test data with changing shares outstanding
        data = pd.DataFrame({
            'date': pd.to_datetime(['2021-12-31', '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31']),
            'filing_date': pd.to_datetime(['2022-01-15', '2022-04-15', '2022-07-15', '2022-10-15', '2023-01-15']),
            'net_income': [100.0, 120.0, 140.0, 160.0, 180.0],
            'weighted_average_shares_outstanding_diluted': [40.0, 45.0, 50.0, 55.0, 60.0]  # Increasing shares
        })
        
        # Calculate EPS TTM
        result = eps_ttm(data)
        
        # Expected values using the most recent shares outstanding (60.0)
        expected_3 = (100.0 + 120.0 + 140.0 + 160.0) / 55.0
        expected_4 = (120.0 + 140.0 + 160.0 + 180.0) / 60.0
        
        # Check results
        assert pytest.approx(result.iloc[3]) == expected_3
        assert pytest.approx(result.iloc[4]) == expected_4
