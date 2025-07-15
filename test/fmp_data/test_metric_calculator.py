"""
Tests for the metric_calculator module.

This module tests the metric calculation functions in the metric_calculator module.
"""

import pytest
import pandas as pd
import numpy as np
from fmp_data.metric_calculator import *


class TestRatioMetric:
    """Tests for the ratio_metric function which is the core calculation function."""
    
    def test_basic_calculation(self):
        """Test basic calculation of ratio metric."""
        # Create test data with 5 quarters in ascending order (oldest to newest)
        data = pd.DataFrame({
            'date': pd.to_datetime(['2021-12-31', '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31']),
            'filing_date': pd.to_datetime(['2022-01-15', '2022-04-15', '2022-07-15', '2022-10-15', '2023-01-15']),
            'numerator': [10.0, 20.0, 30.0, 40.0, 50.0],
            'denominator': [100.0, 200.0, 300.0, 400.0, 500.0]
        })
        
        # Calculate ratio with window=4
        result = ratio_metric(data, 'numerator', 'denominator', 4)
        
        # Expected values for the last two rows (which have 4 quarters of data)
        expected_3 = (10.0 + 20.0 + 30.0 + 40.0) / (100.0 + 200.0 + 300.0 + 400.0)
        expected_4 = (20.0 + 30.0 + 40.0 + 50.0) / (200.0 + 300.0 + 400.0 + 500.0)
        
        # Check results
        assert len(result) == 5
        assert pytest.approx(result.iloc[3]) == expected_3
        assert pytest.approx(result.iloc[4]) == expected_4
        
        # First 3 rows should be NaN (not enough data for window=4)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])
    
    def test_different_window_sizes(self):
        """Test ratio metric with different window sizes."""
        # Create test data with 5 quarters
        data = pd.DataFrame({
            'date': pd.to_datetime(['2021-12-31', '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31']),
            'filing_date': pd.to_datetime(['2022-01-15', '2022-04-15', '2022-07-15', '2022-10-15', '2023-01-15']),
            'numerator': [10.0, 20.0, 30.0, 40.0, 50.0],
            'denominator': [100.0, 200.0, 300.0, 400.0, 500.0]
        })
        
        # Test with window=2
        result_window_2 = ratio_metric(data, 'numerator', 'denominator', 2)
        
        # Expected values for window=2
        expected_1 = (10.0 + 20.0) / (100.0 + 200.0)
        expected_2 = (20.0 + 30.0) / (200.0 + 300.0)
        expected_3 = (30.0 + 40.0) / (300.0 + 400.0)
        expected_4 = (40.0 + 50.0) / (400.0 + 500.0)
        
        # Check results for window=2
        assert len(result_window_2) == 5
        assert pytest.approx(result_window_2.iloc[1]) == expected_1
        assert pytest.approx(result_window_2.iloc[2]) == expected_2
        assert pytest.approx(result_window_2.iloc[3]) == expected_3
        assert pytest.approx(result_window_2.iloc[4]) == expected_4
        assert np.isnan(result_window_2.iloc[0])  # First row should be NaN
        
        # Test with window=3
        result_window_3 = ratio_metric(data, 'numerator', 'denominator', 3)
        
        # Expected values for window=3
        expected_2 = (10.0 + 20.0 + 30.0) / (100.0 + 200.0 + 300.0)
        expected_3 = (20.0 + 30.0 + 40.0) / (200.0 + 300.0 + 400.0)
        expected_4 = (30.0 + 40.0 + 50.0) / (300.0 + 400.0 + 500.0)
        
        # Check results for window=3
        assert pytest.approx(result_window_3.iloc[2]) == expected_2
        assert pytest.approx(result_window_3.iloc[3]) == expected_3
        assert pytest.approx(result_window_3.iloc[4]) == expected_4
        assert np.isnan(result_window_3.iloc[0])  # First two rows should be NaN
        assert np.isnan(result_window_3.iloc[1])
    
    def test_zero_denominator(self):
        """Test handling of zero denominator."""
        # Create test data with all zero denominators
        data = pd.DataFrame({
            'date': pd.to_datetime(['2021-12-31', '2022-03-31', '2022-06-30', '2022-09-30']),
            'filing_date': pd.to_datetime(['2022-01-15', '2022-04-15', '2022-07-15', '2022-10-15']),
            'numerator': [10.0, 20.0, 30.0, 40.0],
            'denominator': [0.0, 0.0, 0.0, 0.0]  # All zero denominators
        })
        
        # Calculate ratio with window=4
        result = ratio_metric(data, 'numerator', 'denominator', 4)
        
        # Should be NaN due to division by zero
        assert np.isnan(result.iloc[3])
        
        # Test with some zero denominators that sum to zero
        data = pd.DataFrame({
            'date': pd.to_datetime(['2021-12-31', '2022-03-31', '2022-06-30', '2022-09-30']),
            'filing_date': pd.to_datetime(['2022-01-15', '2022-04-15', '2022-07-15', '2022-10-15']),
            'numerator': [10.0, 20.0, 30.0, 40.0],
            'denominator': [10.0, -10.0, 5.0, -5.0]  # Sum equals zero
        })
        
        # Calculate ratio with window=4
        result = ratio_metric(data, 'numerator', 'denominator', 4)
        
        # Should be NaN due to division by zero
        assert np.isnan(result.iloc[3])
    
    def test_negative_values(self):
        """Test ratio metric with negative values."""
        # Create test data with negative values
        data = pd.DataFrame({
            'date': pd.to_datetime(['2021-12-31', '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31']),
            'filing_date': pd.to_datetime(['2022-01-15', '2022-04-15', '2022-07-15', '2022-10-15', '2023-01-15']),
            'numerator': [-10.0, 20.0, -30.0, 40.0, -50.0],  # Mixed positive and negative
            'denominator': [100.0, -200.0, 300.0, -400.0, 500.0]  # Mixed positive and negative
        })
        
        # Calculate ratio with window=4
        result = ratio_metric(data, 'numerator', 'denominator', 4)
        
        # Expected values
        expected_3 = (-10.0 + 20.0 + -30.0 + 40.0) / (100.0 + -200.0 + 300.0 + -400.0)
        expected_4 = (20.0 + -30.0 + 40.0 + -50.0) / (-200.0 + 300.0 + -400.0 + 500.0)
        
        # Check results
        assert pytest.approx(result.iloc[3]) == expected_3
        assert pytest.approx(result.iloc[4]) == expected_4


class TestMetricCalculator:
    """Tests for all metric calculation functions in metric_calculator module."""
    
    def test_gross_margin_ttm(self):
        """Test calculation of gross margin TTM."""
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
    
    def test_operating_margin_ttm(self):
        """Test calculation of operating margin TTM."""
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
    
    def test_sga_margin_ttm(self):
        """Test calculation of SGA margin TTM."""
        # Create test data with 5 quarters in ascending order (oldest to newest)
        data = pd.DataFrame({
            'date': pd.to_datetime(['2021-12-31', '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31']),
            'filing_date': pd.to_datetime(['2022-01-15', '2022-04-15', '2022-07-15', '2022-10-15', '2023-01-15']),
            'selling_general_and_administrative_expenses': [25.0, 20.0, 22.0, 24.0, 26.0],
            'revenue': [95.0, 70.0, 80.0, 90.0, 100.0]
        })
        
        # Calculate SGA margin TTM
        result = sga_margin_ttm(data)
        
        # Expected values for the last two rows (which have 4 quarters of data)
        expected_3 = (25.0 + 20.0 + 22.0 + 24.0) / (95.0 + 70.0 + 80.0 + 90.0)
        expected_4 = (20.0 + 22.0 + 24.0 + 26.0) / (70.0 + 80.0 + 90.0 + 100.0)
        
        # Check results
        assert len(result) == 5
        assert pytest.approx(result.iloc[3]) == expected_3
        assert pytest.approx(result.iloc[4]) == expected_4
        
        # First 3 rows should be NaN (not enough data for TTM)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])

    def test_rd_margin_ttm(self):
        """Test calculation of R&D margin TTM."""
        # Create test data with 5 quarters in ascending order (oldest to newest)
        data = pd.DataFrame({
            'date': pd.to_datetime(['2021-12-31', '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31']),
            'filing_date': pd.to_datetime(['2022-01-15', '2022-04-15', '2022-07-15', '2022-10-15', '2023-01-15']),
            'research_and_development_expenses': [25.0, 20.0, 22.0, 24.0, 26.0],
            'revenue': [95.0, 70.0, 80.0, 90.0, 100.0]
        })
        
        # Calculate R&D margin TTM
        result = rd_margin_ttm(data)
        
        # Expected values for the last two rows (which have 4 quarters of data)
        expected_3 = (25.0 + 20.0 + 22.0 + 24.0) / (95.0 + 70.0 + 80.0 + 90.0)
        expected_4 = (20.0 + 22.0 + 24.0 + 26.0) / (70.0 + 80.0 + 90.0 + 100.0)
        
        # Check results
        assert len(result) == 5
        assert pytest.approx(result.iloc[3]) == expected_3
        assert pytest.approx(result.iloc[4]) == expected_4
        
        # First 3 rows should be NaN (not enough data for TTM)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])
    
    def test_net_earning_yoy_growth(self):
        """Test calculation of net earning YOY growth."""
        # Create test data with 8 quarters (2 years) in ascending order
        data = pd.DataFrame({
            'date': pd.to_datetime(['2021-03-31', '2021-06-30', '2021-09-30', '2021-12-31', 
                                  '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31']),
            'filing_date': pd.to_datetime(['2021-04-15', '2021-07-15', '2021-10-15', '2022-01-15', 
                                         '2022-04-15', '2022-07-15', '2022-10-15', '2023-01-15']),
            'net_income': [100.0, 120.0, 90.0, 150.0, 110.0, 150.0, 100.0, 180.0]
        })
        
        # Calculate net earning YOY growth
        result = net_earning_yoy_growth(data)
        
        # Expected values for the second year (which have year-ago data)
        # Q1 2022 vs Q1 2021: (110 - 100) / 100 = 0.10
        expected_4 = (110.0 - 100.0) / abs(100.0)
        # Q2 2022 vs Q2 2021: (150 - 120) / 120 = 0.25
        expected_5 = (150.0 - 120.0) / abs(120.0)
        # Q3 2022 vs Q3 2021: (100 - 90) / 90 = 0.111...
        expected_6 = (100.0 - 90.0) / abs(90.0)
        # Q4 2022 vs Q4 2021: (180 - 150) / 150 = 0.20
        expected_7 = (180.0 - 150.0) / abs(150.0)
        
        # Check results
        assert len(result) == 8
        assert pytest.approx(result.iloc[4]) == expected_4
        assert pytest.approx(result.iloc[5]) == expected_5
        assert pytest.approx(result.iloc[6]) == expected_6
        assert pytest.approx(result.iloc[7]) == expected_7
        
        # First 4 rows should be NaN (no year-ago data)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])
        assert np.isnan(result.iloc[3])
    
    def test_operating_income_yoy_growth(self):
        """Test calculation of operating income YOY growth."""
        # Create test data with 8 quarters (2 years) in ascending order
        data = pd.DataFrame({
            'date': pd.to_datetime(['2021-03-31', '2021-06-30', '2021-09-30', '2021-12-31', 
                                  '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31']),
            'filing_date': pd.to_datetime(['2021-04-15', '2021-07-15', '2021-10-15', '2022-01-15', 
                                         '2022-04-15', '2022-07-15', '2022-10-15', '2023-01-15']),
            'operating_income': [200.0, 220.0, 190.0, 250.0, 230.0, 270.0, 200.0, 300.0]
        })
        
        # Calculate operating income YOY growth
        result = operating_income_yoy_growth(data)
        
        # Expected values for the second year (which have year-ago data)
        # Q1 2022 vs Q1 2021: (230 - 200) / 200 = 0.15
        expected_4 = (230.0 - 200.0) / abs(200.0)
        # Q2 2022 vs Q2 2021: (270 - 220) / 220 = 0.227...
        expected_5 = (270.0 - 220.0) / abs(220.0)
        # Q3 2022 vs Q3 2021: (200 - 190) / 190 = 0.0526...
        expected_6 = (200.0 - 190.0) / abs(190.0)
        # Q4 2022 vs Q4 2021: (300 - 250) / 250 = 0.20
        expected_7 = (300.0 - 250.0) / abs(250.0)
        
        # Check results
        assert len(result) == 8
        assert pytest.approx(result.iloc[4]) == expected_4
        assert pytest.approx(result.iloc[5]) == expected_5
        assert pytest.approx(result.iloc[6]) == expected_6
        assert pytest.approx(result.iloc[7]) == expected_7
        
        # First 4 rows should be NaN (no year-ago data)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])
        assert np.isnan(result.iloc[3])
    
    def test_yoy_growth_with_negative_values(self):
        """Test YOY growth calculations with negative values."""
        # Create test data with 8 quarters (2 years) in ascending order
        data = pd.DataFrame({
            'date': pd.to_datetime(['2021-03-31', '2021-06-30', '2021-09-30', '2021-12-31', 
                                  '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31']),
            'filing_date': pd.to_datetime(['2021-04-15', '2021-07-15', '2021-10-15', '2022-01-15', 
                                         '2022-04-15', '2022-07-15', '2022-10-15', '2023-01-15']),
            'net_income': [-100.0, 120.0, -90.0, 150.0, -50.0, 150.0, -45.0, 180.0],
            'operating_income': [200.0, -220.0, 190.0, -250.0, 230.0, -110.0, 200.0, -125.0]
        })
        
        # Calculate net earning YOY growth
        net_income_result = net_earning_yoy_growth(data)
        
        # Expected values for net_income
        # Q1 2022 vs Q1 2021: (-50 - (-100)) / |-100| = 0.50
        expected_net_4 = (-50.0 - (-100.0)) / abs(-100.0)
        # Q3 2022 vs Q3 2021: (-45 - (-90)) / |-90| = 0.50
        expected_net_6 = (-45.0 - (-90.0)) / abs(-90.0)
        
        # Check results for net_income
        assert pytest.approx(net_income_result.iloc[4]) == expected_net_4
        assert pytest.approx(net_income_result.iloc[6]) == expected_net_6
        
        # Calculate operating income YOY growth
        op_income_result = operating_income_yoy_growth(data)
        
        # Expected values for operating_income
        # Q2 2022 vs Q2 2021: (-110 - (-220)) / |-220| = 0.50
        expected_op_5 = (-110.0 - (-220.0)) / abs(-220.0)
        # Q4 2022 vs Q4 2021: (-125 - (-250)) / |-250| = 0.50
        expected_op_7 = (-125.0 - (-250.0)) / abs(-250.0)
        
        # Check results for operating_income
        assert pytest.approx(op_income_result.iloc[5]) == expected_op_5
        assert pytest.approx(op_income_result.iloc[7]) == expected_op_7
    
    def test_long_term_debt_to_ttm_operating_income(self):
        """Test calculation of long term debt to TTM operating income ratio."""
        # Create test data with 5 quarters in ascending order (oldest to newest)
        data = pd.DataFrame({
            'date': pd.to_datetime(['2021-12-31', '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31']),
            'filing_date': pd.to_datetime(['2022-01-15', '2022-04-15', '2022-07-15', '2022-10-15', '2023-01-15']),
            'long_term_debt': [1000.0, 1100.0, 1050.0, 950.0, 900.0],
            'operating_income': [100.0, 120.0, 130.0, 140.0, 150.0]
        })
        
        # Calculate long term debt to TTM operating income
        result = long_term_debt_to_ttm_operating_income(data)
        
        # Expected values for the last two rows (which have 4 quarters of data)
        # For index 3: long_term_debt = 950.0, operating_income_ttm = 100.0 + 120.0 + 130.0 + 140.0 = 490.0
        # ratio = 950.0 / 490.0 = 1.9387...
        expected_3 = 950.0 / (100.0 + 120.0 + 130.0 + 140.0)
        
        # For index 4: long_term_debt = 900.0, operating_income_ttm = 120.0 + 130.0 + 140.0 + 150.0 = 540.0
        # ratio = 900.0 / 540.0 = 1.6666...
        expected_4 = 900.0 / (120.0 + 130.0 + 140.0 + 150.0)
        
        # Check results
        assert len(result) == 5
        assert pytest.approx(result.iloc[3]) == expected_3
        assert pytest.approx(result.iloc[4]) == expected_4
        
        # First 3 rows should be NaN (not enough data for TTM)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])
    
    def test_roe_ttm(self):
        """Test calculation of Return on Equity (ROE) TTM."""
        # Create test data with 5 quarters in ascending order (oldest to newest)
        data = pd.DataFrame({
            'date': pd.to_datetime(['2021-12-31', '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31']),
            'filing_date': pd.to_datetime(['2022-01-15', '2022-04-15', '2022-07-15', '2022-10-15', '2023-01-15']),
            'net_income': [50.0, 60.0, 70.0, 80.0, 90.0],
            'total_equity': [500.0, 520.0, 550.0, 580.0, 600.0]
        })
        
        # Calculate ROE TTM
        result = roe_ttm(data)
        
        # Expected values for the last two rows (which have 4 quarters of data)
        # For index 3: net_income_ttm = 50.0 + 60.0 + 70.0 + 80.0 = 260.0, total_equity = 580.0
        # roe = 260.0 / 580.0 = 0.4483...
        expected_3 = (50.0 + 60.0 + 70.0 + 80.0) / 580.0
        
        # For index 4: net_income_ttm = 60.0 + 70.0 + 80.0 + 90.0 = 300.0, total_equity = 600.0
        # roe = 300.0 / 600.0 = 0.5
        expected_4 = (60.0 + 70.0 + 80.0 + 90.0) / 600.0
        
        # Check results
        assert len(result) == 5
        assert pytest.approx(result.iloc[3]) == expected_3
        assert pytest.approx(result.iloc[4]) == expected_4
        
        # First 3 rows should be NaN (not enough data for TTM)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])
    
    def test_roe_ttm_with_zero_equity(self):
        """Test ROE TTM calculation with zero equity values."""
        # Create test data with zero equity values
        data = pd.DataFrame({
            'date': pd.to_datetime(['2021-12-31', '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31']),
            'filing_date': pd.to_datetime(['2022-01-15', '2022-04-15', '2022-07-15', '2022-10-15', '2023-01-15']),
            'net_income': [50.0, 60.0, 70.0, 80.0, 90.0],
            'total_equity': [500.0, 520.0, 550.0, 0.0, 0.0]  # Zero equity for the last two quarters
        })
        
        # Calculate ROE TTM
        result = roe_ttm(data)
        
        # Should be NaN due to division by zero
        assert np.isnan(result.iloc[3])
        assert np.isnan(result.iloc[4])
        
        # Test with negative equity
        data = pd.DataFrame({
            'date': pd.to_datetime(['2021-12-31', '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31']),
            'filing_date': pd.to_datetime(['2022-01-15', '2022-04-15', '2022-07-15', '2022-10-15', '2023-01-15']),
            'net_income': [50.0, 60.0, 70.0, 80.0, 90.0],
            'total_equity': [500.0, 520.0, 550.0, -100.0, -200.0]  # Negative equity for the last two quarters
        })
        
        # Calculate ROE TTM
        result = roe_ttm(data)
        
        # Expected values for negative equity
        # For index 3: net_income_ttm = 50.0 + 60.0 + 70.0 + 80.0 = 260.0, total_equity = -100.0
        # roe = 260.0 / -100.0 = -2.6
        expected_3 = (50.0 + 60.0 + 70.0 + 80.0) / -100.0
        
        # For index 4: net_income_ttm = 60.0 + 70.0 + 80.0 + 90.0 = 300.0, total_equity = -200.0
        # roe = 300.0 / -200.0 = -1.5
        expected_4 = (60.0 + 70.0 + 80.0 + 90.0) / -200.0
        
        # Check results for negative equity
        assert pytest.approx(result.iloc[3]) == expected_3
        assert pytest.approx(result.iloc[4]) == expected_4
        
    def test_debt_to_equity(self):
        """Test calculation of debt to equity ratio."""
        # Create test data
        data = pd.DataFrame({
            'date': pd.to_datetime(['2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31', '2023-03-31']),
            'filing_date': pd.to_datetime(['2022-04-15', '2022-07-15', '2022-10-15', '2023-01-15', '2023-04-15']),
            'total_liabilities': [500.0, 550.0, 600.0, 650.0, 700.0],
            'total_equity': [1000.0, 1100.0, 1200.0, 1300.0, 1400.0]
        })
        
        # Calculate debt to equity ratio
        result = debt_to_equity(data)
        
        # Expected values
        # For index 0: total_liabilities = 500.0, total_equity = 1000.0
        # debt_to_equity = 500.0 / 1000.0 = 0.5
        expected_0 = 500.0 / 1000.0
        
        # For index 4: total_liabilities = 700.0, total_equity = 1400.0
        # debt_to_equity = 700.0 / 1400.0 = 0.5
        expected_4 = 700.0 / 1400.0
        
        # Check results
        assert len(result) == 5
        assert pytest.approx(result.iloc[0]) == expected_0
        assert pytest.approx(result.iloc[4]) == expected_4
    
    def test_eps_ttm(self):
        """Test calculation of Earnings Per Share (EPS) TTM."""
        # Create test data with 5 quarters in ascending order (oldest to newest)
        data = pd.DataFrame({
            'date': pd.to_datetime(['2021-12-31', '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31']),
            'filing_date': pd.to_datetime(['2022-01-15', '2022-04-15', '2022-07-15', '2022-10-15', '2023-01-15']),
            'net_income': [1000000.0, 1200000.0, 1300000.0, 1400000.0, 1500000.0],
            'weighted_average_shares_outstanding_diluted': [10000000.0, 10100000.0, 10200000.0, 10300000.0, 10400000.0]
        })
        
        # Calculate EPS TTM
        result = eps_ttm(data)
        
        # Expected values for the last two rows (which have 4 quarters of data)
        # For index 3: net_income_ttm = 1000000.0 + 1200000.0 + 1300000.0 + 1400000.0 = 4900000.0
        # weighted_average_shares_outstanding_diluted = 10300000.0
        # eps = 4900000.0 / 10300000.0 = 0.4757...
        expected_3 = (1000000.0 + 1200000.0 + 1300000.0 + 1400000.0) / 10300000.0
        
        # For index 4: net_income_ttm = 1200000.0 + 1300000.0 + 1400000.0 + 1500000.0 = 5400000.0
        # weighted_average_shares_outstanding_diluted = 10400000.0
        # eps = 5400000.0 / 10400000.0 = 0.5192...
        expected_4 = (1200000.0 + 1300000.0 + 1400000.0 + 1500000.0) / 10400000.0
        
        # Check results
        assert len(result) == 5
        assert pytest.approx(result.iloc[3]) == expected_3
        assert pytest.approx(result.iloc[4]) == expected_4
        
        # First 3 rows should be NaN (not enough data for TTM)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])
    
    def test_book_value_per_share(self):
        """Test calculation of book value per share."""
        # Create test data with small, easy to verify numbers
        data = pd.DataFrame({
            'date': pd.to_datetime(['2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31', '2023-03-31']),
            'filing_date': pd.to_datetime(['2022-04-15', '2022-07-15', '2022-10-15', '2023-01-15', '2023-04-15']),
            'total_assets': [100.0, 110.0, 120.0, 130.0, 140.0],
            'total_liabilities': [40.0, 42.0, 44.0, 46.0, 48.0],
            'goodwill_and_intangible_assets': [10.0, 11.0, 12.0, 13.0, 14.0],
            'weighted_average_shares_outstanding_diluted': [10.0, 11.0, 12.0, 13.0, 14.0]  # Different share counts
        })
        
        # Calculate book value per share
        result = book_value_per_share(data)
        
        # Expected values for each row
        # For index 0: (100.0 - 40.0 - 10.0) / 10.0 = 5.0
        expected_0 = (100.0 - 40.0 - 10.0) / 10.0
        
        # For index 1: (110.0 - 42.0 - 11.0) / 11.0 = 5.18...
        expected_1 = (110.0 - 42.0 - 11.0) / 11.0
        
        # For index 2: (120.0 - 44.0 - 12.0) / 12.0 = 5.33...
        expected_2 = (120.0 - 44.0 - 12.0) / 12.0
        
        # For index 3: (130.0 - 46.0 - 13.0) / 13.0 = 5.46...
        expected_3 = (130.0 - 46.0 - 13.0) / 13.0
        
        # For index 4: (140.0 - 48.0 - 14.0) / 14.0 = 5.57...
        expected_4 = (140.0 - 48.0 - 14.0) / 14.0
        
        # Check results for all rows
        assert len(result) == 5
        assert pytest.approx(result.iloc[0]) == expected_0
        assert pytest.approx(result.iloc[1]) == expected_1
        assert pytest.approx(result.iloc[2]) == expected_2
        assert pytest.approx(result.iloc[3]) == expected_3
        assert pytest.approx(result.iloc[4]) == expected_4
        
    def test_fcf_per_share_ttm(self):
        """Test calculation of free cash flow per share TTM with changing share counts."""
        # Create test data with changing share counts
        data = pd.DataFrame({
            'date': pd.to_datetime(['2021-12-31', '2022-03-31', '2022-06-30', '2022-09-30', '2022-12-31']),
            'filing_date': pd.to_datetime(['2022-01-15', '2022-04-15', '2022-07-15', '2022-10-15', '2023-01-15']),
            'free_cash_flow': [100.0, 120.0, 140.0, 160.0, 180.0],
            'weighted_average_shares_outstanding_diluted': [40.0, 45.0, 50.0, 55.0, 60.0]  # Increasing share count
        })
        
        # Calculate FCF per share TTM
        result = fcf_per_share_ttm(data)
        
        # Expected values for the last two rows
        # For index 3: free_cash_flow_ttm = 100.0 + 120.0 + 140.0 + 160.0 = 520.0
        # weighted_average_shares_outstanding_diluted = 55.0
        # fcf_per_share = 520.0 / 55.0 = 9.45...
        expected_3 = (100.0 + 120.0 + 140.0 + 160.0) / 55.0
        
        # For index 4: free_cash_flow_ttm = 120.0 + 140.0 + 160.0 + 180.0 = 600.0
        # weighted_average_shares_outstanding_diluted = 60.0
        # fcf_per_share = 600.0 / 60.0 = 10.0
        expected_4 = (120.0 + 140.0 + 160.0 + 180.0) / 60.0
        
        # Check results
        assert pytest.approx(result.iloc[3]) == expected_3
        assert pytest.approx(result.iloc[4]) == expected_4

        # First 3 rows should be NaN (not enough data for TTM)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[1])
        assert np.isnan(result.iloc[2])        
        



