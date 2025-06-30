"""
Tests for the Price calculator.

This module tests the PriceCalculator class which is a simple pass-through
for daily price data.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from fmp_offline.metrics.daily.price import PriceCalculator


class TestPriceCalculator:
    """Tests for the PriceCalculator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = PriceCalculator()
        
        # Create test price data
        self.daily_prices = pd.DataFrame({
            'symbol': ['AAPL', 'AAPL', 'AAPL', 'MSFT', 'MSFT', 'MSFT'],
            'date': pd.to_datetime([
                '2023-01-15', '2023-01-14', '2023-01-13',
                '2023-01-15', '2023-01-14', '2023-01-13'
            ]),
            'close': [150.82, 151.73, 149.24, 240.22, 242.58, 238.51],
            'volume': [70000000, 68000000, 72000000, 25000000, 24000000, 26000000],
            'open': [149.48, 150.64, 148.32, 239.48, 241.22, 237.55],
            'high': [151.95, 152.87, 150.40, 242.36, 243.82, 239.96],
            'low': [148.75, 149.59, 147.85, 238.22, 240.15, 236.89]
        })
        
        self.data_sources = {
            'daily_prices': self.daily_prices
        }
    
    def test_initialization(self):
        """Test initialization of the calculator."""
        assert self.calculator.name == 'price'
        assert 'daily_prices' in self.calculator.dependencies
        required_columns = self.calculator.dependencies['daily_prices']
        assert 'symbol' in required_columns
        assert 'date' in required_columns
        assert 'close' in required_columns
    
    def test_calculation_basic(self):
        """Test basic calculation of Price (pass-through)."""
        result = self.calculator.calculate(self.data_sources)
        
        # Check that result has the expected columns
        assert 'symbol' in result.columns
        assert 'date' in result.columns
        assert 'price' in result.columns
        
        # Check that we have results for all rows
        assert len(result) == len(self.daily_prices)
        
        # Check that the price values match the close prices
        for i, row in result.iterrows():
            symbol = row['symbol']
            date = row['date']
            price = row['price']
            
            # Find the corresponding row in the input data
            original_row = self.daily_prices[
                (self.daily_prices['symbol'] == symbol) & 
                (self.daily_prices['date'] == date)
            ]
            
            assert len(original_row) == 1
            assert pytest.approx(price) == original_row.iloc[0]['close']
    
    def test_missing_close_prices(self):
        """Test handling of missing close prices."""
        # Create data with missing values
        prices_with_missing = pd.DataFrame({
            'symbol': ['GOOG', 'GOOG', 'GOOG'],
            'date': pd.to_datetime(['2023-01-15', '2023-01-14', '2023-01-13']),
            'close': [2230.55, np.nan, 2210.24],
            'volume': [1500000, 1400000, 1600000]
        })
        
        data_sources = {
            'daily_prices': prices_with_missing
        }
        
        result = self.calculator.calculate(data_sources)
        
        # Should have results for all rows, but NaN for the missing close price
        assert len(result) == 3
        
        # Check the row with missing close price
        missing_row = result[result['date'] == pd.Timestamp('2023-01-14')]
        assert len(missing_row) == 1
        assert np.isnan(missing_row.iloc[0]['price'])
        
        # Check the rows with valid close prices
        valid_rows = result[result['date'] != pd.Timestamp('2023-01-14')]
        assert len(valid_rows) == 2
        
        for i, row in valid_rows.iterrows():
            date = row['date']
            price = row['price']
            
            original_row = prices_with_missing[prices_with_missing['date'] == date]
            assert pytest.approx(price) == original_row.iloc[0]['close']
    
    def test_empty_data(self):
        """Test handling of empty price data."""
        # Create empty data
        empty_prices = pd.DataFrame({
            'symbol': [],
            'date': [],
            'close': [],
            'volume': []
        })
        
        data_sources = {
            'daily_prices': empty_prices
        }
        
        result = self.calculator.calculate(data_sources)
        
        # Should return an empty DataFrame with the expected columns
        assert len(result) == 0
        assert 'symbol' in result.columns
        assert 'date' in result.columns
        assert 'price' in result.columns
    
    def test_additional_columns_preserved(self):
        """Test that additional columns are preserved in the output."""
        # Create data with additional columns
        prices_with_extra = pd.DataFrame({
            'symbol': ['AMZN', 'AMZN'],
            'date': pd.to_datetime(['2023-01-15', '2023-01-14']),
            'close': [98.12, 97.25],
            'volume': [35000000, 34000000],
            'market_cap': [1000000000000, 990000000000],
            'sector': ['Technology', 'Technology']
        })
        
        data_sources = {
            'daily_prices': prices_with_extra
        }
        
        result = self.calculator.calculate(data_sources)
        
        # Should have results for all rows
        assert len(result) == 2
        
        # Check that the price values match the close prices
        for i, row in result.iterrows():
            symbol = row['symbol']
            date = row['date']
            price = row['price']
            
            original_row = prices_with_extra[
                (prices_with_extra['symbol'] == symbol) & 
                (prices_with_extra['date'] == date)
            ]
            
            assert pytest.approx(price) == original_row.iloc[0]['close']
            
        # The additional columns should not be in the result
        assert 'market_cap' not in result.columns
        assert 'sector' not in result.columns
