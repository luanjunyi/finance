"""
Daily price metric calculator.

This module provides a simple pass-through calculator for daily price data.
"""

import pandas as pd
from typing import Dict, Set
from ..metrics import DailyMetricCalculator


class PriceCalculator(DailyMetricCalculator):
    """Calculator for daily price data."""
    
    def __init__(self):
        """Initialize the price calculator."""
        super().__init__(
            name='price',
            dependencies={
                'daily_prices': {
                    'symbol', 'date', 'close'
                }
            },
            required_metrics=[]  # No dependency on other metrics
        )
    
    def _perform_calculation(self, data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Extract price data for each symbol and date.
        
        Args:
            data_sources: Dictionary with daily_prices DataFrame
            
        Returns:
            DataFrame with columns: symbol, date, price
        """
        daily_prices = data_sources['daily_prices']
        
        # Create a copy with renamed columns
        result = daily_prices[['symbol', 'date', 'close']].copy()
        result.rename(columns={'close': 'price'}, inplace=True)
        
        return result


# Register the calculator
price_calculator = PriceCalculator().register()
