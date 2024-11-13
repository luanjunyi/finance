import pandas as pd
import os
from datetime import datetime, date
import logging


class PriceLoader:
    def __init__(self):
        """Initialize with an empty cache for price data"""
        self.price_cache = {}

    def get_price(self, symbol, date_str, price_type='Close'):
        """Get price for a symbol on a specific date

        Args:
            symbol (str): Stock symbol
            date_str (str): Date to get price for in YYYY-MM-DD format
            price_type (str): Type of price ('Open', 'High', 'Low', 'Close')

        Returns:
            float: Price for the specified symbol and date
        """
        # Convert datetime or date object to string format if needed
        if isinstance(date_str, (datetime, date)):
            date_str = date_str.strftime('%Y-%m-%d')

        # Load data into cache if not already present
        if symbol not in self.price_cache:
            file_path = f'stock_data/{symbol}_daily.csv'
            if not os.path.exists(file_path):
                raise ValueError(f"No data file found for symbol {symbol}")

            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(
                df['Date'], utc=True).dt.strftime('%Y-%m-%d')
            self.price_cache[symbol] = df.set_index('Date')

        # Get price from cache
        try:
            return self.price_cache[symbol].loc[date_str, price_type]
        except KeyError:
            # Try to find the closest previous date within 4 days
            for i in range(1, 5):
                prev_date = (datetime.strptime(date_str, '%Y-%m-%d') -
                             pd.Timedelta(days=i)).strftime('%Y-%m-%d')
                try:
                    return self.price_cache[symbol].loc[prev_date, price_type]
                except KeyError:
                    continue
            raise KeyError(
                f"No price data found for {symbol} on {date_str} or within 4 previous days")
