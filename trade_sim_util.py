import pandas as pd
import os
from datetime import datetime, date
import logging
import sqlite3
from utils.logging_config import setup_logging

# Set up logging with filename and line numbers
setup_logging()

class PriceLoader:
    """
    **DEPRECATED**

    This class is deprecated because the data quality from Yahoo finance is inferior to FMP's data.
    Please use `FMPPriceLoader` instead.
    """
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


class FMPPriceLoader:
    def __init__(self):
        """Initialize database connection"""
        self.conn = sqlite3.connect('fmp_data.db')
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self.price_cache = {}

    def get_last_available_price(self, symbol, start_date, price_type='Close'):
        """Get the last available price before or on the given date.
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Date to search from in YYYY-MM-DD format
            price_type (str): Type of price ('Open', 'High', 'Low', 'Close')
            
        Returns:
            tuple: (price, date) - The price and the date it was found on
            
        Raises:
            ValueError: If price_type is invalid
            KeyError: If no price data found for the symbol
        """
        # Convert datetime or date object to string format if needed
        if isinstance(start_date, (datetime, date)):
            start_date = start_date.strftime('%Y-%m-%d')

        # Convert price_type to database column name (lowercase)
        price_column = price_type.lower()
        if price_column not in ['open', 'high', 'low', 'close']:
            raise ValueError("price_type must be one of 'Open', 'High', 'Low', 'Close'")

        # For Close price type, use adjusted_close
        if price_type == 'Close':
            self.cursor.execute("""
                SELECT adjusted_close as price, date
                FROM daily_price
                WHERE symbol = ? AND date <= ?
                ORDER BY date DESC
                LIMIT 1
            """, (symbol, start_date))
        else:
            # For other price types, get both raw price and adjustment ratio
            self.cursor.execute(f"""
                SELECT {price_column} as raw_price,
                       adjusted_close / close as adj_ratio,
                       date
                FROM daily_price
                WHERE symbol = ? AND date <= ?
                ORDER BY date DESC
                LIMIT 1
            """, (symbol, start_date))

        result = self.cursor.fetchone()
        if not result:
            raise KeyError(f"No price data found for {symbol}")

        if price_type == 'Close':
            return float(result['price']), result['date']
        else:
            return float(result['raw_price'] * result['adj_ratio']), result['date']

    def get_price(self, symbol, date_str, price_type='Close'):
        """Get adjusted price for a symbol on a specific date.
        
        All prices returned are adjusted for splits and dividends to be consistent
        with the CSV data. The adjustment uses the formula:
            adjusted_price = price * adjusted_close / close
        
        For Close price type, returns the adjusted_close directly.
        For other price types (Open, High, Low), applies the adjustment formula.

        Args:
            symbol (str): Stock symbol
            date_str (str): Date to get price for in YYYY-MM-DD format
            price_type (str): Type of price ('Open', 'High', 'Low', 'Close')

        Returns:
            float: Adjusted price for the specified symbol and date

        Raises:
            ValueError: If price_type is invalid
            KeyError: If no price data found within 4 days
        """
        # Convert datetime or date object to string format if needed
        if isinstance(date_str, (datetime, date)):
            date_str = date_str.strftime('%Y-%m-%d')

        # Convert price_type to database column name (lowercase)
        price_column = price_type.lower()
        if price_column not in ['open', 'high', 'low', 'close']:
            raise ValueError("price_type must be one of 'Open', 'High', 'Low', 'Close'")

        # For Close price type, return adjusted_close directly
        if price_type == 'Close':
            self.cursor.execute("""
                SELECT adjusted_close as price
                FROM daily_price
                WHERE symbol = ? AND date = ?
            """, (symbol, date_str))
            
            result = self.cursor.fetchone()
            if result and result['price'] is not None:
                return float(result['price'])

            # Try to find the closest previous date within 4 days
            self.cursor.execute("""
                SELECT adjusted_close as price
                FROM daily_price
                WHERE symbol = ?
                AND date < ?
                AND date >= date(?, '-4 days')
                ORDER BY date DESC
                LIMIT 1
            """, (symbol, date_str, date_str))

        else:
            # For other price types, get both unadjusted price and adjustment ratio
            self.cursor.execute(f"""
                SELECT {price_column} as raw_price,
                       adjusted_close / close as adj_ratio
                FROM daily_price
                WHERE symbol = ? AND date = ?
            """, (symbol, date_str))
            
            result = self.cursor.fetchone()
            if result and result['raw_price'] is not None:
                return float(result['raw_price'] * result['adj_ratio'])

            # Try to find the closest previous date within 4 days
            self.cursor.execute(f"""
                SELECT {price_column} as raw_price,
                       adjusted_close / close as adj_ratio
                FROM daily_price
                WHERE symbol = ?
                AND date < ?
                AND date >= date(?, '-4 days')
                ORDER BY date DESC
                LIMIT 1
            """, (symbol, date_str, date_str))

        result = self.cursor.fetchone()
        if result:
            if price_type == 'Close':
                return float(result['price'])
            else:
                return float(result['raw_price'] * result['adj_ratio'])

        raise KeyError(f"No price data found for {symbol} on {date_str} or within 4 previous days")

    def __del__(self):
        """Close database connection when object is destroyed"""
        if hasattr(self, 'cursor'):
            self.cursor.close()
        if hasattr(self, 'conn'):
            self.conn.close()
