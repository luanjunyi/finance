import pandas as pd
import os
from datetime import datetime, date
import logging
from typing import List, Dict, Union
import sqlite3
from utils.logging_config import setup_logging

# Set up logging with filename and line numbers
setup_logging()

BEFORE_PRICE = 'before_price'
AFTER_PRICE = 'after_price'
PRICE_METRICS = {BEFORE_PRICE, AFTER_PRICE}

class Dataset:
    def __init__(self, symbol: Union[str, List[str]], metrics: Dict[str, str], db_path: str = '/Users/jluan/code/finance/data/fmp_data.db'):
        """Initialize Dataset object.
        
        Args:
            symbol (Union[str, List[str]]): Stock symbol or list of stock symbols
            metrics (Dict[str, str]): Dictionary mapping metrics to their renamed columns. 
                                    If value is empty string or None, the original metric name is used.
            db_path (str): Path to SQLite database
        
        Attributes:
            data (pd.DataFrame): The dataset as a pandas DataFrame.
            symbol (Union[str, List[str]]): Stock symbol(s).
            metrics (Dict[str, str]): Dictionary mapping metrics to their renamed columns.
            db_path (str): Path to SQLite database.
        """

        self.symbol = [symbol] if isinstance(symbol, str) else symbol
        self.metrics = metrics
        self.db_path = db_path
        self.data = self.build()

    def __getattr__(self, name):
        """Delegate any unknown attributes/methods to the underlying pandas DataFrame."""
        return getattr(self.data, name)
        
    def __getitem__(self, key):
        """Enable DataFrame-style indexing."""
        return self.data.__getitem__(key)

    def _get_table_columns(self) -> dict:
        """Get all columns from all tables in the database."""
        tables = ['daily_price', 'income_statement', 'balance_sheet', 'cash_flow', 'metrics']
        columns = {}
        
        with sqlite3.connect(self.db_path) as conn:
            for table in tables:
                cursor = conn.execute(f'PRAGMA table_info({table})')
                columns[table] = [row[1] for row in cursor.fetchall()]
                
        return columns

    def _find_metric_locations(self) -> dict:
        """Find which tables contain each requested metric."""
        table_columns = self._get_table_columns()
        metric_locations = {}
        
        for metric in self.metrics.keys():
            if metric in PRICE_METRICS:
                continue
            locations = []
            for table, columns in table_columns.items():
                if metric in columns:
                    locations.append(table)
            
            if not locations:
                logging.fatal(f"Metric '{metric}' not found in any table")
            if len(locations) > 1:
                logging.warning(f"Metric '{metric}' found in multiple tables: {locations}. Using {locations[0]}")
            metric_locations[metric] = locations[0]
                
        return metric_locations

    def _handle_price_metrics(self, symbol: str, dates) -> pd.DataFrame:
        """Handle special price metrics using FMPPriceLoader.

        Args:
            symbol (str): Stock symbol to get prices for
            dates (pd.Series): Series of dates to get prices for

        Returns:
            pd.DataFrame: DataFrame with price metrics for each date
        """
        requested_price_metrics = set(self.metrics.keys()) & PRICE_METRICS

        price_loader = FMPPriceLoader(self.db_path)
        price_data = []

        for date in dates:
            row = {'date': date, 'symbol': symbol}
            for metric in requested_price_metrics:
                if metric == BEFORE_PRICE:
                    price, _ = price_loader.get_last_available_price(symbol, date)
                else:
                    assert metric == AFTER_PRICE
                    price, _ = price_loader.get_next_available_price(symbol, date)
                row[metric] = price
            price_data.append(row)

        return pd.DataFrame(price_data)

    def build(self) -> pd.DataFrame:
        """Build and return the dataset as a pandas DataFrame."""

        # Handle regular metrics through database queries
        metric_locations = self._find_metric_locations()

        # Group metrics by table
        table_metrics = {}
        for metric, table in metric_locations.items():
            if table not in table_metrics:
                table_metrics[table] = []
            table_metrics[table].append(metric)        

        # Query each table for regular metrics
        dfs = []
        with sqlite3.connect(self.db_path) as conn:
            for table, metrics in table_metrics.items():
                placeholders = ','.join(['?' for _ in self.symbol])
                query = f"""
                SELECT date, symbol, {', '.join(metrics)}
                FROM {table}
                WHERE symbol IN ({placeholders})
                ORDER BY date
                """
                df = pd.read_sql_query(query, conn, params=tuple(self.symbol))
                dfs.append(df)

        # dfs can't be empty because either there are regular prices (when we just want historical price data) or 
        # metrics are requested with before or after price
        assert len(dfs) > 0, f"No data found for requested metrics: {self.metrics}"

        # Merge all dataframes
        result = dfs[0]
        for df in dfs[1:]:
            result = pd.merge(result, df, on=['date', 'symbol'], how='outer')

        # Handle price metrics if requested
        if len(set(PRICE_METRICS) & set(self.metrics.keys())) > 0:
            # Get all dates from the result DataFrame or query them if no regular metrics
            assert not result.empty, f"Requested {self.metrics} for {self.symbol} which are " \
                "pure for joining with metrics. For pure price data, use close, open, adjusted close, etc."

            # Process price metrics for each symbol separately
            price_dfs = []
            for symbol in self.symbol:
                symbol_dates = result[result['symbol'] == symbol]['date'].unique()
                price_df = self._handle_price_metrics(symbol, symbol_dates)
                price_dfs.append(price_df)

            # Combine all price dataframes
            if price_dfs:
                price_df = pd.concat(price_dfs, ignore_index=True)
                if result.empty:
                    result = price_df
                else:
                    result = pd.merge(result, price_df, on=['date', 'symbol'], how='outer')

        # Rename columns if specified
        rename_dict = {k: v for k, v in self.metrics.items() if v and v is not None}
        if rename_dict:
            result = result.rename(columns=rename_dict)

        return result.sort_values(['symbol', 'date'])

    def get_data(self) -> pd.DataFrame:
        """Return the dataset."""
        return self.data


class FMPPriceLoader:
    def __init__(self, db_path: str = '/Users/jluan/code/finance/data/fmp_data.db'):
        """Initialize database connection

        Args:
            db_path (str): Path to the SQLite database file. Defaults to '/Users/jluan/code/finance/data/fmp_data.db'

        Raises:
            FileNotFoundError: If the database file doesn't exist
        """
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"FMP Database file not found: {db_path}")

        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()


    def get_price(self, symbol, date_str, price_type='close'):
        """Get adjusted price for a symbol on a specific date.

        All prices returned are adjusted for splits and dividends to be consistent
        with the CSV data. The adjustment uses the formula:
            adjusted_price = price * adjusted_close / close

        For Close price type, returns the adjusted_close directly.
        For other price types (Open, High, Low), applies the adjustment formula.

        Args:
            symbol (str): Stock symbol
            date_str (str): Date to get price for in YYYY-MM-DD format
            price_type (str): Type of price ('open', 'high', 'low', 'close')

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
        if price_column not in ['open', 'high', 'low', 'close', 'adjusted_close']:
            raise ValueError(
                "price_type must be one of 'Open', 'High', 'Low', 'Close', 'adjusted_close'")

        
        # Try exact date match first
        self.cursor.execute(f"""
            SELECT {price_column} * (adjusted_close / close) as adj_price
            FROM daily_price
            WHERE symbol = ? AND date = ?
        """, (symbol, date_str))

        result = self.cursor.fetchone()

        return float(result['adj_price'])

    def get_last_available_price(self, symbol, start_date, price_type='close'):
        """Get the last available price before or on the start date.

        Args:
            symbol (str): Stock symbol
            start_date (str): Date to get price for in YYYY-MM-DD format
            price_type (str): Type of price ('open', 'high', 'low', 'close')

        Returns:
            tuple: (price, date) where price is the adjusted price and date is
                  the actual date of the price

        Raises:
            ValueError: If price_type is invalid
            KeyError: If no price data found
        """
        # Convert datetime or date object to string format if needed
        if isinstance(start_date, (datetime, date)):
            start_date = start_date.strftime('%Y-%m-%d')

        # Convert price_type to database column name (lowercase)
        price_column = price_type.lower()
        if price_column not in ['open', 'high', 'low', 'close', 'adjusted_close']:
            raise ValueError(
                "price_type must be one of 'Open', 'High', 'Low', 'Close', 'adjusted_close'")

        self.cursor.execute(f"""
            SELECT {price_column} * (adjusted_close / close) as adj_price, date
            FROM daily_price
            WHERE symbol = ? AND date <= ?
            ORDER BY date DESC
            LIMIT 1
        """, (symbol, start_date))

        result = self.cursor.fetchone()
        return float(result['adj_price']), result['date']        


    def get_close_price_during(self, symbol, start_date, end_date,):
        """Get price range for a symbol between start_date and end_date

        Args:
            symbol (str): Stock symbol
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format

        Returns:
            dict: Dictionary keyed by date with price data

        Raises:
            KeyError: If no price data found for the symbol
        """
        # Convert datetime or date object to string format if needed
        if isinstance(start_date, (datetime, date)):
            start_date = start_date.strftime('%Y-%m-%d')

        if isinstance(end_date, (datetime, date)):
            end_date = end_date.strftime('%Y-%m-%d')

        self.cursor.execute("""
            SELECT date, adjusted_close
            FROM daily_price
            WHERE symbol = ? AND date >= ? AND date <= ?
            ORDER BY date ASC
        """, (symbol, start_date, end_date))

        return {row['date']: row['adjusted_close'] for row in self.cursor.fetchall()}

    def get_close_price_for_the_last_days(self, symbol, last_date, num_days):
        """
        Get price range for a symbol for the last num_days days starting from
        last_date (inclusive)

        Args:
            symbol (str): Stock symbol
            last_date (str): Last date in YYYY-MM-DD format
            num_days (int): Number of days to get price for

        Returns:
            dict: Dictionary keyed by date with price data
        """
        # Convert datetime or date object to string format if needed
        if isinstance(last_date, (datetime, date)):
            last_date = last_date.strftime('%Y-%m-%d')


        self.cursor.execute("""
            SELECT date, adjusted_close
            FROM daily_price
            WHERE symbol = ? AND date <= ?
            ORDER BY date DESC
            LIMIT ?
        """, (symbol, last_date, num_days))

        return {row['date']: row['adjusted_close'] for row in self.cursor.fetchall()}        

    def get_next_available_price(self, symbol, start_date, price_type='close'):
        """Get the next available price after or on the start date.

        Args:
            symbol (str): Stock symbol
            start_date (str): Date to get price for in YYYY-MM-DD format
            price_type (str): Type of price ('open', 'high', 'low', 'close')

        Returns:
            tuple: (price, date) where price is the adjusted price and date is
                  the actual date of the price

        Raises:
            ValueError: If price_type is invalid
            KeyError: If no price data found
        """
        # Convert datetime or date object to string format if needed
        if isinstance(start_date, (datetime, date)):
            start_date = start_date.strftime('%Y-%m-%d')

        # Convert price_type to database column name (lowercase)
        price_column = price_type.lower()
        if price_column not in ['open', 'high', 'low', 'close', 'adjusted_close']:
            raise ValueError(
                "price_type must be one of 'Open', 'High', 'Low', 'Close', 'adjusted_close'")

        self.cursor.execute(f"""
            SELECT {price_column} * (adjusted_close / close) as adj_price, date
            FROM daily_price
            WHERE symbol = ? AND date >= ?
            ORDER BY date ASC
            LIMIT 1
        """, (symbol, start_date))

        result = self.cursor.fetchone()
        return float(result['adj_price']), result['date']

    def __del__(self):
        """Close database connection when object is destroyed"""
        if hasattr(self, 'cursor'):
            self.cursor.close()
        if hasattr(self, 'conn'):
            self.conn.close()
