import pandas as pd
import os
from datetime import datetime, date, timedelta
import logging
from typing import List, Dict, Union
import sqlite3


BEFORE_PRICE = 'before_price'
AFTER_PRICE = 'after_price'
PRICE_METRICS = {BEFORE_PRICE, AFTER_PRICE}

class Dataset:
    def __init__(self, symbol: Union[str, List[str]], metrics: Dict[str, str], for_date: Union[str, List[str]] = None, db_path: str = '/Users/jluan/code/finance/data/fmp_data.db'):
        """Initialize Dataset object.
        
        Args:
            symbol (Union[str, List[str]]): Stock symbol or list of stock symbols
            metrics (Dict[str, str]): Dictionary mapping metrics to their renamed columns. 
                                    If value is empty string or None, the original metric name is used.
            for_date (Union[str, List[str]]): A specific date or list of dates to filter by in 'YYYY-MM-DD' format.
                                           Data will be filtered to include only exact date matches.
            db_path (str): Path to SQLite database
        
        Attributes:
            data (pd.DataFrame): The dataset as a pandas DataFrame.
            symbol (Union[str, List[str]]): Stock symbol(s).
            metrics (Dict[str, str]): Dictionary mapping metrics to their renamed columns.
            for_date (Union[str, List[str]]): Date(s) to filter by.
            db_path (str): Path to SQLite database.
        """

        self.symbol = [symbol] if isinstance(symbol, str) else symbol
        self.metrics = metrics
        self.db_path = db_path
        
        # Convert for_date to a list if it's a single string or None
        if isinstance(for_date, str):
            self.for_date = [for_date,]
        else:
            self.for_date = for_date
            
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

        price_loader = FMPPriceLoader(db_path=self.db_path)
        price_data = []

        for date in dates:
            row = {'date': date, 'symbol': symbol}
            for metric in requested_price_metrics:
                try:
                    if metric == BEFORE_PRICE:
                        price, _ = price_loader.get_last_available_price(symbol, date)
                    else:
                        assert metric == AFTER_PRICE
                        price, _ = price_loader.get_next_available_price(symbol, date)
                except KeyError:
                    logging.warning(f'Failed to fetch historical price for {symbol} on {date}')
                else:
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
                # Create placeholders for symbols
                symbol_placeholders = ','.join(['?' for _ in self.symbol])
                
                # Build the query with date filter if for_date is provided
                query = f"""
                SELECT date, symbol, {', '.join(metrics)}
                FROM {table}
                WHERE symbol IN ({symbol_placeholders})
                """
                
                params = list(self.symbol)
                
                # Add date filter if for_date is provided
                if self.for_date:
                    date_placeholders = ','.join(['?' for _ in self.for_date])
                    query += f" AND date IN ({date_placeholders})"
                    params.extend(self.for_date)
                
                query += " ORDER BY date"
                
                df = pd.read_sql_query(query, conn, params=tuple(params))
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

        result = result.sort_values(['symbol', 'date'])

        return result

    def get_data(self) -> pd.DataFrame:
        """Return the dataset."""
        return self.data


class FMPPriceLoader:
    def __init__(self, price_tolerance_days: int = 4, db_path: str = '/Users/jluan/code/finance/data/fmp_data.db'):
        """Initialize database connection

        Args:
            price_tolerance_days (int): Maximum number of days to search for a price when given date has no data. Used
                in get_last_available_price and get_next_available_price
            db_path (str): Path to the SQLite database file. Defaults to '/Users/jluan/code/finance/data/fmp_data.db'

        Raises:
            FileNotFoundError: If the database file doesn't exist
        """

        self.price_tolerance_days = price_tolerance_days 
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"FMP Database file not found: {db_path}")

        # Connect in read-only mode using URI
        self.conn = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()


    def get_price(self, symbol, date_str, price_type='close'):
        """Get adjusted price for a symbol on a specific date.

        All prices returned are adjusted for splits and dividends to be consistent
        with the CSV data. The adjustment uses the formula:
            adjusted_price = price * float(adjusted_close) / close

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
            SELECT {price_column} * (CAST(adjusted_close AS FLOAT) / close) as adj_price
            FROM daily_price
            WHERE symbol = ? AND date = ? AND close > 0 AND volume > 0
        """, (symbol, date_str))

        result = self.cursor.fetchone()

        return float(result['adj_price'])

    def get_eps(self, symbol, date):
        """Get the EPS for a given stock on a specific date.

        Args:
            symbol (str): Stock symbol
            date (str): Date to get EPS for in YYYY-MM-DD format

        Returns:
            float: EPS for the specified symbol and date

        Raises:
            ValueError: If no EPS data found for the specified symbol and date
        """
        self.cursor.execute("""
            SELECT date, net_income_per_share as eps
            FROM metrics
            WHERE symbol = ? AND date <= ?
            ORDER BY date DESC
            LIMIT 1
        """, (symbol, date))

        result = self.cursor.fetchone()
        if not result:
            raise ValueError(
                f"Can't find EPS for {symbol} on {date}")
        return result['date'], float(result['eps'])

    def get_last_available_price(self, symbol, start_date, price_type='close', max_window_days=None):
        """Get the last available **adjusted** price before or on the start date.

        Args:
            symbol (str): Stock symbol
            start_date (str): Date to get price for in YYYY-MM-DD format
            price_type (str): Type of price ('open', 'high', 'low', 'close')
            max_window_days (int): Maximum number of "look back" days when the given date is not found

        Returns:
            tuple: (price, date) where price is the adjusted price and date is
                  the actual date of the price

        Raises:
            ValueError: If price_type is invalid
            KeyError: If no price data found
        """
        if max_window_days is None:
            max_window_days = self.price_tolerance_days

        # Convert datetime or date object to string format if needed
        if isinstance(start_date, (datetime, date)):
            start_date = start_date.strftime('%Y-%m-%d')

        # Convert price_type to database column name (lowercase)
        price_column = price_type.lower()
        if price_column not in ['open', 'high', 'low', 'close', 'adjusted_close']:
            raise ValueError(
                "price_type must be one of 'Open', 'High', 'Low', 'Close', 'adjusted_close'")

        self.cursor.execute(f"""
            SELECT {price_column} * (CAST(adjusted_close AS FLOAT) / close) as adj_price, date
            FROM daily_price
            WHERE symbol = ? AND date <= ? AND close > 0 AND volume > 0
            ORDER BY date DESC
            LIMIT 1
        """, (symbol, start_date))

        result = self.cursor.fetchone()
        if not result:
            raise KeyError(
                f"Can't find historical price of {symbol} for {start_date} or earlier")
        date_used = result['date']
        diff = (pd.to_datetime(date_used) - pd.to_datetime(start_date)).days
        if abs(diff) > max_window_days:
            raise KeyError(
                f"Can't find historical price of {symbol} for {start_date} within {max_window_days} days, last available is {date_used}")
        return float(result['adj_price']), date_used


    def get_next_available_price(self, symbol, start_date, price_type='close', max_window_days=None):
        """Get the next available price after or on the start date.

        Args:
            symbol (str): Stock symbol
            start_date (str): Date to get price for in YYYY-MM-DD format
            price_type (str): Type of price ('open', 'high', 'low', 'close')
            max_window_days (int): Maximum number of "look ahead" days when the given date is not found

        Returns:
            tuple: (price, date) where price is the adjusted price and date is
                  the actual date of the price

        Raises:
            ValueError: If price_type is invalid
            KeyError: If no price data found
        """
        if max_window_days is None:
            max_window_days = self.price_tolerance_days

        # Convert datetime or date object to string format if needed
        if isinstance(start_date, (datetime, date)):
            start_date = start_date.strftime('%Y-%m-%d')

        # Convert price_type to database column name (lowercase)
        price_column = price_type.lower()
        if price_column not in ['open', 'high', 'low', 'close', 'adjusted_close']:
            raise ValueError(
                "price_type must be one of 'Open', 'High', 'Low', 'Close', 'adjusted_close'")

        self.cursor.execute(f"""
            SELECT {price_column} * (CAST(adjusted_close AS FLOAT) / close) as adj_price, date
            FROM daily_price
            WHERE symbol = ? AND date >= ? AND close > 0 AND volume > 0
            ORDER BY date ASC
            LIMIT 1
        """, (symbol, start_date))

        result = self.cursor.fetchone()
        if not result:
            raise KeyError(
                f"Can't find historical price of {symbol} for {start_date} or earlier")
        date_used = result['date']
        diff = (pd.to_datetime(date_used) - pd.to_datetime(start_date)).days
        if abs(diff) > max_window_days:
            raise KeyError(
                f"Can't find historical price of {symbol} for {start_date} within {max_window_days} days, next available is {date_used}")
        return float(result['adj_price']), date_used 


    def get_price_for_stocks_during(self, symbols: List[str], begin_date: str, end_date: str, price_type='close'):
        """Get all prices for given stocks during the date range.

        Args:
            symbols (list): List of stock symbols
            begin_date (str): Start date in YYYY-MM-DD format (inclusive)
            end_date (str): End date in YYYY-MM-DD format (inclusive)
            price_type (str): Type of price ('open', 'high', 'low', 'close')

        Returns:
            dict: Dictionary mapping symbol to list of (date, price) tuples, sorted by date DESC
        """
        if price_type not in ['open', 'high', 'low', 'close', 'adjusted_close']:
            raise ValueError(
                "price_type must be one of 'open', 'high', 'low', 'close', 'adjusted_close'")

        # Simple query to get all prices in the date range
        placeholders = ','.join('?' * len(symbols))
        query = f"""
            SELECT 
                symbol,
                date,
                {price_type} * (CAST(adjusted_close AS FLOAT) / close) as adj_price
            FROM daily_price
            WHERE symbol IN ({placeholders})
            AND date >= ?
            AND date <= ?
            AND close > 0
            AND volume > 0
            ORDER BY symbol, date DESC
        """
        
        # Execute query
        self.cursor.execute(query, (*symbols, begin_date, end_date))
        
        # Group prices by symbol
        prices_by_symbol = {}
        for row in self.cursor.fetchall():
            symbol = row['symbol']
            if symbol not in prices_by_symbol:
                prices_by_symbol[symbol] = []
            prices_by_symbol[symbol].append((row['date'], float(row['adj_price'])))
        
        return prices_by_symbol

    def get_last_available_price_for_stocks_on_dates(self, symbols: List[str], dates: List[str], price_type='close'):
        """Get the last available **adjusted** price before or on the start date.

        Args:
            symbols (list): List of stock symbols
            start_dates (list): List of dates to get prices for in YYYY-MM-DD format
            price_type (str): Type of price ('open', 'high', 'low', 'close')

        Returns:
            dict: Dictionary mapping (symbol, start_date) pairs to (price, actual_date) tuples
        """
        if not dates or not symbols:
            return {}

        # Find min and max dates to create an efficient date range
        min_date = min(dates)
        max_date = max(dates)
        min_search_date = self.cursor.execute(
            "SELECT date(?, '-5 days')", (min_date,)
        ).fetchone()[0]

        # Get all prices in the date range
        prices_by_symbol = self.get_price_for_stocks_during(
            symbols, min_search_date, max_date, price_type)
        
        # Sort dates in descending order to match with prices
        sorted_dates = sorted(dates, reverse=True)
        results = {}
        
        # For each symbol, merge sort with dates to find matches
        for symbol, prices in prices_by_symbol.items():
            date_idx = 0
            price_idx = 0
            
            while date_idx < len(sorted_dates) and price_idx < len(prices):
                request_date = sorted_dates[date_idx]
                price_date, price = prices[price_idx]
                
                if price_date <= request_date:
                    # Found a match - this is the most recent price for this date
                    results[(symbol, request_date)] = (price, price_date)
                    date_idx += 1
                else:
                    # Price is too recent, try next price
                    price_idx += 1
        
        return results

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
            WHERE symbol = ? AND date BETWEEN ? AND ? AND close > 0 AND volume > 0
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
            WHERE symbol = ? AND date <= ? AND close > 0 AND volume > 0
            ORDER BY date DESC
            LIMIT ?
        """, (symbol, last_date, num_days))

        return {row['date']: row['adjusted_close'] for row in self.cursor.fetchall()}        


    def num_stocks_at(self, date):
        cursor = self.conn.cursor()
        cursor.execute(f"""
            SELECT COUNT(DISTINCT symbol) as num_stocks
            FROM daily_price
            WHERE date = ? AND close > 0 AND volume > 0
        """, (date,))

        return cursor.fetchone()['num_stocks']

    def get_us_stock_symbols(self):
        self.cursor.execute("""
            SELECT symbol FROM stock_symbol 
            WHERE exchange_short_name IN ('NYSE', 'NASDAQ', 'AMEX')
            AND type = 'stock'
        """)
        return [row['symbol'] for row in self.cursor.fetchall()]

    def active_us_stocks_on(self, date_str) -> set[str]:

        five_days_ago = (datetime.strptime(date_str, '%Y-%m-%d') - timedelta(days=5)).strftime('%Y-%m-%d')
        
        self.cursor.execute('''
            SELECT DISTINCT A.symbol
            FROM stock_symbol A
            JOIN daily_price B ON A.symbol = B.symbol
            WHERE A.exchange_short_name IN ('NYSE', 'NASDAQ', 'AMEX')
                AND A.type = 'stock'
                AND B.adjusted_close < 10000
                AND B.volume > 0
                AND B.date BETWEEN ? AND ?
        ''', (five_days_ago, date_str))
        
        return set(row['symbol'] for row in self.cursor.fetchall())        

def fmp(sql):
    db = sqlite3.connect('/Users/jluan/code/finance/data/fmp_data.db')
    df = pd.read_sql_query(sql, db)
    return df        