"""
Offline data access for financial metrics.

This module provides access to financial data stored in the database.
"""

import logging
import os
import numpy as np
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional, Union, Set, Tuple
import pandas as pd
import sqlite3
from utils.config import FMP_DB_PATH

class OfflineData:
    def __init__(self, symbol: Union[str, List[str]], metrics: List[str], for_date: Union[str, List[str]] = None, 
                 start_date: str = None, end_date: str = None, db_path: str = FMP_DB_PATH):
        """Initialize Dataset object.
        
        Args:
            symbol (Union[str, List[str]]): Stock symbol or list of stock symbols
            metrics (List[str]): List of metrics to retrieve. 
            for_date (Union[str, List[str]]): A specific date or list of dates to filter by in 'YYYY-MM-DD' format.
                                           Data will be filtered to include only exact date matches.
            start_date (str): Start date in 'YYYY-MM-DD' format for date range queries (inclusive).
            end_date (str): End date in 'YYYY-MM-DD' format for date range queries (inclusive).
            db_path (str): Path to SQLite database
        
        Attributes:
            data (pd.DataFrame): The dataset as a pandas DataFrame.
            symbol (Union[str, List[str]]): Stock symbol(s).
            metrics (Dict[str, str]): Dictionary mapping metrics to their renamed columns.
            for_date (Union[str, List[str]]): Date(s) to filter by.
            start_date (str): Start date for date range queries.
            end_date (str): End date for date range queries.
            db_path (str): Path to SQLite database.
            
        Note:
            Either for_date or (start_date, end_date) must be provided, but not both.
        """
        # Validate that either for_date or (start_date, end_date) is provided, but not both
        if for_date is not None and (start_date is not None or end_date is not None):
            raise ValueError("Either for_date or (start_date, end_date) must be provided, but not both.")
            

        self.symbol = [symbol] if isinstance(symbol, str) else symbol
        self.metrics = metrics
        self.db_path = db_path
        self.start_date = start_date
        self.end_date = end_date
        
        # Convert for_date to a list if it's a single string or None
        if isinstance(for_date, str):
            self.for_date = [for_date,]
        else:
            self.for_date = for_date

        self.data = None

    
    def _get_table_columns(self) -> dict:
        """Get all columns from all tables in the database."""
        tables = ['income_statement', 'balance_sheet', 'cash_flow']
        columns = {}
        
        with sqlite3.connect(f'file:{self.db_path}?mode=ro', uri=True) as conn:
            for table in tables:
                cursor = conn.execute(f'PRAGMA table_info({table})')
                columns[table] = [row[1] for row in cursor.fetchall()]
                
        return columns

    def _find_metric_locations(self) -> dict:
        """Find which tables contain each requested metric."""
        table_columns = self._get_table_columns()
        metric_locations = {}
        
        for metric in self.metrics:
            locations = []
            for table, columns in table_columns.items():
                if metric in columns:
                    locations.append(table)
            
            if len(locations) > 1:
                logging.warning(f"Metric '{metric}' found in multiple tables: {locations}. Using {locations[0]}")
            if len(locations) == 0:
                raise ValueError(f"Metric '{metric}' not found in any table")
            metric_locations[metric] = locations[0]
                
        return metric_locations


    def build(self) -> pd.DataFrame:
        """Build and return the dataset as a pandas DataFrame.
        
        This method uses the gen() method to collect data for all symbols and then
        combines them into a single DataFrame.
        """
        # Collect results from the generator
        symbol_dfs = []
        
        # Use the gen() method to get data for each symbol
        for symbol, df in self.gen():
            # Add the symbol column back
            df['symbol'] = symbol
            symbol_dfs.append(df)
        
        # If no data was found for any symbol, return an empty DataFrame
        if not symbol_dfs:
            # Create an empty DataFrame with the expected columns
            columns = ['symbol', 'filing_date'] + self.metrics
            return pd.DataFrame(columns=columns)
        
        # Combine all symbol DataFrames
        result = pd.concat(symbol_dfs, ignore_index=True)
        
        # Sort by symbol and date
        result = result.sort_values(['symbol', 'filing_date'])
        
        return result

    # Use this method instead of calling .data so that it's easier to write unit tests
    def get_data(self) -> pd.DataFrame:
        """Return the dataset."""
        if self.data is None:
            self.data = self.build()
        return self.data
        
    def gen(self):
        """Generator that yields metrics for each symbol.
        
        This method queries the database for each symbol individually and yields the results,
        which is more memory-efficient than loading all data at once.
        
        Yields:
            tuple: A tuple containing (symbol, dataframe) where dataframe contains the metrics for that symbol.
                  The dataframe is indexed by date and contains the requested metrics as columns.
        """
        # Find which tables contain each requested metric
        metric_locations = self._find_metric_locations()
        
        # Group metrics by table
        table_metrics = {}
        for metric, table in metric_locations.items():
            table_metrics.setdefault(table, []).append(metric)
        
        # Process each symbol individually
        for symbol in self.symbol:
            # Query each table for this symbol
            symbol_dfs = []
            
            with sqlite3.connect(f'file:{self.db_path}?mode=ro', uri=True) as conn:
                for table, metrics in table_metrics.items():
                    # Build the query with date filter if for_date is provided
                    query = f"""
                    SELECT filing_date, {', '.join(metrics)}
                    FROM {table}
                    WHERE symbol = ?
                    """
                    
                    params = [symbol]
                    
                    # Add date filter if for_date is provided
                    if self.for_date:
                        date_placeholders = ','.join(['?' for _ in self.for_date])
                        query += f" AND filing_date IN ({date_placeholders})"
                        params.extend(self.for_date)
                    elif self.start_date and self.end_date:
                        query += " AND filing_date BETWEEN ? AND ?"
                        params.extend([self.start_date, self.end_date])
                    
                    query += " ORDER BY filing_date"
                    
                    df = pd.read_sql_query(query, conn, params=tuple(params))
                    if not df.empty:
                        symbol_dfs.append(df)
            
            # Skip if no data found for this symbol
            if not symbol_dfs:
                continue
                
            # Merge all dataframes for this symbol
            result = symbol_dfs[0]
            for df in symbol_dfs[1:]:
                result = pd.merge(result, df, on='filing_date', how='outer')
            
            yield symbol, result
    
    @classmethod
    def historical_tradable_price_fmp(cls, symbol: str, start_date: str, end_date: str, db_path: str = FMP_DB_PATH):
        """Get historical tradable price for a symbol in a date range.
        
        If there are gaps in the price data (days without data), the method will fill
        these gaps using the last available price data.
        """
        with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as conn:
            query = """
            SELECT symbol, date, high, low
            FROM daily_price
            WHERE symbol = ?
            AND date BETWEEN ? AND ?
            ORDER BY date
            """
            params = [symbol, start_date, end_date]
            df = pd.read_sql_query(query, conn, params=tuple(params))
            
            if df.empty:
                return df.drop(columns=['high', 'low'])
            
            # Calculate tradable price as average of high and low
            df['tradable_price'] = df.apply(lambda row: (row['high'] + row['low']) / 2, axis=1)
            df.drop(columns=['high', 'low'], inplace=True)
            
            # Create a continuous date range to fill gaps
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            template = pd.DataFrame({'date': date_range})
            template['symbol'] = symbol
            
            # Convert date columns to datetime for proper merging
            df['date'] = pd.to_datetime(df['date'])
            template['date'] = pd.to_datetime(template['date'])
            
            # Merge with the template to include all dates
            merged = pd.merge(template, df, on=['symbol', 'date'], how='left')
            
            # Forward fill the tradable_price column to use last available price for gaps
            merged['tradable_price'] = merged['tradable_price'].ffill()
            
            # Convert date back to string format for consistency
            merged['date'] = merged['date'].dt.strftime('%Y-%m-%d')
            
            return merged

    @classmethod
    def historical_tradable_price(cls, symbol: str, start_date: str, end_date: str, db_path: str = FMP_DB_PATH):
        """Get historical tradable price for a symbol in a date range using EODHD data.
        
        Uses adjusted_close prices from daily_price_eodhd table. If there are gaps in the 
        price data (days without data), the method will fill these gaps using the last 
        available price data.
        
        Args:
            symbol (str): Stock symbol
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
            db_path (str): Path to SQLite database
            
        Returns:
            pd.DataFrame: DataFrame with columns ['symbol', 'date', 'tradable_price']
        """
        with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as conn:
            query = """
            SELECT symbol, date, high, low, close, adjusted_close
            FROM daily_price_eodhd
            WHERE symbol = ?
            AND date BETWEEN ? AND ?
            ORDER BY date
            """
            params = [symbol, start_date, end_date]
            df = pd.read_sql_query(query, conn, params=tuple(params))
            
            if df.empty:
                return df.drop(columns=['high', 'low', 'close', 'adjusted_close'] if len(df.columns) > 2 else [])
            
            # Calculate tradable price as average of high and low, then adjust it
            # Adjustment factor is adjusted_close / close
            df['adjustment_factor'] = df['adjusted_close'] / df['close']
            df['tradable_price'] = ((df['high'] + df['low']) / 2) * df['adjustment_factor']
            df.drop(columns=['high', 'low', 'close', 'adjusted_close', 'adjustment_factor'], inplace=True)
            
            # Create a continuous date range to fill gaps
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            template = pd.DataFrame({'date': date_range})
            template['symbol'] = symbol
            
            # Convert date columns to datetime for proper merging
            df['date'] = pd.to_datetime(df['date'])
            template['date'] = pd.to_datetime(template['date'])
            
            # Merge with the template to include all dates
            merged = pd.merge(template, df, on=['symbol', 'date'], how='left')
            
            # Forward fill the tradable_price column to use last available price for gaps
            merged['tradable_price'] = merged['tradable_price'].ffill()
            
            # Convert date back to string format for consistency
            merged['date'] = merged['date'].dt.strftime('%Y-%m-%d')
            
            return merged

    @classmethod
    def get_us_active_stocks(cls, db_path: str = FMP_DB_PATH) -> List[str]:
        with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as conn:
            query = """
            SELECT symbol
            FROM valid_us_stocks_der
            """
            df = pd.read_sql_query(query, conn)
            return df['symbol'].tolist()

    @classmethod
    def get_suspicious_symbols(cls, db_path: str = FMP_DB_PATH) -> List[str]:
        with sqlite3.connect(f'file:{db_path}?mode=ro', uri=True) as conn:
            query = """
            SELECT DISTINCT symbol
            FROM suspicious_symbols
            """
            df = pd.read_sql_query(query, conn)
            return df['symbol'].tolist()
        
