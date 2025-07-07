"""
Dataset interface for financial metrics.

This module provides the main Dataset class that serves as the interface
for accessing financial metrics, either from the database or by calculation.
"""

import logging
from typing import List, Union
import pandas as pd

from fmp_data.offline_data import OfflineData
from fmp_data.metric_calculator import DERIVED_METRICS
from utils.logging_config import setup_logging as setup_global_logging

class Dataset:
    """
    Interface for accessing financial metrics.
    
    This class provides a unified interface for accessing financial metrics,
    either directly from the database or by calculation using metric calculators.
    """

    
    def __init__(self, symbols: Union[str, List[str]], metrics: List[str], start_date: str, end_date: str,
                 with_price: bool = False,
                 db_path: str = '/Users/jluan/code/finance/data/fmp_data.db'):
        """
        Initialize the dataset.
        
        Args:
            symbols: List of symbols to get data for
            metrics: List of metrics to retrieve
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            db_path: Path to SQLite database
        """
        self.db_path = db_path
        self.symbols = [symbols] if isinstance(symbols, str) else symbols
        self.metrics = metrics
        self.start_date = start_date
        self.end_date = end_date
        self.with_price = with_price
        setup_global_logging()
        self.logger = logging.getLogger(__name__)
        self._plan_query() 

    def _plan_query(self):
        """
        Plan the query to get the data.
        """
        self.direct_metrics = []
        self.derived_metrics = []
        self.dependencies = set()
        for metric in self.metrics:
            if metric in DERIVED_METRICS:
                self.derived_metrics.append(metric)
                self.dependencies.update(DERIVED_METRICS[metric]['dependencies'])
            else:
                self.direct_metrics.append(metric)

    def gen(self):
        """
        Generate the dataset.
        """
        for symbol in self.symbols:
            symbol_df = None
            derived_df = None
            if self.direct_metrics:
                symbol_df = OfflineData(symbol, self.direct_metrics, db_path=self.db_path).build()
            if self.derived_metrics:
                data_source = OfflineData(symbol, list(self.dependencies), db_path=self.db_path).build().sort_values('filing_date', ascending=True)
                derived_df = data_source[['symbol', 'filing_date']].copy()
                for metric in self.derived_metrics:
                    metric_series = DERIVED_METRICS[metric]['function'](data_source)
                    derived_df[metric] = metric_series
            if symbol_df is None:
                symbol_df = derived_df
            elif derived_df is not None:
                symbol_df = pd.merge(symbol_df, derived_df, on=['symbol', 'filing_date'], how='outer')
            symbol_df = self.fill_date_gaps(symbol_df)
            if self.with_price:
                price_df = OfflineData.historical_tradable_price(symbol, self.start_date, self.end_date, db_path=self.db_path)
                price_df['date'] = pd.to_datetime(price_df['date'])
                symbol_df = pd.merge(symbol_df, price_df, on=['symbol', 'date'], how='left') 
            yield symbol, symbol_df

    def build(self) -> pd.DataFrame:
        """
        Build and return the dataset as a pandas DataFrame.
        
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
            columns = ['symbol', 'date'] + self.metrics
            return pd.DataFrame(columns=columns)
        
        # Combine all symbol DataFrames
        result = pd.concat(symbol_dfs, ignore_index=True)
        
        # Sort by symbol and date
        result = result.sort_values(['symbol', 'date'])
        
        return result

            
    def fill_date_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill gaps in the time series data by forward-filling values between effective dates.
        
        Args:
            df: DataFrame with data from a single symbol.
            
        Returns:
            DataFrame with filled date gaps where each date shows the latest data available
        """
        if df.empty:
            return df

        assert 'symbol' in df.columns and 'first_effective_date' not in df.columns
        assert df['symbol'].nunique() == 1
        
        # Add first_effective_date as the day after filing_date
        df['first_effective_date'] = pd.to_datetime(df['filing_date']) + pd.Timedelta(days=1)
        if 'date' in df.columns:
            df.drop(columns=['date'], inplace=True)
        
        # Sort by first_effective_date
        df = df.sort_values('first_effective_date')
        
        # Create a continuous date range
        # Get the largest first_effective_date that's <= start_date
        min_date = self.start_date
        effective_dates = pd.to_datetime(df['first_effective_date'])
        dates_after_start = effective_dates[effective_dates <= min_date]
        if not dates_after_start.empty:
            min_date = dates_after_start.max().strftime('%Y-%m-%d')
        
        date_range = pd.date_range(start=min_date, end=self.end_date, freq='D')
        
        # Get the symbol value
        symbol = df['symbol'].iloc[0]
        
        # Create a template DataFrame with all dates
        template = pd.DataFrame({'date': date_range})
        template['symbol'] = symbol
        
        # Merge with the original data on symbol and date=first_effective_date
        # This correctly aligns the data with the dates when it becomes effective
        merged = pd.merge(template, df, 
                         left_on=['symbol', 'date'], 
                         right_on=['symbol', 'first_effective_date'], 
                         how='left')
        
        # Forward fill all columns except symbol and date
        merged = merged.ffill()
        
        # Sort by date for the final result
        merged = merged[merged.date.between(self.start_date, self.end_date)].sort_values('date')
        
        return merged
