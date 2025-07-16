from functools import cached_property
from fmp_data import FMPPriceLoader
import pandas as pd
from typing import Optional, Dict, List
import sqlite3
from utils.config import FMP_DB_PATH
from datetime import datetime
import numpy as np
from fmp_data import Dataset
import time
from tqdm import tqdm
from abc import ABC, abstractmethod

class BaseFeature:
    def __init__(self, symbol):
        self.symbol = symbol
        self.price_loader = FMPPriceLoader()
    
    @cached_property
    def value(self):
        return self.calculate()

    def calculate(self):
        raise NotImplementedError

    def get_close_price_during(self, start_date, end_date):
        return self.price_loader.get_close_price_during(self.symbol, start_date, end_date)

    def get_close_price_for_the_last_days(self, last_date, num_days):
        return self.price_loader.get_close_price_for_the_last_days(self.symbol, last_date, num_days)


class FinancialFeatureBase(ABC):
    """
    Base class for financial features that calculate metrics across stocks.
    
    This base class provides common functionality for feature generators,
    including database access, symbol retrieval, and sector quantile calculation.
    """
    
    def __init__(self, db_path: str = FMP_DB_PATH):
        """
        Initialize the feature generator.
        
        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
    
    def _get_valid_symbols(self) -> pd.DataFrame:
        """
        Get valid symbols and their sectors from the valid_us_stocks_der table.
        
        Returns:
            DataFrame with columns: symbol, sector
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
            SELECT symbol, sector
            FROM valid_us_stocks_der
            """
            return pd.read_sql_query(query, conn)
    
    @abstractmethod
    def _create_features_table(self):
        """
        Create the features table if it doesn't exist.
        This method must be implemented by subclasses.
        """
        pass
    
    def _get_year_quarter(self, date_str: str) -> str:
        """
        Convert a date string to year-quarter format (e.g., '2024-09-30' -> '2024Q3').
        
        Args:
            date_str: Date string in format 'YYYY-MM-DD'
            
        Returns:
            String in format 'YYYYQN' where N is the quarter number (1-4)
        """
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        year = date_obj.year
        month = date_obj.month
        quarter = (month - 1) // 3 + 1
        return f"{year}Q{quarter}"
    
    def calculate_sector_quantiles(self, df: pd.DataFrame, feature_column: str) -> pd.DataFrame:
        """
        Calculate sector quantiles for a feature within each sector and year-quarter.
        
        Args:
            df: DataFrame with columns: symbol, sector, date, [feature_column], year_quarter
            feature_column: Name of the column to calculate quantiles for
            
        Returns:
            DataFrame with sector_quantile column added
        """
        if df.empty:
            return df
            
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Initialize sector_quantile column
        result_df['sector_quantile'] = 0.0
        
        # Group by sector and year_quarter
        sector_year_quarters = result_df.groupby(['sector', 'year_quarter']).size().reset_index()
        print(f"Found {len(sector_year_quarters)} unique sector-quarter combinations")
        
        # Create progress bar for sector quantile calculation
        progress_bar = tqdm(sector_year_quarters.iterrows(), 
                           total=len(sector_year_quarters),
                           desc="Calculating sector quantiles", 
                           unit="sector-quarter")
        
        for _, row in progress_bar:
            sector = row['sector']
            year_quarter = row['year_quarter']
            progress_bar.set_description(f"Processing {sector} {year_quarter}")
            
            # Get group for this sector and year_quarter
            group = result_df[(result_df['sector'] == sector) & 
                             (result_df['year_quarter'] == year_quarter)]
            
            # Calculate the rank as a percentile within this sector and year-quarter
            ranks = group[feature_column].rank(method='average', pct=True)
            
            # Update the sector_quantile values
            for idx, rank in zip(group.index, ranks):
                result_df.loc[idx, 'sector_quantile'] = rank
        
        # Sort by symbol and date
        result_df = result_df.sort_values(['symbol', 'date'], ascending=[True, False])
        
        return result_df
    
    @abstractmethod
    def calculate(self) -> pd.DataFrame:
        """
        Calculate the feature for all valid symbols.
        This method must be implemented by subclasses.
        
        Returns:
            DataFrame with feature data
        """
        pass
    
    def _get_table_name(self) -> str:
        """
        Get the name of the database table where features are stored.
        This method should be implemented by subclasses.
        
        Returns:
            String with the table name
        """
        raise NotImplementedError("Subclasses must implement _get_table_name")
    
    def store_in_database(self):
        """
        Calculate features and store results in the database table returned by _get_table_name().
        """
        # Create the table if it doesn't exist
        self._create_features_table()
        print(f"Created/verified {self._get_table_name()} table in database.")
        
        # Calculate features
        print(f"\n=== Starting {self.__class__.__name__} Calculation ===\n")
        feature_df = self.calculate()
        
        if feature_df.empty:
            print("No feature data to store.")
            return
        
        # Store in database
        print("\n=== Beginning Database Write ===\n")
        with sqlite3.connect(self.db_path) as conn:
            table_name = self._get_table_name()
            
            # First drop the table if it exists
            print(f"Dropping {table_name} table if it exists...")
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
            
            # Then create it again
            self._create_features_table()
            
            print(f"Writing {len(feature_df)} records to database...")
            # index=False means don't include the DataFrame's index as a column in the SQL table
            # if_exists='replace' will replace the table if it exists, which is what we want
            # since we just created a fresh table
            feature_df.to_sql(table_name, conn, if_exists='replace', index=False)
            
            print(f"\n=== Finished! ===\n")
            print(f"Stored {len(feature_df)} records in {table_name} table.")
        
    def get_feature_data(self) -> pd.DataFrame:
        """
        Get the feature data from the database using the table name from _get_table_name().
        
        Returns:
            DataFrame with feature data
        """
        table_name = self._get_table_name()
        with sqlite3.connect(self.db_path) as conn:
            query = f"SELECT * FROM {table_name} ORDER BY symbol, date DESC"
            return pd.read_sql_query(query, conn)