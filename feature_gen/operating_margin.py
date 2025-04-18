import pandas as pd
from typing import Optional
import sqlite3
from datetime import datetime
import numpy as np
from fmp_data import Dataset
import time
from tqdm import tqdm
from feature_gen.base_feature import FinancialFeatureBase

class OperatingMargin(FinancialFeatureBase):
    """
    Feature that calculates operating margin for stocks.
    
    This feature uses the income_statement table to fetch the operating margin
    for stocks from the valid_us_stocks_der table.
    Results are stored in the operating_margin_features_der table in the database.
    """
    
    def _create_features_table(self):
        """
        Create the operating_margin_features_der table if it doesn't exist.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS operating_margin_features_der (
                symbol VARCHAR(10),
                sector VARCHAR(255),
                date DATE,
                operating_margin DECIMAL(10, 4),
                sector_quantile DECIMAL(10, 4),
                year_quarter VARCHAR(6),
                PRIMARY KEY (symbol, date)
            )
            """)
    
    def calculate(self) -> pd.DataFrame:
        """
        Calculate the operating margin for all valid symbols.
        
        Returns:
            DataFrame with columns: symbol, sector, date, operating_margin, sector_quantile, year_quarter
        """
        # Get valid symbols and their sectors
        symbols_df = self._get_valid_symbols()
        symbols = symbols_df['symbol'].tolist()
        print(f"Loaded {len(symbols)} symbols from valid_us_stocks_der. Beginning calculation...")
        
        # Get operating margin data for all symbols
        margin_metrics = {'operating_income_ratio': 'operating_margin'}
        
        margin_dataset = Dataset(
            symbol=symbols,
            metrics=margin_metrics,
            db_path=self.db_path
        )
        
        # Get the data (already contains quarterly reports)
        margin_df = margin_dataset.get_data()
        print(f"Loaded operating margin data with {len(margin_df)} records.")
        
        # Make a copy of the DataFrame to avoid SettingWithCopyWarning
        margin_df = margin_df.copy()

        # Convert date to datetime
        margin_df.loc[:, 'date'] = pd.to_datetime(margin_df['date'])
        
        # Prepare results
        results = []
        
        # Create progress bar
        print("Processing operating margin data for each symbol...")
        progress_bar = tqdm(symbols, desc="Processing symbols", unit="symbol")
        
        for symbol in progress_bar:
            # Update progress bar description
            progress_bar.set_description(f"Processing {symbol}")
            
            # Get data for this symbol
            symbol_data = margin_df[margin_df['symbol'] == symbol]
            
            if len(symbol_data) < 1:  # Need at least one record
                continue
                
            # Sort by date in descending order
            symbol_data = symbol_data.sort_values('date', ascending=False)
            
            # Process each quarter's data
            for i in range(len(symbol_data)):
                current_quarter = symbol_data.iloc[i]
                current_date = current_quarter['date']
                
                # Get operating margin
                operating_margin = current_quarter['operating_margin']
                
                # Skip if operating margin is None or NaN
                if operating_margin is None or pd.isna(operating_margin):
                    continue
                
                # Get sector for this symbol
                sector = symbols_df[symbols_df['symbol'] == symbol]['sector'].iloc[0]
                
                # Format date as string
                date_str = current_date.strftime('%Y-%m-%d')
                
                # Get year-quarter
                year_quarter = self._get_year_quarter(date_str)
                
                results.append({
                    'symbol': symbol,
                    'sector': sector,
                    'date': date_str,
                    'operating_margin': operating_margin,
                    'year_quarter': year_quarter
                })
        
        # Convert results to DataFrame
        result_df = pd.DataFrame(results)
        
        if result_df.empty:
            print("No results found. Returning empty DataFrame.")
            return result_df
        
        print(f"Generated {len(result_df)} operating margin records. Calculating sector quantiles...")
        
        # Use the base class method to calculate sector quantiles
        result_df = self.calculate_sector_quantiles(result_df, 'operating_margin')
        
        print(f"Calculation complete. Generated {len(result_df)} total records.")
        return result_df
    
    def _get_table_name(self) -> str:
        """
        Get the name of the database table where operating margin features are stored.
        
        Returns:
            String with the table name
        """
        return 'operating_margin_features_der'
