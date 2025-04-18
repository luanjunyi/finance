import pandas as pd
from typing import Optional, Dict, List
import sqlite3
from datetime import datetime
import numpy as np
from fmp_data import Dataset
import time
from tqdm import tqdm
from feature_gen.base_feature import FinancialFeatureBase


class RevenueGrowth(FinancialFeatureBase):
    """
    Feature that calculates year-over-year (YoY) growth of quarterly revenue for stocks.
    
    This feature uses the income_statement table to calculate the YoY growth rate
    of quarterly revenue for stocks from the valid_us_stocks_der table.
    Results are stored in the growth_features_der table in the database.
    """
    
    def _create_features_table(self):
        """
        Create the growth_features_der table if it doesn't exist.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS growth_features_der (
                symbol VARCHAR(10),
                sector VARCHAR(255),
                date DATE,
                yoy DECIMAL(10, 4),
                sector_quantile DECIMAL(10, 4),
                year_quarter VARCHAR(6),
                PRIMARY KEY (symbol, date)
            )
            """)
    
    def calculate(self) -> pd.DataFrame:
        """
        Calculate the year-over-year revenue growth for all valid symbols.
        
        Returns:
            DataFrame with columns: symbol, sector, date, yoy, sector_quantile, year_quarter
        """
        # Get valid symbols and their sectors
        symbols_df = self._get_valid_symbols()
        symbols = symbols_df['symbol'].tolist()
        print(f"Loaded {len(symbols)} symbols from valid_us_stocks_der. Beginning calculation...")
        
        # Get quarterly revenue data for all symbols
        revenue_metrics = {'revenue': 'revenue'}
        
        revenue_dataset = Dataset(
            symbol=symbols,
            metrics=revenue_metrics,
            db_path=self.db_path
        )
        
        # Get the data (already contains quarterly reports)
        revenue_df = revenue_dataset.get_data()
        print(f"Loaded revenue data with {len(revenue_df)} records.")
        
        # Make a copy of the DataFrame to avoid SettingWithCopyWarning
        revenue_df = revenue_df.copy()

        # Convert date to datetime
        revenue_df.loc[:, 'date'] = pd.to_datetime(revenue_df['date'])
        
        # Calculate YoY growth
        results = []
        
        # Create progress bar
        print("Calculating YoY growth for each symbol...")
        progress_bar = tqdm(symbols, desc="Processing symbols", unit="symbol")
        
        for symbol in progress_bar:
            # Update progress bar description
            progress_bar.set_description(f"Processing {symbol}")
            
            # Get data for this symbol
            symbol_data = revenue_df[revenue_df['symbol'] == symbol]
            
            if len(symbol_data) < 5:  # Need at least 5 quarters to calculate YoY growth
                continue
                
            # Sort by date in descending order
            symbol_data = symbol_data.sort_values('date', ascending=False)
            
            # For each quarter, find the corresponding quarter from previous year
            for i in range(len(symbol_data) - 4):  # -4 to ensure we have data from previous year
                current_quarter = symbol_data.iloc[i]
                current_date = current_quarter['date']
                
                # Find the quarter from previous year (approximately 4 quarters back)
                # We need to find the exact same quarter from previous year
                prev_year_date = current_date - pd.DateOffset(years=1)
                
                # Find the closest quarter to prev_year_date
                prev_year_quarters = symbol_data[
                    (symbol_data['date'] <= prev_year_date + pd.DateOffset(days=45)) & 
                    (symbol_data['date'] >= prev_year_date - pd.DateOffset(days=45))
                ]
                assert len(prev_year_quarters) <= 1, "Multiple previous quarters found for the same date"
                if len(prev_year_quarters) == 0:
                    continue
                
                prev_year_quarter = prev_year_quarters.iloc[0]
                
                # Calculate YoY growth
                current_revenue = current_quarter['revenue']
                prev_year_revenue = prev_year_quarter['revenue']
                
                if prev_year_revenue and prev_year_revenue != 0:
                    yoy_growth = (current_revenue - prev_year_revenue) / abs(prev_year_revenue)
                    
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
                        'yoy': yoy_growth,
                        'year_quarter': year_quarter
                    })
        
        # Convert results to DataFrame
        result_df = pd.DataFrame(results)
        
        if result_df.empty:
            print("No results found. Returning empty DataFrame.")
            return result_df
        
        print(f"Generated {len(result_df)} growth records. Calculating sector quantiles...")
        
        # Use the base class method to calculate sector quantiles
        result_df = self.calculate_sector_quantiles(result_df, 'yoy')
        
        print(f"Calculation complete. Generated {len(result_df)} total records.")
        return result_df
    
    def _get_table_name(self) -> str:
        """
        Get the name of the database table where revenue growth features are stored.
        
        Returns:
            String with the table name
        """
        return 'growth_features_der'

