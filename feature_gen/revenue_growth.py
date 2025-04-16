import pandas as pd
from typing import Optional
import sqlite3
from datetime import datetime
import numpy as np
from fmp_data import Dataset
import time
from tqdm import tqdm

class RevenueGrowth:
    """
    Feature that calculates year-over-year (YoY) growth of quarterly revenue for stocks.
    
    This feature uses the income_statement table to calculate the YoY growth rate
    of quarterly revenue for stocks from the valid_us_stocks_der table.
    Results are stored in the growth_features_der table in the database.
    """
    
    def __init__(self, db_path: str = '/Users/jluan/code/finance/data/fmp_data.db'):
        """
        Initialize the RevenueGrowth feature.
        
        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
    
    def _get_valid_symbols(self):
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
    
    def _create_growth_features_table(self):
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
    
    def _get_year_quarter(self, date_str):
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
                    sector = symbols_df[symbols_df['symbol'] == symbol]['sector'].iloc[0] if not symbols_df[symbols_df['symbol'] == symbol].empty else None
                    
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
        
        # Calculate sector_quantile for each sector and year-quarter
        result_df['sector_quantile'] = 0.0  # Default value
        
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
            ranks = group['yoy'].rank(method='average', pct=True)
            
            # Update the sector_quantile values
            for idx, rank in zip(group.index, ranks):
                result_df.loc[idx, 'sector_quantile'] = rank
        
        # Sort by symbol and date
        result_df = result_df.sort_values(['symbol', 'date'], ascending=[True, False])
        
        print(f"Calculation complete. Generated {len(result_df)} total records.")
        return result_df
    
    def store_in_database(self):
        """
        Calculate revenue growth and store results in the growth_features_der table.
        """
        start_time = time.time()
        
        # Create the table if it doesn't exist
        self._create_growth_features_table()
        print("Created/verified growth_features_der table in database.")
        
        # Calculate growth features
        print("\n=== Starting Revenue Growth Calculation ===\n")
        growth_df = self.calculate()
        
        if growth_df.empty:
            print("No growth data to store.")
            return
        
        # Store in database
        print("\n=== Beginning Database Write ===\n")
        with sqlite3.connect(self.db_path) as conn:
            print("Clearing existing data from growth_features_der table...")
            conn.execute("DELETE FROM growth_features_der")
            
            print(f"Writing {len(growth_df)} records to database...")
            growth_df.to_sql('growth_features_der', conn, if_exists='append', index=False)
            
            end_time = time.time()
            duration = end_time - start_time
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            
            print(f"\n=== Finished! ===\n")
            print(f"Stored {len(growth_df)} growth records in database.")
            print(f"Total processing time: {minutes} minutes and {seconds} seconds.")
    
    def get_revenue_growth(self) -> pd.DataFrame:
        """
        Get the revenue growth DataFrame from the database.
        
        Returns:
            DataFrame with columns: symbol, sector, date, yoy, sector_quantile, year_quarter
        """
        with sqlite3.connect(self.db_path) as conn:
            query = "SELECT * FROM growth_features_der ORDER BY symbol, date DESC"
            return pd.read_sql_query(query, conn)
