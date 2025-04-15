import pandas as pd
from typing import List, Optional
from datetime import datetime, timedelta
import sqlite3
from fmp_data import Dataset

class RevenueGrowth:
    """
    Feature that calculates year-over-year (YoY) growth of quarterly revenue for stocks.
    
    This feature uses the income_statement table to calculate the YoY growth rate
    of quarterly revenue for each stock in the provided list of symbols.
    """
    
    def __init__(self, symbols: List[str] | str, calculation_date: Optional[str] = None, 
                 db_path: str = '/Users/jluan/code/finance/data/fmp_data.db'):
        """
        Initialize the RevenueGrowth feature.
        
        Args:
            symbols: Stock symbol or list of stock symbols to calculate revenue growth for
            calculation_date: Date for which to calculate the growth (format: 'YYYY-MM-DD')
                              If None, all available dates will be included
            db_path: Path to the SQLite database
        """
        self.symbols = symbols if isinstance(symbols, list) else [symbols]
        self.calculation_date = calculation_date
        self.db_path = db_path
        
    def calculate(self) -> pd.DataFrame:
        """
        Calculate the year-over-year revenue growth for each symbol.
        
        Returns:
            DataFrame with columns: symbol, sector, date, yoy
        """
        # Get sector information for each symbol using SQL
        # (since sector is not available in Dataset yet)
        with sqlite3.connect(self.db_path) as conn:
            symbol_placeholders = ','.join(['?' for _ in self.symbols])
            sector_query = f"""
            SELECT symbol, sector
            FROM stock_symbol
            WHERE symbol IN ({symbol_placeholders})
            """
            sector_df = pd.read_sql_query(sector_query, conn, params=tuple(self.symbols))
        
        # Get quarterly revenue data for each symbol
        # We need historical data, not just the most recent point
        # So we'll create a dataset without for_date to get all historical data
        revenue_metrics = {'revenue': 'revenue'}
        
        revenue_dataset = Dataset(
            symbol=self.symbols,
            metrics=revenue_metrics,
            db_path=self.db_path
        )
        
        # Get the data (already contains quarterly reports)
        revenue_df = revenue_dataset.get_data()
        
        # If calculation_date is provided, filter the data
        if self.calculation_date:
            revenue_df = revenue_df[revenue_df['date'] <= self.calculation_date]
        
        # Convert date to datetime
        revenue_df.loc[:, 'date'] = pd.to_datetime(revenue_df['date'])
        
        # Calculate YoY growth
        results = []
        
        for symbol in self.symbols:
            # Get data for this symbol
            symbol_data = revenue_df[revenue_df['symbol'] == symbol].copy()
            
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
                
                if len(prev_year_quarters) == 0:
                    continue
                
                prev_year_quarter = prev_year_quarters.iloc[0]
                
                # Calculate YoY growth
                current_revenue = current_quarter['revenue']
                prev_year_revenue = prev_year_quarter['revenue']
                
                if prev_year_revenue and prev_year_revenue != 0:
                    yoy_growth = (current_revenue - prev_year_revenue) / abs(prev_year_revenue)
                    
                    # Get sector for this symbol
                    sector = sector_df[sector_df['symbol'] == symbol]['sector'].iloc[0] if not sector_df[sector_df['symbol'] == symbol].empty else None
                    
                    results.append({
                        'symbol': symbol,
                        'sector': sector,
                        'date': current_date.strftime('%Y-%m-%d'),
                        'yoy': yoy_growth
                    })
        
        # Convert results to DataFrame
        result_df = pd.DataFrame(results)
        
        # If calculation_date was provided, filter to only include data up to that date
        if self.calculation_date:
            result_df = result_df[result_df['date'] <= self.calculation_date]
            
        # Sort by symbol and date
        result_df = result_df.sort_values(['symbol', 'date'], ascending=[True, False])
        
        return result_df
    
    def get_revenue_growth(self) -> pd.DataFrame:
        """
        Get the revenue growth DataFrame.
        
        Returns:
            DataFrame with columns: symbol, sector, date, yoy
        """
        return self.calculate()
