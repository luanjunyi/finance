import pandas as pd
from typing import List
from datetime import datetime
import matplotlib.pyplot as plt

from fmp_data import FMPPriceLoader


class SMA:
    def __init__(self, symbols: List[str], begin_date: str, end_date: str):
        self.symbols = symbols
        self.price_loader = FMPPriceLoader()
        self.prices = self.price_loader.get_price_for_stocks_during(
            self.symbols, begin_date, end_date)
        # Sort prices by date for each symbol
        self.prices = {symbol: sorted(prices, key=lambda x: x[0]) for symbol, prices in self.prices.items()}

    def values(self, window_sizes: List[int]):
        # Create a list to store DataFrames for each symbol
        symbol_dfs = []
        window_sizes.sort()
        
        for symbol in self.symbols:
            prices = self.prices[symbol]
            # Create DataFrame for this symbol
            df = pd.DataFrame(prices, columns=['date', 'price'])  # type: ignore
            
            # Calculate SMA for each window size
            for window in window_sizes:
                df[f'sma_{window}'] = df['price'].rolling(window=window).mean()
            
            df['symbol'] = symbol
            symbol_dfs.append(df)

        # Concatenate all symbol DataFrames
        result_df = pd.concat(symbol_dfs, ignore_index=True)
        # Drop the price column and reorder columns
        cols = ['symbol', 'date'] + [f'sma_{w}' for w in window_sizes]
        self.values_cache = result_df[cols]  # type: ignore
        return self.values_cache

    def has_open_mouth_trend(self, symbol: str, date_str: str, for_days: int,
        windows: List[int], tolerance: float=0.0):
        """
        Check if SMAs are in decreasing order for each date in the range.
        
        Args:
            symbol: The stock symbol
            date_str: The reference date
            for_days: Number of days to look back
            windows: List of window sizes to check
            
        Returns:
            bool: True if SMAs are in decreasing order for all dates in range
            
        Raises:
            ValueError: If the requested date range is outside the available data range
        """
        # Sort windows to ensure we check in ascending order
        windows = sorted(windows)
        
        # Get data for the symbol
        df = self.values_cache[self.values_cache.symbol == symbol].copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        # If date_str not in dataframe, use the last date before it
        target_date = pd.to_datetime(date_str)
        if target_date not in df['date'].values:
            target_date = df[df['date'] < target_date]['date'].max()
            if pd.isna(target_date):  # No valid date found
                return False
                
        # Get the start date for our check
        start_date = target_date - pd.Timedelta(days=for_days)
        
        # Check if date range is within available data
        assert min_date <= start_date <=  target_date <= max_date, \
            f"Requested date range [{start_date}, {target_date}] is outside available data range [{min_date}, {max_date}]"
        # Filter data to our date range
        mask = (df['date'] >= start_date) & (df['date'] <= target_date)
        df = df[mask]
            
        # For each date in our range
        for _, row in df.iterrows():
            # Check if SMAs are in decreasing order
            sma_values = [row[f'sma_{w}'] for w in windows]
            
            # Return false if any SMA is NaN
            if any(pd.isna(v) for v in sma_values):
                return False
                
            # Check if values are strictly decreasing
            if not all(sma_values[i] >= sma_values[i+1] * (1-tolerance) for i in range(len(sma_values)-1)):
                return False
                
        return True

    def plot(self, symbol):
        assert symbol in self.prices
        df = self.values_cache[self.values_cache.symbol == symbol]
        
        plt.figure(figsize=(12, 6))
        # Plot the original price first
        prices_df = pd.DataFrame(self.prices[symbol], columns=['date', 'price'])  # type: ignore
        plt.plot(prices_df['date'], prices_df['price'], label='price', color='black', alpha=0.3)
        
        sma_columns = [col for col in df.columns if col.startswith('sma_')] # type: ignore
        
        for col in sma_columns:
            plt.plot(df['date'], df[col], label=col)
            
        plt.title(f'{symbol} SMAs')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        # Reduce x-axis label crowding
        plt.gca().xaxis.set_major_locator(plt.MaxNLocator(30))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()