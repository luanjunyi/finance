import numpy as np
from typing import Tuple
from datetime import datetime, timedelta
from .base_feature import BaseFeature
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse

class AUC_SMA(BaseFeature):
    def __init__(self, symbol: str, current_date: str, look_back_days: int, slow_days: int, fast_days: int):
        """
        Initialize AUC_SMA feature calculator
        
        Args:
            symbol: Stock symbol
            current_date: The date to calculate the feature for (YYYY-MM-DD format)
            look_back_days: Number of trading days to look back for calculating the feature
            slow_days: Window size for slow SMA
            fast_days: Window size for fast SMA
        """
        super().__init__(symbol)
        if fast_days >= slow_days:
            raise ValueError("fast_days must be less than slow_days")
        if fast_days <= 0 or slow_days <= 0:
            raise ValueError("Both fast_days and slow_days must be positive integers")
        if look_back_days <= 0:
            raise ValueError("look_back_days must be positive")
            
        self.current_date = current_date if isinstance(current_date, str) else current_date.strftime('%Y-%m-%d')
        self.look_back_days = look_back_days
        self.slow_days = slow_days
        self.fast_days = fast_days
        
    def calculate_sma(self, prices: np.ndarray, window: int) -> np.ndarray:
        """
        Calculate Simple Moving Average for the given window size.
        Returns array of same length as input, with first (window-1) elements removed
        to avoid partial windows.
        """
        if prices is None or len(prices) == 0:
            raise ValueError("Input prices array cannot be None or empty")
        if np.any(np.isnan(prices)):
            raise ValueError("Input prices array contains NaN values")
        if len(prices) < window:
            raise ValueError(f"Input array length ({len(prices)}) must be >= window size ({window})")
            
        weights = np.ones(window) / window
        return np.convolve(prices, weights, mode='valid')

    def calculate_auc(self, curve1: np.ndarray, curve2: np.ndarray) -> float:
        """
        Calculate the area under the curve between two curves, but only where curve1 > curve2.
        
        Args:
            curve1: First curve
            curve2: Second curve
            
        Returns:
            float: Area where curve1 > curve2, 0 otherwise
        """
        if curve1 is None or curve2 is None:
            raise ValueError("Input curves cannot be None")
        if len(curve1) != len(curve2):
            raise ValueError("Input curves must have the same length")
            
        if np.isnan(curve1).any() or np.isnan(curve2).any():
            raise ValueError("Input curves contain NaN values")
            
        # Only consider regions where curve1 > curve2
        diff = np.where(curve1 > curve2, curve1 - curve2, 0)
        return np.trapz(diff)

    def calculate(self) -> float:
        """
        Calculate the AUC-based momentum metric.
        
        Returns:
            float: (A_high - A_low) / A_s where:
                  A_high: AUC between fast and slow SMA where fast > slow
                  A_low: AUC between slow and fast SMA where slow > fast
                  A_s: AUC of slow SMA itself
                  
        Raises:
            ValueError: If input data is invalid or calculations result in undefined values
        """
        # Need enough days for both SMA calculation and the look-back window
        required_days = self.slow_days + self.look_back_days - 1
        price_dict = self.get_close_price_for_the_last_days(self.current_date, required_days)
        
        if len(price_dict) < required_days:
            raise ValueError(f"Insufficient price data. Need {required_days} trading days, got {len(price_dict)}")
        
        # Convert dict to sorted numpy array (oldest to newest)
        dates = sorted(price_dict.keys())
        prices = np.array([price_dict[date] for date in dates])
        
        # Calculate SMAs
        slow_sma = self.calculate_sma(prices, self.slow_days)
        fast_sma = self.calculate_sma(prices, self.fast_days)
        
        # Trim the arrays to the look_back_days window
        # Since calculate_sma removes (window-1) values from the beginning,
        # we need to take the last look_back_days values
        slow_sma = slow_sma[-self.look_back_days:]
        fast_sma = fast_sma[-self.look_back_days:]
        
        # Calculate areas
        A_high = self.calculate_auc(fast_sma, slow_sma)  # Only counts where fast > slow
        A_low = self.calculate_auc(slow_sma, fast_sma)   # Only counts where slow > fast
        A_s = np.trapz(slow_sma)  # Total area under slow SMA
        
        if A_s == 0:
            raise ValueError("A_s (AUC of slow SMA) is zero, cannot normalize the result")
            
        return (A_high - A_low) / abs(A_s)
        
    def plot_analysis(self):
        """Plot price history, SMAs, and AUCs"""
        # Need enough days for both SMA calculation and the look-back window
        required_days = max(self.slow_days, self.fast_days) + self.look_back_days - 1
        price_dict = self.get_close_price_for_the_last_days(self.current_date, required_days)
        
        if len(price_dict) < required_days:
            raise ValueError(f"Insufficient price data. Need {required_days} trading days, got {len(price_dict)}")
        
        # Convert to sorted arrays (oldest to newest)
        dates = sorted(price_dict.keys())
        prices = np.array([price_dict[date] for date in dates])
        dates = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
        
        # Calculate SMAs
        slow_sma = self.calculate_sma(prices, self.slow_days)
        fast_sma = self.calculate_sma(prices, self.fast_days)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(15, 8))
        
        # Plot only the look_back_days window
        plot_dates = dates[-(self.look_back_days + max(self.slow_days, self.fast_days) - 1):]
        plot_prices = prices[-(self.look_back_days + max(self.slow_days, self.fast_days) - 1):]
        
        ax.plot(plot_dates, plot_prices, label='Price', color='gray', alpha=0.5)
        ax.plot(plot_dates[self.slow_days-1:], slow_sma, label=f'{self.slow_days}-day SMA', color='blue')
        ax.plot(plot_dates[self.fast_days-1:], fast_sma, label=f'{self.fast_days}-day SMA', color='red')
        
        # Fill areas between curves for the look_back_days window
        valid_dates = plot_dates[max(self.slow_days, self.fast_days)-1:]
        
        # Fill areas between curves
        ax.fill_between(valid_dates, slow_sma[-self.look_back_days:], fast_sma[-self.look_back_days:],
                       where=fast_sma[-self.look_back_days:] >= slow_sma[-self.look_back_days:],
                       color='green', alpha=0.3, label='A_high')
        ax.fill_between(valid_dates, slow_sma[-self.look_back_days:], fast_sma[-self.look_back_days:],
                       where=fast_sma[-self.look_back_days:] <= slow_sma[-self.look_back_days:],
                       color='red', alpha=0.3, label='A_low')
        
        # Customize plot
        ax.set_title(f'{self.symbol} Price and SMA Analysis\n{self.current_date} (Past {self.look_back_days} trading days)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        

        plt.show()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate AUC-SMA momentum metric')
    parser.add_argument('symbol', type=str, help='Stock symbol')
    parser.add_argument('current_date', type=str, help='Current date (YYYY-MM-DD)')
    parser.add_argument('--look-back', type=int, default=120, help='Number of trading days to look back (default: 120)')
    parser.add_argument('--slow', type=int, default=90, help='Slow SMA window (default: 90)')
    parser.add_argument('--fast', type=int, default=30, help='Fast SMA window (default: 30)')
    
    args = parser.parse_args()
    
    try:
        # Create and calculate feature
        feature = AUC_SMA(args.symbol, args.current_date, args.look_back, args.slow, args.fast)
        metric = feature.calculate()
        
        # Print results
        print(f"\nResults for {args.symbol} on {args.current_date}:")
        print(f"Look-back period: {args.look_back} trading days")
        print(f"Slow SMA: {args.slow} days")
        print(f"Fast SMA: {args.fast} days")
        print(f"Momentum Metric: {metric:.4f}")
        
        # Generate plot
        feature.plot_analysis()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)