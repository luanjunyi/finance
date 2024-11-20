#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from trade_sim_util import FMPPriceLoader
import logging
from statistics import mean, median
from datetime import datetime, timedelta
import argparse
from utils.logging_config import setup_logging

# Set up logging with filename and line numbers
setup_logging()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Calculate returns after holding stocks for specified days')
    parser.add_argument('input_file', help='Path to CSV file containing trading operations')
    parser.add_argument('--days', type=int, default=5, help='Number of days to hold (default: 5)')
    return parser.parse_args()

def hold_days_after_buy(csv_file, days):
    """
    Calculate the normalized daily return after holding for specified number of days after each BUY operation.
    Return is normalized using the formula: (1 + total_return)^(1/days_held) - 1
    If the target close date is not available, uses the last available price.
    
    Args:
        csv_file (str): Path to CSV file containing trading operations
        days (int): Number of days to hold after buying
        
    Returns:
        DataFrame: DataFrame containing normalized daily returns for each BUY operation
    """
    # Read the CSV file
    df = pd.read_csv(csv_file, names=['Symbol', 'Date', 'Operation', 'Amount'])
    
    # Filter for BUY operations only
    buy_ops = df[df['Operation'] == 'BUY']
    
    # Initialize price loader
    price_loader = FMPPriceLoader()
    
    returns = []
    failed_symbols = []
    
    # Calculate return for each BUY operation
    for _, row in buy_ops.iterrows():
        symbol = row['Symbol']
        buy_date = row['Date']
        
        try:
            # Get buy day open price
            open_price = price_loader.get_price(symbol, buy_date, 'Open')
            
            # Calculate the target date after holding period
            buy_datetime = datetime.strptime(buy_date, '%Y-%m-%d')
            target_date = (buy_datetime + timedelta(days=days)).strftime('%Y-%m-%d')
            
            try:
                # First try to get the exact target date close price
                close_price = price_loader.get_price(symbol, target_date, 'Close')
                actual_date = target_date
            except KeyError:
                # If target date not available, get the last available price
                close_price, actual_date = price_loader.get_last_available_price(symbol, target_date, 'Close')
            
            # Calculate return
            total_return = (close_price / open_price) - 1
            
            # Calculate actual days held
            actual_datetime = datetime.strptime(actual_date, '%Y-%m-%d')
            actual_days_held = (actual_datetime - buy_datetime).days
            
            # Calculate normalized daily return
            if actual_days_held <= 0:
                raise ValueError(f"Actual days held must be greater than 0, but got {actual_days_held}")
            normalized_return = ((1 + total_return) ** (1.0 / actual_days_held) - 1)
            
            returns.append({
                'symbol': symbol,
                'buy_date': buy_date,
                'sell_date': actual_date,
                'target_date': target_date,
                'days_held': actual_days_held,
                'raw_return': total_return * 100,  # Convert to percentage
                'normalized_return': normalized_return * 100,  # Convert to percentage
                'used_last_available': actual_date != target_date
            })
        except Exception as e:
            failed_symbols.append((symbol, buy_date, str(e)))
            logging.warning(f"Failed to get prices for {symbol} bought on {buy_date}: {e}")
    
    # Convert to DataFrame for easier analysis
    returns_df = pd.DataFrame(returns)
    
    # Calculate statistics
    avg_return = mean(returns_df['normalized_return'])
    med_return = median(returns_df['normalized_return'])
    
    # Print summary statistics
    logging.info(f"\nSummary Statistics for {days}-day hold period:")
    logging.info(f"Number of BUY operations analyzed: {len(returns_df)}")
    logging.info(f"Average normalized daily return: {avg_return:.2f}%")
    logging.info(f"Median normalized daily return: {med_return:.2f}%")
    
    # Count operations using last available price
    if not returns_df.empty:
        last_available_count = returns_df['used_last_available'].sum()
        if last_available_count > 0:
            logging.info(f"Operations using last available price: {last_available_count}")
    
    if failed_symbols:
        logging.info("\nFailed to get prices for:")
        for symbol, date, error in failed_symbols:
            logging.info(f"  {symbol} bought on {date}: {error}")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    
    # Calculate symmetric range around 0 for x-axis
    max_abs_return = max(abs(returns_df['normalized_return'].max()), abs(returns_df['normalized_return'].min()))
    bin_range = (-max_abs_return, max_abs_return)
    
    plt.hist(returns_df['normalized_return'], bins=30, edgecolor='black', range=bin_range)
    plt.title(f'Distribution of {days}-Day Hold Normalized Daily Returns')
    plt.xlabel('Normalized Daily Return (%)')
    plt.ylabel('Frequency')
    
    # Add vertical lines for mean, median, and zero
    plt.axvline(0, color='k', linestyle='solid', linewidth=1, label='Zero')
    plt.axvline(avg_return, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {avg_return:.2f}%')
    plt.axvline(med_return, color='g', linestyle='dashed', linewidth=2, label=f'Median: {med_return:.2f}%')
    plt.legend()
    
    # Set x-axis ticks to be symmetric around 0
    plt.xlim(bin_range)
    
    plt.grid(True, alpha=0.3)
    plt.show()
    
    return returns_df

if __name__ == '__main__':
    args = parse_args()
    returns = hold_days_after_buy(args.input_file, args.days)
