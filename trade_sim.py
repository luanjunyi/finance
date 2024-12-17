import os
import csv
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
from tqdm.auto import tqdm
from fmp_data import FMPPriceLoader
from utils.logging_config import setup_logging

stock_data_folder = 'stock_data'
initial_fund = 1000000

# Set up logging with filename and line numbers
setup_logging()

def plot_portfolio_value(dates, portfolio_values):
    plt.figure(figsize=(10, 6))
    plt.plot(dates, portfolio_values, marker='o')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def read_trading_ops(trading_ops_file):
    """Read trading operations from a CSV file and return as a list."""
    with open(trading_ops_file, 'r') as file:
        reader = csv.reader(file)
        return [row for row in reader]


def backtest_trading(trading_ops, initial_fund, end_day, use_open_price_for_buy=False, plot=False, return_history=False):
    """
    Run trading simulation based on a list of trading operations.

    Args:
        trading_ops: List of operations, where each operation is a list containing
                    [symbol, date_str, action, fraction]
        initial_fund: Initial cash amount
        end_day: End date for the simulation
        plot: Whether to plot the results
        return_history: Whether to return history data
    """

    fund = initial_fund
    portfolio = {}
    portfolio_values = []
    dates = []

    price_loader = FMPPriceLoader()

    # Process trading operations
    for op in tqdm(trading_ops, desc="Processing trades"):
        assert len(
            op) == 4, f"Each operation must have 4 elements, got {len(op)}, op is {op}"

        symbol, date_str, action, fraction = op
        date = datetime.strptime(date_str, '%Y-%m-%d').date()
        if date > end_day:
            break

        # Calculate current total portfolio value (including cash) BEFORE the trade
        current_value = fund
        for sym, sh in portfolio.items():
            if sh > 0:
                try:
                    current_price, used_date = price_loader.get_last_available_price(sym, date, 'close')
                except Exception as e:
                    raise Exception(
                        f"Error getting price for {sym} on {date}: {str(e)}")
                if date.strftime('%Y-%m-%d') != used_date:
                    logging.warning(
                        f"Before execution portfolio value calculation used prevoious day price for {sym}, used {used_date} instead of {date}")
                current_value += sh * current_price

        price_type = 'close'
        try:
            price, date_used = price_loader.get_next_available_price(symbol, date, price_type)
        except Exception as e:
            raise Exception(
                f"Error getting price for {symbol} on {date}: {str(e)}")
        if date.strftime('%Y-%m-%d') != date_used:
            logging.warning(
                f"Buying used next day price for {symbol}, used {date_used} instead of {date}")

        # Calculate and store the portfolio value BEFORE the trade

        if action == 'BUY':
            # Convert fraction to float and calculate dollar amount to invest
            fraction = float(fraction)
            if use_open_price_for_buy:
                price, _ = price_loader.get_next_available_price(symbol, date, 'open')
            dollar_amount = current_value * fraction
            if dollar_amount > fund:
                raise Exception(
                    f"Insufficient funds to invest {fraction:.1%} of portfolio (${dollar_amount:.2f})")

            # Calculate shares to buy based on dollar amount
            shares = int(dollar_amount / price)
            cost = price * shares
            fund -= cost
            if symbol in portfolio:
                portfolio[symbol] += shares
            else:
                portfolio[symbol] = shares

        else:  # SELL
            assert action == 'SELL', "Action must be 'SELL' or 'BUY'"
            if symbol not in portfolio or portfolio[symbol] <= 0:
                raise Exception(f"No shares of {symbol} to sell")

            current_shares = portfolio[symbol]
            if fraction == 'ALL':
                shares_to_sell = current_shares
            else:
                # Convert fraction to float and calculate shares to sell
                fraction = float(fraction)
                shares_to_sell = int(current_shares * fraction)

            if shares_to_sell > current_shares:
                raise Exception(
                    f"Not enough shares of {symbol} to sell {shares_to_sell}")

            fund += price * shares_to_sell
            portfolio[symbol] -= shares_to_sell

        # Calculate and store the portfolio value AFTER the trade
        updated_value = fund
        for sym, sh in portfolio.items():
            if sh > 0:
                try:
                    current_price, date_used = price_loader.get_next_available_price(sym, date, 'close')
                except Exception as e:
                    raise Exception(
                        f"Error getting price for {sym} on {date}: {str(e)}")
                if date.strftime('%Y-%m-%d') != date_used:
                    logging.warning(
                        f"After execution portfolio value calculation used next day price for {sym}, used {date_used} instead of {date}")
                updated_value += sh * current_price

        portfolio_values.append(updated_value)  # Store the post-trade value
        dates.append(date)

    # Evaluate final portfolio value
    logging.info(f"Calculating final portfolio value")
    final_value = fund
    for symbol, shares in portfolio.items():
        if shares > 0:
            try:
                last_price, date_used = price_loader.get_next_available_price(symbol, end_day, 'close')
            except Exception as e:
                raise Exception(
                    f"Error getting price for {symbol} on {end_day}: {str(e)}")
            if end_day.strftime('%Y-%m-%d') != date_used:
                logging.warning(
                    f"Final portfolio value calculation used next day price for {symbol}, used {date_used} instead of {end_day}")
            final_value += shares * last_price

    # Call the plotting function
    if plot:
        plot_portfolio_value(dates, portfolio_values)
        print("graph plotted")

    if return_history:
        return portfolio_values, dates
    return portfolio_values[-1]  # Original return value


def parse_args():
    parser = argparse.ArgumentParser(description='Stock trading simulator')
    parser.add_argument('--trades', type=str, default='trading_ops.csv',
                        help='Path to the trading operations CSV file')
    parser.add_argument('--end_date', type=str, required=True,
                        help='End date for simulation in YYYY-MM-DD format')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Plot the portfolio value over time')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    end_day = datetime.strptime(args.end_date, '%Y-%m-%d').date()

    # Read operations from file, then pass to backtest_trading
    trading_ops = read_trading_ops(args.trades)
    final_fund = backtest_trading(
        trading_ops, initial_fund, end_day, plot=args.plot)
    print(f"Final fund value: ${final_fund:.2f}")
