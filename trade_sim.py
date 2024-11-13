import os
import csv
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import logging
from tqdm.auto import tqdm
from trade_sim_util import PriceLoader

stock_data_folder = 'stock_data'
initial_fund = 1000000


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


def get_price(symbol, date, price_type):
    assert price_type in ['Open', 'High', 'Low',
                          'Close'], "price_type must be one of 'Open', 'High', 'Low', 'Close'"

    stock_file = os.path.join(stock_data_folder, f"{symbol}_daily.csv")
    if not os.path.exists(stock_file):
        raise FileNotFoundError(
            f"Error: Stock data file not found for {symbol}")

    try:
        stock_data = pd.read_csv(stock_file)
        stock_data['Date'] = pd.to_datetime(
            stock_data['Date'], utc=True).dt.date
        stock_data = stock_data.set_index('Date')
    except Exception as e:
        raise Exception(f"Error reading stock data file for {symbol}: {e}")

    if date not in stock_data.index:
        for i in range(1, 5):
            new_date = date - pd.Timedelta(days=i)
            if new_date in stock_data.index:
                date = new_date
                logging.debug(
                    f"No date {date} for {symbol}, using {new_date} instead")
                break
        else:
            raise KeyError(
                f"Date {date} not found in historical data for {symbol} and no recent data within 4 days")

    return stock_data.loc[date, price_type]


def get_price_with_delisted(price_loader, symbol, date, price_type, portfolio):
    """
    Get price data with fallback handling for delisted stocks.
    
    Args:
        price_loader: PriceLoader instance
        symbol: Stock symbol
        date: Date to get price for
        price_type: Type of price ('Open', 'High', 'Low', 'Close')
        portfolio: Current portfolio dictionary
    """
    default_prices = {
        'TA': 156.87,
        'LTHM': 68.26,
    }

    try:
        return price_loader.get_price(symbol, date, price_type)
    except KeyError:
        if symbol in portfolio:
            if symbol in default_prices:
                logging.debug(f"Used default price for listed stocks found for {symbol}")
                return default_prices[symbol]
            else:
                raise KeyError(f"Default price for delisted stocks not found for {symbol}")
        else:
            raise FileNotFoundError(f"Error: Stock data file not found for {symbol} and symbol not in portfolio")


def read_trading_ops(trading_ops_file):
    """Read trading operations from a CSV file and return as a list."""
    with open(trading_ops_file, 'r') as file:
        reader = csv.reader(file)
        return [row for row in reader]


def backtest_trading(trading_ops, initial_fund, end_day, plot=False, return_history=False):
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

    price_loader = PriceLoader()

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
                current_price = get_price_with_delisted(
                    price_loader, sym, date, 'Open', portfolio)
                current_value += sh * current_price

        price_type = 'Open' if action == 'BUY' else 'Close'
        price = get_price_with_delisted(price_loader, symbol, date, price_type, portfolio)

        if action == 'BUY':
            # Convert fraction to float and calculate dollar amount to invest
            fraction = float(fraction)
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
                current_price = get_price_with_delisted(
                    price_loader, sym, date, 'Close', portfolio)
                updated_value += sh * current_price

        portfolio_values.append(updated_value)  # Store the post-trade value
        dates.append(date)

    # Evaluate final portfolio value
    logging.info(f"Calculating final portfolio value")
    final_value = fund
    for symbol, shares in portfolio.items():
        if shares > 0:
            last_price = get_price_with_delisted(
                price_loader, symbol, end_day, 'Close', portfolio)
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
