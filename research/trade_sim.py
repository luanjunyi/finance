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
import numpy as np

stock_data_folder = 'stock_data'
initial_fund = 1000000

class Backtest:
    def __init__(self, initial_fund=0, tolerance_days=4):
        self.tolerance_days = tolerance_days
        self.initial_fund = initial_fund
        self._reset()

    def _reset(self):
        self.cash = self.initial_fund
        self.price_loader = FMPPriceLoader(price_tolerance_days=self.tolerance_days)
        self.portfolio = {}
        self.returns = []
        self.dates = []
        self.portfolio_values = []

        
    def plot_portfolio_value(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.dates, self.portfolio_values, marker='o')
        plt.title('Portfolio Value Over Time')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def read_trading_ops(trading_ops_file):
        """Read trading operations from a CSV file and return as a list."""
        with open(trading_ops_file, 'r') as file:
            reader = csv.reader(file)
            return [row for row in reader]

    def get_portfolio_value(self, date):   
        value = self.cash
        logging.debug(f'Calculating portfolio value for {date}, cash: {self.cash}')
        for sym, sh in self.portfolio.items():
            if sh > 0:
                try:
                    current_price, date_used = self.price_loader.get_last_available_price(sym, date, 'close', max_window_days=13)
                except Exception as e:
                    raise Exception(
                        f"Error getting price for {sym} on {date}: {str(e)}")
                if date.strftime('%Y-%m-%d') != date_used:
                    logging.warning(
                        f"After execution portfolio value calculation used previous day price for {sym}, used {date_used} instead of {date}")
                logging.debug(f'Current price for {sym} is {current_price}, value: {sh * current_price}')
                value += sh * current_price
        return value        

    def run(self, trading_ops, end_day, use_open_price_for_buy=False, plot=False, return_history=False):
        """
        Run trading simulation based on a list of trading operations.

        Args:
            trading_ops: List of operations, where each operation is a list containing
                        [symbol, date_str, action, fraction]
            end_day: End date for the simulation, can be str
            plot: Whether to plot the results
            return_history: Whether to return history data
        """
        self._reset()
        if type(trading_ops) == pd.DataFrame:
            trading_ops = trading_ops.values.tolist()



        previous_fund_value = self.initial_fund
        current_cumulative_return = 1.0

        if type(end_day) == str:
            end_day = datetime.strptime(end_day, '%Y-%m-%d').date()

        last_transaction_date = ''

        # Process trading operations
        for op in tqdm(trading_ops, desc="Processing trades"):
            assert len(
                op) == 4, f"Each operation must have 4 elements, got {len(op)}, op is {op}"

            symbol, date_str, action, fraction = op
            date = datetime.strptime(date_str, '%Y-%m-%d').date()
            updated_value = self.get_portfolio_value(date)

            # if date_str == '2024-05-15':
            #     import pdb; pdb.set_trace()

            if last_transaction_date != date_str:
                last_transaction_date = date_str
                # Calculate and store the portfolio value before the trade
                period_return = updated_value / previous_fund_value if previous_fund_value != 0 else 1
                self.returns.append(period_return)
                current_cumulative_return *= period_return
                self.portfolio_values.append(current_cumulative_return)
                self.dates.append(date)
            
            previous_fund_value = updated_value

            assert date <= end_day

            price_type = 'close'

            price, date_used = self.price_loader.get_next_available_price(symbol, date, price_type)

            if date.strftime('%Y-%m-%d') != date_used:
                logging.warning(
                    f"Buying used next day price for {symbol}, used {date_used} instead of {date}")

            if action == 'BUY':
                # Convert fraction to float and calculate dollar amount to invest
                fraction = float(fraction)
                if use_open_price_for_buy:
                    price, _ = self.price_loader.get_next_available_price(symbol, date, 'open')
                dollar_amount = max(previous_fund_value * fraction, 10000)
                if dollar_amount > self.cash:
                    logging.info(
                        f"Added ${dollar_amount-self.cash} funds to invest {fraction:.1%} of portfolio (${dollar_amount:.2f})")
                    self.cash = dollar_amount

                # Calculate shares to buy based on dollar amount
                shares = int(dollar_amount / price)
                cost = price * shares
                self.cash -= cost

                self.portfolio.setdefault(symbol, 0)
                self.portfolio[symbol] += shares

            else:  # SELL
                assert action == 'SELL', "Action must be 'SELL' or 'BUY'"
                if symbol not in self.portfolio or self.portfolio[symbol] <= 0:
                    raise Exception(f"No shares of {symbol} to sell on {date_str}")

                current_shares = self.portfolio[symbol]
                if fraction == 'ALL':
                    shares_to_sell = current_shares
                else:
                    # Convert fraction to float and calculate shares to sell
                    fraction = float(fraction)
                    shares_to_sell = int(current_shares * fraction)

                if shares_to_sell > current_shares:
                    raise Exception(
                        f"Not enough shares of {symbol} to sell {shares_to_sell}")

                self.cash += price * shares_to_sell
                self.portfolio[symbol] -= shares_to_sell
            previous_fund_value = self.get_portfolio_value(date)    

            # Calculate and store the portfolio value after the trade

        # Evaluate final portfolio value
        logging.info(f"Calculating final portfolio value")
        final_value = self.get_portfolio_value(end_day)
        final_return = final_value / previous_fund_value
        self.returns.append(final_return)
        current_cumulative_return *= final_return
        self.portfolio_values.append(current_cumulative_return)
        self.dates.append(end_day)

        # Call the plotting function
        if plot:
            self.plot_portfolio_value()
            print("graph plotted")

        if return_history:
            return self.portfolio_values, self.dates
        return self.portfolio_values[-1]

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
    trading_ops = Backtest.read_trading_ops(args.trades)
    backtest = Backtest(initial_fund=initial_fund)
    final_fund = backtest.run(trading_ops, end_day, plot=args.plot)
    print(f"Final fund value: ${final_fund:.2f}")
