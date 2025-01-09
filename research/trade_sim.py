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


class Backtest:
    def __init__(self, initial_fund=0, tolerance_days=4, begin_eval_date=None, end_eval_date=None, eval_interval=7, min_position_dollar=10000):
        self.tolerance_days = tolerance_days
        self.initial_fund = initial_fund
        self.begin_eval_date = begin_eval_date
        self.end_eval_date = end_eval_date
        self.eval_interval = eval_interval
        self.min_position_dollar = min_position_dollar
        self._reset()

    def _reset(self):
        self.cash = self.initial_fund
        self.price_loader = FMPPriceLoader(price_tolerance_days=self.tolerance_days)
        self.portfolio = {}
        self.portfolio_values = []
        self.dates = []

        
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
                current_price, date_used = self.price_loader.get_last_available_price(sym, date, 'close', max_window_days=13)

                if date.strftime('%Y-%m-%d') != date_used:
                    logging.warning(
                        f"After execution portfolio value calculation used previous day price for {sym}, used {date_used} instead of {date}")
                logging.debug(f'Current price for {sym} is {current_price}, value: {sh * current_price}')
                value += sh * current_price
        return value        

    def run(self, trading_ops, use_open_price_for_buy=False, plot=False, return_history=False):
        """
        Run trading simulation based on a list of trading operations.

        Args:
            trading_ops: List of operations, where each operation is a list containing
                        [symbol, date_str, action, fraction]
            use_open_price_for_buy: Whether to use open price for buy operations
            plot: Whether to plot the results
            return_history: Whether to return history data
        """
        assert self.begin_eval_date is not None, "begin_eval_date must be set"
        assert self.end_eval_date is not None, "end_eval_date must be set"
        assert self.eval_interval is not None, "eval_interval must be set"

        self._reset()
        if type(trading_ops) == pd.DataFrame:
            trading_ops = trading_ops.values.tolist()

        # Convert dates to datetime if they are strings
        if type(self.begin_eval_date) == str:
            self.begin_eval_date = datetime.strptime(self.begin_eval_date, '%Y-%m-%d').date()
        if type(self.end_eval_date) == str:
            self.end_eval_date = datetime.strptime(self.end_eval_date, '%Y-%m-%d').date()

        # Sort trading operations by date
        trading_ops.sort(key=lambda x: x[1])  # Sort by date_str (index 1)
        
        # Convert trading ops dates to datetime objects for easier comparison
        trading_dates = []
        for op in trading_ops:
            date = datetime.strptime(op[1], '%Y-%m-%d').date()
            trading_dates.append(date)
            assert date <= self.end_eval_date, f"Trading date {date} is after end_eval_date {self.end_eval_date}"

        # Generate evaluation dates
        eval_dates = pd.date_range(self.begin_eval_date, self.end_eval_date, freq=f'{self.eval_interval}D').date
        
        # Initialize tracking variables
        previous_fund_value = self.initial_fund
        current_cumulative_return = 1.0
        trading_idx = 0
        eval_idx = 0
        
        # Process events in chronological order
        with tqdm(total=len(trading_ops) + len(eval_dates), desc="Processing events") as pbar:
            while trading_idx < len(trading_ops) or eval_idx < len(eval_dates):
                # Determine which event comes next
                trade_date = trading_dates[trading_idx] if trading_idx < len(trading_ops) else None
                eval_date = eval_dates[eval_idx] if eval_idx < len(eval_dates) else None
                
                # Process trade if it's next, or if we're done with evaluations
                if not eval_date or (trade_date and trade_date <= eval_date):
                    op = trading_ops[trading_idx]
                    symbol, date_str, action, fraction = op
                    
                    price_type = 'close'
                    price, date_used = self.price_loader.get_next_available_price(symbol, trade_date, price_type)

                    if trade_date.strftime('%Y-%m-%d') != date_used:
                        logging.warning(
                            f"Trading used next day price for {symbol}, used {date_used} instead of {trade_date}")

                    if action == 'BUY':
                        fraction = float(fraction)
                        if use_open_price_for_buy:
                            price, _ = self.price_loader.get_next_available_price(symbol, trade_date, 'open')
                        # Get current portfolio value at trade date
                        current_value = self.get_portfolio_value(trade_date)
                        dollar_amount = max(current_value * fraction, self.min_position_dollar)
                        if dollar_amount > self.cash:                            
                            # Calculate return up to this poin
                            period_return = current_value / previous_fund_value
                            current_cumulative_return *= period_return
                            
                            # Add cash and update the base value for next period
                            additional_cash = dollar_amount - self.cash
                            self.cash = dollar_amount
                            previous_fund_value = current_value + additional_cash
                            logging.info(f"Added ${additional_cash:.2f} cash on {trade_date}")

                        shares = int(dollar_amount / price)
                        assert shares > 0, f"Allocated fund can't buy even one share of {symbol}, price is {price}"
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
                            fraction = float(fraction)
                            shares_to_sell = int(current_shares * fraction)

                        if shares_to_sell > current_shares:
                            raise Exception(
                                f"Not enough shares of {symbol} to sell {shares_to_sell}")

                        self.cash += price * shares_to_sell
                        self.portfolio[symbol] -= shares_to_sell
                    
                    trading_idx += 1
                    pbar.update(1)
                
                # Process evaluation if it's next, or if we're done with trades
                else:
                    assert not trade_date or (eval_date and eval_date <= trade_date)
                    # Evaluate portfolio at this evaluation date
                    current_value = self.get_portfolio_value(eval_date)
                    period_return = current_value / previous_fund_value if previous_fund_value > 0 else 1.0
                    current_cumulative_return *= period_return
                    previous_fund_value = current_value
                    
                    self.portfolio_values.append(current_cumulative_return)
                    self.dates.append(eval_date)
                    
                    eval_idx += 1
                    pbar.update(1)

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
    parser.add_argument('--begin_eval_date', type=str, required=True,
                        help='Start date for evaluation in YYYY-MM-DD format')
    parser.add_argument('--end_eval_date', type=str, required=True,
                        help='End date for evaluation in YYYY-MM-DD format')
    parser.add_argument('--eval_interval', type=int, required=True,
                        help='Number of days between evaluations')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='Plot the portfolio value over time')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    backtest = Backtest(initial_fund=initial_fund, begin_eval_date=args.begin_eval_date, end_eval_date=args.end_eval_date, eval_interval=args.eval_interval)
    # Read operations from file, then pass to backtest_trading
    trading_ops = Backtest.read_trading_ops(args.trades)
    final_fund = backtest.run(trading_ops, plot=args.plot)
    print(f"Final fund value: ${final_fund:.2f}")
