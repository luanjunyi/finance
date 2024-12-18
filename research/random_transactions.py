from functools import cache
from typing import Set
import pandas as pd
import os
import random
import logging
from datetime import datetime, timedelta
from fmp_data import FMPPriceLoader, Dataset

"""
RandomTransactionGenerator:
    Generate random buy and sell transactions according to the input transaction dates. This is used
    to generate random transactions that can be compared to a given portfolio management strategy. We
    can then gauge how likely the portfolio's result is due to luck V.S. skill.
"""
class RandomTransactionGenerator:
    def __init__(self, original_transactions_file):
        self.original_transactions_file = original_transactions_file
        self.price_loader = FMPPriceLoader()
        self.symbols = self.price_loader.get_us_stock_symbols()

    @cache
    def get_last_available_price(self, symbol, date):
        return self.price_loader.get_last_available_price(symbol, date)

    @cache
    def find_momentum_stocks(self, date_str):
        date = datetime.strptime(date_str, '%Y-%m-%d').date()
        last_3m = (date - timedelta(days=90)).strftime('%Y-%m-%d')
        last_6m = (date - timedelta(days=180)).strftime('%Y-%m-%d')
        last_9m = (date - timedelta(days=270)).strftime('%Y-%m-%d')
        last_12m = (date - timedelta(days=360)).strftime('%Y-%m-%d')
        candidates = set()
        sell_triggered = set()
        for sym in self.symbols:
            try:
                p, _ = self.get_last_available_price(sym, date_str)
                p_3m, _ = self.get_last_available_price(sym, last_3m)
                p_6m, _ = self.get_last_available_price(sym, last_6m)
                p_9m, _ = self.get_last_available_price(sym, last_9m)
                p_12m, _ = self.get_last_available_price(sym, last_12m)
            except KeyError:
                continue
            
            returns = [p/p_3m-1, p_3m/p_6m-1, p_6m/p_9m-1, p_9m/p_12m-1]
            if p > p_12m * 1.40 and min(returns) > 0.05:
                candidates.add(sym)
        
        return list(candidates)


    def check_momentum_at(self, date_str: str, portfolio: Set[str]):
        candidates = self.find_momentum_stocks(date_str)

        date = datetime.strptime(date_str, '%Y-%m-%d').date()
        last_6m = (date - timedelta(days=180)).strftime('%Y-%m-%d')

        sell_triggered = set()
        for sym in portfolio:
            p, _ = self.get_last_available_price(sym, date_str)
            p_6m, _ = self.get_last_available_price(sym, last_6m)

            if p < p_6m and sym in portfolio:
                sell_triggered.add(sym)

        logging.info(f"Found {len(candidates)} symbol candidates {date_str}")
        return random.choice(candidates), sell_triggered

    
    def prefetch_prices(self):
        logging.info("Prefetching price data...")
        dates = self.extract_buy_dates(self.original_transactions_file)
        self.price_history = \
            self.price_loader.get_last_available_price_for_stocks_on_dates(
                self.symbols,
                dates)
        logging.info(f"Prefetched {len(self.price_history)} price history data")

    def extract_buy_dates(self, input_file):
        dates = set()
        ops = pd.read_csv(input_file, names=[
                          'symbol', 'date', 'action', 'amount'])
        for _, row in ops.iterrows():
            date = str(row['date'])
            action = row['action']
            if action == 'BUY':
                dates.add(date)
                date = datetime.strptime(date, '%Y-%m-%d').date()
                dates.add((date - timedelta(days=90)).strftime('%Y-%m-%d'))
                dates.add((date - timedelta(days=180)).strftime('%Y-%m-%d'))
                dates.add((date - timedelta(days=270)).strftime('%Y-%m-%d'))
                dates.add((date - timedelta(days=360)).strftime('%Y-%m-%d'))

        return list(dates)

    def original_transactions(self):
        ops = pd.read_csv(self.original_transactions_file, names=[
                          'symbol', 'date', 'action', 'amount'])
        return ops


    def generate_operations(self, num=1):
        """
        Generate random buy and sell transactions. The logic is simple: at each buy transaction date
        1. Randomly choose a stock to add to the portfolio, assign 1% of total portfolio value
        2. Remove any stock that meet sell critierion
        3. Evaluate the return in the last period before those operation. Those returns will be multifplied together
           to get the final return (time-weighted return as in https://en.wikipedia.org/wiki/Time-weighted_return ).
        Ignore the sell operations in the original file.
        """       
        ret = []

        for iter in range(num):
            ops = pd.read_csv(self.original_transactions_file, names=[
                            'symbol', 'date', 'action', 'amount'])
            portfolio = set()
            new_ops = []                                        
            for _, row in ops.iterrows():
                date = str(row['date'])
                action = row['action']
                amount = str(row['amount'])

                if action == 'BUY':
                    sym, sell_triggered = self.check_momentum_at(date, portfolio)
                    portfolio.add(sym)
                    new_ops.append([sym, date, 'BUY', amount])
                    for sym in sell_triggered:
                        new_ops.append([sym, date, 'SELL', 'ALL'])
                        portfolio.remove(sym)
                        logging.debug(f"SELL {sym} at {date}")

                elif action == 'SELL':
                    pass # Ignore sell transactions as sell are managed by ourself

            output_df = pd.DataFrame(new_ops)
            output_df.columns = ['symbol', 'date', 'action', 'amount']
            ret.append(output_df)
        return ret  

    def buy_single(self, symbol):
        ops = pd.read_csv(self.original_transactions_file, names=[
                        'symbol', 'date', 'action', 'amount'])
        new_ops = []                                        
        for _, row in ops.iterrows():
            date = str(row['date'])
            action = row['action']
            amount = str(row['amount'])

            if action == 'BUY':
                new_ops.append([symbol, date, 'BUY', amount])
            elif action == 'SELL':
                pass # Ignore sell transactions 

        output_df = pd.DataFrame(new_ops)
        output_df.columns = ['symbol', 'date', 'action', 'amount']
        return output_df   


if __name__ == "__main__":
    input_file = "ap_ops.csv"
    # Read the input file to get the start and end dates
    ops_df = pd.read_csv(input_file, names=['symbol', 'date', 'action', 'amount'])
    start_date = ops_df['date'].min()
    end_date = ops_df['date'].max()

    # Create generator instance
    generator = RandomTransactionGenerator()

    # Can now call generate_operations multiple times efficiently
    random_ops = generator.generate_operations(input_file)
    print(random_ops)
