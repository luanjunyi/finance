from functools import cache
from re import fullmatch
from typing import List, Set
import pandas as pd
import random
import logging
from datetime import datetime, timedelta

from fmp_data import FMPPriceLoader, Dataset, AFTER_PRICE, fmp
from feature_gen.sma import SMA

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
        self.at_least_increase_1y = 0.8

    @cache
    def get_last_available_price(self, symbol, date):
        return self.price_loader.get_last_available_price(symbol, date)

    @cache
    def price_increased_by(self, date_str: str, looking_back_days: int, at_least: float) -> List[str]:
        date = datetime.strptime(date_str, '%Y-%m-%d')
        previous_date = date - timedelta(days=looking_back_days)
        candidates = set()
        
        for sym in self.symbols:
            try:
                p1, used_date = self.get_last_available_price(sym, date_str)
                if date - datetime.strptime(used_date, '%Y-%m-%d') > timedelta(days=5):
                    continue
                p0, used_date = self.get_last_available_price(sym, previous_date.strftime('%Y-%m-%d'))
                if previous_date - datetime.strptime(used_date, '%Y-%m-%d') > timedelta(days=5):
                    continue
            except KeyError:
                continue
            
            if p1 > p0 * (1 + at_least):
                candidates.add(sym)
        
        return list(candidates)

    @cache
    def active_stocks_on(self, date_str: str) -> set[str]:
        return self.price_loader.active_us_stocks_on(date_str)
        

    @cache
    def find_momentum_stocks(self, date_str) -> frozenset[str]:

        candidates = self.price_increased_by(date_str, 365, self.at_least_increase_1y)
        ret = set()
        
        for sym in candidates:
            if self.sma.has_open_mouth_trend(sym, date_str, 60, [20, 60, 120], tolerance=0.01):
                ret.add(sym)
        
        return frozenset(ret)

    @cache
    def get_fundamentals(self, symbols: set[str], date_str: str) -> pd.DataFrame:
        fundamentals = Dataset(list(symbols), {
            'revenue': '',
            'free_cash_flow_per_share': 'fcf_per_share',
            'operating_profit_margin': 'profit',
        }).data.copy()

        fundamentals['price'] = fundamentals['symbol'].apply(
            lambda sym: self.get_last_available_price(sym, date_str)[0])

        fundamentals['value'] = fundamentals['fcf_per_share'] / fundamentals['price']
        fundamentals['growth'] = fundamentals.groupby('symbol')['revenue'].pct_change(fill_method=None, periods=4)

        # Filter only the most recent result before date_str
        fundamentals = fundamentals[fundamentals['date'] <= date_str]
        fundamentals = fundamentals.loc[fundamentals.groupby('symbol')['date'].idxmax()]   
        return fundamentals     


    def check_momentum_at(self, date_str: str, portfolio: set[str]):
        candidates = self.find_momentum_stocks(date_str)
        fundamentals = self.get_fundamentals(candidates, date_str).copy()

        selected = []
        for symbol in candidates:
            if symbol in portfolio:
                continue
            data = fundamentals[fundamentals['symbol'] == symbol]
            if data.empty:
                continue
            data = data.iloc[0]
            if True:#data.growth >= 0.2 and data.value >= 0.05 and data.profit >= 0.3:
                selected.append([data.value, symbol])
        selected.sort(reverse=True)

        date = datetime.strptime(date_str, '%Y-%m-%d').date()
        last_6m = (date - timedelta(days=180)).strftime('%Y-%m-%d')
        last_12m = (date - timedelta(days=360)).strftime('%Y-%m-%d')

        sell_triggered = set()
        for sym in portfolio:
            p, _ = self.get_last_available_price(sym, date_str)
            p_12m, _ = self.get_last_available_price(sym, last_12m)

            if p < p_12m:
                sell_triggered.add(sym)

        logging.info(f"Found {len(selected)} symbol candidates {date_str}, need to sell {len(sell_triggered)} in portfolio")
        return [sym[1] for sym in selected], sell_triggered

    
    def prefetch(self):
        logging.info("Prefetching data to cache...")
        dates = self.extract_buy_dates(self.original_transactions_file)
        stocks = set()
        for date in dates:
            symbols = self.price_increased_by(date, 365, self.at_least_increase_1y)
            logging.info(f"Found {len(symbols)} stock increased by at least {self.at_least_increase_1y} on {date}")
            stocks |= set(symbols)

        logging.info(f"Found {len(stocks)} that passed minimum price increase filter in total. Will prefetch SMA")

        min_date, max_date = min(dates), max(dates)
        min_date = datetime.strptime(min_date, '%Y-%m-%d') - timedelta(days=365)
        min_date = min_date.strftime('%Y-%m-%d')  
        
        self.sma = SMA(list(stocks), begin_date=min_date, end_date=max_date)
        self.sma.values([20, 60, 120])

    def extract_buy_dates(self, input_file):
        dates = set()
        ops = pd.read_csv(input_file, names=[
                          'symbol', 'date', 'action', 'amount'])
        for _, row in ops.iterrows():
            date = str(row['date'])
            action = row['action']
            if action == 'BUY':
                dates.add(date)

        return list(dates)

    def original_transactions(self):
        ops = pd.read_csv(self.original_transactions_file, names=[
                          'symbol', 'date', 'action', 'amount'])
        return ops

    def pure_random(self, date, portfolio):
        active_stocks = list(self.active_stocks_on(date))
        return random.sample(active_stocks, 5), [random.choice(list(portfolio)),] if len(portfolio) > 0 else []

    def generate_pure_random(self, num=1):
        ret = []
        #self.prefetch()
        ops = self.original_transactions()        

        for iter in range(num):
            portfolio = set()
            new_ops = []                                        
            for _, row in ops.iterrows():
                date = str(row['date'])
                action = row['action']
                amount = str(row['amount'])

                if action == 'BUY':
                    selected, sell_triggered = self.pure_random(date, portfolio)
                    random.shuffle(selected)
                    for sym in selected[:3]:
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
        #self.prefetch()
        ops = self.original_transactions()        

        for iter in range(num):
            portfolio = set()
            new_ops = []                                        
            for _, row in ops.iterrows():
                date = str(row['date'])
                action = row['action']
                amount = str(row['amount'])

                if action == 'BUY':
                    selected, sell_triggered = self.check_momentum_at(date, portfolio)
                    random.shuffle(selected)
                    for sym in selected[:3]:
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
