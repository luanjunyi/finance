import pandas as pd
import os
import random
import logging
from datetime import datetime
from trade_sim_util import PriceLoader


class RandomOperationsGenerator:
    def __init__(self, start_date, end_date):
        """Initialize with start_date and cache available symbols
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
        """
        self.start_date = start_date
        self.end_date = end_date
        self.price_loader = PriceLoader()
        self.available_symbols = self._get_available_symbols()
        
        if not self.available_symbols:
            raise ValueError(f"No symbols found with data available since {start_date}")

    def _get_available_symbols(self):
        """Private method to get available symbols with price growth filter"""
        logging.info(f"Getting available symbols for {self.start_date} to {self.end_date}")
        stock_files = os.listdir('stock_data')
        available_symbols = []

        for f in stock_files:
            if '_daily.csv' not in f:
                continue

            symbol = f.replace('_daily.csv', '')
            try:
                # Get start and end prices using PriceLoader
                start_price = self.price_loader.get_price(symbol, self.start_date)
                end_price = self.price_loader.get_price(symbol, self.end_date)
                
                # Check if price increased by at least 2%
                price_increase = (end_price - start_price) / start_price
                
                if start_price > 20 and price_increase >= 0.02:
                    available_symbols.append(symbol)
                    
            except (ValueError, KeyError) as e:
                logging.debug(f"Skipping {symbol}: {str(e)}")
                continue

        logging.info(f"Found {len(available_symbols)} symbols meeting criteria")
        return available_symbols

    def generate_operations(self, input_file):
        """Generate random operations (formerly generate_random_operations)"""
        ops = pd.read_csv(input_file, names=[
                          'symbol', 'date', 'action', 'amount'])
        portfolio = set()
        new_ops = []

        for _, row in ops.iterrows():
            date = row['date']
            action = row['action']
            amount = str(row['amount'])

            if action == 'BUY':
                available_to_buy = list(
                    set(self.available_symbols) - portfolio)
                if not available_to_buy:
                    raise ValueError(f"No available symbols to buy on {date}")

                random_symbol = random.choice(available_to_buy)
                portfolio.add(random_symbol)
                new_ops.append([random_symbol, date, 'BUY', amount])

            elif action == 'SELL':
                if not portfolio:
                    raise ValueError(
                        f"No symbols in portfolio to sell on {date}")

                random_symbol = random.choice(list(portfolio))
                portfolio.remove(random_symbol)
                new_ops.append([random_symbol, date, 'SELL', amount])

        output_df = pd.DataFrame(
            new_ops, columns=['symbol', 'date', 'action', 'amount'])
        return output_df


if __name__ == "__main__":
    input_file = "ap_ops.csv"
    # Read the input file to get the start and end dates
    ops_df = pd.read_csv(input_file, names=['symbol', 'date', 'action', 'amount'])
    start_date = ops_df['date'].min()
    end_date = ops_df['date'].max()

    # Create generator instance
    generator = RandomOperationsGenerator(start_date, end_date)

    # Can now call generate_operations multiple times efficiently
    random_ops = generator.generate_operations(input_file)
    print(random_ops)
