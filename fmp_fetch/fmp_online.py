import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from utils.logging_config import setup_logging as setup_global_logging
from .fmp_api import FMPAPI

class FMPOnline:
    """
    A non-async wrapper for FMP API functions.
    This class provides synchronous versions of the functions used in fmp_crawler.
    """
    def __init__(self):
        self.api = FMPAPI()

        # Setup logging with filename and line numbers
        self.setup_logging()

    def setup_logging(self):
        """Configure logging for the fetcher"""
        setup_global_logging()

 
    def get_price(self, symbol: str, date: str):
        prices = self.api.get_prices(symbol, date, date)
        if not prices:
            raise ValueError(f"Can't find price for {symbol} on {date}")
        return prices[0]

    def get_close_price(self, symbol: str, date: str):
        return self.get_price(symbol, date)['adjClose']  
      

    def get_pe_ratio(self, symbol: str, date: str):
        ratios = self.api.get_ratios(symbol, 'quarter', 120)

        eps = [(pd.to_datetime(r['date']), r['netIncomePerShare']) for r in ratios if pd.to_datetime(r['date']) + pd.Timedelta(days=31 * 3) <= pd.to_datetime(date)]
        if len(eps) < 4:
            raise ValueError(f"{symbol} has only {len(eps)} EPS data points before {date}")
        eps = sorted(eps, key=lambda x: x[0], reverse=True)
        eps_1y = sum([r[1] for r in eps[:4]])

        price = self.get_close_price(symbol, date)
        pe_ratio = price / eps_1y
        return pe_ratio

    

# Example usage
if __name__ == "__main__":
    fmp = FMPOnline()
    
    # Example: Get price data for AAPL
    from_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')
    prices = fmp.get_price('AAPL', from_date, to_date)
    
    if prices:
        print(f"Got {len(prices)} price records for AAPL")
        print(f"Latest price: {prices[0]}")
