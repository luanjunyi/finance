import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from utils.logging_config import setup_logging as setup_global_logging
from .fmp_api import FMPAPI
from tqdm import tqdm

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
        self.logger = logging.getLogger(__name__)

 
    def get_price(self, symbol: str, date: str):
        if symbol == 'ABNB' and date == '2020-12-09':
            self.logger.info("Querying ABNB at 2020-12-09 which is pre-IPO date will return IPO price ($68)")
            return {"date": "2020-12-09", "open": 68.0, "high": 68.0, "low": 68.0, "close": 68.0, "adjClose": 68.0, "volume": 1000000}
            
        prices = self.api.get_prices(symbol, date, date)
        if not prices:
            if symbol == 'TWTR' and date >= '2022-10-05':
                return {"date": "2022-10-05", "open": 53.7, "high": 53.7, "low": 53.7, "close": 53.7, "adjClose": 53.7, "volume": 1000000}
            raise ValueError(f"Can't find price for {symbol} on {date}")
        return prices[0]

    def get_close_price(self, symbol: str, date: str):
        return self.get_price(symbol, date)['adjClose']  
      

    def get_pe_ratio(self, symbol: str, date: str):
        """Get the PE ratio for a given symbol and date, using the last 4 quarters of EPS data.
        
        Args:
            symbol (str): The stock symbol.
            date (str): The date to get the PE ratio for.
        
        Returns:
            float: The PE ratio from the last 4 quarters.
        """
        if symbol == 'VOO':
            self.logger.info("Using PE ratio of 1.0 for VOO")
            return 1.0
        ratios = self.api.get_ratios(symbol, 'quarter', 120)

        eps = [(pd.to_datetime(r['date']), r['netIncomePerShare']) for r in ratios if \
                pd.to_datetime(r['date']) + pd.Timedelta(days=30 * 3) <= pd.to_datetime(date) and \
                    pd.to_datetime(r['date']) >= pd.to_datetime(date) - pd.Timedelta(days=31 * 15)]
        
        if len(eps) < 4:
            raise ValueError(f"{symbol} has only {len(eps)} EPS data points before {date}: {eps}")
        eps = sorted(eps, key=lambda x: x[0], reverse=True)
        eps_1y = sum([r[1] for r in eps[:4]])

        price = self.get_close_price(symbol, date)
        pe_ratio = price / eps_1y
        return pe_ratio

    def get_price_to_fcf(self, symbol: str, date: str):
        """Get the price-to-FCF ratio for a given symbol and date, using the last 4 quarters of FCF data.
        
        Args:
            symbol (str): The stock symbol.
            date (str): The date to get the price-to-FCF ratio for.
        
        Returns:
            float: The price-to-FCF ratio from the last 4 quarters.
        """
        if symbol == 'VOO':
            self.logger.info("Using price-to-FCF ratio of 1.0 for VOO")
            return 1.0
        ratios = self.api.get_ratios(symbol, 'quarter', 120)

        fcf = [(pd.to_datetime(r['date']), r['freeCashFlowPerShare']) for r in ratios if \
                pd.to_datetime(r['date']) + pd.Timedelta(days=30 * 3) <= pd.to_datetime(date) and \
                    pd.to_datetime(r['date']) >= pd.to_datetime(date) - pd.Timedelta(days=31 * 15)]
        if len(fcf) < 4:
            raise ValueError(f"{symbol} has only {len(fcf)} FCF data points before {date}: {fcf}")
        fcf = sorted(fcf, key=lambda x: x[0], reverse=True)
        fcf_1y = sum([r[1] for r in fcf[:4]])

        price = self.get_close_price(symbol, date)
        price_to_fcf = price / fcf_1y
        return price_to_fcf

    def get_close_prices_during(self, symbols: List[str], start_date: str, end_date: str):
        """Get close prices for multiple symbols during a date range.
        
        Args:
            symbols (List[str]): List of stock symbols.
            start_date (str): Start date in format YYYY-MM-DD.
            end_date (str): End date in format YYYY-MM-DD.
        
        Returns:
            pd.DataFrame: DataFrame with columns: symbol, date, close_price.
        """
        self.logger.info(f"Fetching close prices for {len(symbols)} symbols from {start_date} to {end_date}")
        
        all_data = []
        for symbol in tqdm(symbols):
            if symbol.startswith('^'): # symbols start with '^' are index symbols need a different API endpoint
                prices = self.api.index_prices(symbol, start_date, end_date)
                for price in prices:
                    all_data.append({
                            'symbol': price['symbol'],
                            'date': price['date'],
                            'close_price': price['price']
                    })                  
            else:                
                prices = self.api.get_prices(symbol, start_date, end_date)
                for price in prices:
                    all_data.append({
                            'symbol': price['symbol'],
                            'date': price['date'],
                            'close_price': price['adjClose']
                    })  
        
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['symbol', 'date'])
        return df

    def get_index_price(self, symbol: str, date: str):
        date = pd.to_datetime(date).strftime('%Y-%m-%d')
        prices = self.api.index_prices(symbol, date, date)
        if not prices:
            raise ValueError(f"Can't find price for {symbol} on {date}")
        assert prices[0]['date'] == date, f"Date mismatch: received {prices[0]['date']}, but requested {date}"
        assert prices[0]['symbol'] == symbol, f"Symbol mismatch: received {prices[0]['symbol']}, but requested {symbol}"
        return prices[0]['price']
    

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
