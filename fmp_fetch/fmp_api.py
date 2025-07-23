import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from functools import cache
from utils.logging_config import setup_logging as setup_global_logging
from utils.config import FMP_API_KEY


class FMPAPI:
    """
    A non-async wrapper for FMP API functions.
    This class provides synchronous versions of the functions used in fmp_crawler.
    """
    def __init__(self):
        self.api_key = FMP_API_KEY

        self.base_url = "https://financialmodelingprep.com/stable"
        self.last_request_time = 0
        self.rate_limit_delay = 0.2  # 5 requests per second

        # Setup logging with filename and line numbers
        self.setup_logging()

        # No database connection needed for online queries

    def setup_logging(self):
        """Configure logging for the fetcher"""
        setup_global_logging()

    def make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, base_url: Optional[str] = None) -> Optional[Any]:
        """
        Make a synchronous HTTP request to the FMP API
        
        Args:
            endpoint: API endpoint path
            params: Dictionary of query parameters
            
        Returns:
            JSON response or None if the request fails
        """
        # Initialize params if None
        if params is None:
            params = {}
            
        # Add API key to params
        params['apikey'] = self.api_key
        
        # Rate limiting
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last)

        url = f"{base_url or self.base_url}/{endpoint}"

        for attempt in range(3):
            try:
                response = requests.get(url, params=params)
                self.last_request_time = time.time()

                if response.status_code == 200:
                    return response.json()
                else:
                    error_msg = f"API request failed: {response.status_code} - {response.text}"
                    logging.warning(f"URL: {url} {params} - {error_msg}")

                if attempt < 2:  # Don't sleep on last attempt
                    time.sleep(5)

            except Exception as e:
                logging.error(f"URL: {url} - Error: {str(e)}")
                if attempt < 2:
                    time.sleep(5)

        return None

    # All symbols list
    @cache
    def get_all_symbols(self):
        return self.make_request('stock/list', base_url='https://financialmodelingprep.com/api/v3')

    @cache
    def get_all_tradable_symbols(self):
        return self.make_request('available-traded/list', base_url='https://financialmodelingprep.com/api/v3')
    


    # Price fetching functions
    @cache
    def get_prices(self, symbol: str, from_date: str, to_date: str):
        """
        Get historical price data for a symbol between two dates
        
        Args:
            symbol: Stock symbol
            from_date: Start date in format YYYY-MM-DD
            to_date: End date in format YYYY-MM-DD
            
        Returns:
            List of price data dictionaries or None if the request fails
        """
        prices = self.make_request(
            'historical-price-eod/dividend-adjusted',
            {
                'symbol': symbol,
                'from': from_date,
                'to': to_date
            }
        )

        return prices

    @cache
    def get_ratios(self, symbol: str, period: str = 'quarter', limit: int = 120):
        ratios = self.make_request(
            'ratios',
            {
                'symbol': symbol,
                'period': period,
                'limit': limit
            }
        )
        return ratios

    @cache
    def get_income_statement(self, symbol: str, period: str = 'quarter', limit: int = 120):
        income_statement = self.make_request(
            'income-statement',
            {
                'symbol': symbol,
                'period': period,
                'limit': limit
            }
        )
        return income_statement

    @cache
    def get_cashflow_statement(self, symbol: str, period: str = 'quarter', limit: int = 120):
        cashflow_statement = self.make_request(
            'cash-flow-statement',
            {
                'symbol': symbol,
                'period': period,
                'limit': limit
            }
        )
        return cashflow_statement
        
    @cache
    def get_balance_sheet(self, symbol: str, period: str = 'quarter', limit: int = 120):
        balance_sheet = self.make_request(
            'balance-sheet-statement',
            {
                'symbol': symbol,
                'period': period,
                'limit': limit
            }
        )
        return balance_sheet

    @cache
    def get_balance_sheet(self, symbol: str, period: str = 'quarter', limit: int = 120):
        balance_sheet = self.make_request(
            'balance-sheet-statement',
            {
                'symbol': symbol,
                'period': period,
                'limit': limit
            }
        )
        return balance_sheet

    @cache
    def index_prices(self, symbol: str, from_date: str, to_date: str):
        return self.make_request(
            'historical-price-eod/light',
            {
                'symbol': symbol,
                'from': from_date,
                'to': to_date
            }
        )

    @cache
    def spx_constituents(self):
        spx = self.make_request('sp500-constituent')
        return [x['symbol'] for x in spx]

    def batch_price_quote(self, symbols: List[str]):
        assert len(symbols) > 0
        return self.make_request(
            'batch-quote-short',
            {
                'symbols': ','.join(symbols),
            }
        )

    def batch_market_cap(self, symbols: List[str]):
        assert len(symbols) > 0
        return self.make_request(
            'market-capitalization-batch',
            {
                'symbols': ','.join(symbols),
            }
        )
    

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
