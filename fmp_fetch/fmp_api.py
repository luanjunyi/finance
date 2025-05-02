import os
import time
import logging
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from utils.logging_config import setup_logging as setup_global_logging


class FMPAPI:
    """
    A non-async wrapper for FMP API functions.
    This class provides synchronous versions of the functions used in fmp_crawler.
    """
    def __init__(self):
        self.api_key = os.getenv('FMP_API_KEY')
        if not self.api_key:
            raise ValueError("FMP_API_KEY environment variable not set")

        self.base_url = "https://financialmodelingprep.com/stable"
        self.last_request_time = 0
        self.rate_limit_delay = 0.25  # 4 requests per second

        # Setup logging with filename and line numbers
        self.setup_logging()

        # No database connection needed for online queries

    def setup_logging(self):
        """Configure logging for the fetcher"""
        setup_global_logging()

    def make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
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

        url = f"{self.base_url}/{endpoint}"

        for attempt in range(3):
            try:
                response = requests.get(url, params=params)
                self.last_request_time = time.time()

                if response.status_code == 200:
                    return response.json()
                else:
                    error_msg = f"API request failed: {response.status_code} - {response.text}"
                    logging.error(f"URL: {url} {params} - {error_msg}")

                if attempt < 2:  # Don't sleep on last attempt
                    time.sleep(5)

            except Exception as e:
                logging.error(f"URL: {url} - Error: {str(e)}")
                if attempt < 2:
                    time.sleep(5)

        return None




    # Price fetching functions
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
