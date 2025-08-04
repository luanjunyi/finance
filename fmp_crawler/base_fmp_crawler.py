import os.path
import time
import logging
import aiohttp
import asyncio
import sqlite3
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
from utils.logging_config import setup_logging as setup_global_logging
from utils.config import FMP_API_KEY


class BaseFMPCrawler:
    def __init__(self, db_path: str):
        self.skip_existing = True
        self.api_key = FMP_API_KEY

        self.base_url = "https://financialmodelingprep.com/stable"
        self.last_request_time = 0
        self.rate_limit_delay = 0.25  # 4 requests per second

        # Setup logging with filename and line numbers
        self.setup_logging()

        # Setup database
        self.create_tables(db_path)
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row

    def setup_logging(self):
        """Configure logging for the crawler"""
        setup_global_logging()

    async def make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None, base_url: Optional[str] = None) -> Optional[Any]:
        if params is None:
            params = {}
        params['apikey'] = self.api_key

        # Rate limiting
        now = time.time()
        time_since_last = now - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)

        url = f"{base_url or self.base_url}/{endpoint}"

        for attempt in range(3):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params) as response:
                        self.last_request_time = time.time()

                        if response.status == 200:
                            # response can be empty, but if status is 200, it is legit response. For example,
                            # requesting financial statements for a stock that's newly listed.
                            return await response.json()
                        else:
                            error_msg = f"API request failed: {response.status} params: [{params}], Error {await response.text()}"
                            logging.error(f"URL: {url} - {error_msg}")

                        if attempt < 2:  # Don't sleep on last attempt
                            await asyncio.sleep(5)

            except Exception as e:
                logging.error(f"URL: {url} - Error: {str(e)}")
                if attempt < 2:
                    await asyncio.sleep(5)

        return None

    def check_missing_fields(self, data: Dict[str, Any], required_fields: List[str], symbol: str):
        missing = [
            field for field in required_fields if field not in data or data[field] is None]
        if missing:
            logging.warning(
                f"Symbol {symbol} missing fields: {', '.join(missing)}")
            # Log to file
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open('missing_data.log', 'a') as f:
                f.write(
                    f"{timestamp} - Symbol: {symbol}, Missing fields: {', '.join(missing)}\n")

    def close(self):
        """Close the database connection"""
        if hasattr(self, 'db'):
            self.db.close()

    def get_symbols_to_crawl(self):
        """Get list of symbols to crawl from the database"""
        cursor = self.db.cursor()
        cursor.execute('''
            SELECT symbol FROM stock_symbol 
            WHERE exchange_short_name IN ('NYSE', 'NASDAQ', 'AMEX', 'OTC')
                AND type = 'stock'
        ''')
        return [row['symbol'] for row in cursor.fetchall()]

    @staticmethod
    def create_tables(db_path: str):
        parent = os.path.dirname(os.path.dirname(__file__))
        sql_file_path = os.path.join(parent, 'db', 'db_creation.sql')

        # Connect to the database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Read and execute the SQL script
        with open(sql_file_path, 'r') as f:
            sql_script = f.read()

        cursor.executescript(sql_script)
        conn.commit()
        conn.close()
