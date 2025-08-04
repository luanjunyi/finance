import argparse
from datetime import datetime, timedelta

from base_fmp_crawler import BaseFMPCrawler
import time
import asyncio
import logging
from tqdm import tqdm

INSIDER_TRADING_MAX_LIMIT = 1000
INSIDER_TRADING_MIN_PAGE = 0
INSIDER_TRADING_MAX_PAGE = 100


class InsiderCrawler(BaseFMPCrawler):
    async def crawl_insider_trading(self, symbol: str):
        if self.skip_existing:
            cursor = self.db.cursor()
            # Check if the record already exists
            cursor.execute("SELECT 1 FROM insider_trading WHERE symbol = ? LIMIT 1",
                           (symbol,))
            if cursor.fetchone():
                return

        endpoint = 'insider-trading/search'
        params = {'transactionType': 'P-Purchase', 'page': 0, 'limit': INSIDER_TRADING_MAX_LIMIT, 'symbol': symbol}
        data = await self.make_request(endpoint, params=params)

        if not data:
            logging.warning(f"Failed to fetch income statement for {symbol}")
            return

        cursor = self.db.cursor()
        for statement in data:
            self.check_missing_fields(
                statement,
                ['filingDate', 'transactionDate', 'transactionType', 'reportingName', 'typeOfOwner', 'securitiesTransacted', 'price'],
                symbol
            )

            cursor.execute('''
                INSERT OR REPLACE INTO insider_trading 
                (symbol, filingDate, transactionDate, reportingCik, companyCik,
                 transactionType, securitiesOwned, reportingName, typeOfOwner, acquisitionOrDisposition, 
                directOrIndirect, formType, securitiesTransacted, price, securityName)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                statement.get('filingDate'),
                statement.get('transactionDate'),
                statement.get('reportingCik'),
                statement.get('companyCik'),
                statement.get('transactionType'),
                statement.get('securitiesOwned'),
                statement.get('reportingName'),
                statement.get('typeOfOwner'),
                statement.get('acquisitionOrDisposition'),
                statement.get('directOrIndirect'),
                statement.get('formType'),
                statement.get('securitiesTransacted'),
                statement.get('price'),
                statement.get('securityName'),
            ))

    async def crawl(self, symbols=None):
        logging.info("Starting insider crawling...")
        start_time = time.time()

        if symbols is None:
            symbols = self.get_symbols_to_crawl()

        for symbol in tqdm(symbols, desc="Crawling symbols"):
            await self.crawl_insider_trading(symbol)
            self.db.commit()

        elapsed = time.time() - start_time
        logging.info(f"Insider crawling completed in {elapsed:.2f} seconds")


async def main():
    parser = argparse.ArgumentParser(description='Crawl insider trading from FMP')
    parser.add_argument('--symbols', nargs='+', help='List of stock symbols to crawl. If not provided, will crawl all symbols from the database.')
    parser.add_argument('--db-path', required=True, help='Path to the database file that store the crawled result')
    args = parser.parse_args()
    print(args)

    crawler = InsiderCrawler(args.db_path)
    await crawler.crawl(args.symbols)
    crawler.close()


if __name__ == "__main__":
    asyncio.run(main())
