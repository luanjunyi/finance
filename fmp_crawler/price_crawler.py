import asyncio
import argparse
from .base_fmp_crawler import BaseFMPCrawler
from datetime import datetime, timedelta
import time
import logging
from tqdm import tqdm


class PriceCrawler(BaseFMPCrawler):
    async def crawl_symbol_prices(self, symbol: str, from_date: str, to_date: str):
        prices = await self.make_request(
            f'historical-price-full/{symbol}',
            {'from': from_date, 'to': to_date}
        )

        if not prices or 'historical' not in prices:
            logging.error(f"Failed to fetch prices for {symbol}")
            return

        cursor = self.db.cursor()
        for price in prices['historical']:
            self.check_missing_fields(
                price,
                ['date', 'open', 'high', 'low', 'close', 'adjClose', 'volume'],
                symbol
            )

            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO daily_price 
                    (symbol, date, open, high, low, close, adjusted_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    price.get('date'),
                    price.get('open'),
                    price.get('high'),
                    price.get('low'),
                    price.get('close'),
                    price.get('adjClose'),
                    price.get('volume')
                ))
            except Exception as e:
                logging.error(
                    f"Error inserting price for {symbol} on {price.get('date')}: {str(e)}")

    async def crawl(self, symbols=None):
        logging.info("Starting price crawling...")
        start_time = time.time()

        if symbols is None:
            symbols = self.get_symbols_to_crawl()
        
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=365*30)
                     ).strftime('%Y-%m-%d')

        for symbol in tqdm(symbols, desc="Crawling symbols"):
            await self.crawl_symbol_prices(symbol, from_date, to_date)
            self.db.commit()

        elapsed = time.time() - start_time
        logging.info(f"Price crawling completed in {elapsed:.2f} seconds")


async def main():
    parser = argparse.ArgumentParser(description='Crawl stock prices from FMP')
    parser.add_argument('--symbols', nargs='+', help='List of stock symbols to crawl. If not provided, will crawl all symbols from the database.')
    args = parser.parse_args()

    crawler = PriceCrawler()
    await crawler.crawl(args.symbols)
    crawler.close()

if __name__ == "__main__":
    asyncio.run(main())
