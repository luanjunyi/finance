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
            'historical-price-eod/dividend-adjusted',
            {'symbol': symbol, 'from': from_date, 'to': to_date}
        )

        if not prices:
            logging.warning(f"Failed to fetch prices for {symbol}")
            return

        cursor = self.db.cursor()
        for price in prices:
            self.check_missing_fields(
                price,
                ['date', 'adjOpen', 'adjHigh', 'adjLow', 'adjClose', 'volume'],
                symbol
            )

            try:
                if int(price.get('volume')) > 0 and float(price.get('adjClose') > 0.0):
                    cursor.execute('''
                        INSERT OR REPLACE INTO daily_price 
                        (symbol, date, open, high, low, close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        price.get('date'),
                        price.get('adjOpen'),
                        price.get('adjHigh'),
                        price.get('adjLow'),
                        price.get('adjClose'),
                        price.get('volume')
                    ))
            except Exception as e:
                logging.error(
                    f"Error inserting price for {symbol} on {price.get('date')}: {str(e)}")

    async def crawl(self, symbols=None):
        logging.info("Starting price crawling...")

        if symbols is None:
            symbols = self.get_symbols_to_crawl()
        
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=365*30)
                     ).strftime('%Y-%m-%d')

        for symbol in tqdm(symbols, desc="Crawling symbols"):
            await self.crawl_symbol_prices(symbol, from_date, to_date)
            self.db.commit()


async def main():
    parser = argparse.ArgumentParser(description='Crawl stock prices from FMP')
    parser.add_argument('--symbols', nargs='+', help='List of stock symbols to crawl. If not provided, will crawl all symbols from the database.')
    args = parser.parse_args()

    crawler = PriceCrawler()
    await crawler.crawl(args.symbols)
    crawler.close()

if __name__ == "__main__":
    asyncio.run(main())
