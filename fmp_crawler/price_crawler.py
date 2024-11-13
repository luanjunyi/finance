import asyncio
from .base_fmp_crawler import BaseFMPCrawler
from datetime import datetime, timedelta
import time
from tqdm import tqdm


class PriceCrawler(BaseFMPCrawler):
    def get_symbols_to_crawl(self):
        cursor = self.db.cursor()
        cursor.execute('''
            SELECT symbol FROM stock_symbol 
            WHERE exchange_short_name IN ('NYSE', 'NASDAQ', 'AMEX')
            AND type = 'stock'
        ''')
        return [row['symbol'] for row in cursor.fetchall()]

    async def crawl_symbol_prices(self, symbol: str, from_date: str, to_date: str):
        prices = await self.make_request(
            f'historical-price-full/{symbol}',
            {'from': from_date, 'to': to_date}
        )

        if not prices or 'historical' not in prices:
            self.logger.error(f"Failed to fetch prices for {symbol}")
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
                self.logger.error(
                    f"Error inserting price for {symbol} on {price.get('date')}: {str(e)}")

    async def crawl(self):
        self.logger.info("Starting price crawling...")
        start_time = time.time()

        # Create table if not exists
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS daily_price (
                symbol VARCHAR(10),
                date DATE NOT NULL,
                open DECIMAL(10,2),
                high DECIMAL(10,2),
                low DECIMAL(10,2),
                close DECIMAL(10,2),
                adjusted_close DECIMAL(10,2),
                volume BIGINT,
                UNIQUE(symbol, date)
            )
        ''')

        symbols = self.get_symbols_to_crawl()
        to_date = datetime.now().strftime('%Y-%m-%d')
        from_date = (datetime.now() - timedelta(days=365*30)
                     ).strftime('%Y-%m-%d')

        for symbol in tqdm(symbols, desc="Crawling symbols"):
            await self.crawl_symbol_prices(symbol, from_date, to_date)
            self.db.commit()

        elapsed = time.time() - start_time
        self.logger.info(f"Price crawling completed in {elapsed:.2f} seconds")


async def main():
    crawler = PriceCrawler()
    await crawler.crawl()
    crawler.close()

if __name__ == "__main__":
    asyncio.run(main())
