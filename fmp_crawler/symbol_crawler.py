from .base_fmp_crawler import BaseFMPCrawler
from typing import List, Dict, Any
import time


class SymbolCrawler(BaseFMPCrawler):
    async def crawl(self):
        self.logger.info("Starting symbol crawling...")
        start_time = time.time()

        # Create table if not exists
        self.db.execute('''
            CREATE TABLE IF NOT EXISTS stock_symbol (
                symbol VARCHAR(10) PRIMARY KEY,
                name VARCHAR(255),
                exchange VARCHAR(255),
                exchange_short_name VARCHAR(255),
                type VARCHAR(255)
            )
        ''')

        # Fetch symbols
        symbols = await self.make_request('stock/list')
        if not symbols:
            self.logger.error("Failed to fetch symbols")
            return

        # Filter and insert symbols
        cursor = self.db.cursor()
        for symbol in symbols:
            self.check_missing_fields(
                symbol,
                ['symbol', 'name', 'exchange', 'exchangeShortName', 'type'],
                symbol.get('symbol', 'UNKNOWN')
            )

            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO stock_symbol 
                    (symbol, name, exchange, exchange_short_name, type)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    symbol.get('symbol'),
                    symbol.get('name'),
                    symbol.get('exchange'),
                    symbol.get('exchangeShortName'),
                    symbol.get('type')
                ))
            except Exception as e:
                self.logger.error(
                    f"Error inserting symbol {symbol.get('symbol')}: {str(e)}")

        self.db.commit()

        elapsed = time.time() - start_time
        self.logger.info(f"Symbol crawling completed in {elapsed:.2f} seconds")


async def main():
    crawler = SymbolCrawler()
    await crawler.crawl()
    crawler.close()

if __name__ == "__main__":
    asyncio.run(main())
