import asyncio
from .base_fmp_crawler import BaseFMPCrawler
from typing import List, Dict, Any
import time
import logging


class SymbolCrawler(BaseFMPCrawler):
    async def crawl(self):
        logging.info("Starting symbol crawling...")
        start_time = time.time()

        # Fetch symbols
        symbols = await self.make_request('stock/list')
        if not symbols:
            logging.error("Failed to fetch symbols")
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
                logging.error(
                    f"Error inserting symbol {symbol.get('symbol')}: {str(e)}")

        self.db.commit()

        elapsed = time.time() - start_time
        logging.info(f"Symbol crawling completed in {elapsed:.2f} seconds")


async def main():
    crawler = SymbolCrawler()
    await crawler.crawl()
    crawler.close()

if __name__ == "__main__":
    asyncio.run(main())
