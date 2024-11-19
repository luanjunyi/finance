import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import logging
import time
from tqdm import tqdm

from .base_fmp_crawler import BaseFMPCrawler


class OwnerEarningsCrawler(BaseFMPCrawler):
    def __init__(self):
        super().__init__(api_version='v4')  # Specify v4 API version

    async def crawl_symbol_owner_earnings(self, symbol: str):
        owner_earnings = await self.make_request(
            'owner_earnings',  # The API version is already in base_url
            {'symbol': symbol}
        )

        if not owner_earnings:
            logging.error(f"Failed to fetch owner earnings for {symbol}")
            with open('missing_data.log', 'a') as log_file:
                log_file.write(f"{symbol}: Missing owner earnings\n")
            return

        cursor = self.db.cursor()

        for earnings in owner_earnings:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO owner_earnings (
                        symbol, date, averagePPE, maintenanceCapex, ownersEarnings,
                        growthCapex, ownersEarningsPerShare
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    earnings.get('date'),
                    earnings.get('averagePPE'),
                    earnings.get('maintenanceCapex'),
                    earnings.get('ownersEarnings'),
                    earnings.get('growthCapex'),
                    earnings.get('ownersEarningsPerShare')
                ))
            except Exception as e:
                logging.error(
                    f"Error inserting owner earnings for {symbol} on {earnings.get('date')}: {str(e)}")

    async def crawl(self):
        logging.info("Starting owner earnings crawling...")
        start_time = time.time()

        symbols = self.get_symbols_to_crawl()

        for symbol in tqdm(symbols, desc="Crawling owner earnings"):
            await self.crawl_symbol_owner_earnings(symbol)
            self.db.commit()

        elapsed = time.time() - start_time
        logging.info(
            f"Owner earnings crawling completed in {elapsed:.2f} seconds")


async def main():
    crawler = OwnerEarningsCrawler()
    await crawler.crawl()
    crawler.close()

if __name__ == "__main__":
    asyncio.run(main())