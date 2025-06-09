import asyncio
from fmp_crawler.base_fmp_crawler import BaseFMPCrawler
from typing import List, Dict, Any
import time
import logging
import yfinance as yf


class SymbolCrawler(BaseFMPCrawler):
    async def crawl(self):
        logging.info("Starting symbol crawling...")
        start_time = time.time()

        # Fetch symbols
        symbols = await self.make_request('stock/list', base_url='https://financialmodelingprep.com/api/v3')
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
                    (symbol, name, exchange, exchange_short_name, type, sector, industry)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol.get('symbol'),
                    symbol.get('name'),
                    symbol.get('exchange'),
                    symbol.get('exchangeShortName'),
                    symbol.get('type'),
                    None,  # sector will be backfilled
                    None   # industry will be backfilled
                ))
            except Exception as e:
                logging.error(
                    f"Error inserting symbol {symbol.get('symbol')}: {str(e)}")

        self.db.commit()

        elapsed = time.time() - start_time
        logging.info(f"Symbol crawling completed in {elapsed:.2f} seconds")

    def backfill_sector_industry(self):
        """Backfill sector and industry data using Yahoo Finance."""
        logging.info("Starting sector and industry backfill...")
        start_time = time.time()
        requests_per_second = 2
        delay = 1.0 / requests_per_second
        
        cursor = self.db.cursor()
        # Get all symbols that need backfilling
        cursor.execute('''
            SELECT symbol 
            FROM stock_symbol 
            WHERE type = 'stock' AND (sector IS NULL OR industry IS NULL)
        ''')
        symbols = [row[0] for row in cursor.fetchall()]
        logging.info(f"Found {len(symbols)} symbols to backfill")
        for symbol in symbols:           
            try:
                logging.info(f"Updating sector and industry for {symbol}")
                ticker = yf.Ticker(symbol)
                info = ticker.info
                sector = info.get('sector')
                industry = info.get('industry')
                
                if sector or industry:
                    cursor.execute('''
                        UPDATE stock_symbol 
                        SET sector = ?, industry = ?
                        WHERE symbol = ?
                    ''', (sector, industry, symbol))
                    
                    logging.info(f"Updated {symbol}: sector={sector}, industry={industry}")
                
            except Exception as e:
                logging.error(f"Error updating sector/industry for {symbol}: {str(e)}")
                continue
                
            # Commit every 100 symbols to avoid holding transaction too long
            if symbols.index(symbol) % 100 == 0:
                self.db.commit()

            # Rate limiting
            time.sleep(delay)
        
        self.db.commit()
        elapsed = time.time() - start_time
        logging.info(f"Sector and industry backfill completed in {elapsed:.2f} seconds")


async def main():
    crawler = SymbolCrawler()
    #await crawler.crawl()
    crawler.backfill_sector_industry()
    crawler.close()

if __name__ == "__main__":
    #asyncio.run(main())
    import sys
    crawler = SymbolCrawler(db_path=sys.argv[1])
    crawler.backfill_sector_industry()
    crawler.close()