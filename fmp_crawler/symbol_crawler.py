from fmp_crawler.base_fmp_crawler import BaseFMPCrawler
import time
import logging
from tqdm import tqdm


class SymbolCrawler(BaseFMPCrawler):
    async def crawl(self):
        logging.info("Starting symbol crawling...")

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

    async def backfill_sector_industry(self):
        """Backfill sector and industry data using FMP API."""
        logging.info("Starting sector and industry backfill...")
        
        cursor = self.db.cursor()
        # Get all symbols that need backfilling
        cursor.execute('''
            SELECT symbol 
            FROM stock_symbol 
            WHERE type = 'stock' AND (sector IS NULL OR industry IS NULL)
        ''')
        symbols = [row[0] for row in cursor.fetchall()]
        logging.info(f"Found {len(symbols)} symbols to backfill")
        
        for symbol in tqdm(symbols):
            # Fetch company profile from FMP API
            profile_data = await self.make_request('profile', {'symbol': symbol})
            
            if not profile_data or len(profile_data) == 0:
                logging.warning(f"No profile data found for {symbol}")
                continue
                
            # FMP returns a list with one item for the profile
            profile = profile_data[0]
            sector = profile.get('sector')
            industry = profile.get('industry')
            
            if sector or industry:
                cursor.execute('''
                    UPDATE stock_symbol 
                        SET sector = ?, industry = ?
                        WHERE symbol = ?
                    ''', (sector, industry, symbol))
                    
                # Commit immediately after each update
                self.db.commit()
        
        logging.info("Completed backfilling sector and industry")
    

if __name__ == "__main__":
    import sys
    import asyncio
    
    async def main():
        crawler = SymbolCrawler(db_path=sys.argv[1])
        await crawler.backfill_sector_industry()
        crawler.close()
    
    asyncio.run(main())
