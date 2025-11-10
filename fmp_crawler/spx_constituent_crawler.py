from fmp_crawler.base_fmp_crawler import BaseFMPCrawler
import logging
from datetime import datetime


class SpxConstituentCrawler(BaseFMPCrawler):
    async def crawl(self):
        """Crawl S&P 500 constituent changes and store them in the database."""
        logging.info("Starting S&P 500 constituent changes crawling...")

        # First, fetch historical constituent changes
        await self._crawl_historical_changes()
        
        # Then, fetch current constituents and fill gaps
        await self._crawl_current_constituents()
        
        logging.info("Completed S&P 500 constituent changes crawling")

    async def _crawl_historical_changes(self):
        """Crawl historical S&P 500 constituent changes."""
        logging.info("Fetching historical S&P 500 constituent changes...")
        
        # Fetch constituent changes data
        data = await self.make_request('historical-sp500-constituent')
        if not data:
            logging.fatal("Failed to fetch S&P 500 constituent changes")
            return

        logging.info(f"Fetched {len(data)} historical constituent change records")

        cursor = self.db.cursor()
        processed_count = 0

        for entry in data:
            try:
                # Parse the dateAdded field (e.g., "March 24, 2025")
                date_added_str = entry.get('dateAdded')
                if not date_added_str:
                    logging.fatal(f"Missing dateAdded field in entry: {entry}")
                    continue

                # Convert date format from "March 24, 2025" to "2025-03-24"
                date_added = self._parse_date(date_added_str)
                if not date_added:
                    logging.fatal(f"Could not parse date: {date_added_str}")
                    continue

                # Get symbols for add and remove operations
                add_symbol = entry.get('symbol')
                remove_symbol = entry.get('removedTicker')

                if not add_symbol and not remove_symbol:
                    logging.warning(f"Missing both added and removed symbol data in entry: {entry}")
                    continue

                # Insert add record
                if add_symbol:
                    cursor.execute('''
                        INSERT OR REPLACE INTO spx_constituent_changes 
                        (symbol, date, type)
                        VALUES (?, ?, ?)
                    ''', (add_symbol, date_added, 'add'))

                # Insert remove record
                if remove_symbol:
                    cursor.execute('''
                        INSERT OR REPLACE INTO spx_constituent_changes 
                        (symbol, date, type)
                        VALUES (?, ?, ?)
                    ''', (remove_symbol, date_added, 'remove'))

                processed_count += 1

            except Exception as e:
                logging.fatal(f"Error processing historical entry {entry}: {str(e)}")

        self.db.commit()
        logging.info(f"Successfully processed {processed_count} historical constituent change records")

    async def _crawl_current_constituents(self):
        """Crawl current S&P 500 constituents and fill gaps in historical data."""
        logging.info("Fetching current S&P 500 constituents...")
        
        # Fetch current constituents
        current_data = await self.make_request('sp500-constituent')
        if not current_data:
            logging.fatal("Failed to fetch current S&P 500 constituents")
            return

        logging.info(f"Fetched {len(current_data)} current S&P 500 constituents")

        cursor = self.db.cursor()
        
        missing_count = 0
        
        for constituent in current_data:
            try:
                symbol = constituent.get('symbol')
                if not symbol:
                    logging.fatal(f"Missing symbol in current constituent: {constituent}")
                    continue
                
                # Check if this symbol is missing from historical data
                # A symbol should be in historical adds if it's currently in S&P 500
                # Unless it was there from the beginning and never changed
                cursor.execute('''
                    SELECT type 
                    FROM spx_constituent_changes 
                    WHERE symbol = ? 
                    ORDER BY date DESC 
                    LIMIT 1
                ''', (symbol,))
                last_type = cursor.fetchone()
                
                if last_type is None or last_type[0] != 'add':
                    # This symbol is currently in S&P 500 but not in historical adds
                    # We need to add it with the dateFirstAdded if available
                    
                    date_first_added_str = constituent.get('dateFirstAdded')
                    if date_first_added_str:
                        # Parse the date (assuming it's in YYYY-MM-DD format)
                        try:
                            # Validate the date format
                            parsed_date = datetime.strptime(date_first_added_str, "%Y-%m-%d")
                            add_date = date_first_added_str
                        except ValueError:
                            # If it's not in YYYY-MM-DD format, try to parse it
                            add_date = self._parse_date(date_first_added_str)
                            if not add_date:
                                logging.fatal(f"Could not parse dateFirstAdded '{date_first_added_str}' for {symbol}")
                                continue
                    else:
                        # If no dateFirstAdded, we can't determine when it was added
                        logging.fatal(f"No dateFirstAdded for current constituent {symbol}")
                        continue
                    
                    # Insert the missing add record
                    cursor.execute('''
                        INSERT OR REPLACE INTO spx_constituent_changes 
                        (symbol, date, type)
                        VALUES (?, ?, ?)
                    ''', (symbol, add_date, 'add'))
                    
                    missing_count += 1
                    logging.info(f"Added missing constituent: {symbol} on {add_date}")
                    
            except Exception as e:
                logging.fatal(f"Error processing current constituent {constituent}: {str(e)}")
        
        self.db.commit()
        logging.info(f"Successfully added {missing_count} missing constituent records from current data")

    def _parse_date(self, date_str: str) -> str:
        """
        Parse date from format 'March 24, 2025' to 'yyyy-mm-dd'.
        
        Args:
            date_str: Date string in format like "March 24, 2025"
            
        Returns:
            Date string in format "yyyy-mm-dd" or None if parsing fails
        """
        try:
            # Parse the date string
            parsed_date = datetime.strptime(date_str, "%B %d, %Y")
            # Return in yyyy-mm-dd format
            return parsed_date.strftime("%Y-%m-%d")
        except ValueError as e:
            logging.fatal(f"Failed to parse date '{date_str}': {str(e)}")
            return None


if __name__ == "__main__":
    import sys
    import asyncio
    
    async def main():
        if len(sys.argv) < 2:
            print("Usage: python spx_constituent_crawler.py <db_path>")
            sys.exit(1)
            
        crawler = SpxConstituentCrawler(db_path=sys.argv[1])
        await crawler.crawl()
        crawler.close()
    
    asyncio.run(main())