"""
EODHD Batch Price History Crawler

This module fetches historical price data for all US stocks from EODHD API
and stores it in the daily_price_eodhd table. It fetches data from yesterday
back to 30 years ago in anti-chronological order.
"""

import sqlite3
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any
from tqdm import tqdm
import time
import os

from eodhd import APIClient
from utils.logging_config import setup_logging as setup_global_logging
from utils.config import EODHD_API_KEY


class EODHDBatchPriceCrawler:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.api_client = APIClient(api_key=EODHD_API_KEY)
        setup_global_logging()
        
        # Setup database connection
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        
        # Rate limiting - be conservative with EODHD API
        self.rate_limit_delay = 60 / 1000 # 1000 requests per minute
        self.last_request_time = 0
        
        logging.info(f"Initialized EODHD Batch Price Crawler with database: {db_path}")

    def rate_limit(self):
        """Ensure we don't exceed API rate limits"""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        
        if time_since_last_request < self.rate_limit_delay:
            sleep_time = self.rate_limit_delay - time_since_last_request
            logging.info(f"Rate limiting: {sleep_time} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

    def get_date_range(self, days_back: int = 30 * 365) -> List[str]:
        """
        Generate list of dates from yesterday back to specified days
        Returns dates in anti-chronological order (newest first)
        """
        end_date = datetime.now() - timedelta(days=1)  # Yesterday
        start_date = end_date - timedelta(days=days_back)
        
        dates = []
        current_date = end_date
        
        while current_date >= start_date:
            dates.append(current_date.strftime('%Y-%m-%d'))
            current_date -= timedelta(days=1)
        
        return dates

    def date_exists_in_db(self, date: str) -> bool:
        """Check if we already have data for this date"""
        cursor = self.db.cursor()
        cursor.execute(
            "SELECT COUNT(*) as count FROM daily_price_eodhd WHERE date = ?",
            (date,)
        )
        result = cursor.fetchone()
        return result['count'] > 0

    def save_price_data(self, price_data: List[Dict[str, Any]], date: str):
        """Save price data to the database"""
        cursor = self.db.cursor()
        saved_count = 0
        
        for price in price_data:
            try:
                # Map EODHD fields to our database schema
                symbol = price.get('code')
                if not symbol:
                    continue
                
                # Skip if volume is 0 and close price is 0 or negative
                volume = price.get('volume', 0)
                close_price = price.get('close', 0)
                
                if volume == 0 and (close_price <= 0):
                    continue
                
                cursor.execute('''
                    INSERT OR REPLACE INTO daily_price_eodhd 
                    (symbol, date, open, high, low, close, adjusted_close, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
                    date,
                    price.get('open'),
                    price.get('high'),
                    price.get('low'),
                    price.get('close'),
                    price.get('adjusted_close'),
                    price.get('volume', 0)
                ))
                saved_count += 1
                
            except Exception as e:
                logging.warning(f"Error saving price data for {price.get('code', 'unknown')}: {e}")
                continue
        
        self.db.commit()
        logging.info(f"Saved {saved_count} price records for {date}")
        return saved_count

    def fetch_and_save_date(self, date: str) -> int:
        """Fetch and save price data for a specific date"""      
        try:
            # Rate limit before API call
            self.rate_limit()
            
            logging.debug(f"Fetching price data for {date}")
            price_data = self.api_client.get_eod_splits_dividends_data(country='US', date=date)
            
            if not price_data:
                logging.info(f"No price data returned for {date}")
                return 0
            
            logging.debug(f"Retrieved {len(price_data)} price records for {date}")
            return self.save_price_data(price_data, date)
            
        except Exception as e:
            logging.error(f"Error fetching data for {date}: {e}")
            return 0

    def crawl_historical_prices(self, days_back: int = 30 * 365, skip_existing: bool = True):
        """
        Main method to crawl historical price data
        
        Args:
            days_back: Number of days to go back (default: 30 years)
            skip_existing: Whether to skip dates that already exist in DB
        """
        logging.info(f"Starting historical price crawl for {days_back} days back")
        
        dates = self.get_date_range(days_back)
        total_saved = 0
        
        with tqdm(total=len(dates), desc="Crawling price history from EODHD") as pbar:
            for date in dates:
                if date >= '2023-12-01':
                    print(f"[TEMP FIX, REMOVE AFTER RUN] Skipping {date}")
                    continue
                pbar.set_description(f"Processing {date}")
                
                if skip_existing and self.date_exists_in_db(date):
                    pbar.update(1)
                    continue
                
                saved_count = self.fetch_and_save_date(date)
                total_saved += saved_count
                
                pbar.update(1)
                pbar.set_postfix({"Total saved": total_saved})
        
        logging.info(f"Completed crawl. Total records saved: {total_saved}")

    def close(self):
        """Close database connection"""
        if self.db:
            self.db.close()


def main():
    """Main function to run the crawler"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EODHD Batch Price History Crawler')
    parser.add_argument('--db-path', required=True, help='Path to SQLite database')
    parser.add_argument('--days-back', type=int, default=30*365, 
                       help='Number of days to go back (default: 30 years)')
    parser.add_argument('--no-skip-existing', action='store_true',
                       help='Do not skip dates that already exist in database')
    
    args = parser.parse_args()
    
    crawler = None
    try:
        crawler = EODHDBatchPriceCrawler(args.db_path)
        crawler.crawl_historical_prices(
            days_back=args.days_back,
            skip_existing=not args.no_skip_existing
        )
    except KeyboardInterrupt:
        logging.info("Crawl interrupted by user")
    except Exception as e:
        logging.error(f"Crawl failed: {e}")
        raise
    finally:
        if crawler:
            crawler.close()


if __name__ == "__main__":
    main()
