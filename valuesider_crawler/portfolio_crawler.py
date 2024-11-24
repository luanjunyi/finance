from playwright.async_api import async_playwright
import pandas as pd
from typing import List, Dict
import time
from datetime import datetime
import logging
import argparse
import re
import asyncio

from utils.logging_config import setup_logging

setup_logging(logging.DEBUG)


class PortfolioCrawler:
    def __init__(self, start_url: str):
        self.start_url = start_url
        self.data_by_date = {}
        
    def extract_date_from_url(self, url: str) -> str:
        """Extract date from URL in YYYY-MM format."""
        url = url.split('?')[0]
        pattern = r'/(\d{4})/(\d+)$'
        match = re.search(pattern, url)

        year, quarter = match.groups()
        if quarter == '1':
            return datetime(int(year), 3, 31).strftime('%Y-%m-%d')
        elif quarter == '2':
            return datetime(int(year), 6, 30).strftime('%Y-%m-%d')
        elif quarter == '3':
            return datetime(int(year), 9, 30).strftime('%Y-%m-%d')
        elif quarter == '4':
            return datetime(int(year), 12, 31).strftime('%Y-%m-%d')
        else:
            raise ValueError(f"Expecting to have year and quarter in URL, but got {url}")

    async def extract_row_data(self, row) -> Dict:
        """Extract data from a single table row."""
        try:
            return {
                'symbol': await (await row.query_selector('.guru_table_column >> nth=1')).inner_text(),
                'company': await (await row.query_selector('.guru_table_column >> nth=2')).inner_text(),
                'portfolio_weight': await (await row.query_selector('.guru_table_column >> nth=3')).inner_text(),
                'shares': await (await row.query_selector('.guru_table_column >> nth=4')).inner_text(),
                'buy_price': await (await row.query_selector('.guru_table_column >> nth=5')).inner_text(),
                'current_price': await (await row.query_selector('.guru_table_column >> nth=6')).inner_text(),
                'value': await (await row.query_selector('.guru_table_column >> nth=7')).inner_text(),
                'change': await (await row.query_selector('.guru_table_column >> nth=9')).inner_text()
            }
        except Exception as e:
            logging.error(f"Error extracting row data: {e}")
            return None

    async def crawl_quarter(self) -> str:
        """Crawl all pages within a single quarter."""
        async with async_playwright() as p:
            browser = await p.firefox.launch(headless=True, firefox_user_prefs={
                "javascript.enabled": True
            })
            page = await browser.new_page()
            current_url = self.start_url
            page_num = 1
            current_page_data = []
            quarter_date = self.extract_date_from_url(current_url)
            
            try:
                while True:
                    logging.info(f"Processing page {page_num} for quarter {quarter_date}: {current_url}")
                    await page.goto(current_url)
                    # Wait for the table to load
                    await page.wait_for_selector(".guru_table_body")
                    
                    # Extract data from current page
                    rows = await page.query_selector_all(".guru_table_body .guru_table_row")
                    for row in rows:
                        row_data = await self.extract_row_data(row)
                        if row_data:
                            current_page_data.append(row_data)
                    
                    # Check for next page
                    next_link = await page.query_selector('a[rel="next"]')
                    if not next_link:
                        logging.info("No more pages to process")
                        break
                        
                    current_url = await next_link.get_attribute('href')
                    if not current_url:
                        break
                        
                    current_url = 'https://valuesider.com' + current_url if not current_url.startswith('http') else current_url
                    page_num += 1
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logging.error(f"Error during crawling: {e}")
            finally:
                # Create DataFrame for this quarter's data
                if current_page_data and quarter_date:
                    df = pd.DataFrame(current_page_data)
                    self._clean_dataframe(df)
                    self.data_by_date[quarter_date] = df
                
                next_quarter_link = await page.query_selector('a.arrow.arrow-right')
                next_quarter_url = await next_quarter_link.get_attribute('href') if next_quarter_link else None
                await browser.close()
                
        return next_quarter_url

    async def crawl(self) -> Dict[str, pd.DataFrame]:
        """Crawl all quarters starting from the initial URL."""
        self.data_by_date = {}  # Reset data for fresh crawl
        current_url = self.start_url
        quarter_num = 1
        
        while current_url:
            logging.info(f"Crawling quarter: {current_url}")
            self.start_url = current_url  # Update start_url for the next quarter
            next_quarter_url = await self.crawl_quarter()
            
            # Move to next quarter
            current_url = next_quarter_url
            if current_url and not current_url.startswith('http'):
                current_url = 'https://valuesider.com' + current_url

            quarter_num += 1
        
        return self.data_by_date
    
    def _clean_dataframe(self, df: pd.DataFrame) -> None:
        """Clean up the DataFrame numeric columns."""
        if not df.empty:
            # Clean up numeric columns
            df['portfolio_weight'] = df['portfolio_weight'].str.rstrip('%').astype('float') / 100.0
            df['shares'] = df['shares'].str.replace(',', '').astype('float')
            df['buy_price'] = df['buy_price'].str.lstrip('$').str.replace(',', '').astype('float')
            
            try:
                df['current_price'] = pd.to_numeric(df['current_price'].str.extract(r'\$(\d+\.?\d*)')[0], errors='coerce')
            except Exception as e:
                logging.error(f"Error parsing current price: {e}")
                df['current_price'] = float('nan')
            
            df['value'] = df['value'].str.lstrip('$').str.replace(',', '').astype('float') * 1000
            try:
                df['change'] = df['change'].str.rstrip('%').astype('float') / 100.0
            except Exception as e:
                logging.error(f"Error parsing change: {e}")
                df['change'] = float('nan')

async def main():   
    parser = argparse.ArgumentParser(description='Crawl portfolio data from ValueSider')
    parser.add_argument('start_url', help=
    'Starting URL for crawling (e.g., https://valuesider.com/guru/warren-buffett-berkshire-hathaway/portfolio/2023/2)')
    args = parser.parse_args()
    
    crawler = PortfolioCrawler(args.start_url)
    data_by_date = await crawler.crawl()
    
    # Save each date's data to a separate CSV file
    for date, df in data_by_date.items():
        print(date)
        print(df)
    
    return data_by_date

if __name__ == "__main__":
    asyncio.run(main())