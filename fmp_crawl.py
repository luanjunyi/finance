import asyncio
import argparse
from fmp_crawler.symbol_crawler import SymbolCrawler
from fmp_crawler.price_crawler import PriceCrawler
from fmp_crawler.financial_crawler import FinancialCrawler

async def main():
    parser = argparse.ArgumentParser(description='FMP Data Crawler')
    parser.add_argument('--type', choices=['symbol', 'price', 'financial'],
                      required=True, help='Type of data to crawl')
    
    args = parser.parse_args()
    
    if args.type == 'symbol':
        crawler = SymbolCrawler()
    elif args.type == 'price':
        crawler = PriceCrawler()
    else:  # financial
        crawler = FinancialCrawler()
        
    try:
        await crawler.crawl()
    finally:
        crawler.close()

if __name__ == "__main__":
    asyncio.run(main()) 