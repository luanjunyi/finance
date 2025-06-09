
import asyncio
import argparse
from fmp_crawler.symbol_crawler import SymbolCrawler
from fmp_crawler.price_crawler import PriceCrawler
from fmp_crawler.financial_crawler import FinancialCrawler
from fmp_crawler.metrics_crawler import MetricsCrawler

async def main():
    parser = argparse.ArgumentParser(description='FMP Data Crawler')
    parser.add_argument('--type', choices=['symbol', 'price', 'financial', 'metrics'],
                        required=True, help='Type of data to crawl')
    parser.add_argument('--db-path', required=True, help='Path to the database file taht store the crawled result')

    args = parser.parse_args()
    db_path = args.db_path

    if args.type == 'symbol':
        crawler = SymbolCrawler(db_path)
    elif args.type == 'price':
        crawler = PriceCrawler(db_path)
    elif args.type == 'financial':
        crawler = FinancialCrawler(db_path)
    elif args.type == 'metrics':
        crawler = MetricsCrawler(db_path)
    else:
        raise ValueError(f"Invalid type: {args.type}")

    try:
        await crawler.crawl()
    finally:
        crawler.close()

if __name__ == "__main__":
    asyncio.run(main())
