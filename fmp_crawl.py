
import asyncio
import argparse
from fmp_crawler.symbol_crawler import SymbolCrawler
from fmp_crawler.price_crawler import PriceCrawler
from fmp_crawler.financial_crawler import FinancialCrawler
from fmp_crawler.metrics_crawler import MetricsCrawler
from fmp_crawler.owner_earnings_crawler import OwnerEarningsCrawler

async def main():
    parser = argparse.ArgumentParser(description='FMP Data Crawler')
    parser.add_argument('--type', choices=['symbol', 'price', 'financial', 'metrics', 'owner'],
                        required=True, help='Type of data to crawl')

    args = parser.parse_args()

    if args.type == 'symbol':
        crawler = SymbolCrawler()
    elif args.type == 'price':
        crawler = PriceCrawler()
    elif args.type == 'financial':
        crawler = FinancialCrawler()
    elif args.type == 'metrics':
        crawler = MetricsCrawler()
    elif args.type == 'owner':
        crawler = OwnerEarningsCrawler()
    else:
        raise ValueError(f"Invalid type: {args.type}")

    try:
        await crawler.crawl()
    finally:
        crawler.close()

if __name__ == "__main__":
    asyncio.run(main())
