import pandas as pd
import requests
import yfinance as yf
from io import StringIO
from datetime import datetime
import time
from tqdm import tqdm
import logging
import random
import os
from concurrent.futures import ThreadPoolExecutor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('stock_pipeline.log'),
        logging.StreamHandler()
    ]
)


def download_stock_data(symbol, start_date, end_date, output_dir='stock_data', wait_time=0.5):
    """Download historical data for a single stock"""
    tries = 0
    max_tries = 3
    backoff_factor = 1.5

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    while tries < max_tries:
        try:
            time.sleep(wait_time * (backoff_factor ** tries))
            stock = yf.Ticker(symbol)
            df = stock.history(start=start_date, end=end_date)

            if not df.empty:
                filename = f"{output_dir}/{symbol.replace('/', '_')}_daily.csv"
                df.to_csv(filename)
                return True
            else:
                tries += 1

        except Exception as e:
            logging.warning(
                f"Attempt {tries + 1} failed for {symbol}: {str(e)}")
            tries += 1

    logging.error(f"Failed to download {symbol} after {max_tries} attempts")
    return False


class StockDataPipeline:
    def __init__(self, output_dir='stock_data', rate_limit=2):
        self.output_dir = output_dir
        self.rate_limit = rate_limit
        self.wait_time = 1.0 / rate_limit

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def get_all_symbols(self):
        """Download symbols from NASDAQ's FTP site for all exchanges"""
        logging.info("Downloading symbols from NASDAQ directory...")
        url = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt"
        response = requests.get(url)
        data = StringIO(response.text)
        df = pd.read_csv(data, delimiter='|', skipfooter=1, engine='python')

        # Filter for common stocks
        df = df[
            (df['ETF'] == 'N') &  # Not an ETF
            (df['Test Issue'] == 'N') &  # Not a test issue
            (df['NextShares'] == 'N')  # Not a NextShares issue
        ]

        # Create clean exchange labels
        exchange_map = {
            'Q': 'NASDAQ',
            'N': 'NYSE',
            'A': 'NYSE American',
            'P': 'NYSE Arca',
            'Z': 'BATS',
            'V': 'IEX'
        }
        df['Exchange'] = df['Listing Exchange'].map(exchange_map)
        # Convert Symbol column to string type and handle NaN values
        df['Symbol'] = df['Symbol'].astype(str)
        # Keep only relevant columns and rename them
        df = df[['Symbol', 'Security Name', 'Exchange',
                 'Financial Status', 'Round Lot Size']]
        df = df.rename(columns={'Security Name': 'Name'})

        # Filter out preferred stocks, warrants, units, etc.
        df = df[~df['Symbol'].str.contains(
            '[\$\+\*\=\@]', regex=True)]  # Special characters
        # Dots (often indicate preferred shares)
        df = df[~df['Symbol'].str.contains('\.', regex=True)]
        df = df[~df['Name'].str.contains('Preferred|Partnership|Trust|Warrant|Right|Unit',
                                         case=False, regex=True)]

        # Log exchange breakdown
        logging.info("\nExchange breakdown:")
        logging.info(df['Exchange'].value_counts().to_string())

        return df

    def get_stock_info(self, symbol):
        """Get additional stock information using yfinance"""
        try:
            time.sleep(self.wait_time)
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return pd.Series({
                'Symbol': symbol,
                'Market_Cap': info.get('marketCap'),
                'Sector': info.get('sector'),
                'Industry': info.get('industry')
            })
        except Exception as e:
            logging.warning(f"Error getting info for {symbol}: {str(e)}")
            return pd.Series({
                'Symbol': symbol,
                'Market_Cap': None,
                'Sector': None,
                'Industry': None
            })

    def run_pipeline(self, start_date='2022-01-01', end_date=None,
                     min_market_cap=None, exchanges=None, max_concurrent=5):
        """
        Run the complete pipeline to get symbols and download historical data

        Parameters:
        start_date (str): Start date for historical data
        end_date (str): End date for historical data
        min_market_cap (float): Minimum market cap filter
        exchanges (list): List of exchanges to include (e.g., ['NYSE', 'NASDAQ'])
        max_concurrent (int): Maximum number of concurrent downloads
        """
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')

        # Get symbols
        logging.info("Starting symbol collection...")
        stocks_df = self.get_all_symbols()

        # Filter by exchange if specified
        if exchanges:
            stocks_df = stocks_df[stocks_df['Exchange'].isin(exchanges)]
            logging.info(
                f"Filtered to {len(stocks_df)} stocks from specified exchanges: {exchanges}")

        # Get additional info
        logging.info("Getting stock info...")
        with tqdm(total=len(stocks_df), desc="Getting stock info") as pbar:
            stock_info = []
            for symbol in stocks_df['Symbol']:
                info = self.get_stock_info(symbol)
                stock_info.append(info)
                pbar.update(1)

        stock_info_df = pd.DataFrame(stock_info)

        # Merge with original data
        full_df = stocks_df.merge(stock_info_df, on='Symbol', how='left')

        # Apply market cap filter if specified
        if min_market_cap:
            full_df = full_df[full_df['Market_Cap'] >= min_market_cap]
            logging.info(
                f"Filtered to {len(full_df)} stocks with market cap >= ${min_market_cap:,.0f}")

        # Save stock info
        full_df.to_csv(f"{self.output_dir}/stock_info.csv", index=False)
        logging.info(f"Found {len(full_df)} valid stocks")

        # Download historical data using ThreadPoolExecutor
        logging.info("Starting historical data download...")
        successful = 0
        failed = 0

        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = []
            for symbol in full_df['Symbol']:
                futures.append(
                    executor.submit(
                        download_stock_data,
                        symbol,
                        start_date,
                        end_date,
                        self.output_dir,
                        self.wait_time
                    )
                )

            with tqdm(total=len(futures), desc="Downloading historical data") as pbar:
                for future in futures:
                    if future.result():
                        successful += 1
                    else:
                        failed += 1
                    pbar.update(1)

        # Final summary
        logging.info(f"""
        Pipeline Complete:
        - Total stocks processed: {len(full_df)}
        - Successful downloads: {successful}
        - Failed downloads: {failed}
        - Data saved to: {self.output_dir}
        """)


# Example usage:
if __name__ == "__main__":
    pipeline = StockDataPipeline(
        output_dir='stock_data',
        rate_limit=2  # 2 requests per second
    )

    pipeline.run_pipeline(
        start_date='2022-01-01',
        # min_market_cap=1e9,  # $1 billion minimum market cap
        exchanges=['NYSE', 'NASDAQ'],  # Only NYSE and NASDAQ
        max_concurrent=5  # 5 concurrent downloads
    )
