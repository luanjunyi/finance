import sqlite3
import random
import os
from datetime import datetime, timedelta
from trade_sim_util import PriceLoader, FMPPriceLoader
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def get_random_symbols(n=400):
    """Get n random stock symbols from the CSV files in stock_data directory"""
    stock_data_dir = 'stock_data'
    
    # Get all CSV files in the stock_data directory
    csv_files = [f for f in os.listdir(stock_data_dir) if f.endswith('_daily.csv')]
    
    # Extract symbols from filenames (remove _daily.csv suffix)
    symbols = [f.replace('_daily.csv', '') for f in csv_files]
    
    # Return random selection of symbols
    return random.sample(symbols, min(n, len(symbols)))

def get_date_range():
    """Get 10 random dates between 2022-02-01 and 2024-11-01 that exist in the database"""
    conn = sqlite3.connect('data/fmp_data.db')
    cursor = conn.cursor()
    
    # Get all dates in our range that exist in the database
    cursor.execute('''
        SELECT DISTINCT date 
        FROM daily_price 
        WHERE date BETWEEN '2022-02-01' AND '2024-11-01'
        ORDER BY date
    ''')
    
    available_dates = [row[0] for row in cursor.fetchall()]
    cursor.close()
    conn.close()
    
    if len(available_dates) < 10:
        raise ValueError("Not enough dates available in the database")
    
    # Select 10 random dates from available dates
    return sorted(random.sample(available_dates, 10))

def compare_prices():
    """Compare prices between PriceLoader and FMPPriceLoader"""
    csv_loader = PriceLoader()
    db_loader = FMPPriceLoader()
    
    symbols = get_random_symbols()
    dates = get_date_range()
    price_types = ['Open', 'High', 'Low', 'Close']
    
    discrepancies = []
    total_comparisons = 0
    valid_symbols = []
    
    # First filter out symbols that don't exist in CSV data
    for symbol in symbols:
        try:
            # Try to get any price to verify symbol exists
            csv_loader.get_price(symbol, dates[0], 'Close')
            valid_symbols.append(symbol)
        except (KeyError, ValueError):
            continue
    
    logging.info(f"Found {len(valid_symbols)} valid symbols out of {len(symbols)} total")
    
    # Compare prices only for valid symbols
    for symbol in valid_symbols:
        for date_str in dates:
            for price_type in price_types:
                total_comparisons += 1
                try:
                    csv_price = csv_loader.get_price(symbol, date_str, price_type)
                    db_price = db_loader.get_price(symbol, date_str, price_type)
                    
                    # Compare with some tolerance for floating point differences
                    if abs(csv_price - db_price) > 0.01:  # 1 cent tolerance
                        discrepancies.append({
                            'symbol': symbol,
                            'date': date_str,
                            'price_type': price_type,
                            'csv_price': csv_price,
                            'db_price': db_price,
                            'difference': abs(csv_price - db_price),
                            'pct_difference': abs(csv_price - db_price) / csv_price * 100
                        })
                except (KeyError, ValueError) as e:
                    # Only log when we have data in one source but not the other
                    logging.debug(f"Missing data for {symbol} on {date_str} for {price_type}: {str(e)}")
                except Exception as e:
                    logging.error(f"Unexpected error for {symbol} on {date_str} for {price_type}: {str(e)}")
    
    # Print summary
    logging.info(f"\nComparison Summary:")
    logging.info(f"Valid symbols compared: {len(valid_symbols)}")
    logging.info(f"Total comparisons made: {total_comparisons}")
    logging.info(f"Total discrepancies found: {len(discrepancies)}")
    
    if discrepancies:
        logging.info("\nDiscrepancies found:")
        for d in discrepancies:
            logging.info(
                f"Symbol: {d['symbol']}, Date: {d['date']}, Type: {d['price_type']}")
            logging.info(
                f"  CSV Price: ${d['csv_price']:.2f}, DB Price: ${d['db_price']:.2f}")
            logging.info(
                f"  Absolute Difference: ${d['difference']:.2f}")
            logging.info(
                f"  Percentage Difference: {d['pct_difference']:.2f}%")
            logging.info("-" * 50)

if __name__ == '__main__':
    compare_prices()
