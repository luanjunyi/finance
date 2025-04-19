import sqlite3
import csv
import pandas as pd
import logging
from datetime import datetime, date, timedelta
from tqdm import tqdm
from fmp_data import FMPPriceLoader, Dataset


class InterdayTrading:
    def __init__(self, begin_date: str, end_date: str):
        """
        Initialize InterdayTrading with date range and load valid stocks.
        Args:
            begin_date: Start date in 'YYYY-MM-DD'
            end_date: End date in 'YYYY-MM-DD'
        """
        # Parse date strings
        self.begin_date = datetime.strptime(begin_date, '%Y-%m-%d').date()
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        # Price loader for fetching close prices
        self.price_loader = FMPPriceLoader()
        self.db_path = '/Users/jluan/code/finance/data/fmp_data.db'
        # Load valid symbols, sectors, industries
        self.stocks = self._load_valid_stocks()
        # Load free_cash_flow_per_share history
        self.fundamentals = self._load_fundamentals(self.stocks['symbol'].tolist())
    
    def _load_valid_stocks(self) -> pd.DataFrame:
        """Load valid stock symbols, sectors, and industries from database."""
        with sqlite3.connect(self.db_path) as conn:
            stocks = pd.read_sql_query(
                "SELECT symbol, sector, industry FROM valid_us_stocks_der", conn
            )
        return stocks

    def _is_trading_day(self, check_date: date) -> bool:
        """
        Check if the given date is a trading day by trying to get AAPL's price.
        
        Args:
            check_date: Date to check
            
        Returns:
            True if it's a trading day, False otherwise
        """
        try:
            _ = self.price_loader.get_price('AAPL', check_date)
            return True
        except Exception:
            return False        
    
    def _load_fundamentals(self, symbols: list) -> pd.DataFrame:
        """Load free cash flow per share data for the given symbols."""
        assert len(symbols) > 0, "No symbols provided"
        ds = Dataset(symbols, metrics={'free_cash_flow_per_share':'free_cash_flow_per_share'})
        fundamentals = ds.get_data()
        fundamentals['date'] = pd.to_datetime(fundamentals['date']).dt.date
        return fundamentals[['symbol','date','free_cash_flow_per_share']]
    
    def _get_price_data(self, symbols: list, current_date: date) -> pd.DataFrame:
        """
        Get price data for the given symbols on the given date.
        
        Args:
            symbols: List of symbols to get prices for
            current_date: Date to get prices for
            
        Returns:
            DataFrame with symbol and price columns
        """
        price_ds = Dataset(
            symbols,
            metrics={'adjusted_close':'price'},
            for_date=current_date.strftime('%Y-%m-%d')
        )
        price_data = price_ds.get_data()
        price_data['date'] = pd.to_datetime(price_data['date']).dt.date
        return price_data[price_data['date'] == current_date][['symbol','price']]
    
    def get_price_to_fcf(self, price_data: pd.DataFrame, current_date: date) -> pd.DataFrame:
        """
        Calculate 4-quarter sum of free cash flow and merge with valid symbols.
        Returns DataFrame with columns ['symbol','free_cash_flow','price','price_to_fcf'].
        
        Args:
            price_data: DataFrame with symbol and price columns
            current_date: Date to calculate for
            
        Returns:
            DataFrame with symbol, free_cash_flow, price, and price_to_fcf columns
        """
        # Compute FCF window bounds
        threshold = current_date - timedelta(days=90)
        low_bound = threshold - timedelta(days=460)
        window = self.fundamentals[
            (self.fundamentals['date'] >= low_bound) &
            (self.fundamentals['date'] <= threshold)
        ]
        
        # Find symbols with enough data (at least 4 records)
        counts = window.groupby('symbol').size().reset_index(name='count')
        valid_symbols = counts[counts['count'] >= 4]['symbol'].tolist()
        
        # Calculate FCF sum for each symbol (sum of 4 most recent quarters)
        window_valid = window[window['symbol'].isin(valid_symbols)]
        sorted_win = window_valid.sort_values(['symbol', 'date'], ascending=[True, False])
        last4 = sorted_win.groupby('symbol').head(4)
        sum_fcf = last4.groupby('symbol')['free_cash_flow_per_share'] \
            .sum().reset_index(name='free_cash_flow')
        
        # Merge with stocks data
        fcf_df = pd.merge(sum_fcf, self.stocks[['symbol']], on='symbol', how='inner')
        
        # Merge FCF with price data and calculate ratio
        df = pd.merge(fcf_df, price_data, on='symbol', how='inner')
        df['price_to_fcf'] = df['price'] / df['free_cash_flow']
        
        return df
    
    def generate(self, output_csv: str = "", verbose: bool = True) -> list:
        """
        Generate trading operations over the date range and write to CSV.
        
        Args:
            output_csv: Path to output CSV file (no header, rows: symbol,date,action,fraction)
            verbose: Whether to display progress bar and detailed logs
            
        Returns:
            List of operations [symbol, date_str, action, fraction]
        """
        operations = []
        current_date = self.begin_date
        
        # Calculate total days for progress bar
        total_days = (self.end_date - self.begin_date).days + 1
        
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        logger = logging.getLogger('InterdayTrading')
        
        logger.info(f"Processing {total_days} days with {len(self.stocks)} stocks")
        
        # Create progress bar
        with tqdm(total=total_days, desc="Processing days", disable=not verbose) as pbar:
            while current_date <= self.end_date:
                logger.info(f"Processing date: {current_date}")
                
                # Skip non-trading days
                if not self._is_trading_day(current_date):
                    logger.info(f"{current_date} is not a trading day, skipping")
                    current_date += timedelta(days=1)
                    pbar.update(1)
                    continue
                
                # Get price data for all stocks on current_date
                logger.info(f"Fetching price data for {len(self.stocks)} stocks on {current_date}")
                price_data = self._get_price_data(self.stocks['symbol'].tolist(), current_date)
                logger.info(f"Retrieved prices for {len(price_data)} stocks")
                
                # Combine free cash flow and price into price_to_fcf
                logger.info("Calculating price-to-FCF ratios")
                df = self.get_price_to_fcf(price_data, current_date)

                
                # Determine buy/sell operations
                logger.info("Determining trading operations")
                ops = self._determine_operations(df, current_date)
                logger.info(f"Generated {len(ops)} operations for {current_date}")
                operations.extend(ops)
                
                # Move to next day
                current_date += timedelta(days=1)
                pbar.update(1)
        # Write operations to CSV
        if output_csv:
            with open(output_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                for op in operations:
                    writer.writerow(op)
        return operations

    def _determine_operations(self, df: pd.DataFrame, current_date: date) -> list:
        """
        Placeholder for trading logic. Return list of [symbol, date_str, action, fraction].
        Args:
            df: DataFrame with columns symbol, free_cash_flow_per_share, price, price_to_fcf
            current_date: The date for this batch
        Returns:
            List of operations
        """
        # TODO: implement strategy logic to generate buy/sell signals
        return []

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate interday trading operations')
    parser.add_argument('--begin-date', type=str, default='2024-01-01', help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str, default='2024-06-30', help='End date in YYYY-MM-DD format')
    parser.add_argument('--output', type=str, default='', help='Output CSV file path')
    
    args = parser.parse_args()
    
    # Create InterdayTrading instance and generate operations
    trader = InterdayTrading(args.begin_date, args.end_date)
    operations = trader.generate(args.output)
    
    print(f"Generated {len(operations)} trading operations from {args.begin_date} to {args.end_date}")
    print(f"Results saved to {args.output}")
