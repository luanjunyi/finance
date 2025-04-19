import sqlite3
import csv
import pandas as pd
from datetime import datetime, date, timedelta
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
        self.daily_features = self._load_fcf_data(self.stocks['symbol'].tolist())
    
    def _load_valid_stocks(self) -> pd.DataFrame:
        """Load valid stock symbols, sectors, and industries from database."""
        with sqlite3.connect(self.db_path) as conn:
            stocks = pd.read_sql_query(
                "SELECT symbol, sector, industry FROM valid_us_stocks_der", conn
            )
        return stocks
    
    def _load_fcf_data(self, symbols: list) -> pd.DataFrame:
        """Load free cash flow per share data for the given symbols."""
        assert len(symbols) > 0, "No symbols provided"
        ds = Dataset(symbols, metrics={'free_cash_flow_per_share':'free_cash_flow_per_share'})
        daily = ds.get_data()
        daily['date'] = pd.to_datetime(daily['date']).dt.date
        return daily[['symbol','date','free_cash_flow_per_share']]

    def _get_fcf_window(self, current_date: date) -> pd.DataFrame:
        """
        Get the window of FCF data to use for calculations.
        
        Args:
            current_date: The current date to calculate the window for
            
        Returns:
            DataFrame with FCF data in the appropriate window
        """
        # Compute threshold and window bounds
        threshold = current_date - timedelta(days=90)
        low_bound = threshold - timedelta(days=460)
        window = self.daily_features[
            (self.daily_features['date'] >= low_bound) &
            (self.daily_features['date'] <= threshold)
        ]
        return window
    
    def _get_symbols_with_enough_data(self, window_df: pd.DataFrame, min_records: int = 4) -> list:
        """
        Get symbols that have at least min_records in the window.
        
        Args:
            window_df: DataFrame with the FCF data window
            min_records: Minimum number of records required
            
        Returns:
            List of symbols with enough data
        """
        counts = window_df.groupby('symbol').size().reset_index(name='count')
        valid_syms = counts[counts['count'] >= min_records]['symbol']
        return valid_syms.tolist()
    
    def _calculate_fcf_sum(self, window_df: pd.DataFrame, valid_symbols: list) -> pd.DataFrame:
        """
        Calculate the sum of the 4 most recent FCF values for each symbol.
        
        Args:
            window_df: DataFrame with the FCF data window
            valid_symbols: List of symbols to include
            
        Returns:
            DataFrame with symbol and free_cash_flow columns
        """
        window_valid = window_df[window_df['symbol'].isin(valid_symbols)]
        sorted_win = window_valid.sort_values(['symbol', 'date'], ascending=[True, False])
        last4 = sorted_win.groupby('symbol').head(4)
        sum_fcf = last4.groupby('symbol')['free_cash_flow_per_share'] \
            .sum().reset_index(name='free_cash_flow')
        return sum_fcf
    
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
        # Get FCF window and valid symbols
        window = self._get_fcf_window(current_date)
        valid_symbols = self._get_symbols_with_enough_data(window)
        
        # Calculate FCF sum and merge with stocks
        sum_fcf = self._calculate_fcf_sum(window, valid_symbols)
        fcf_df = pd.merge(sum_fcf, self.stocks[['symbol']], on='symbol', how='inner')
        
        # Merge FCF with price data and calculate ratio
        df = pd.merge(fcf_df, price_data, on='symbol', how='inner')
        df['price_to_fcf'] = df['price'] / df['free_cash_flow']
        
        return df

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
    
    def generate(self, output_csv: str = "") -> list:
        """
        Generate trading operations over the date range and write to CSV.
        
        Args:
            output_csv: Path to output CSV file (no header, rows: symbol,date,action,fraction)
            
        Returns:
            List of operations [symbol, date_str, action, fraction]
        """
        operations = []
        current_date = self.begin_date
        
        while current_date <= self.end_date:
            # Skip non-trading days
            if not self._is_trading_day(current_date):
                current_date += timedelta(days=1)
                continue
                
            # Get price data for all stocks on current_date
            price_data = self._get_price_data(self.stocks['symbol'].tolist(), current_date)
            
            # Combine free cash flow and price into price_to_fcf
            df = self.get_price_to_fcf(price_data, current_date)
            
            # Determine buy/sell operations
            ops = self._determine_operations(df, current_date)
            operations.extend(ops)
            
            # Move to next day
            current_date += timedelta(days=1)
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
