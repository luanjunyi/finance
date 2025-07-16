import sqlite3
import csv
import pandas as pd
import logging
from datetime import datetime, date, timedelta
from tqdm import tqdm
from fmp_data_legacy import FMPPriceLoader, Dataset
import numpy as np
from utils.config import FMP_DB_PATH


class InterdayTrading:
    def __init__(self, begin_date: str, end_date: str, db_path: str = FMP_DB_PATH):
        """
        Initialize InterdayTrading with date range and load valid stocks.
        Args:
            begin_date: Start date in 'YYYY-MM-DD'
            end_date: End date in 'YYYY-MM-DD'
        """
        # Configure logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger('InterdayTrading')
        self.db_path = db_path
        # Parse date strings
        self.begin_date = datetime.strptime(begin_date, '%Y-%m-%d').date()
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        # Price loader for fetching close prices
        self.price_loader = FMPPriceLoader(db_path=db_path)
        # Load valid symbols, sectors, industries
        self.stocks = self._load_valid_stocks()
        # Load fundamentals from financial statements
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
        """Load fundamentals from financial statements"""
        assert len(symbols) > 0, "No symbols provided"
        ds = Dataset(symbols, metrics={
            'free_cash_flow_per_share': '',
            'revenue': '',
            'operating_profit_margin': 'operating_margin',
            'total_current_assets': '',
            'total_debt': '',
            'weighted_average_shares_outstanding_diluted': 'num_shares',
        }, db_path=self.db_path)
        fundamentals = ds.get_data()
        fundamentals['date'] = pd.to_datetime(fundamentals['date']).dt.date
        return fundamentals[['symbol', 'date', 'free_cash_flow_per_share', 'revenue', 'operating_margin', 'total_current_assets', 'total_debt', 'num_shares']]
    
    def _get_price_data(self, symbols: list, current_date: date) -> pd.DataFrame:
        """
        Get price data for the given symbols on the given date.
        
        Args:
            symbols: List of symbols to get prices for
            current_date: Date to get prices for
            
        Returns:
            DataFrame with symbol, price, and market_cap columns
        """
        price_ds = Dataset(
            symbols,
            metrics={'close':'price'},
            for_date=current_date.strftime('%Y-%m-%d'),
            db_path=self.db_path
        )
        price_data = price_ds.get_data()
        price_data['date'] = pd.to_datetime(price_data['date']).dt.date
        price_data = price_data[price_data['date'] == current_date][['symbol','price']]
        
        # Use _filter_fundamentals to get the most recent data for each symbol
        # Use a large window to ensure we capture data for all symbols
        # and min_records=1 to include symbols with at least one record
        filtered_data = self._filter_fundamentals(current_date, 100, 1)
        
        # Take the first row for each symbol (most recent)
        latest_shares = filtered_data.groupby('symbol').first().reset_index()[['symbol', 'num_shares']]
        
        # Merge price data with shares data
        result = pd.merge(price_data, latest_shares, on='symbol', how='inner')
        
        # Calculate market cap as price * num_shares
        result['market_cap'] = result['price'] * result['num_shares']
        
        return result
        
    def _get_price_before(self, symbols: list, target_date: date) -> pd.DataFrame:
        """
        Get price data for the given symbols on the most recent trading day on or before the target date.
        
        Args:
            symbols: List of symbols to get prices for
            target_date: Target date to get prices for or before
            
        Returns:
            DataFrame with symbol, price, and actual_date columns
        """
        # Start with the target date
        check_date = target_date
        
        # Find the most recent trading day (going backward in time)
        max_attempts = 5  # Limit the number of days to check to avoid infinite loop
        attempts = 0
        
        while attempts < max_attempts:
            if self._is_trading_day(check_date):
                # Found a trading day, get price data
                self.logger.debug(f"Found trading day {check_date} for target date {target_date}")
                price_data = self._get_price_data(symbols, check_date)
                
                # Add the actual date to the result
                price_data['actual_date'] = check_date
                price_data.drop(columns=['market_cap', 'num_shares'], inplace=True)
                return price_data
            
            # Move to previous day
            check_date = check_date - timedelta(days=1)
            attempts += 1
        
        # If we couldn't find a trading day after max_attempts, return empty DataFrame
        raise ValueError(f"Could not find trading day within {max_attempts} days before {target_date}")
    
    def _filter_fundamentals(self, current_date: date, window_days: int, min_records: int) -> pd.DataFrame:
        """
        Filter fundamentals data to a specific window and ensure minimum record count.
        
        Args:
            current_date: Date to calculate for
            window_days: Number of days to look back from the last quarter date
            min_records: Minimum number of records required per symbol
            
        Returns:
            DataFrame with filtered fundamentals data, sorted by symbol and date (descending)
        """
        # Compute window bounds
        last_quarter_date = current_date - timedelta(days=90)
        first_quarter_date = last_quarter_date - timedelta(days=window_days)
        window = self.fundamentals[
            (self.fundamentals['date'] >= first_quarter_date) &
            (self.fundamentals['date'] <= last_quarter_date)
        ]
        
        # Find symbols with enough data
        counts = window.groupby('symbol').size().reset_index(name='count')
        valid_symbols = counts[counts['count'] >= min_records]['symbol'].tolist()
        
        # Filter for valid symbols and sort by date (descending)
        window_valid = window[window['symbol'].isin(valid_symbols)]
        sorted_window = window_valid.sort_values(['symbol', 'date'], ascending=[True, False])
        
        return sorted_window
        
    def get_price_to_fcf(self, price_data: pd.DataFrame, current_date: date) -> pd.DataFrame:
        """
        Calculate free cash flow metrics and price-to-FCF ratio for stocks.
        
        This method computes multiple FCF metrics using the 4 most recent quarters of data:
        - Sum of FCF over 4 quarters (free_cash_flow)
        - Minimum quarterly FCF (min_fcf)
        - Most recent quarter's FCF (last_fcf)
        
        It then calculates the price-to-FCF ratio based on the sum of FCF.
        
        Args:
            price_data: DataFrame with symbol and price columns
            current_date: Date to calculate for
            
        Returns:
            DataFrame with columns:
            - symbol: Stock symbol
            - free_cash_flow: Sum of FCF over 4 quarters
            - min_fcf: Minimum quarterly FCF
            - last_fcf: Most recent quarter's FCF
            - price_to_fcf: Price divided by sum of FCF
        """
        COL = ['symbol', 'free_cash_flow', 'min_fcf', 'last_fcf', 'price_to_fcf']
        # Filter fundamentals data for the required window (4 quarters)
        sorted_window = self._filter_fundamentals(current_date, 400, 4)
        
        # Take the 4 most recent quarters for each symbol
        last4 = sorted_window.groupby('symbol').head(4)
        
        # Calculate sum of FCF over the 4 quarters
        sum_fcf = last4.groupby('symbol')['free_cash_flow_per_share'] \
            .sum().reset_index(name='free_cash_flow')
            
        # Calculate minimum FCF from the 4 quarters
        min_fcf = last4.groupby('symbol')['free_cash_flow_per_share'] \
            .min().reset_index(name='min_fcf')
            
        # Get the most recent FCF value (first row for each symbol since we sorted descending by date)
        last_fcf = last4.groupby('symbol').first().reset_index()[['symbol', 'free_cash_flow_per_share']]
        last_fcf = last_fcf.rename(columns={'free_cash_flow_per_share': 'last_fcf'})
        
        # Merge all FCF metrics together
        fcf_metrics = pd.merge(sum_fcf, min_fcf, on='symbol')
        fcf_metrics = pd.merge(fcf_metrics, last_fcf, on='symbol')
        
        # Merge FCF metrics with price data and calculate price-to-fcf ratio
        df = pd.merge(fcf_metrics, price_data, on='symbol', how='inner')
        df['price_to_fcf'] = df['price'] / df['free_cash_flow']
        
        return df[COL]

    def get_revenue_growth(self, current_date: date) -> pd.DataFrame:
        """
        Calculate revenue growth metrics for stocks.
        
        This method computes multiple revenue growth metrics using the past 8 quarters of data:
        - Median YoY growth over the past 4 quarters (median_yoy)
        - Minimum YoY growth over the past 4 quarters (min_yoy)
        - Most recent quarter's YoY growth (last_yoy)
        
        Args:
            current_date: Date to calculate for
            
        Returns:
            DataFrame with columns:
            - symbol: Stock symbol
            - median_yoy: Median YoY revenue growth over past 4 quarters
            - min_yoy: Minimum YoY revenue growth over past 4 quarters
            - last_yoy: Most recent quarter's YoY revenue growth
        """
        COL = ['symbol', 'median_yoy', 'min_yoy', 'last_yoy']
        # Filter fundamentals data for the required window (8 quarters)
        window_valid = self._filter_fundamentals(current_date, 90 * 8, 8)
        
        # Get the list of valid symbols
        valid_symbols = window_valid['symbol'].unique().tolist()
        
        # Calculate YoY growth for each symbol
        growth_results = []
        
        for symbol in valid_symbols:
            symbol_data = window_valid[window_valid['symbol'] == symbol]
            symbol_data = symbol_data.sort_values('date', ascending=False)
            
            # Need at least 8 quarters of data to calculate 4 YoY growth values
            if len(symbol_data) < 8:
                continue
                
            # Calculate YoY growth for the 4 most recent quarters
            yoy_values = []
            
            for i in range(4):  # Calculate 4 YoY values
                if i + 4 >= len(symbol_data):
                    break
                    
                current_quarter = symbol_data.iloc[i]
                prev_year_quarter = symbol_data.iloc[i + 4]  # Same quarter, previous year
                
                current_revenue = current_quarter['revenue']
                prev_year_revenue = prev_year_quarter['revenue']
                
                if prev_year_revenue and prev_year_revenue != 0:
                    yoy_growth = (current_revenue - prev_year_revenue) / abs(prev_year_revenue)
                    yoy_values.append(yoy_growth)
            
            # Only include symbols with at least 4 YoY growth values
            if len(yoy_values) >= 4:
                growth_results.append({
                    'symbol': symbol,
                    'median_yoy': np.median(yoy_values),
                    'min_yoy': min(yoy_values),
                    'last_yoy': yoy_values[0]  # Most recent quarter's YoY growth
                })
        
        # Convert results to DataFrame
        growth_df = pd.DataFrame(growth_results)
        
        if growth_df.empty:
            return pd.DataFrame(columns=COL)
        
        return growth_df[COL]
        
    def get_profit_margin(self, current_date: date) -> pd.DataFrame:
        """
        Calculate operating profit margin metrics for stocks.
        
        This method retrieves operating profit margins for different time periods:
        - Most recent quarter (opm_3m)
        - 2 quarters ago (opm_6m)
        - 3 quarters ago (opm_9m)
        - 4 quarters ago (opm_12m)
        
        Args:
            current_date: Date to calculate for
            
        Returns:
            DataFrame with columns:
            - symbol: Stock symbol
            - opm_3m: Operating profit margin from most recent quarter
            - opm_6m: Operating profit margin from 2 quarters ago
            - opm_9m: Operating profit margin from 3 quarters ago
            - opm_12m: Operating profit margin from 4 quarters ago
        """
        COL = ['symbol', 'opm_3m', 'opm_6m', 'opm_9m', 'opm_12m']
        # Filter fundamentals data for the required window (4 quarters)
        sorted_window = self._filter_fundamentals(current_date, 400, 4)
        
        # Get the list of valid symbols
        valid_symbols = sorted_window['symbol'].unique().tolist()
        
        # Calculate profit margins for each symbol
        profit_margin_results = []
        
        for symbol in valid_symbols:
            symbol_data = sorted_window[sorted_window['symbol'] == symbol]
            symbol_data = symbol_data.sort_values('date', ascending=False)
            
            # Need at least 4 quarters of data
            if len(symbol_data) < 4:
                continue
                
            # Get operating margins for the 4 most recent quarters
            quarters_data = {}
            
            for i in range(min(4, len(symbol_data))):
                quarter_data = symbol_data.iloc[i]
                quarters_data[f'opm_{(i+1)*3}m'] = quarter_data['operating_margin']
            
            # Only include symbols with all 4 quarters of data
            if len(quarters_data) == 4:
                quarters_data['symbol'] = symbol
                profit_margin_results.append(quarters_data)
        
        # Convert results to DataFrame
        profit_margin_df = pd.DataFrame(profit_margin_results)
        
        if profit_margin_df.empty:
            return pd.DataFrame(columns=COL)
        
        return profit_margin_df[COL]
        
    def get_price_to_ncav(self, price_data: pd.DataFrame, current_date: date) -> pd.DataFrame:
        """
        Calculate price-to-NCAV (Net Current Asset Value) ratio for stocks.
        
        NCAV = total_current_assets - total_debt
        price_to_ncav = market_cap / NCAV = (num_shares * close_price) / (total_current_assets - total_debt)
        
        Args:
            price_data: DataFrame with symbol and price columns
            current_date: Date to calculate for
            
        Returns:
            DataFrame with columns:
            - symbol: Stock symbol
            - ncav: Net Current Asset Value (total_current_assets - total_debt)
            - price_to_ncav: Price-to-NCAV ratio
        """
        COL = ['symbol', 'ncav', 'price_to_ncav']
        # Filter fundamentals data for the required window
        sorted_window = self._filter_fundamentals(current_date, 400, 1)
        
        # Take the most recent quarter for each symbol
        latest = sorted_window.groupby('symbol').first().reset_index()
        
        # Calculate NCAV and price-to-NCAV
        ncav_df = latest[['symbol', 'total_current_assets', 'total_debt']].copy()
        ncav_df['ncav'] = ncav_df['total_current_assets'] - ncav_df['total_debt']
        
        # Merge with price data
        df = pd.merge(ncav_df, price_data, on='symbol', how='inner')
        
        # Calculate price-to-NCAV ratio
        df['price_to_ncav'] = (df['num_shares'] * df['price']) / df['ncav']
        
        # Return only the required columns
        return df[COL]
    
    def get_price_momentum(self, current_date: date) -> pd.DataFrame:
        """
        Calculate price momentum metrics for stocks.
        
        This method computes price momentum metrics by comparing current price (p0) with:
        - Price 3 months ago (p1) -> m3
        - Price 6 months ago (p2) -> m6
        - Price 9 months ago (p3) -> m9
        - Price 12 months ago (p4) -> m12
        
        Args:
            current_date: Date to calculate for
            
        Returns:
            DataFrame with columns:
            - symbol: Stock symbol
            - m3: Price momentum over 3 months (p0/p1 - 1)
            - m6: Price momentum over 6 months (p0/p2 - 1)
            - m9: Price momentum over 9 months (p0/p3 - 1)
            - m12: Price momentum over 12 months (p0/p4 - 1)
        """
        COL = ['symbol', 'm3', 'm6', 'm9', 'm12']
        symbols = self.stocks['symbol'].tolist()
        self.logger.info(f"Calculating price momentum for {len(symbols)} stocks")
        
        # Calculate dates for 3, 6, 9, and 12 months ago
        date_3m_ago = current_date - timedelta(days=90)
        date_6m_ago = current_date - timedelta(days=180)
        date_9m_ago = current_date - timedelta(days=270)
        date_12m_ago = current_date - timedelta(days=365)
        
        # Get price data for current date and 3, 6, 9, 12 months ago using _get_price_before
        # to ensure we get data even if the dates are not trading days
        p0_data = self._get_price_before(symbols, current_date)
        p1_data = self._get_price_before(symbols, date_3m_ago)
        p2_data = self._get_price_before(symbols, date_6m_ago)
        p3_data = self._get_price_before(symbols, date_9m_ago)
        p4_data = self._get_price_before(symbols, date_12m_ago)
        
        # Rename price columns to avoid confusion when merging
        p0_data = p0_data.rename(columns={'price': 'p0', 'actual_date': 'date_p0'}).set_index('symbol')
        p1_data = p1_data.rename(columns={'price': 'p1', 'actual_date': 'date_p1'}).set_index('symbol')
        p2_data = p2_data.rename(columns={'price': 'p2', 'actual_date': 'date_p2'}).set_index('symbol')
        p3_data = p3_data.rename(columns={'price': 'p3', 'actual_date': 'date_p3'}).set_index('symbol')
        p4_data = p4_data.rename(columns={'price': 'p4', 'actual_date': 'date_p4'}).set_index('symbol')
        
        # Merge all price data
        merged_data = p0_data.merge(p1_data, on='symbol', how='inner')
        merged_data = merged_data.merge(p2_data, on='symbol', how='inner')
        merged_data = merged_data.merge(p3_data, on='symbol', how='inner')
        merged_data = merged_data.merge(p4_data, on='symbol', how='inner')
        
        # Calculate price momentum metrics
        merged_data['m3'] = merged_data['p0'] / merged_data['p1'] - 1
        merged_data['m6'] = merged_data['p0'] / merged_data['p2'] - 1
        merged_data['m9'] = merged_data['p0'] / merged_data['p3'] - 1
        merged_data['m12'] = merged_data['p0'] / merged_data['p4'] - 1
        
        # Select only the required columns
        result_df = merged_data.reset_index()[COL]
        
        self.logger.info(f"Calculated price momentum for {len(result_df)} stocks")
        return result_df

    def build_features_for_date(self, date: date, skip_signals: set = set(), use_return_after_days: int = -1) -> pd.DataFrame:
        """
        Build features for a specific date, including price-to-FCF ratios and sector/industry information.
        
        Args:
            date: Date to build features for
        
        Returns:
            DataFrame with features for the given date, including:
            - symbol: Stock symbol
            - sector: Stock sector
            - industry: Stock industry
            - free_cash_flow: Sum of FCF over 4 quarters
            - min_fcf: Minimum quarterly FCF
            - last_fcf: Most recent quarter's FCF
            - price: Current stock price
            - price_to_fcf: Price divided by sum of FCF
            - median_yoy: Median YoY revenue growth over past 4 quarters
            - min_yoy: Minimum YoY revenue growth over past 4 quarters
            - last_yoy: Most recent quarter's YoY revenue growth
            - opm_3m, opm_6m, opm_9m, opm_12m: Operating profit margins for different time periods
            - price_to_ncav: Price to NCAV (Net Current Asset Value) ratio
            - return_date: Date to calculate the return (optional)
            - return_day_price: Price on return_date (optional)
            - date: The date for which features were built
        """
        assert self._is_trading_day(date), f"Date {date} is not a trading day"
        # Get price data for all stocks on current_date
        self.logger.info(f"Fetching price data for {len(self.stocks)} stocks on {date}")
        price_data = self._get_price_data(self.stocks['symbol'].tolist(), date).set_index('symbol')
        self.logger.info(f"Retrieved prices for {len(price_data)} stocks")
        
        if 'price_to_fcf' not in skip_signals:
            # Combine free cash flow and price into price_to_fcf
            self.logger.info("Calculating price-to-FCF ratios")
            fcf_df = self.get_price_to_fcf(price_data, date).set_index('symbol')
        else:
            fcf_df = pd.DataFrame()
        
        # Calculate revenue growth metrics
        if 'revenue_growth' not in skip_signals:
            self.logger.info("Calculating revenue growth metrics")
            growth_df = self.get_revenue_growth(date).set_index('symbol')
        else:
            growth_df = pd.DataFrame()
        
        # Calculate price momentum metrics
        if 'momentum' not in skip_signals:
            self.logger.info("Calculating price momentum metrics")
            momentum_df = self.get_price_momentum(date).set_index('symbol')
        else:
            momentum_df = pd.DataFrame()
        
        # Calculate profit margin metrics
        if 'operation_margin' not in skip_signals:
            self.logger.info("Calculating profit margin metrics")
            profit_margin_df = self.get_profit_margin(date).set_index('symbol')
        else:
            profit_margin_df = pd.DataFrame()
            
        # Calculate price-to-NCAV ratio
        if 'price_to_ncav' not in skip_signals:
            self.logger.info("Calculating price-to-NCAV ratios")
            ncav_df = self.get_price_to_ncav(price_data, date).set_index('symbol')
        else:
            ncav_df = pd.DataFrame()

        if use_return_after_days > 1:
            return_date = date + timedelta(days=use_return_after_days)
            while not self._is_trading_day(return_date) and return_date < date + timedelta(days=use_return_after_days + 10):
                self.logger.warning(f"Return date {return_date} is not a trading day, trying next day")
                return_date += timedelta(days=1)

            assert self._is_trading_day(return_date), f"Can't find trading day for {date} after {use_return_after_days} days"
            
            self.logger.info(f"Calculating return after {use_return_after_days} days: from {date} to {return_date}")
            return_after_days_df = self._get_price_data(self.stocks['symbol'].tolist(), return_date)
            return_after_days_df['return_date'] = return_date.strftime('%Y-%m-%d')
            return_after_days_df.rename(columns={'price': 'return_day_price', 'num_shares': 'return_num_shares', 'market_cap': 'return_market_cap'}, inplace=True)
        else:
            return_after_days_df = pd.DataFrame()
        
        # Merge all feature dataframes
        self.logger.info("Merging feature data")
        df = price_data
        if not fcf_df.empty:
            df = pd.merge(df, fcf_df, on='symbol', how='outer')
        if not growth_df.empty:
            df = pd.merge(df, growth_df, on='symbol', how='outer')
        if not momentum_df.empty:
            df = pd.merge(df, momentum_df, on='symbol', how='outer')
        if not profit_margin_df.empty:
            df = pd.merge(df, profit_margin_df, on='symbol', how='outer')
        if not ncav_df.empty:
            df = pd.merge(df, ncav_df, on='symbol', how='outer')
        if not return_after_days_df.empty:
            df = pd.merge(df, return_after_days_df, on='symbol', how='inner')
        
        # Add sector and industry information
        self.logger.info("Adding sector and industry information")
        df = pd.merge(df, self.stocks[['symbol', 'sector', 'industry']], on='symbol', how='left')
        
        # Add the date column
        df['date'] = date.strftime('%Y-%m-%d')
        
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
        
        self.logger.info(f"Processing {total_days} days with {len(self.stocks)} stocks")
        
        # Create progress bar
        with tqdm(total=total_days, desc="Processing days", disable=not verbose) as pbar:
            while current_date <= self.end_date:
                self.logger.info(f"Processing date: {current_date}")
                
                # Skip non-trading days
                if not self._is_trading_day(current_date):
                    self.logger.info(f"{current_date} is not a trading day, skipping")
                    current_date += timedelta(days=1)
                    pbar.update(1)
                    continue
                
                # Build features for the current date
                df = self.build_features_for_date(current_date)
                # Determine buy/sell operations
                self.logger.info("Determining trading operations")

                ops = self._determine_operations(df, current_date)
                self.logger.info(f"Generated {len(ops)} operations for {current_date}")
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
