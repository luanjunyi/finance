"""FMP Data Cleaner

This module detects and cleans data quality issues in the FMP database.
Currently focuses on detecting unaccounted stock splits in daily_price table.
"""

import argparse
import logging
import sqlite3
import pandas as pd
from datetime import datetime
from typing import List, Set
from tqdm import tqdm
from utils.config import FMP_DB_PATH

class FMPDataCleaner:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.bad_data_file = "bad_data.txt"
        
    def detect_unaccounted_splits(self, multiple_threshold: float) -> dict[str, str]:
        """
        Detect symbols with potential unaccounted stock splits by looking for large volume changes.
        
        Args:
            multiple_threshold: Minimum percentage change in volume to flag as potential split
            
        Returns:
            Dict mapping symbols to their detailed reason for being flagged
        """
        # Clear the bad_data.txt file at the start of each run
        with open(self.bad_data_file, 'w') as f:
            f.write("")  # Clear the file
        
        suspicious_symbols = {}
        
        with sqlite3.connect(f'file:{self.db_path}?mode=ro', uri=True) as conn:
            # Get all unique symbols first with their stock information in one efficient query
            cursor = conn.cursor()
            cursor.execute("""
                SELECT DISTINCT dp.symbol, ss.exchange_short_name, ss.sector, ss.industry
                FROM daily_price dp
                LEFT JOIN stock_symbol ss ON dp.symbol = ss.symbol
            """)
            self.symbols_info = {row[0]: {
                'exchange_short_name': row[1] or 'N/A',
                'sector': row[2] or 'N/A', 
                'industry': row[3] or 'N/A'
            } for row in cursor.fetchall()}
            
            symbols = list(self.symbols_info.keys())
            self.logger.info(f"Checking {len(symbols)} symbols for unaccounted splits")
            
            # Process each symbol individually to avoid memory issues
            for symbol in tqdm(symbols, desc="Processing symbols"):
                
                # Get daily price data for this symbol only
                query = """
                SELECT date, volume, close
                FROM daily_price
                WHERE symbol = ? AND volume > 0
                ORDER BY date
                """
                
                df = pd.read_sql_query(query, conn, params=(symbol,))
                
                if len(df) < 2:  # Need at least 2 days to compare
                    continue
                    
                # Calculate volume and close price ratios
                df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
                df['close_ratio'] = df['close'] / df['close'].shift(1)
                
                # Detect potential splits/reverse splits:
                # One ratio should be abnormally high (> 1 + threshold) 
                # AND the other should be abnormally low (< 1 - threshold)
                # This covers both normal splits and reverse splits
                
                suspicious_changes = df[
                    ((df['volume_ratio'] >= multiple_threshold) & (1 / df['close_ratio'] >= multiple_threshold)) |
                    ((df['volume_ratio'] >= multiple_threshold) & (1 / df['close_ratio'] >= multiple_threshold))
                ]
                
                if not suspicious_changes.empty:
                    # Get the first suspicious change
                    first_change = suspicious_changes.iloc[0]
                    
                    # Calculate previous values for logging
                    prev_volume = first_change['volume'] / first_change['volume_ratio']
                    prev_close = first_change['close'] / first_change['close_ratio']
                    
                    # Create detailed reason for the suspicious symbol
                    bad_data_entry = (
                        f"on {first_change['date']}: "
                        f"volume {prev_volume:,.0f} -> {first_change['volume']:,.0f} "
                        f"(ratio: {first_change['volume_ratio']:.2f}), "
                        f"close {prev_close:.2f} -> {first_change['close']:.2f} "
                        f"(ratio: {first_change['close_ratio']:.2f})"
                    )
                    
                    # Store symbol and its detailed reason
                    suspicious_symbols[symbol] = bad_data_entry
                            
        return suspicious_symbols
    
    def write_suspicious_symbols(self, suspicious_symbols: dict[str, str]) -> int:
        """
        Write suspicious symbols to the suspicious_symbols table.
        
        Args:
            suspicious_symbols: Dict mapping symbols to their detailed reasons
            
        Returns:
            Number of records inserted
        """
        if not suspicious_symbols:
            return 0
            
        inserted_count = 0
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            for symbol, reason in suspicious_symbols.items():
                # Insert or replace to handle duplicates
                cursor.execute(
                    """INSERT OR REPLACE INTO suspicious_symbols (symbol, reason, bad_type, date_added) 
                       VALUES (?, ?, ?, DATE('now'))""",
                    (symbol, reason, "PRICE_HOLE")
                )
                inserted_count += cursor.rowcount
                
            conn.commit()
            
        return inserted_count
    
    def clean_unaccounted_splits(self, dry_run: bool, multiple_threshold: float) -> None:
        """
        Main method to detect and optionally clean unaccounted splits.
        
        Args:
            dry_run: If True, only report issues without writing to database
            multiple_threshold: Minimum change in volume/price to flag as potential split
        """
        self.logger.info(f"Starting unaccounted split detection (dry_run={dry_run})")
        
        suspicious_symbols = self.detect_unaccounted_splits(multiple_threshold)
        
        if not suspicious_symbols:
            self.logger.info("No suspicious symbols found")
            return
            
        self.logger.info(f"Found {len(suspicious_symbols)} suspicious symbols:")
            
        if not dry_run:
            inserted_count = self.write_suspicious_symbols(suspicious_symbols)
            self.logger.info(f"Inserted {inserted_count} suspicious symbols into database")
        else:
            with open(self.bad_data_file, 'w') as f:
                for symbol, reason in suspicious_symbols.items():
                    f.write(f"{symbol} | {self.symbols_info[symbol]['exchange_short_name']} | {self.symbols_info[symbol]['sector']} | {self.symbols_info[symbol]['industry']}: {reason}\n")
            self.logger.info("Dry run mode - no records were written to database")

def main():
    parser = argparse.ArgumentParser(description='Clean FMP data quality issues')
    parser.add_argument('--no-dry-run', action='store_true',
                       help='Actually write data (default is dry-run mode)')
    parser.add_argument('--multiple-threshold', type=float, default=0.8,
                       help='Minimum change in multiple to flag as suspicious (default: 0.8)')
    parser.add_argument('--db-path', type=str, default=FMP_DB_PATH,
                       help='Path to the FMP database')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create cleaner and run
    cleaner = FMPDataCleaner(args.db_path)
    cleaner.clean_unaccounted_splits(
        dry_run=not args.no_dry_run,  # Default is dry-run unless --no-dry-run is specified
        multiple_threshold=args.multiple_threshold
    )

if __name__ == '__main__':
    main()