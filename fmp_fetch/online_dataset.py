import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Union, Set
from tqdm import tqdm
from utils.logging_config import setup_logging as setup_global_logging
from .fmp_api import FMPAPI
from functools import cache

INCOME_STATEMENT = "income_statement"
CASHFLOW_STATEMENT = "cashflow_statement"
BALANCE_SHEET = "balance_sheet"

def filter_us_stock(sym):
    return sym['type'] == 'stock' and sym['exchangeShortName'] in ['NYSE', 'NASDAQ', 'AMEX']

class Dataset:
    """
    A class for fetching financial data for multiple symbols using the FMP API.
    This class provides similar functionality to fmp_data.Dataset but uses online API calls
    instead of a database.
    """
    def __init__(self, symbols: Union[str, List[str]], metrics: List[str], 
                 start_date: str, end_date: str):
        """Initialize Dataset object.
        
        Args:
            symbols (Union[str, List[str]]): Stock symbol or list of stock symbols
            metrics (List[str]): List of metrics to fetch
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
        
        Attributes:
            symbols (List[str]): List of stock symbols
            metrics (List[str]): List of metrics to fetch
            start_date (str): Start date
            end_date (str): End date
            api (FMPAPI): FMP API client
            data (pd.DataFrame): The dataset as a pandas DataFrame
        """
        self.symbols = [symbols] if isinstance(symbols, str) else symbols
        self.metrics = metrics
        self.start_date = start_date
        self.end_date = end_date
        self.api = FMPAPI()
        self.financial_statements = {}
        self._data = None
        
        # Setup logging
        self.setup_logging()

    @classmethod
    @cache
    def us_stock_symbols(cls):
        symbols = FMPAPI().get_all_symbols()
        return [s['symbol'] for s in symbols if filter_us_stock(s)]

    @classmethod
    @cache
    def us_active_stocks(cls):
        symbols = FMPAPI().get_all_tradable_symbols()
        return [s['symbol'] for s in symbols if filter_us_stock(s)]
    
    @property
    def data(self) -> pd.DataFrame:
        """Lazily build and return the dataset.
        
        Returns:
            pd.DataFrame: The dataset with all requested metrics
        """
        if self._data is None:
            self._data = self.build()
        return self._data
    
    def setup_logging(self):
        """Configure logging for the dataset"""
        setup_global_logging()
        self.logger = logging.getLogger(__name__)
    
    def _categorize_metrics(self) -> Dict[str, List[str]]:
        """Categorize metrics by data source and handle derived metrics dependencies.
        
        Returns:
            Dict[str, List[str]]: Dictionary mapping data sources to lists of metrics
        """
        income_statement_metrics = {
            'revenue', 'costOfRevenue', 'grossProfit', 'researchAndDevelopmentExpenses',
            'generalAndAdministrativeExpenses', 'sellingAndMarketingExpenses',
            'sellingGeneralAndAdministrativeExpenses', 'otherExpenses', 'operatingExpenses',
            'costAndExpenses', 'netInterestIncome', 'interestIncome', 'interestExpense',
            'depreciationAndAmortization', 'ebitda', 'ebit', 'nonOperatingIncomeExcludingInterest',
            'operatingIncome', 'totalOtherIncomeExpensesNet', 'incomeBeforeTax', 'incomeTaxExpense',
            'netIncomeFromContinuingOperations', 'netIncomeFromDiscontinuedOperations',
            'otherAdjustmentsToNetIncome', 'netIncome', 'netIncomeDeductions', 'bottomLineNetIncome',
            'eps', 'epsDiluted', 'weightedAverageShsOut', 'weightedAverageShsOutDil'
        }
        
        cashflow_statement_metrics = {
            'netIncome', 'depreciationAndAmortization', 'deferredIncomeTax', 'stockBasedCompensation',
            'changeInWorkingCapital', 'accountsReceivables', 'inventory', 'accountsPayables',
            'otherWorkingCapital', 'otherNonCashItems', 'netCashProvidedByOperatingActivities',
            'investmentsInPropertyPlantAndEquipment', 'acquisitionsNet', 'purchasesOfInvestments',
            'salesMaturitiesOfInvestments', 'otherInvestingActivities', 'netCashProvidedByInvestingActivities',
            'netDebtIssuance', 'longTermNetDebtIssuance', 'shortTermNetDebtIssuance',
            'netStockIssuance', 'netCommonStockIssuance', 'commonStockIssuance', 'commonStockRepurchased',
            'netPreferredStockIssuance', 'netDividendsPaid', 'commonDividendsPaid', 'preferredDividendsPaid',
            'otherFinancingActivities', 'netCashProvidedByFinancingActivities',
            'effectOfForexChangesOnCash', 'netChangeInCash', 'cashAtEndOfPeriod', 'cashAtBeginningOfPeriod',
            'operatingCashFlow', 'capitalExpenditure', 'freeCashFlow', 'incomeTaxesPaid', 'interestPaid'
        }
        
        balance_sheet_metrics = {
            'cashAndCashEquivalents', 'shortTermInvestments', 'cashAndShortTermInvestments',
            'netReceivables', 'accountsReceivables', 'otherReceivables', 'inventory', 'prepaids',
            'otherCurrentAssets', 'totalCurrentAssets', 'propertyPlantEquipmentNet', 'goodwill',
            'intangibleAssets', 'goodwillAndIntangibleAssets', 'longTermInvestments', 'taxAssets',
            'otherNonCurrentAssets', 'totalNonCurrentAssets', 'otherAssets', 'totalAssets',
            'totalPayables', 'accountPayables', 'otherPayables', 'accruedExpenses', 'shortTermDebt',
            'capitalLeaseObligationsCurrent', 'taxPayables', 'deferredRevenue', 'otherCurrentLiabilities',
            'totalCurrentLiabilities', 'longTermDebt', 'capitalLeaseObligationsNonCurrent',
            'deferredRevenueNonCurrent', 'deferredTaxLiabilitiesNonCurrent', 'otherNonCurrentLiabilities',
            'totalNonCurrentLiabilities', 'otherLiabilities', 'capitalLeaseObligations', 'totalLiabilities',
            'treasuryStock', 'preferredStock', 'commonStock', 'retainedEarnings', 'additionalPaidInCapital',
            'accumulatedOtherComprehensiveIncomeLoss', 'otherTotalStockholdersEquity', 'totalStockholdersEquity',
            'totalEquity', 'minorityInterest', 'totalLiabilitiesAndTotalEquity', 'totalInvestments',
            'totalDebt', 'netDebt'
        }
        
        price_metrics = {'close_price'}
        
        derived_metrics = {'pe', 'price_to_fcf', 'price_to_owner_earning'}
        
        # Categorize requested metrics
        categorized_metrics = {
            INCOME_STATEMENT: set(),
            CASHFLOW_STATEMENT: set(),
            BALANCE_SHEET: set(),
            'price': set(),
            'derived': set()
        }
        
        # First pass: identify direct metrics and derived metrics
        for metric in self.metrics:
            if metric in income_statement_metrics:
                categorized_metrics[INCOME_STATEMENT].add(metric)
            elif metric in cashflow_statement_metrics:
                categorized_metrics[CASHFLOW_STATEMENT].add(metric)
            elif metric in balance_sheet_metrics:
                categorized_metrics[BALANCE_SHEET].add(metric)
            elif metric in price_metrics:
                categorized_metrics['price'].add(metric)
            elif metric in derived_metrics:
                categorized_metrics['derived'].add(metric)
            else:
                raise ValueError(f"Metric '{metric}' not recognized")
        
        # Add dependencies for derived metrics with simple hard-coded logic
        if 'pe' in self.metrics:
            # PE ratio needs price and EPS data
            categorized_metrics['price'].add('close_price')
            categorized_metrics[INCOME_STATEMENT].add('netIncome')
            categorized_metrics[INCOME_STATEMENT].add('weightedAverageShsOutDil')
        
        if 'price_to_fcf' in self.metrics:
            # Price to FCF needs price, free cash flow data, and weighted average shares from income statement
            categorized_metrics['price'].add('close_price')
            categorized_metrics[CASHFLOW_STATEMENT].add('freeCashFlow')
            categorized_metrics[INCOME_STATEMENT].add('weightedAverageShsOutDil')
        
        if 'price_to_owner_earning' in self.metrics:
            # Price to owner's earning needs price, net income, depreciation & amortization, capital expenditure, and weighted average shares
            categorized_metrics['price'].add('close_price')
            categorized_metrics[INCOME_STATEMENT].add('netIncome')
            categorized_metrics[CASHFLOW_STATEMENT].add('depreciationAndAmortization')
            categorized_metrics[CASHFLOW_STATEMENT].add('capitalExpenditure')
            categorized_metrics[INCOME_STATEMENT].add('weightedAverageShsOutDil')
        
        return categorized_metrics
    
    def _fetch_financial_statements(self, statement_type: str, required_metrics: Set[str]) -> pd.DataFrame:
        """Fetch financial statements for all symbols.
        
        Args:
            statement_type (str): Type of statement (INCOME_STATEMENT, CASHFLOW_STATEMENT, or BALANCE_SHEET)
            required_metrics (List[str]): List of metrics required from this statement
            
        Returns:
            pd.DataFrame: DataFrame with financial statement data
        """
        self.logger.info(f"Fetching {statement_type} for {len(self.symbols)} symbols")
        assert len(required_metrics) > 0
        all_data = []
        for symbol in tqdm(self.symbols):
            # Call the appropriate API method based on statement type
            if statement_type == INCOME_STATEMENT:
                statements = self.api.get_income_statement(symbol, period='quarter', limit=120)
            elif statement_type == CASHFLOW_STATEMENT:
                statements = self.api.get_cashflow_statement(symbol, period='quarter', limit=120)
            elif statement_type == BALANCE_SHEET:
                statements = self.api.get_balance_sheet(symbol, period='quarter', limit=120)
            else:
                raise ValueError(f"Unknown statement type: {statement_type}")
            
            # Filter statements by date
            end_date_obj = datetime.strptime(self.end_date, '%Y-%m-%d').date()
            
            # Convert each statement to a row in the DataFrame
            for statement in statements:
                filing_date = datetime.strptime(statement['filingDate'], '%Y-%m-%d').date()
                
                # Only include statements filed before or during the date range
                if filing_date <= end_date_obj:
                    # Extract only the required metrics
                    row = {
                        'symbol': symbol,
                        'date': statement['date'],
                        'filing_date': statement['filingDate']
                    }
                    
                    # Add required metrics to the row
                    for metric in required_metrics:
                        if metric in statement:
                            row[metric] = statement[metric]
                    
                    all_data.append(row)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        df['filing_date'] = pd.to_datetime(df['filing_date'])
        
        return df.set_index('symbol')
    
    def _fetch_price_data(self) -> pd.DataFrame:
        """Fetch price data for all symbols.
        
        Returns:
            pd.DataFrame: DataFrame with price data. Missing dates are filled with
            the previous day's close price, or np.nan if no previous price is available.
        """
        import numpy as np
        self.logger.info(f"Fetching price data for {len(self.symbols)} symbols from {self.start_date} to {self.end_date}")
        
        # Create a date range for the specified period
        date_range = pd.date_range(start=self.start_date, end=self.end_date)
        
        all_data = []
        for symbol in tqdm(self.symbols):
            prices = self.api.get_prices(symbol, self.start_date, self.end_date)
            
            # Create a set of dates that are already in the API response
            existing_dates = set()
            price_by_date = {}
            
            for price in prices:
                date_str = price['date']
                existing_dates.add(date_str)
                price_by_date[date_str] = price['adjClose']
                all_data.append({
                    'symbol': symbol,
                    'date': date_str,
                    'close_price': price['adjClose']
                })
            
            # Add entries for missing dates with previous day's price or np.nan
            previous_price = None
            for date in date_range:
                date_str = date.strftime('%Y-%m-%d')
                if date_str not in existing_dates:
                    # If we have a previous price, use it; otherwise use np.nan
                    fill_price = previous_price if previous_price is not None else np.nan
                    all_data.append({
                        'symbol': symbol,
                        'date': date_str,
                        'close_price': fill_price
                    })
                else:
                    # Update previous_price for the next iteration
                    previous_price = price_by_date[date_str]
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values(['symbol', 'date'], inplace=True)
        
        return df.set_index('symbol')
    
    def _compute_pe(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Compute TTM PE ratio for each date in the price DataFrame.
        
        Args:
            price_df (pd.DataFrame): DataFrame with price data
            
        Returns:
            pd.DataFrame: DataFrame with PE ratio data
        """
        self.logger.info("Computing PE ratios")
        income_df = self.financial_statements[INCOME_STATEMENT]
        result_data = []
        
        # For each symbol and date in the price DataFrame
        for symbol in tqdm(self.symbols):
            symbol_prices = price_df.xs(symbol)
            
            for _, price_row in symbol_prices.iterrows():
                date = price_row['date']
                close_price = price_row['close_price']
                
                # Get income statements with filing dates before this date
                valid_statements = income_df.xs(symbol)
                valid_statements = valid_statements[
                    (valid_statements['filing_date'] < date) &
                    (valid_statements['date'] > date - timedelta(days=365 + 31 * 3))
                ].sort_values('filing_date', ascending=False)
                
                if len(valid_statements) < 4:
                    self.logger.warning(f"Not enough income statements for {symbol} on {date.date()}. Found: {valid_statements['filing_date'].tolist() if not valid_statements.empty else []}")
                    continue
                
                # Take the 4 most recent quarters
                recent_quarters = valid_statements.head(4)
                
                # Sum the income for the last 4 quarters
                sum_income = recent_quarters['netIncome'].sum()
                # Make sure we have valid share data
                assert 'weightedAverageShsOutDil' in recent_quarters.columns, f"Missing weightedAverageShsOutDil for {symbol}"
                num_shares = recent_quarters['weightedAverageShsOutDil'].iloc[0]
                assert num_shares > 0, f"Invalid weightedAverageShsOutDil (zero or negative) for {symbol}"
                
                # Calculate EPS and PE ratio
                eps = sum_income / num_shares

                pe_ratio = close_price / eps
                
                result_data.append({
                    'symbol': symbol,
                    'date': date,
                    'pe': pe_ratio
                })
        
        return pd.DataFrame(result_data)
    
    def _compute_price_to_fcf(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Compute price to free cash flow ratio for each date in the price DataFrame.
        
        Args:
            price_df (pd.DataFrame): DataFrame with price data
            
        Returns:
            pd.DataFrame: DataFrame with price to FCF ratio data
        """
        self.logger.info("Computing price to FCF ratios")
        
        # We need income statement data for weightedAverageShsOutDil
        income_df = self.financial_statements[INCOME_STATEMENT]
        cashflow_df = self.financial_statements[CASHFLOW_STATEMENT]
        
        result_data = []
        
        # For each symbol and date in the price DataFrame
        for symbol in tqdm(self.symbols):
            symbol_prices = price_df.xs(symbol)
            
            for _, price_row in symbol_prices.iterrows():
                date = price_row['date']
                close_price = price_row['close_price']
                
                # Get cash flow statements with filing dates before this date
                valid_cashflow_statements = cashflow_df.xs(symbol)
                valid_cashflow_statements = valid_cashflow_statements[
                    (valid_cashflow_statements['filing_date'] < date) &
                    (valid_cashflow_statements['date'] > date - timedelta(days=365 + 31 * 3))
                ].sort_values('filing_date', ascending=False)
                
                # Get income statements with filing dates before this date
                valid_income_statements = income_df.xs(symbol)  
                valid_income_statements = valid_income_statements[
                    (valid_income_statements['filing_date'] < date) &
                    (valid_income_statements['date'] > date - timedelta(days=365 + 31 * 3))
                ].sort_values('filing_date', ascending=False)
                
                if len(valid_cashflow_statements) < 4 or len(valid_income_statements) < 1:
                    self.logger.warning(f"Not enough statements for {symbol} on {date.date()}" + \
                        f" cashflow: {valid_cashflow_statements['filing_date'].tolist()}, income: {valid_income_statements['filing_date'].tolist()}")
                    continue
                
                # Take the 4 most recent quarters for cash flow
                last_4_quarters_cashflow = valid_cashflow_statements.head(4)
                # Take the most recent income statement for shares
                latest_income = valid_income_statements.iloc[0]
                
                # Ensure required columns exist
                assert 'freeCashFlow' in last_4_quarters_cashflow.columns, f"Missing freeCashFlow column for {symbol}"
                assert 'weightedAverageShsOutDil' in latest_income, f"Missing weightedAverageShsOutDil for {symbol}"
                
                # Sum FCF and divide by shares outstanding
                fcf_sum = last_4_quarters_cashflow['freeCashFlow'].sum()
                shares = latest_income['weightedAverageShsOutDil']
                
                # Validate data before calculation
                assert shares > 0, f"Invalid weightedAverageShsOutDil (zero or negative) for {symbol}: {shares}"
                fcf_per_share = fcf_sum / shares
                
                # Calculate price to FCF ratio
                price_to_fcf = close_price / fcf_per_share
                
                result_data.append({
                    'symbol': symbol,
                    'date': date,
                    'price_to_fcf': price_to_fcf
                })
        
        return pd.DataFrame(result_data)
    
    def _get_latest_values_for_dates(self, df: pd.DataFrame, date_range: pd.DatetimeIndex) -> pd.DataFrame:
        """Get the latest values for each date in the date range.
        
        For each date in the date range, find the latest filing before that date.
        
        Args:
            df (pd.DataFrame): DataFrame with financial data
            date_range (pd.DatetimeIndex): Range of dates to get values for
            
        Returns:
            pd.DataFrame: DataFrame with latest values for each date. Indexed by <symbol, date> so that it's easy to
            join with other DataFrames
        """
        if df.empty:
            return pd.DataFrame()
        
        result_data = []
        
        for symbol in pd.unique(df.index.values):
            symbol_data = df.xs(symbol)
            
            for date in date_range:
                # Find the latest filing before this date
                valid_filings = symbol_data[(symbol_data['filing_date'] < date) & (symbol_data['date'] > date - timedelta(days=365 + 31 * 3))]
                
                if valid_filings.empty:
                    self.logger.warning(f"No financial statements for {symbol} on {date.date()}" + \
                        f", found filings: {symbol_data['filing_date'].tolist()}")
                    continue
                
                latest_filing = valid_filings.sort_values('filing_date', ascending=False).iloc[0]
                
                # Create a row for this date with the latest values
                row = {'symbol': symbol, 'date': date}
                
                # Add all metrics from the latest filing
                for col in latest_filing.index:
                    if col not in ['symbol', 'date', 'filing_date']:
                        row[col] = latest_filing[col]
                
                result_data.append(row)
        
        return pd.DataFrame(result_data).set_index(['symbol', 'date'])
    
    def build(self) -> pd.DataFrame:
        """Build and return the dataset as a pandas DataFrame.
        
        Returns:
            pd.DataFrame: The dataset with all requested metrics
        """
        # Categorize metrics by data source (including dependencies for derived metrics)
        categorized_metrics = self._categorize_metrics()
        
        # Create a date range for all dates between start_date and end_date
        date_range = pd.date_range(start=self.start_date, end=self.end_date)
        
        # Fetch data from each source
        dfs = []
        price_df = pd.DataFrame()
        
        # Fetch price data if needed
        if categorized_metrics['price']:
            price_df = self._fetch_price_data()
            dfs.append(price_df)

        for statement_type in [INCOME_STATEMENT, CASHFLOW_STATEMENT, BALANCE_SHEET]:
            if categorized_metrics[statement_type]:
                self.financial_statements[statement_type] = self._fetch_financial_statements(statement_type, categorized_metrics[statement_type])
                # Get the latest values for each date in the date range
                last_one = self._get_latest_values_for_dates(self.financial_statements[statement_type], date_range)
                dfs.append(last_one)
            
        
        # Compute derived metrics if needed
        if 'pe' in self.metrics:
            pe_df = self._compute_pe(price_df)
            dfs.append(pe_df)
        
        if 'price_to_fcf' in self.metrics:
            price_to_fcf_df = self._compute_price_to_fcf(price_df)
            dfs.append(price_to_fcf_df)
            
        if 'price_to_owner_earning' in self.metrics:
            price_to_oe_df = self._compute_price_to_owner_earning(price_df)
            dfs.append(price_to_oe_df)
        
        result = dfs[0]
        for df in dfs[1:]:
            result = pd.merge(result, df, on=['symbol', 'date'], how='outer')
        
        result = result.reset_index()
        all_columns = set(result.columns)
        keep_columns = {'symbol', 'date'}.union(set(self.metrics))
        drop_columns = all_columns - keep_columns
        if drop_columns:
            result = result.drop(columns=list(drop_columns))

        # Sort by symbol and date
        result = result.sort_values(['symbol', 'date']).set_index(['symbol', 'date'])
        
        return result
        
    def _compute_price_to_owner_earning(self, price_df: pd.DataFrame) -> pd.DataFrame:
        """Compute price to TTM owner's earning ratio for each date in the price DataFrame.
        
        Owner's earning is defined as:
        netIncome + depreciationAndAmortization - capitalExpenditure
        
        Args:
            price_df (pd.DataFrame): DataFrame with price data
            
        Returns:
            pd.DataFrame: DataFrame with price to TTM owner's earning ratio data
        """
        self.logger.info("Computing price to TTM owner's earning ratios")
        
        # We need income statement data for netIncome and weightedAverageShsOutDil
        # and cash flow statement for depreciationAndAmortization and capitalExpenditure
        income_df = self.financial_statements[INCOME_STATEMENT]
        cashflow_df = self.financial_statements[CASHFLOW_STATEMENT]
        
        result_data = []
        
        # For each symbol and date in the price DataFrame
        for symbol in tqdm(self.symbols):
            symbol_prices = price_df.xs(symbol)
            
            for _, price_row in symbol_prices.iterrows():
                date = price_row['date']
                close_price = price_row['close_price']
                
                # Get cash flow statements with filing dates before this date
                valid_cashflow_statements = cashflow_df.xs(symbol)
                valid_cashflow_statements = valid_cashflow_statements[
                    (valid_cashflow_statements['filing_date'] < date) &
                    (valid_cashflow_statements['date'] > date - timedelta(days=365 + 31 * 3))
                ].sort_values('filing_date', ascending=False)
                
                # Get income statements with filing dates before this date
                valid_income_statements = income_df.xs(symbol)  
                valid_income_statements = valid_income_statements[
                    (valid_income_statements['filing_date'] < date) &
                    (valid_income_statements['date'] > date - timedelta(days=365 + 31 * 3))
                ].sort_values('filing_date', ascending=False)
                
                if len(valid_cashflow_statements) < 4 or len(valid_income_statements) < 4:
                    self.logger.warning(f"Not enough statements for {symbol} on {date.date()}" + \
                        f" cashflow: {valid_cashflow_statements['filing_date'].tolist()}, income: {valid_income_statements['filing_date'].tolist()}")
                    continue
                
                # Take the 4 most recent quarters for cash flow and income
                last_4_quarters_cashflow = valid_cashflow_statements.head(4)
                last_4_quarters_income = valid_income_statements.head(4)
                # Take the most recent income statement for shares
                latest_income = valid_income_statements.iloc[0]

                
                # Sum net income, depreciation & amortization, and capital expenditure over the last 4 quarters
                net_income_sum = last_4_quarters_income['netIncome'].sum()
                depreciation_sum = last_4_quarters_cashflow['depreciationAndAmortization'].sum()
                capex_sum = abs(last_4_quarters_cashflow['capitalExpenditure'].sum())  # Convert to positive
                
                # Calculate owner's earnings
                owners_earnings = net_income_sum + depreciation_sum - capex_sum
                
                # Get shares outstanding
                shares = latest_income['weightedAverageShsOutDil']
                
                # Calculate owner's earnings per share and price to owner's earnings ratio
                oe_per_share = owners_earnings / shares
                price_to_oe = close_price / oe_per_share
                
                result_data.append({
                    'symbol': symbol,
                    'date': date,
                    'price_to_owner_earning': price_to_oe
                })
        
        return pd.DataFrame(result_data)
