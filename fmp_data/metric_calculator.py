import pandas as pd
import numpy as np


def gross_margin_ttm(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the gross margin TTM for a given symbol.
    
    Args:
        data: DataFrame with required data. Columns: date, filing_date, revenue, cost_of_revenue.
        The rows are sorted by date in ascending order.
        
    Returns:
        Series with the calculated gross margin TTM
    """
    # Calculate rolling sums for revenue and cost_of_revenue over 4 quarters
    revenue_ttm = data['revenue'].rolling(window=4).sum()
    cogs_ttm = data['cost_of_revenue'].rolling(window=4).sum()
    
    # Calculate gross margin TTM
    gross_margin = (revenue_ttm - cogs_ttm) / revenue_ttm
    
    # Replace divisions by zero with np.nan
    gross_margin = gross_margin.replace([np.inf, -np.inf], np.nan)
    
    return gross_margin
    
def operating_margin_ttm(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the operating margin TTM for a given symbol.
    
    Args:
        data: DataFrame with required data. Columns: date, filing_date, operating_income, revenue.
        The rows are sorted by date in ascending order.
        
    Returns:
        Series with the calculated operating margin TTM
    """
    # Calculate rolling sums for operating_income and revenue over 4 quarters
    operating_income_ttm = data['operating_income'].rolling(window=4).sum()
    revenue_ttm = data['revenue'].rolling(window=4).sum()
    
    # Calculate operating margin TTM
    operating_margin = operating_income_ttm / revenue_ttm
    
    # Replace divisions by zero with np.nan
    operating_margin = operating_margin.replace([np.inf, -np.inf], np.nan)
    
    return operating_margin


def eps_ttm(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the earnings per share (EPS) TTM for a given symbol.
    
    Args:
        data: DataFrame with required data. Columns: date, filing_date, net_income, weighted_average_shares_outstanding_diluted.
        The rows are sorted by date in ascending order.
        
    Returns:
        Series with the calculated EPS TTM
    """
    # Calculate rolling sum for net_income over 4 quarters
    net_income_ttm = data['net_income'].rolling(window=4).sum()
    
    # Use the most recent shares_outstanding for each period
    # This is the standard approach for TTM EPS calculation
    shares_outstanding = data['weighted_average_shares_outstanding_diluted']
    
    # Calculate EPS TTM
    eps = net_income_ttm / shares_outstanding
    
    # Replace divisions by zero with np.nan
    eps = eps.replace([np.inf, -np.inf], np.nan)
    
    return eps


DERIVED_METRICS = {
    'gross_margin_ttm': {
        'dependencies': ['revenue', 'cost_of_revenue'],
        'function': gross_margin_ttm
    },
    'operating_margin_ttm': {
        'dependencies': ['operating_income', 'revenue'],
        'function': operating_margin_ttm
    },
    'eps_ttm': {
        'dependencies': ['net_income', 'weighted_average_shares_outstanding_diluted'],
        'function': eps_ttm
    },
}