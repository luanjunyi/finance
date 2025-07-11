import pandas as pd
import numpy as np

def ratio_metric(date: pd.DataFrame, numerator: str, denominator: str, window: int) -> pd.Series:
    """
    Calculate the ratio of numerator to denominator over a given window.
    
    Args:
        date: DataFrame with required data. Columns: date, filing_date, numerator, denominator.
        The rows are sorted by date in ascending order.
        
    Returns:
        Series with the calculated ratio
    """
    # Calculate rolling sums for numerator and denominator over the given window
    numerator_ttm = date[numerator].rolling(window=window).sum()
    denominator_ttm = date[denominator].rolling(window=window).sum()
    
    # Calculate ratio
    ratio = numerator_ttm / denominator_ttm
    
    # Replace divisions by zero with np.nan
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    
    return ratio


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
    return ratio_metric(data, 'operating_income', 'revenue', 4)

def sga_margin_ttm(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the SGA margin TTM for a given symbol.
    
    Args:
        data: DataFrame with required data. Columns: date, filing_date, selling_general_and_administrative_expenses, revenue.
        The rows are sorted by date in ascending order.
        
    Returns:
        Series with the calculated SGA margin TTM
    """
    return ratio_metric(data, 'selling_general_and_administrative_expenses', 'revenue', 4)

def rd_margin_ttm(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the R&D margin TTM for a given symbol.
    
    Args:
        data: DataFrame with required data. Columns: date, filing_date, research_and_development_expenses, revenue.
        The rows are sorted by date in ascending order.
        
    Returns:
        Series with the calculated R&D margin TTM
    """
    return ratio_metric(data, 'research_and_development_expenses', 'revenue', 4)

def depreciation_and_amortization_margin_ttm(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the depreciation and amortization margin TTM for a given symbol.
    
    Args:
        data: DataFrame with required data. Columns: date, filing_date, depreciation_and_amortization, revenue.
        The rows are sorted by date in ascending order.
        
    Returns:
        Series with the calculated depreciation and amortization margin TTM
    """
    return ratio_metric(data, 'depreciation_and_amortization', 'revenue', 4)

def interest_payment_to_operating_income_ttm(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the interest payment to operating income TTM for a given symbol.
    
    Args:
        data: DataFrame with required data. Columns: date, filing_date, net_interest_income, operating_income.
        The rows are sorted by date in ascending order.
        
    Returns:
        Series with the calculated interest payment to operating income TTM
    """
    return ratio_metric(data, 'net_interest_income', 'operating_income', 4)


def net_earning_yoy_growth(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the net earning YOY growth for a given symbol.
    
    Args:
        data: DataFrame with required data. Columns: date, filing_date, net_income.
        The rows are sorted by date in ascending order.
        
    Returns:
        Series with the calculated net earning YOY growth
    """
    # Calculate the difference between current quarter and same quarter previous year
    # diff(periods=4) gives the difference between current value and the value 4 quarters ago
    diff = data['net_income'].diff(periods=4)
    
    # Get the values from 4 quarters ago
    prev_year_values = data['net_income'].shift(4)
    
    # Calculate YOY growth: (current - previous) / abs(previous)
    # Use abs(previous) to handle negative values correctly
    growth = diff / prev_year_values.abs()
    
    # Replace divisions by zero with np.nan
    growth = growth.replace([np.inf, -np.inf], np.nan)
    
    return growth

def operating_income_yoy_growth(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the operating income YOY growth for a given symbol.
    
    Args:
        data: DataFrame with required data. Columns: date, filing_date, operating_income.
        The rows are sorted by date in ascending order.
        
    Returns:
        Series with the calculated operating income YOY growth
    """
    # Calculate the difference between current quarter and same quarter previous year
    # diff(periods=4) gives the difference between current value and the value 4 quarters ago
    diff = data['operating_income'].diff(periods=4)
    
    # Get the values from 4 quarters ago
    prev_year_values = data['operating_income'].shift(4)
    
    # Calculate YOY growth: (current - previous) / abs(previous)
    # Use abs(previous) to handle negative values correctly
    growth = diff / prev_year_values.abs()
    
    # Replace divisions by zero with np.nan
    growth = growth.replace([np.inf, -np.inf], np.nan)
    
    return growth

def long_term_debt_to_ttm_operating_income(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the ratio of long-term debt to trailing twelve months (TTM) operating income.
    
    Args:
        data: DataFrame with required data. Columns: date, filing_date, long_term_debt, operating_income.
        The rows are sorted by date in ascending order.
        
    Returns:
        Series with the calculated ratio of long-term debt to TTM operating income
    """
    # Calculate rolling sums for operating income over the given window
    long_term_debt = data['long_term_debt']
    operating_income_ttm = data['operating_income'].rolling(window=4).sum()
    
    # Calculate ratio
    ratio = long_term_debt / operating_income_ttm
    
    # Replace divisions by zero with np.nan
    ratio = ratio.replace([np.inf, -np.inf], np.nan)
    
    return ratio
    
def roe_ttm(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the return on equity (ROE) trailing twelve months (TTM).
    
    Args:
        data: DataFrame with required data. Columns: date, filing_date, net_income, equity.
        The rows are sorted by date in ascending order.
        
    Returns:
        Series with the calculated ROE TTM
    """
    # Calculate rolling sums for net income and equity over the given window
    net_income_ttm = data['net_income'].rolling(window=4).sum()
    equity = data['total_equity']
    
    # Calculate ROE TTM
    roe = net_income_ttm / equity
    
    # Replace divisions by zero with np.nan
    roe = roe.replace([np.inf, -np.inf], np.nan)
    
    return roe

def debt_to_equity(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the debt to equity ratio for a given symbol.
    
    Args:
        data: DataFrame with required data. Columns: date, filing_date, total_liabilities, total_equity.
        The rows are sorted by date in ascending order.
        
    Returns:
        Series with the calculated debt to equity ratio
    """
    # Calculate debt to equity ratio
    debt_to_equity = data['total_liabilities'] / data['total_equity']
    
    # Replace divisions by zero with np.nan
    debt_to_equity = debt_to_equity.replace([np.inf, -np.inf], np.nan)
    
    return debt_to_equity

def eps_ttm(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the earnings per share (EPS) trailing twelve months (TTM).
    
    Args:
        data: DataFrame with required data. Columns: date, filing_date, net_income, weighted_average_shares_outstanding_diluted.
        The rows are sorted by date in ascending order.
        
    Returns:
        Series with the calculated EPS TTM
    """
    # Calculate rolling sums for net income and weighted average shares outstanding over the given window
    net_income_ttm = data['net_income'].rolling(window=4).sum()
    weighted_average_shares_outstanding_diluted = data['weighted_average_shares_outstanding_diluted']
    
    # Calculate EPS TTM
    eps = net_income_ttm / weighted_average_shares_outstanding_diluted
    
    # Replace divisions by zero with np.nan
    eps = eps.replace([np.inf, -np.inf], np.nan)
    
    return eps

def book_value_per_share(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the book value per share for a given symbol.
    
    Args:
        data: DataFrame with required data. Columns: date, filing_date, total_assets, total_liabilities, goodwill_and_intangible_assets, weighted_average_shares_outstanding_diluted.
        The rows are sorted by date in ascending order.
        
    Returns:
        Series with the calculated book value per share
    """
    # Calculate book value per share
    book_value_per_share = (data['total_assets'] - data['total_liabilities'] - data['goodwill_and_intangible_assets']) \
        / data['weighted_average_shares_outstanding_diluted']
    
    # Replace divisions by zero with np.nan
    book_value_per_share = book_value_per_share.replace([np.inf, -np.inf], np.nan)
    
    return book_value_per_share

def capex_to_operating_income_ttm(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the capital expenditure to operating income trailing twelve months (TTM).
    
    Args:
        data: DataFrame with required data. Columns: date, filing_date, capital_expenditure, operating_income.
        The rows are sorted by date in ascending order.
        
    Returns:
        Series with the calculated capital expenditure to operating income TTM
    """
    return ratio_metric(data, 'capital_expenditure', 'operating_income', 4)

def net_income_margin_ttm(data: pd.DataFrame) -> pd.Series:
    """
    Calculate the net income margin trailing twelve months (TTM).
    
    Args:
        data: DataFrame with required data. Columns: date, filing_date, net_income, revenue.
        The rows are sorted by date in ascending order.
        
    Returns:
        Series with the calculated net income margin TTM
    """
    return ratio_metric(data, 'net_income', 'revenue', 4)

DERIVED_METRICS = {
    'eps_ttm': {
        'dependencies': ['net_income', 'weighted_average_shares_outstanding_diluted'],
        'function': eps_ttm
    },
    'gross_margin_ttm': {
        'dependencies': ['revenue', 'cost_of_revenue'],
        'function': gross_margin_ttm
    },
    'operating_margin_ttm': {
        'dependencies': ['operating_income', 'revenue'],
        'function': operating_margin_ttm
    },
    'sga_margin_ttm': {
        'dependencies': ['selling_general_and_administrative_expenses', 'revenue'],
        'function': sga_margin_ttm
    },
    'rd_margin_ttm': {
        'dependencies': ['research_and_development_expenses', 'revenue'],
        'function': rd_margin_ttm
    },
    'depreciation_and_amortization_margin_ttm': {
        'dependencies': ['depreciation_and_amortization', 'revenue'],
        'function': depreciation_and_amortization_margin_ttm
    },
    'interest_payment_to_operating_income_ttm': {
        'dependencies': ['net_interest_income', 'operating_income'],
        'function': interest_payment_to_operating_income_ttm
    },
    'net_income_margin_ttm': {
        'dependencies': ['net_income', 'revenue'],
        'function': net_income_margin_ttm
    },
    'net_earning_yoy_growth': {
        'dependencies': ['net_income'],
        'function': net_earning_yoy_growth
    },
    'operating_income_yoy_growth': {
        'dependencies': ['operating_income'],
        'function': operating_income_yoy_growth
    },
    'long_term_debt_to_ttm_operating_income': {
        'dependencies': ['long_term_debt', 'operating_income'],
        'function': long_term_debt_to_ttm_operating_income
    },
    'roe_ttm': {
        'dependencies': ['net_income', 'total_equity'],
        'function': roe_ttm
    },
    'debt_to_equity': {
        'dependencies': ['total_liabilities', 'total_equity'],
        'function': debt_to_equity
    },
    'book_value_per_share': {
        'dependencies': ['total_assets', 'total_liabilities', 'goodwill_and_intangible_assets', 'weighted_average_shares_outstanding_diluted'],
        'function': book_value_per_share
    },
    'capex_to_operating_income_ttm': {
        'dependencies': ['capital_expenditure', 'operating_income'],
        'function': capex_to_operating_income_ttm
    }
}