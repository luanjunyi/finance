"""
Base classes and utilities for financial metrics calculation.

This module provides the foundation for calculating financial metrics from FMP data.
It defines a hierarchy of metric calculators for different types of metrics:
- Base MetricCalculator class for common functionality
- QuarterlyMetricCalculator for metrics based on quarterly financial statements
- DailyMetricCalculator for metrics that combine price data with quarterly metrics
"""

import logging
from abc import ABC
from typing import Dict, List, Set, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

# Global registry of metric calculators
METRIC_REGISTRY = {}


class MetricCalculator(ABC):
    """Base class for all metric calculators."""
    
    def __init__(self, name: str, dependencies: Dict[str, Set[str]]):
        """
        Initialize a metric calculator.
        
        Args:
            name: Name of the metric
            dependencies: Dictionary mapping data source names to sets of required columns
        """
        self.name = name
        self.dependencies = dependencies
        self.logger = logging.getLogger(f"{__name__}.{name}")
    
    def calculate(self, data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate the metric using the provided data sources.
        
        Args:
            data_sources: Dictionary mapping data source names to DataFrames
            
        Returns:
            DataFrame with calculated metric values
        """
        self._validate_data_sources(data_sources)
        return self._perform_calculation(data_sources)
    
    def _perform_calculation(self, data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Perform the actual calculation. Must be implemented by subclasses.
        
        Args:
            data_sources: Dictionary mapping data source names to DataFrames
            
        Returns:
            DataFrame with calculated metric values
        """
        raise NotImplementedError("Subclasses must implement _perform_calculation()")
    
    def _validate_data_sources(self, data_sources: Dict[str, pd.DataFrame]) -> None:
        """Validate that all required data sources and columns are present."""
        missing_sources = []
        missing_columns = {}
        
        for source, required_columns in self.dependencies.items():
            if source not in data_sources:
                missing_sources.append(source)
                continue
                
            df = data_sources[source]
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Data source '{source}' must be a DataFrame")
                
            df_columns = set(df.columns) | set(df.index.names) if df.index.names[0] is not None else set(df.columns)
            missing = required_columns - df_columns
            if missing:
                missing_columns[source] = missing
        
        if missing_sources or missing_columns:
            error_msg = f"Missing data for {self.name} calculation:\n"
            if missing_sources:
                error_msg += f"  Missing sources: {', '.join(missing_sources)}\n"
            if missing_columns:
                for source, columns in missing_columns.items():
                    error_msg += f"  Missing columns in {source}: {', '.join(columns)}\n"
            raise ValueError(error_msg)
    
    def register(self):
        """Register this calculator in the global registry."""
        METRIC_REGISTRY[self.name] = self
        self.logger.info(f"Registered {self.name} calculator in registry")
        return self


class QuarterlyMetricCalculator(MetricCalculator):
    """Base class for metrics calculated from quarterly financial statements."""
    
    def __init__(self, name: str, dependencies: Dict[str, Set[str]]):
        """
        Initialize a quarterly metric calculator.
        
        Args:
            name: Name of the metric
            dependencies: Dictionary mapping data source names to sets of required columns
        """
        super().__init__(name, dependencies)
    
    def _perform_calculation(self, data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate the quarterly metric.
        
        Args:
            data_sources: Dictionary mapping data source names to DataFrames
            
        Returns:
            DataFrame with columns: symbol, date, filing_date, and the calculated metric
        """
        # This will be implemented by specific quarterly metric calculators
        raise NotImplementedError("Subclasses must implement _perform_calculation()")
    
    def _prepare_data_for_query(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare a dataframe for efficient querying by symbol and filing_date.
        
        Args:
            df: DataFrame with at least symbol and filing_date columns
            
        Returns:
            DataFrame with indexes for efficient querying
        """
        # Make a copy to avoid modifying the original
        result_df = df.reset_index()
        # Create an index on symbol and filing_date for efficient querying
        result_df = result_df.set_index(['symbol', 'filing_date'])
        # Sort index for faster lookups
        result_df = result_df.sort_index()
            
        return result_df
    
    def _fill_date_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill gaps in the time series data by forward-filling values between effective dates.
        
        Args:
            df: DataFrame with at least symbol and first_effective_date columns
            
        Returns:
            DataFrame with filled date gaps where each date shows the latest data available
        """
        # Make a copy to avoid modifying the original and drop the original date column if it exists
        if 'symbol' in df.index.names:
            result_df = df.reset_index()
        else:
            result_df = df.reset_index(drop=True)
        if 'date' in result_df.columns:
            result_df = result_df.drop(columns=['date'])
        
        # Group by symbol
        filled_dfs = []
        for symbol, group in result_df.groupby('symbol'):
            # Sort by first_effective_date
            group = group.sort_values('first_effective_date')
                
            # Get min and max effective dates for this symbol
            min_date = group['first_effective_date'].min()
            max_date = group['first_effective_date'].max()
            
            # Create a continuous date range
            date_range = pd.date_range(start=min_date, end=max_date, freq='D')
            
            # Create a template DataFrame with all dates
            template = pd.DataFrame({'date': date_range})
            template['symbol'] = symbol
            
            # Merge with the original data on symbol and date=first_effective_date
            # This correctly aligns the data with the dates when it becomes effective
            merged = pd.merge(template, group, 
                             left_on=['symbol', 'date'], 
                             right_on=['symbol', 'first_effective_date'], 
                             how='left')
            
            # Forward fill all columns except symbol and date
            merged = merged.ffill()
            
            # Sort by date for the final result
            merged = merged.sort_values('date')
            filled_dfs.append(merged)
            
        if filled_dfs:
            return pd.concat(filled_dfs, ignore_index=True)
        else:
            return pd.DataFrame(columns=result_df.columns)


class DailyMetricCalculator(MetricCalculator):
    """Base class for metrics that combine price data with quarterly metrics."""
    
    def __init__(self, name: str, dependencies: Dict[str, Set[str]], required_metrics: List[str]):
        """
        Initialize a daily metric calculator.
        
        Args:
            name: Name of the metric
            dependencies: Dictionary mapping data source names to sets of required columns
            required_metrics: List of metric names this calculator depends on
        """
        super().__init__(name, dependencies)
        self.required_metrics = required_metrics
    
    def _perform_calculation(self, data_sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Calculate the daily metric.
        
        Args:
            data_sources: Dictionary mapping data source names to DataFrames
            
        Returns:
            DataFrame with columns: symbol, date, and the calculated metric
        """
        # This will be implemented by specific daily metric calculators
        raise NotImplementedError("Subclasses must implement _perform_calculation()")

def get_dependencies(metrics: List[str]) -> Dict[str, Set[str]]:
    """
    Get combined dependencies for a list of metrics.
    
    Args:
        metrics: List of metric names
        
    Returns:
        Dictionary mapping data source names to sets of required columns
    """
    all_dependencies = {}
    
    for metric in metrics:
        if metric not in METRIC_REGISTRY:
            raise ValueError(f"Metric '{metric}' not found in registry")
            
        calculator = METRIC_REGISTRY[metric]
        
        for source, columns in calculator.dependencies.items():
            if source not in all_dependencies:
                all_dependencies[source] = set()
            all_dependencies[source].update(columns)
    
    return all_dependencies


def calculate_metrics(metrics: List[str], data_sources: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Calculate multiple metrics using the provided data sources.
    
    Args:
        metrics: List of metric names to calculate
        data_sources: Dictionary mapping data source names to DataFrames
        
    Returns:
        Dictionary mapping metric names to DataFrames with calculated values
    """
    results = {}
    
    # First, identify and calculate quarterly metrics
    quarterly_metrics = []
    daily_metrics = []
    
    for metric in metrics:
        if metric not in METRIC_REGISTRY:
            raise ValueError(f"Unknown metric: {metric}")
            
        calculator = METRIC_REGISTRY[metric]
        if isinstance(calculator, QuarterlyMetricCalculator):
            quarterly_metrics.append(metric)
        elif isinstance(calculator, DailyMetricCalculator):
            daily_metrics.append(metric)
        else:
            # Handle other types of metrics
            results[metric] = calculator.calculate(data_sources)
    
    # Calculate quarterly metrics first
    quarterly_results = {}
    for metric in quarterly_metrics:
        calculator = METRIC_REGISTRY[metric]
        quarterly_results[metric] = calculator.calculate(data_sources)
        results[metric] = quarterly_results[metric]
    
    # Add quarterly results to data sources for daily metrics to use
    data_sources_with_quarterly = {**data_sources}
    for metric, df in quarterly_results.items():
        data_sources_with_quarterly[metric] = df
    
    # Calculate daily metrics using the updated data sources
    for metric in daily_metrics:
        calculator = METRIC_REGISTRY[metric]
        results[metric] = calculator.calculate(data_sources_with_quarterly)
    
    return results
