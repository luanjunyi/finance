"""
Financial metrics calculation package for offline FMP data.

This package provides a framework for calculating financial metrics from FMP data,
with separate modules for quarterly and daily metrics.
"""

from .metrics import (
    MetricCalculator,
    QuarterlyMetricCalculator,
    DailyMetricCalculator,
    METRIC_REGISTRY,
    calculate_metrics,
    get_dependencies
)

# Import all metric calculators to register them
from .quarterly import *
from .daily import *
