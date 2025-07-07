"""
Offline financial metrics calculation package for FMP data.

This package provides tools for calculating financial metrics from FMP data
in an offline manner, with a focus on performance and modularity.
"""
from .offline_data import OfflineData
from .dataset import Dataset
# Import from legacy module for backward compatibility
from fmp_data_legacy import FMPPriceLoader, BEFORE_PRICE, AFTER_PRICE, PRICE_METRICS