"""
Configuration module for the finance project.

This module provides centralized access to configuration settings,
primarily loaded from environment variables with sensible defaults.
"""

import os

def read_config_from_env_or_die(name: str):
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Mandatory environment variable {name} is not set")
    return value


# For direct imports like: from utils.config import DB_PATH, API_KEY
FMP_DB_PATH = read_config_from_env_or_die("FMP_DB_PATH")
FMP_API_KEY = read_config_from_env_or_die("FMP_API_KEY")
EODHD_API_KEY = read_config_from_env_or_die("EODHD_API_KEY")
