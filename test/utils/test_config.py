"""
Test for the config module.

This test simply tries to import the mandatory configuration variables.
If the test fails, it means the current environment is not properly configured.
"""

import pytest


def test_mandatory_configs_are_set():
    """
    Test that mandatory configuration variables are set in the environment.
    
    This test will fail if the environment variables are not set, indicating
    that the environment is not properly configured for running the application.
    """
    from utils.config import FMP_DB_PATH, FMP_API_KEY
    
    # Simply check that the values are not empty
    assert FMP_DB_PATH, "FMP_DB_PATH environment variable is not set. This is mandatory for running the application."
    assert FMP_API_KEY, "FMP_API_KEY environment variable is not set. This is mandatory for running the application."
