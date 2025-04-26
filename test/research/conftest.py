import pytest
import tempfile
import sqlite3
import os
import pandas as pd
from datetime import date, timedelta


@pytest.fixture(scope="session")
def trading_test_db():
    """Create a temporary test database with sample data for trading tests."""
    temp_db = tempfile.NamedTemporaryFile(delete=False)
    conn = sqlite3.connect(temp_db.name)
    
    # Create necessary tables
    conn.executescript("""
        -- Create valid_us_stocks_der table
        CREATE TABLE IF NOT EXISTS valid_us_stocks_der (
            symbol TEXT PRIMARY KEY,
            sector TEXT,
            industry TEXT
        );
        
        -- Create daily_price table
        CREATE TABLE IF NOT EXISTS daily_price (
            symbol TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            adjusted_close REAL,
            volume INTEGER,
            price REAL,
            PRIMARY KEY (symbol, date)
        );
        
        -- Create income_statement table
        CREATE TABLE IF NOT EXISTS income_statement (
            symbol TEXT,
            date TEXT,
            period TEXT,
            calendar_year TEXT,
            reported_currency TEXT,
            revenue REAL,
            cost_of_revenue REAL,
            gross_profit REAL,
            operating_income REAL,
            net_income REAL,
            eps REAL,
            eps_diluted REAL,
            PRIMARY KEY (symbol, date)
        );
        
        -- Create cash_flow table
        CREATE TABLE IF NOT EXISTS cash_flow (
            symbol TEXT,
            date TEXT,
            period TEXT,
            calendar_year TEXT,
            reported_currency TEXT,
            operating_cash_flow REAL,
            capital_expenditure REAL,
            free_cash_flow REAL,
            dividends_paid REAL,
            net_income REAL,
            PRIMARY KEY (symbol, date)
        );
        
        -- Create metrics table for per-share values
        CREATE TABLE IF NOT EXISTS metrics (
            symbol TEXT,
            date TEXT,
            calendar_year INTEGER,
            period TEXT,
            revenue_per_share REAL,
            net_income_per_share REAL,
            operating_cash_flow_per_share REAL,
            free_cash_flow_per_share REAL,
            book_value_per_share REAL,
            roe REAL,
            operating_profit_margin REAL,
            PRIMARY KEY (symbol, date)
        );
    """)
    
    # Insert sample stock data
    conn.executemany(
        """INSERT INTO valid_us_stocks_der (
            symbol, sector, industry
        ) VALUES (?, ?, ?)""",
        [
            ('AAPL', 'Technology', 'Consumer Electronics'),
            ('MSFT', 'Technology', 'Software'),
            ('GOOGL', 'Technology', 'Internet Services'),
        ]
    )
    
    # Insert sample price data with literal values for better readability
    # We'll include just a few key dates for each stock instead of 100 days
    conn.executemany(
        """INSERT INTO daily_price (
            symbol, date, open, high, low, close, adjusted_close, volume, price
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            # AAPL price data
            ('AAPL', '2024-01-01', 148.5, 153.0, 147.0, 150.0, 150.0, 1000000, 150.0),
            ('AAPL', '2023-12-01', 143.5, 148.0, 142.0, 145.0, 145.0, 950000, 145.0),
            ('AAPL', '2023-11-01', 138.5, 143.0, 137.0, 140.0, 140.0, 900000, 140.0),
            ('AAPL', '2023-10-01', 133.5, 138.0, 132.0, 135.0, 135.0, 850000, 135.0),
            
            # MSFT price data
            ('MSFT', '2024-01-01', 297.0, 306.0, 294.0, 300.0, 300.0, 800000, 300.0),
            ('MSFT', '2023-12-01', 287.0, 296.0, 284.0, 290.0, 290.0, 780000, 290.0),
            ('MSFT', '2023-11-01', 277.0, 286.0, 274.0, 280.0, 280.0, 760000, 280.0),
            ('MSFT', '2023-10-01', 267.0, 276.0, 264.0, 270.0, 270.0, 740000, 270.0),
            
            # GOOGL price data
            ('GOOGL', '2024-01-01', 1980.0, 2040.0, 1960.0, 2000.0, 2000.0, 600000, 2000.0),
            ('GOOGL', '2023-12-01', 1930.0, 1990.0, 1910.0, 1950.0, 1950.0, 580000, 1950.0),
            ('GOOGL', '2023-11-01', 1880.0, 1940.0, 1860.0, 1900.0, 1900.0, 560000, 1900.0),
            ('GOOGL', '2023-10-01', 1830.0, 1890.0, 1810.0, 1850.0, 1850.0, 540000, 1850.0)
        ]
    )
    
    # Insert sample fundamental data with literal values
    # Income statement data
    conn.executemany(
        """INSERT INTO income_statement (
            symbol, date, period, calendar_year, reported_currency,
            revenue, cost_of_revenue, gross_profit, operating_income,
            net_income, eps, eps_diluted
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            # AAPL quarterly data - 2023
            ('AAPL', '2023-10-01', 'Q', '2023', 'USD', 120e9, 60e9, 60e9, 36e9, 30e9, 3.0, 2.9),
            ('AAPL', '2023-07-01', 'Q', '2023', 'USD', 115e9, 57.5e9, 57.5e9, 34.5e9, 28.75e9, 2.9, 2.8),
            ('AAPL', '2023-04-01', 'Q', '2023', 'USD', 110e9, 55e9, 55e9, 33e9, 27.5e9, 2.8, 2.7),
            ('AAPL', '2023-01-01', 'Q', '2023', 'USD', 105e9, 52.5e9, 52.5e9, 31.5e9, 26.25e9, 2.7, 2.6),
            
            # AAPL quarterly data - 2022
            ('AAPL', '2022-10-01', 'Q', '2022', 'USD', 100e9, 50e9, 50e9, 30e9, 25e9, 2.6, 2.5),
            ('AAPL', '2022-07-01', 'Q', '2022', 'USD', 95e9, 47.5e9, 47.5e9, 28.5e9, 23.75e9, 2.5, 2.4),
            ('AAPL', '2022-04-01', 'Q', '2022', 'USD', 90e9, 45e9, 45e9, 27e9, 22.5e9, 2.4, 2.3),
            ('AAPL', '2022-01-01', 'Q', '2022', 'USD', 85e9, 42.5e9, 42.5e9, 25.5e9, 21.25e9, 2.3, 2.2),
            
            # MSFT quarterly data - 2023
            ('MSFT', '2023-10-01', 'Q', '2023', 'USD', 58e9, 23.2e9, 34.8e9, 23.2e9, 17.4e9, 3.2, 3.1),
            ('MSFT', '2023-07-01', 'Q', '2023', 'USD', 56e9, 22.4e9, 33.6e9, 22.4e9, 16.8e9, 3.15, 3.05),
            ('MSFT', '2023-04-01', 'Q', '2023', 'USD', 54e9, 21.6e9, 32.4e9, 21.6e9, 16.2e9, 3.1, 3.0),
            ('MSFT', '2023-01-01', 'Q', '2023', 'USD', 52e9, 20.8e9, 31.2e9, 20.8e9, 15.6e9, 3.05, 2.95),
            
            # MSFT quarterly data - 2022
            ('MSFT', '2022-10-01', 'Q', '2022', 'USD', 50e9, 20e9, 30e9, 20e9, 15e9, 3.0, 2.9),
            ('MSFT', '2022-07-01', 'Q', '2022', 'USD', 48e9, 19.2e9, 28.8e9, 19.2e9, 14.4e9, 2.95, 2.85),
            ('MSFT', '2022-04-01', 'Q', '2022', 'USD', 46e9, 18.4e9, 27.6e9, 18.4e9, 13.8e9, 2.9, 2.8),
            ('MSFT', '2022-01-01', 'Q', '2022', 'USD', 44e9, 17.6e9, 26.4e9, 17.6e9, 13.2e9, 2.85, 2.75),
            
            # GOOGL quarterly data - 2023
            ('GOOGL', '2023-10-01', 'Q', '2023', 'USD', 76e9, 34.2e9, 41.8e9, 26.6e9, 21.28e9, 4.8, 4.7),
            ('GOOGL', '2023-07-01', 'Q', '2023', 'USD', 74e9, 33.3e9, 40.7e9, 25.9e9, 20.72e9, 4.6, 4.5),
            ('GOOGL', '2023-04-01', 'Q', '2023', 'USD', 72e9, 32.4e9, 39.6e9, 25.2e9, 20.16e9, 4.4, 4.3),
            ('GOOGL', '2023-01-01', 'Q', '2023', 'USD', 70e9, 31.5e9, 38.5e9, 24.5e9, 19.6e9, 4.2, 4.1),
            
            # GOOGL quarterly data - 2022
            ('GOOGL', '2022-10-01', 'Q', '2022', 'USD', 68e9, 30.6e9, 37.4e9, 23.8e9, 19.04e9, 4.0, 3.9),
            ('GOOGL', '2022-07-01', 'Q', '2022', 'USD', 66e9, 29.7e9, 36.3e9, 23.1e9, 18.48e9, 3.8, 3.7),
            ('GOOGL', '2022-04-01', 'Q', '2022', 'USD', 64e9, 28.8e9, 35.2e9, 22.4e9, 17.92e9, 3.6, 3.5),
            ('GOOGL', '2022-01-01', 'Q', '2022', 'USD', 62e9, 27.9e9, 34.1e9, 21.7e9, 17.36e9, 3.4, 3.3)
        ]
    )
    
    # Insert sample cash flow data with literal values
    conn.executemany(
        """INSERT INTO cash_flow (
            symbol, date, period, calendar_year, reported_currency,
            operating_cash_flow, capital_expenditure, free_cash_flow,
            dividends_paid, net_income
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            # AAPL cash flow data - 2023
            ('AAPL', '2023-10-01', 'Q', '2023', 'USD', 42.0e9, -6.0e9, 36.0e9, -3.6e9, 30.0e9),
            ('AAPL', '2023-07-01', 'Q', '2023', 'USD', 40.25e9, -5.75e9, 34.5e9, -3.45e9, 28.75e9),
            ('AAPL', '2023-04-01', 'Q', '2023', 'USD', 38.5e9, -5.5e9, 33.0e9, -3.3e9, 27.5e9),
            ('AAPL', '2023-01-01', 'Q', '2023', 'USD', 36.75e9, -5.25e9, 31.5e9, -3.15e9, 26.25e9),
            
            # AAPL cash flow data - 2022
            ('AAPL', '2022-10-01', 'Q', '2022', 'USD', 35.0e9, -5.0e9, 30.0e9, -3.0e9, 25.0e9),
            ('AAPL', '2022-07-01', 'Q', '2022', 'USD', 33.25e9, -4.75e9, 28.5e9, -2.85e9, 23.75e9),
            ('AAPL', '2022-04-01', 'Q', '2022', 'USD', 31.5e9, -4.5e9, 27.0e9, -2.7e9, 22.5e9),
            ('AAPL', '2022-01-01', 'Q', '2022', 'USD', 29.75e9, -4.25e9, 25.5e9, -2.55e9, 21.25e9),
            
            # MSFT cash flow data - 2023
            ('MSFT', '2023-10-01', 'Q', '2023', 'USD', 23.2e9, -3.48e9, 19.72e9, -1.74e9, 17.4e9),
            ('MSFT', '2023-07-01', 'Q', '2023', 'USD', 22.4e9, -3.36e9, 19.04e9, -1.68e9, 16.8e9),
            ('MSFT', '2023-04-01', 'Q', '2023', 'USD', 21.6e9, -3.24e9, 18.36e9, -1.62e9, 16.2e9),
            ('MSFT', '2023-01-01', 'Q', '2023', 'USD', 20.8e9, -3.12e9, 17.68e9, -1.56e9, 15.6e9),
            
            # MSFT cash flow data - 2022
            ('MSFT', '2022-10-01', 'Q', '2022', 'USD', 20.0e9, -3.0e9, 17.0e9, -1.5e9, 15.0e9),
            ('MSFT', '2022-07-01', 'Q', '2022', 'USD', 19.2e9, -2.88e9, 16.32e9, -1.44e9, 14.4e9),
            ('MSFT', '2022-04-01', 'Q', '2022', 'USD', 18.4e9, -2.76e9, 15.64e9, -1.38e9, 13.8e9),
            ('MSFT', '2022-01-01', 'Q', '2022', 'USD', 17.6e9, -2.64e9, 14.96e9, -1.32e9, 13.2e9),
            
            # GOOGL cash flow data - 2023
            ('GOOGL', '2023-10-01', 'Q', '2023', 'USD', 28.88e9, -5.32e9, 23.56e9, 0.0, 21.28e9),
            ('GOOGL', '2023-07-01', 'Q', '2023', 'USD', 28.12e9, -5.18e9, 22.94e9, 0.0, 20.72e9),
            ('GOOGL', '2023-04-01', 'Q', '2023', 'USD', 27.36e9, -5.04e9, 22.32e9, 0.0, 20.16e9),
            ('GOOGL', '2023-01-01', 'Q', '2023', 'USD', 26.6e9, -4.9e9, 21.7e9, 0.0, 19.6e9),
            
            # GOOGL cash flow data - 2022
            ('GOOGL', '2022-10-01', 'Q', '2022', 'USD', 25.84e9, -4.76e9, 21.08e9, 0.0, 19.04e9),
            ('GOOGL', '2022-07-01', 'Q', '2022', 'USD', 25.08e9, -4.62e9, 20.46e9, 0.0, 18.48e9),
            ('GOOGL', '2022-04-01', 'Q', '2022', 'USD', 24.32e9, -4.48e9, 19.84e9, 0.0, 17.92e9),
            ('GOOGL', '2022-01-01', 'Q', '2022', 'USD', 23.56e9, -4.34e9, 19.22e9, 0.0, 17.36e9)
        ]
    )
    
    # Insert sample metrics data with literal values
    conn.executemany(
        """INSERT INTO metrics (
            symbol, date, calendar_year, period,
            revenue_per_share, net_income_per_share, operating_cash_flow_per_share,
            free_cash_flow_per_share, book_value_per_share, roe, operating_profit_margin
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            # AAPL metrics - 2023
            ('AAPL', '2023-10-01', 2023, 'Q', 12.5, 3.0, 4.2, 3.6, 32.0, 0.09375, 0.30),
            ('AAPL', '2023-07-01', 2023, 'Q', 12.0, 2.875, 4.025, 3.45, 31.5, 0.09127, 0.29),
            ('AAPL', '2023-04-01', 2023, 'Q', 11.5, 2.75, 3.85, 3.3, 31.0, 0.08871, 0.28),
            ('AAPL', '2023-01-01', 2023, 'Q', 11.0, 2.625, 3.675, 3.15, 30.5, 0.08607, 0.27),
            
            # AAPL metrics - 2022
            ('AAPL', '2022-10-01', 2022, 'Q', 10.5, 2.5, 3.5, 3.0, 30.0, 0.08333, 0.26),
            ('AAPL', '2022-07-01', 2022, 'Q', 10.0, 2.375, 3.325, 2.85, 29.5, 0.08051, 0.25),
            ('AAPL', '2022-04-01', 2022, 'Q', 9.5, 2.25, 3.15, 2.7, 29.0, 0.07759, 0.24),
            ('AAPL', '2022-01-01', 2022, 'Q', 9.0, 2.125, 2.975, 2.55, 28.5, 0.07456, 0.23),
            
            # MSFT metrics - 2023
            ('MSFT', '2023-10-01', 2023, 'Q', 7.0, 2.175, 2.9, 2.465, 25.5, 0.08529, 0.40),
            ('MSFT', '2023-07-01', 2023, 'Q', 6.75, 2.1, 2.8, 2.38, 25.0, 0.084, 0.39),
            ('MSFT', '2023-04-01', 2023, 'Q', 6.5, 2.025, 2.7, 2.295, 24.5, 0.08265, 0.38),
            ('MSFT', '2023-01-01', 2023, 'Q', 6.25, 1.95, 2.6, 2.21, 24.0, 0.08125, 0.37),
            
            # MSFT metrics - 2022
            ('MSFT', '2022-10-01', 2022, 'Q', 6.0, 1.875, 2.5, 2.125, 23.5, 0.07979, 0.36),
            ('MSFT', '2022-07-01', 2022, 'Q', 5.75, 1.8, 2.4, 2.04, 23.0, 0.07826, 0.35),
            ('MSFT', '2022-04-01', 2022, 'Q', 5.5, 1.725, 2.3, 1.955, 22.5, 0.07667, 0.34),
            ('MSFT', '2022-01-01', 2022, 'Q', 5.25, 1.65, 2.2, 1.87, 22.0, 0.075, 0.33),
            
            # GOOGL metrics - 2023
            ('GOOGL', '2023-10-01', 2023, 'Q', 12.5, 3.55, 4.81, 3.93, 28.0, 0.12679, 0.35),
            ('GOOGL', '2023-07-01', 2023, 'Q', 12.0, 3.45, 4.69, 3.82, 27.5, 0.12545, 0.34),
            ('GOOGL', '2023-04-01', 2023, 'Q', 11.5, 3.36, 4.56, 3.72, 27.0, 0.12444, 0.33),
            ('GOOGL', '2023-01-01', 2023, 'Q', 11.0, 3.27, 4.43, 3.62, 26.5, 0.12340, 0.32),
            
            # GOOGL metrics - 2022
            ('GOOGL', '2022-10-01', 2022, 'Q', 10.5, 3.17, 4.31, 3.51, 26.0, 0.12192, 0.31),
            ('GOOGL', '2022-07-01', 2022, 'Q', 10.0, 3.08, 4.18, 3.41, 25.5, 0.12078, 0.30),
            ('GOOGL', '2022-04-01', 2022, 'Q', 9.5, 2.99, 4.05, 3.31, 25.0, 0.1196, 0.29),
            ('GOOGL', '2022-01-01', 2022, 'Q', 9.0, 2.89, 3.93, 3.2, 24.5, 0.11796, 0.28)
        ]
    )
    
    # Commit changes and close connection
    
    conn.commit()
    conn.close()
    
    yield temp_db.name
    
    # Cleanup after all tests are done
    os.unlink(temp_db.name)


@pytest.fixture
def sample_symbols():
    """Return a list of sample stock symbols."""
    return ['AAPL', 'MSFT', 'GOOGL']


@pytest.fixture
def sample_dates():
    """Return a list of sample dates."""
    return [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]
