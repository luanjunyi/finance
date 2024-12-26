import pytest
import tempfile
import sqlite3
import os

PRODUCTION_DB_PATH = '/Users/jluan/code/finance/data/fmp_data.db'

def get_production_schema():
    """Get schema from production database."""

    conn = sqlite3.connect(PRODUCTION_DB_PATH)
    cursor = conn.cursor()
    
    # Get all table creation SQL and index creation SQL
    cursor.execute("""
        SELECT sql FROM sqlite_master 
        WHERE sql IS NOT NULL 
        AND type IN ('table', 'index')
        ORDER BY type DESC;
    """)
    
    schema = []
    for row in cursor.fetchall():
        if row[0]:  # sql can be NULL for some system tables
            schema.append(row[0] + ';')
    
    conn.close()
    return '\n'.join(schema)


@pytest.fixture(scope="session")
def test_db():
    """Create a temporary test database with sample data."""
    temp_db = tempfile.NamedTemporaryFile(delete=False)
    conn = sqlite3.connect(temp_db.name)
    
    # Create schema from production database
    schema_sql = get_production_schema()
    conn.executescript(schema_sql)
    
    # Insert sample data
    conn.executemany(
        """INSERT INTO daily_price (
            symbol, date, open, high, low, close, adjusted_close, volume
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            ('AAPL', '2024-01-01', 100.0, 105.0, 99.0, 102.0, 102.0, 1000000),
            ('AAPL', '2024-01-02', 102.0, 106.0, 101.0, 105.0, 105.0, 1200000),
            ('GOOGL', '2024-01-01', 150.0, 155.0, 149.0, 152.0, 152.0, 500000),
        ]
    )
    
    conn.executemany(
        """INSERT INTO income_statement (
            symbol, date, period, calendar_year, reported_currency,
            revenue, cost_of_revenue, gross_profit, operating_income,
            net_income, eps, eps_diluted
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            ('AAPL', '2024-01-01', 'FY', '2024', 'USD', 
             100000000, 50000000, 50000000, 30000000, 25000000, 2.5, 2.4),
            ('GOOGL', '2024-01-01', 'FY', '2024', 'USD',
             80000000, 40000000, 40000000, 25000000, 20000000, 3.0, 2.9),
        ]
    )
    
    conn.executemany(
        """INSERT INTO balance_sheet (
            symbol, date, period, calendar_year, reported_currency,
            total_assets, total_current_assets, total_non_current_assets,
            total_liabilities, total_current_liabilities, total_non_current_liabilities,
            total_equity
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            ('AAPL', '2024-01-01', 'FY', '2024', 'USD',
             500000000, 200000000, 300000000, 200000000, 100000000, 100000000, 300000000),
            ('GOOGL', '2024-01-01', 'FY', '2024', 'USD',
             400000000, 150000000, 250000000, 150000000, 75000000, 75000000, 250000000),
        ]
    )

    conn.executemany(
        """INSERT INTO cash_flow (
            symbol, date, period, calendar_year, reported_currency,
            operating_cash_flow, capital_expenditure, free_cash_flow,
            dividends_paid, net_income
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            ('AAPL', '2024-01-01', 'FY', '2024', 'USD',
             35000000, -5000000, 30000000, -2000000, 25000000),
            ('GOOGL', '2024-01-01', 'FY', '2024', 'USD',
             28000000, -4000000, 24000000, 0, 20000000),
        ]
    )

    conn.executemany(
        """INSERT INTO metrics (
            symbol, date, calendar_year, period,
            revenue_per_share, net_income_per_share, operating_cash_flow_per_share,
            free_cash_flow_per_share, book_value_per_share, roe
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            ('AAPL', '2024-01-01', 2024, 'FY',
             10.0, 2.5, 3.5, 3.0, 30.0, 0.15),
            ('GOOGL', '2024-01-01', 2024, 'FY',
             8.0, 2.0, 2.8, 2.4, 25.0, 0.12),
        ]
    )

    # Insert stock symbols
    conn.executemany(
        """INSERT INTO stock_symbol (
            symbol, name, exchange, exchange_short_name, type, sector, industry
        ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
        [
            ('AAPL', 'Apple Inc.', 'NASDAQ', 'NASDAQ', 'stock', 'Technology', 'Consumer Electronics'),
            ('GOOGL', 'Alphabet Inc.', 'NASDAQ', 'NASDAQ', 'stock', 'Technology', 'Internet Content & Information'),
        ]
    )
    
    conn.commit()
    conn.close()
    
    yield temp_db.name
    
    # Cleanup after all tests are done
    os.unlink(temp_db.name)


@pytest.fixture
def sample_symbols():
    """Return a list of sample stock symbols."""
    return ['AAPL', 'GOOGL']


@pytest.fixture
def sample_dates():
    """Return a list of sample dates."""
    return ['2024-01-01', '2024-01-02']
