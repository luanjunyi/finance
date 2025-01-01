CREATE VIEW IF NOT EXISTS valid_us_stocks AS
WITH us_stocks AS (
    -- Filter for US stocks
    SELECT DISTINCT symbol 
    FROM stock_symbol 
    WHERE exchange_short_name IN ('NYSE', 'NASDAQ', 'AMEX')
    AND type = 'stock'
),
aapl_dates AS (
    -- Get all trading dates from AAPL (reference for complete trading days)
    SELECT date 
    FROM daily_price 
    WHERE symbol = 'AAPL'
),
stock_date_ranges AS (
    -- Get min and max dates for each stock
    SELECT 
        symbol,
        MIN(date) as min_date,
        MAX(date) as max_date,
        COUNT(*) as actual_days
    FROM daily_price
    WHERE symbol IN (SELECT symbol FROM us_stocks)
    GROUP BY symbol
),
expected_days AS (
    -- Calculate expected number of trading days for each stock
    SELECT 
        s.symbol,
        s.min_date,
        s.max_date,
        s.actual_days,
        COUNT(a.date) as expected_days
    FROM stock_date_ranges s
    LEFT JOIN aapl_dates a ON a.date BETWEEN s.min_date AND s.max_date
    GROUP BY s.symbol, s.min_date, s.max_date, s.actual_days
)
-- Final selection of stocks with complete data
SELECT symbol
FROM expected_days
WHERE actual_days = expected_days;
