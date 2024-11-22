CREATE VIEW stock22to24 AS
SELECT A.symbol, min_price * num_share / 1000000 as market_cap
FROM (
    SELECT symbol, min(adjusted_close) as min_price
    FROM daily_price
    WHERE date in ('2022-01-05', '2024-11-05')
    GROUP BY symbol
    HAVING COUNT(DISTINCT date) = 2
) AS A
JOIN (
    SELECT symbol, min(weighted_average_shares_outstanding) as num_share
    FROM income_statement
    WHERE date BETWEEN '2022-01-05' AND '2024-11-05'
    GROUP BY symbol
) AS B
ON A.symbol = B.symbol
WHERE min_price * num_share / 1000000 >= 500
;



SELECT sql
FROM sqlite_master
WHERE type = 'view' AND name = 'stock22to24';