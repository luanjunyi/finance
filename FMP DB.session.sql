CREATE VIEW stock22to24 AS
SELECT A.symbol, min_price * num_share / 1000000 as market_cap
FROM (
    SELECT symbol, min(adjusted_close) as min_price
    FROM daily_price
    WHERE date in ('2021-01-05', '2024-11-05')
    GROUP BY symbol
    HAVING COUNT(DISTINCT date) = 2
) AS A
JOIN (
    SELECT symbol, min(weighted_average_shares_outstanding) as num_share
    FROM income_statement
    WHERE date BETWEEN '2021-01-05' AND '2024-11-05'
    GROUP BY symbol
) AS B
ON A.symbol = B.symbol
WHERE min_price * num_share / 1000000 >= 500
;



SELECT sql
FROM sqlite_master
WHERE type = 'view' AND name = 'stock22to24';


select A.symbol, end_price / start_price - 1 as gain
from (
    select symbol, adjusted_close as start_price
    from daily_price
    where date = '2023-01-05' and symbol  in ('^SPX', 'QQQ', '^DJI')
) AS A
join (
    select symbol, adjusted_close as end_price
    from daily_price
    where date = '2024-01-05' and symbol in ('^SPX', 'QQQ', '^DJI')
) AS B
ON B.symbol = A.symbol