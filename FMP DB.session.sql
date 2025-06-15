












CREATE VIEW stock22to24 AS
SELECT A.symbol, C.sector, C.industry, min_price * num_share / 1000000 as market_cap
FROM (
    SELECT symbol, min(close) as min_price
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
JOIN (
    SELECT symbol, sector, industry
    FROM stock_symbol
    WHERE sector is not NULL and industry is not NULL and type = 'stock'
) AS C
ON A.symbol = B.symbol AND B.symbol = C.symbol
WHERE min_price * num_share / 1000000 >= 100
;



SELECT sql
FROM sqlite_master
WHERE type = 'view' AND name = 'stock22to24';


select A.symbol, end_price / start_price - 1 as gain
from (
    select symbol, close as start_price
    from daily_price
    where date = '2023-01-05' and symbol  in ('^SPX', 'QQQ', '^DJI')
) AS A
join (
    select symbol, close as end_price
    from daily_price
    where date = '2024-01-05' and symbol in ('^SPX', 'QQQ', '^DJI')
) AS B
ON B.symbol = A.symbol