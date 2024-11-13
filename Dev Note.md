# Goal

Build a crawler that crawl historical stock daily price and historical financial statements from FMP API.

- Crawl all data in the past 30 years.
- Store the data in a local sqlite database.
- Use python to call the FMP API with requests package. The whole codebase should be in Python.

# FMP API

## Getting all stock symbols

https://financialmodelingprep.com/api/v3/stock/list?apikey=API_KEY

This return all stock symbols.

```json
[
  {
    "symbol": "PMGL",
    "name": "Perth Mint Gold",
    "price": 17.94,
    "exchange": "Australian Securities Exchange",
    "exchangeShortName": "ASX",
    "type": "etf"
  },
  {
    "symbol": "IWR",
    "name": "iShares Russell Mid-Cap ETF",
    "price": 93.0847,
    "exchange": "New York Stock Exchange Arca",
    "exchangeShortName": "AMEX",
    "type": "etf"
  },
  ...
]
```

## Historical price example

https://financialmodelingprep.com/api/v3/historical-price-full/MSFT?apikey=API_KEY&from=1990-12-01&to=1993-01-03

This return historical price for MSFT from 1990-12-01 to 1993-01-03.

```json
{
  "symbol": "MSFT",
  "historical": [
    {
      "date": "1992-12-31",
      "open": 2.69,
      "high": 2.7,
      "low": 2.66,
      "close": 2.67,
      "adjClose": 1.65,
      "volume": 30851200,
      "unadjustedVolume": 30851200,
      "change": -0.019531,
      "changePercent": -0.74349,
      "vwap": 2.68,
      "label": "December 31, 92",
      "changeOverTime": -0.0074349
    },
    {
      "date": "1992-12-30",
      "open": 2.71,
      "high": 2.72,
      "low": 2.66,
      "close": 2.68,
      "adjClose": 1.65,
      "volume": 50860800,
      "unadjustedVolume": 50860800,
      "change": -0.027344,
      "changePercent": -1.11,
      "vwap": 2.6925,
      "label": "December 30, 92",
      "changeOverTime": -0.0111
    },
    ...
  ]
}
```

## Historical financial statements example

### Income statement

https://financialmodelingprep.com/api/v3/income-statement/AAPL?apikey=API_KEY&period=quarter&limit=120

This return the income statement for AAPL of the last 120 quarters, which is 30 years.

```json
[
  {
    "date": "2024-09-28",
    "symbol": "AAPL",
    "reportedCurrency": "USD",
    "cik": "0000320193",
    "fillingDate": "2024-11-01",
    "acceptedDate": "2024-11-01 06:01:36",
    "calendarYear": "2024",
    "period": "FY",
    "revenue": 391035000000,
    "costOfRevenue": 210352000000,
    "grossProfit": 180683000000,
    "grossProfitRatio": 0.4620634982,
    "researchAndDevelopmentExpenses": 31370000000,
    "generalAndAdministrativeExpenses": 0,
    "sellingAndMarketingExpenses": 0,
    "sellingGeneralAndAdministrativeExpenses": 26097000000,
    "otherExpenses": 0,
    "operatingExpenses": 57467000000,
    "costAndExpenses": 267819000000,
    "interestIncome": 0,
    "interestExpense": 0,
    "depreciationAndAmortization": 11445000000,
    "ebitda": 134661000000,
    "ebitdaratio": 0.3443707085,
    "operatingIncome": 123216000000,
    "operatingIncomeRatio": 0.3151022287,
    "totalOtherIncomeExpensesNet": 269000000,
    "incomeBeforeTax": 123485000000,
    "incomeBeforeTaxRatio": 0.3157901467,
    "incomeTaxExpense": 29749000000,
    "netIncome": 93736000000,
    "netIncomeRatio": 0.2397125577,
    "eps": 6.11,
    "epsdiluted": 6.08,
    "weightedAverageShsOut": 15343783000,
    "weightedAverageShsOutDil": 15408095000,
    "link": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/0000320193-24-000123-index.htm",
    "finalLink": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
  },
  ...
]
```

### Balance sheet

https://financialmodelingprep.com/api/v3/balance-sheet-statement/AAPL?period=annual&apikey=4KbPA5SFE5xuvmiYYxGd0rz163fMbyzN&limit=120&period=quarter

This return the balance sheet for AAPL of the last 120 quarters, which is 30 years.

```json
[
  {
    "date": "2024-09-28",
    "symbol": "AAPL",
    "reportedCurrency": "USD",
    "cik": "0000320193",
    "fillingDate": "2024-11-01",
    "acceptedDate": "2024-11-01 06:01:36",
    "calendarYear": "2024",
    "period": "FY",
    "cashAndCashEquivalents": 29943000000,
    "shortTermInvestments": 35228000000,
    "cashAndShortTermInvestments": 65171000000,
    "netReceivables": 66243000000,
    "inventory": 7286000000,
    "otherCurrentAssets": 14287000000,
    "totalCurrentAssets": 152987000000,
    "propertyPlantEquipmentNet": 45680000000,
    "goodwill": 0,
    "intangibleAssets": 0,
    "goodwillAndIntangibleAssets": 0,
    "longTermInvestments": 91479000000,
    "taxAssets": 19499000000,
    "otherNonCurrentAssets": 55335000000,
    "totalNonCurrentAssets": 211993000000,
    "otherAssets": 0,
    "totalAssets": 364980000000,
    "accountPayables": 68960000000,
    "shortTermDebt": 22511000000,
    "taxPayables": 26601000000,
    "deferredRevenue": 8249000000,
    "otherCurrentLiabilities": 50071000000,
    "totalCurrentLiabilities": 176392000000,
    "longTermDebt": 96548000000,
    "deferredRevenueNonCurrent": 0,
    "deferredTaxLiabilitiesNonCurrent": 0,
    "otherNonCurrentLiabilities": 35090000000,
    "totalNonCurrentLiabilities": 131638000000,
    "otherLiabilities": 0,
    "capitalLeaseObligations": 12430000000,
    "totalLiabilities": 308030000000,
    "preferredStock": 0,
    "commonStock": 83276000000,
    "retainedEarnings": -19154000000,
    "accumulatedOtherComprehensiveIncomeLoss": -7172000000,
    "othertotalStockholdersEquity": 0,
    "totalStockholdersEquity": 56950000000,
    "totalEquity": 56950000000,
    "totalLiabilitiesAndStockholdersEquity": 364980000000,
    "minorityInterest": 0,
    "totalLiabilitiesAndTotalEquity": 364980000000,
    "totalInvestments": 126707000000,
    "totalDebt": 106629000000,
    "netDebt": 76686000000,
    "link": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/0000320193-24-000123-index.htm",
    "finalLink": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
  },
  ...
]
```

### Cash flow statement

https://financialmodelingprep.com/api/v3/cash-flow-statement/AAPL?period=annual&apikey=4KbPA5SFE5xuvmiYYxGd0rz163fMbyzN&limit=120&period=quarter

This return the cash flow statement for AAPL of the last 120 quarters, which is 30 years.

```json
[
  {
    "date": "2024-09-28",
    "symbol": "AAPL",
    "reportedCurrency": "USD",
    "cik": "0000320193",
    "fillingDate": "2024-11-01",
    "acceptedDate": "2024-11-01 06:01:36",
    "calendarYear": "2024",
    "period": "FY",
    "netIncome": 93736000000,
    "depreciationAndAmortization": 11445000000,
    "deferredIncomeTax": 0,
    "stockBasedCompensation": 11688000000,
    "changeInWorkingCapital": 3651000000,
    "accountsReceivables": -5144000000,
    "inventory": -1046000000,
    "accountsPayables": 6020000000,
    "otherWorkingCapital": 3821000000,
    "otherNonCashItems": -2266000000,
    "netCashProvidedByOperatingActivities": 118254000000,
    "investmentsInPropertyPlantAndEquipment": -9447000000,
    "acquisitionsNet": 0,
    "purchasesOfInvestments": -48656000000,
    "salesMaturitiesOfInvestments": 62346000000,
    "otherInvestingActivites": -1308000000,
    "netCashUsedForInvestingActivites": 2935000000,
    "debtRepayment": -5998000000,
    "commonStockIssued": 0,
    "commonStockRepurchased": -94949000000,
    "dividendsPaid": -15234000000,
    "otherFinancingActivites": -5802000000,
    "netCashUsedProvidedByFinancingActivities": -121983000000,
    "effectOfForexChangesOnCash": 0,
    "netChangeInCash": -794000000,
    "cashAtEndOfPeriod": 29943000000,
    "cashAtBeginningOfPeriod": 30737000000,
    "operatingCashFlow": 118254000000,
    "capitalExpenditure": -9447000000,
    "freeCashFlow": 108807000000,
    "link": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/0000320193-24-000123-index.htm",
    "finalLink": "https://www.sec.gov/Archives/edgar/data/320193/000032019324000123/aapl-20240928.htm"
  },

```

# DB schema

```sql
CREATE TABLE stock_symbol (
    symbol VARCHAR(10) PRIMARY KEY,
    name VARCHAR(255),
    exchange VARCHAR(255),
    exchange_short_name VARCHAR(255),
    type VARCHAR(255),
);

CREATE TABLE daily_price (
    symbol VARCHAR(10) PRIMARY KEY,
    date DATE NOT NULL,
    open DECIMAL(10,2),
    high DECIMAL(10,2),
    low DECIMAL(10,2),
    close DECIMAL(10,2),
    adjusted_close DECIMAL(10,2),
    volume BIGINT,
    UNIQUE(symbol, date)
);
CREATE INDEX idx_daily_price_symbol_date ON daily_price(symbol, date);

CREATE TABLE income_statement (
    symbol VARCHAR(10),
    date DATE NOT NULL,
    period VARCHAR(10) NOT NULL, -- FY/Q1/Q2/Q3/Q4
    calendar_year VARCHAR(4),
    reported_currency VARCHAR(3),
    revenue DECIMAL(15,2),
    cost_of_revenue DECIMAL(15,2),
    gross_profit DECIMAL(15,2),
    gross_profit_ratio DECIMAL(10,4),
    research_and_development_expenses DECIMAL(15,2),
    general_and_administrative_expenses DECIMAL(15,2),
    selling_and_marketing_expenses DECIMAL(15,2),
    selling_general_and_administrative_expenses DECIMAL(15,2),
    operating_expenses DECIMAL(15,2),
    operating_income DECIMAL(15,2),
    operating_income_ratio DECIMAL(10,4),
    ebitda DECIMAL(15,2),
    ebitda_ratio DECIMAL(10,4),
    net_income DECIMAL(15,2),
    net_income_ratio DECIMAL(10,4),
    eps DECIMAL(10,2),
    eps_diluted DECIMAL(10,2),
    weighted_average_shares_outstanding DECIMAL(15,0),
    weighted_average_shares_outstanding_diluted DECIMAL(15,0),
    link VARCHAR(255),
    UNIQUE(symbol, date, period)
);
CREATE INDEX idx_income_statement_symbol_date ON income_statement(symbol, date);

CREATE TABLE balance_sheet (
    symbol VARCHAR(10),
    date DATE NOT NULL,
    period VARCHAR(10) NOT NULL, -- FY/Q1/Q2/Q3/Q4
    calendar_year VARCHAR(4),
    reported_currency VARCHAR(3),
    cash_and_cash_equivalents DECIMAL(15,2),
    short_term_investments DECIMAL(15,2),
    cash_and_short_term_investments DECIMAL(15,2),
    net_receivables DECIMAL(15,2),
    inventory DECIMAL(15,2),
    total_current_assets DECIMAL(15,2),
    property_plant_equipment_net DECIMAL(15,2),
    goodwill DECIMAL(15,2),
    intangible_assets DECIMAL(15,2),
    long_term_investments DECIMAL(15,2),
    total_non_current_assets DECIMAL(15,2),
    total_assets DECIMAL(15,2),
    accounts_payables DECIMAL(15,2),
    short_term_debt DECIMAL(15,2),
    total_current_liabilities DECIMAL(15,2),
    long_term_debt DECIMAL(15,2),
    total_non_current_liabilities DECIMAL(15,2),
    total_liabilities DECIMAL(15,2),
    total_stockholders_equity DECIMAL(15,2),
    total_equity DECIMAL(15,2),
    total_liabilities_and_stockholders_equity DECIMAL(15,2),
    total_investments DECIMAL(15,2),
    total_debt DECIMAL(15,2),
    net_debt DECIMAL(15,2),
    link VARCHAR(255),
    UNIQUE(symbol, date, period)
);
CREATE INDEX idx_balance_sheet_symbol_date ON balance_sheet(symbol, date);

CREATE TABLE cash_flow (
    symbol VARCHAR(10),
    date DATE NOT NULL,
    period VARCHAR(10) NOT NULL, -- FY/Q1/Q2/Q3/Q4
    calendar_year VARCHAR(4),
    reported_currency VARCHAR(3),
    net_income DECIMAL(15,2),
    depreciation_and_amortization DECIMAL(15,2),
    stock_based_compensation DECIMAL(15,2),
    change_in_working_capital DECIMAL(15,2),
    accounts_receivables DECIMAL(15,2),
    inventory DECIMAL(15,2),
    accounts_payables DECIMAL(15,2),
    net_cash_provided_by_operating_activities DECIMAL(15,2),
    investments_in_property_plant_and_equipment DECIMAL(15,2),
    acquisitions_net DECIMAL(15,2),
    purchases_of_investments DECIMAL(15,2),
    sales_maturities_of_investments DECIMAL(15,2),
    net_cash_used_for_investing_activities DECIMAL(15,2),
    debt_repayment DECIMAL(15,2),
    common_stock_repurchased DECIMAL(15,2),
    dividends_paid DECIMAL(15,2),
    net_cash_used_provided_by_financing_activities DECIMAL(15,2),
    net_change_in_cash DECIMAL(15,2),
    operating_cash_flow DECIMAL(15,2),
    capital_expenditure DECIMAL(15,2),
    free_cash_flow DECIMAL(15,2),
    link VARCHAR(255),
    UNIQUE(symbol, date, period)
);
CREATE INDEX idx_cash_flow_symbol_date ON cash_flow(symbol, date);
```

# Program interface

The program runs in console.

`fmp_crawl --type=symbol` 

To crawl all stock symbols and store them in the database.

The rest crawl back to 30 years in history.

`fmp_crawl --type=price`

To crawl historical price for all stocks from DB and store them in the database. Add filters in the code as a function. For example, filters symbols only NYSE and NASDAQ stocks.

`fmp_crawl --type=financial`

To crawl income statement, balance sheet and cash flow statement for all stocks from DB and store them in the database. Add filters in the code.

# Other requirements

- Rate limit: 4 requests per second.
- Error handling: If the API returns an error, retry for three times with 5 seconds sleep. Then the program should log the error message and continue with the next stock. The log should be appended in a separate log file (error.log). The error log include the failed API call, the error message and the stock symbol, and the link of code.
- Missing data: If the API returned data that has missing fields, the program should log the missing fields and continue with the next stock. The log should be appended in a separate log file (missing_data.log). The missing data log include the stock symbol and what fields are missing.
- Logging: Print the progress and elapsed time in console.
- Use asyncio to speed up without breaking the rate limit, if this will complicate the code significantly, forget about it.
- API key is in env variable `FMP_API_KEY`.