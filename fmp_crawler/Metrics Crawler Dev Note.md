Add metrics crawlers for FMP. 

1. Metrics APIs
We use two endpoints:

https://financialmodelingprep.com/api/v3/key-metrics/AAPL?period=quarter&apikey=4KbPA5SFE5xuvmiYYxGd0rz163fMbyzN&limit=120

Example response:
```json
[
  {
    "symbol": "AAPL",
    "date": "2024-09-28",
    "calendarYear": "2024",
    "period": "Q4",
    "revenuePerShare": 6.2569247672849775,
    "netIncomePerShare": 0.9712634927916509,
    "operatingCashFlowPerShare": 1.7671379957408355,
    "freeCashFlowPerShare": 1.5754690057138188,
    "cashPerShare": 4.295481344240274,
    "bookValuePerShare": 3.753627572915616,
    "tangibleBookValuePerShare": 3.753627572915616,
    "shareholdersEquityPerShare": 3.753627572915616,
    "interestDebtPerShare": 7.028016759831769,
    "marketCap": 3456027602100,
    "enterpriseValue": 3532713602100,
    "peRatio": 58.63239010077361,
    "priceToSalesRatio": 36.406063437269566,
    "pocfratio": 128.90334572004028,
    "pfcfRatio": 144.58551655022382,
    "pbRatio": 60.68529591044776,
    "ptbRatio": 60.68529591044776,
    "evToSales": 37.213879722953756,
    "enterpriseValueOverEBITDA": 108.69219131438065,
    "evToOperatingCashFlow": 131.7635896497706,
    "evToFreeCashFlow": 147.7937330920805,
    "earningsYield": 0.004263854834679534,
    "freeCashFlowYield": 0.006916322076095609,
    "debtToEquity": 1.872326602282704,
    "debtToAssets": 0.29215025480848267,
    "netDebtToEBITDA": 2.359424035443973,
    "currentRatio": 0.8673125765340832,
    "interestCoverage": 0,
    "incomeQuality": 1.8194218241042346,
    "dividendYield": 0.0011006856535776973,
    "payoutRatio": 0.25814332247557004,
    "salesGeneralAndAdministrativeToRevenue": 0,
    "researchAndDdevelopementToRevenue": 0.08179711366269883,
    "intangiblesToTotalAssets": 0,
    "capexToOperatingCashFlow": 0.10846294431390101,
    "capexToRevenue": 0.030633098072263772,
    "capexToDepreciation": 0.9989694263139814,
    "stockBasedCompensationToRevenue": 0.030106394185189088,
    "grahamNumber": 9.057021149912043,
    "roic": 0.09002710785359107,
    "returnOnTangibleAssets": 0.04037481505835936,
    "grahamNetNet": -12.492346093030644,
    "workingCapital": -23405000000,
    "tangibleAssetValue": 56950000000,
    "netCurrentAssetValue": -155043000000,
    "investedCapital": 22275000000,
    "averageReceivables": 54707500000,
    "averagePayables": 58267000000,
    "averageInventory": 6725500000,
    "daysSalesOutstanding": 62.802802064679234,
    "daysPayablesOutstanding": 121.57254510195686,
    "daysOfInventoryOnHand": 12.844802256566961,
    "receivablesTurnover": 1.4330570777289675,
    "payablesTurnover": 0.7402987238979118,
    "inventoryTurnover": 7.0067252264617075,
    "roe": 0.2587532923617208,
    "capexPerShare": 0.1916689900270169
  },
  ...
]
```

https://financialmodelingprep.com/api/v3/ratios/AAPL?period=quarter&apikey=4KbPA5SFE5xuvmiYYxGd0rz163fMbyzN&limit=120

Example response:
```json
[
  {
    "symbol": "AAPL",
    "date": "2024-09-28",
    "calendarYear": "2024",
    "period": "Q4",
    "currentRatio": 0.8673125765340832,
    "quickRatio": 0.8260068483831466,
    "cashRatio": 0.16975259648963673,
    "daysOfSalesOutstanding": 62.802802064679234,
    "daysOfInventoryOutstanding": 12.844802256566961,
    "operatingCycle": 75.64760432124619,
    "daysOfPayablesOutstanding": 121.57254510195686,
    "cashConversionCycle": -45.92494078071067,
    "grossProfitMargin": 0.4622247972190035,
    "operatingProfitMargin": 0.31171389444854103,
    "pretaxProfitMargin": 0.31191404192562944,
    "netProfitMargin": 0.15523016959865163,
    "effectiveTaxRate": 0.5023302938196555,
    "returnOnAssets": 0.04037481505835936,
    "returnOnEquity": 0.2587532923617208,
    "returnOnCapitalEmployed": 0.15690818079623306,
    "netIncomePerEBT": 0.49766970618034445,
    "ebtPerEbit": 1.0006420871210842,
    "ebitPerRevenue": 0.31171389444854103,
    "debtRatio": 0.29215025480848267,
    "debtEquityRatio": 1.872326602282704,
    "longTermDebtToCapitalization": 0.6009110021023125,
    "totalDebtToCapitalization": 0.6518501763673821,
    "interestCoverage": 0,
    "cashFlowToDebtRatio": 0.25144191542638494,
    "companyEquityMultiplier": 6.408779631255487,
    "receivablesTurnover": 1.4330570777289675,
    "payablesTurnover": 0.7402987238979118,
    "inventoryTurnover": 7.0067252264617075,
    "fixedAssetTurnover": 2.0781523642732047,
    "assetTurnover": 0.26009644364074747,
    "operatingCashFlowPerShare": 1.7671379957408355,
    "freeCashFlowPerShare": 1.5754690057138188,
    "cashPerShare": 4.295481344240274,
    "payoutRatio": 0.25814332247557004,
    "operatingCashFlowSalesRatio": 0.28242915832718846,
    "freeCashFlowOperatingCashFlowRatio": 0.891537055686099,
    "cashFlowCoverageRatios": 0.25144191542638494,
    "shortTermCoverageRatios": 1.284113223813401,
    "capitalExpenditureCoverageRatio": 9.219738651994499,
    "dividendPaidAndCapexCoverageRatio": 3.994487485101311,
    "dividendPayoutRatio": 0.25814332247557004,
    "priceBookValueRatio": 60.68529591044776,
    "priceToBookRatio": 60.68529591044776,
    "priceToSalesRatio": 36.406063437269566,
    "priceEarningsRatio": 58.63239010077361,
    "priceToFreeCashFlowsRatio": 144.58551655022382,
    "priceToOperatingCashFlowsRatio": 128.90334572004028,
    "priceCashFlowRatio": 128.90334572004028,
    "priceEarningsToGrowthRatio": -1.9089615381647222,
    "priceSalesRatio": 36.406063437269566,
    "dividendYield": 0.0011006856535776973,
    "enterpriseValueMultiple": 108.69219131438065,
    "priceFairValue": 60.68529591044776
  },
  ...
]
```

2. DB Schema
Update db_creation.sql to create tables for metrics. Create one table that contains all metrics from the above two endpoints (metrics and ratios). Index on <symbol, calendar_year, period>. Make symbol a primary key. 

3. Create a crawler to fetch metrics and insert into the database. 