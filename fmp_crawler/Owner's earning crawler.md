# Goals
- Crawl growth metrics using FMP API
- Crawl owner's earnings using FMP API

# API examples

*cash flow growth*
https://financialmodelingprep.com/api/v3/cash-flow-statement-growth/AAPL?period=quarter&apikey=4KbPA5SFE5xuvmiYYxGd0rz163fMbyzN&limit=120

Response:
```json
[
  {
    "date": "2024-09-28",
    "symbol": "AAPL",
    "calendarYear": "2024",
    "period": "Q4",
    "growthNetIncome": -0.31294293174188736,
    "growthDepreciationAndAmortization": 0.021403508771929824,
    "growthDeferredIncomeTax": 0,
    "growthStockBasedCompensation": -0.0038340885325897525,
    "growthChangeInWorkingCapital": 44.57241379310345,
    "growthAccountsReceivables": -21.272815533980584,
    "growthInventory": -89.58333333333333,
    "growthAccountsPayables": 12.769330734243015,
    "growthOtherWorkingCapital": 3.195912927587739,
    "growthOtherNonCashItems": -1.1953428201811125,
    "growthNetCashProvidedByOperatingActivites": -0.07093353662762492,
    "growthInvestmentsInPropertyPlantAndEquipment": -0.3519293351929335,
    "growthAcquisitionsNet": 0,
    "growthPurchasesOfInvestments": 0.1879987722529159,
    "growthSalesMaturitiesOfInvestments": -0.020590520590520592,
    "growthOtherInvestingActivites": 0.5077319587628866,
    "growthNetCashUsedForInvestingActivites": 12.377952755905511,
    "growthDebtRepayment": -0.348601291115893,
    "growthCommonStockIssued": 0,
    "growthCommonStockRepurchased": 0.05425684337531106,
    "growthDividendsPaid": 0.02336328626444159,
    "growthOtherFinancingActivites": 0.8091180230080954,
    "growthNetCashUsedProvidedByFinancingActivities": 0.3073270955382181,
    "growthEffectOfForexChangesOnCash": 0,
    "growthNetChangeInCash": 1.351653024343574,
    "growthCashAtEndOfPeriod": 0.22146528514318348,
    "growthCashAtBeginningOfPeriod": -0.2147931959553079,
    "growthOperatingCashFlow": -0.07093353662762492,
    "growthCapitalExpenditure": -0.3519293351929335,
    "growthFreeCashFlow": -0.10499120080877672
  },
  ...
]
```

*income statement growth*

https://financialmodelingprep.com/api/v3/income-statement-growth/AAPL?period=quarter&apikey=4KbPA5SFE5xuvmiYYxGd0rz163fMbyzN&limit=120

Response:

```json
[
  {
    "date": "2024-09-28",
    "symbol": "AAPL",
    "calendarYear": "2024",
    "period": "FY",
    "growthRevenue": 0.020219940775141214,
    "growthCostOfRevenue": -0.017675600199872046,
    "growthGrossProfit": 0.06819471705252206,
    "growthGrossProfitRatio": 0.04702395474011335,
    "growthResearchAndDevelopmentExpenses": 0.04863780712017383,
    "growthGeneralAndAdministrativeExpenses": 0,
    "growthSellingAndMarketingExpenses": 0,
    "growthOtherExpenses": -1,
    "growthOperatingExpenses": 0.04776924900176856,
    "growthCostAndExpenses": -0.004331112631234571,
    "growthInterestExpense": -1,
    "growthDepreciationAndAmortization": -0.006424168764649709,
    "growthEBITDA": 0.07026704816404387,
    "growthEBITDARatio": 0.049055213868137715,
    "growthOperatingIncome": 0.07799581805933456,
    "growthOperatingIncomeRatio": 0.05663080556714364,
    "growthTotalOtherIncomeExpensesNet": 1.4761061946902654,
    "growthIncomeBeforeTax": 0.08571604417246959,
    "growthIncomeBeforeTaxRatio": 0.06419802344984436,
    "growthIncomeTaxExpense": 0.7770145152619318,
    "growthNetIncome": -0.033599670086086914,
    "growthNetIncomeRatio": -0.052752949185731576,
    "growthEPS": -0.008116883116883088,
    "growthEPSDiluted": -0.008156606851549727,
    "growthWeightedAverageShsOut": -0.02543458616683152,
    "growthWeightedAverageShsOutDil": -0.02557791606880283
  },
  ...
]
```

*balance sheet growth*

https://financialmodelingprep.com/api/v3/balance-sheet-statement-growth/AAPL?period=quarter&apikey=4KbPA5SFE5xuvmiYYxGd0rz163fMbyzN&limit=120

Response:

```json
[
  {
    "date": "2024-09-28",
    "symbol": "AAPL",
    "calendarYear": "2024",
    "period": "Q4",
    "growthCashAndCashEquivalents": 0.17124975552513202,
    "growthShortTermInvestments": -0.02781763991610553,
    "growthCashAndShortTermInvestments": 0.05452986197634342,
    "growthNetReceivables": 0.5343972945427592,
    "growthInventory": 0.18183292781832927,
    "growthOtherCurrentAssets": -0.0006994474365251451,
    "growthTotalCurrentAssets": 0.21965161238888667,
    "growthPropertyPlantEquipmentNet": 0.026470720417059907,
    "growthGoodwill": 0,
    "growthIntangibleAssets": 0,
    "growthGoodwillAndIntangibleAssets": 0,
    "growthLongTermInvestments": 0.00261946514686541,
    "growthTaxAssets": 0,
    "growthOtherNonCurrentAssets": -0.2143820543763754,
    "growthTotalNonCurrentAssets": 0.028208772074479693,
    "growthOtherAssets": 0,
    "growthTotalAssets": 0.1006236203756197,
    "growthAccountPayables": 0.449531256568714,
    "growthShortTermDebt": 0.3819830553349219,
    "growthTaxPayables": 0,
    "growthDeferredRevenue": 0.024338755743201292,
    "growthOtherCurrentLiabilities": -0.17766755900080475,
    "growthTotalCurrentLiabilities": 0.3401203427946271,
    "growthLongTermDebt": -0.00517425402570885,
    "growthDeferredRevenueNonCurrent": 0,
    "growthDeferrredTaxLiabilitiesNonCurrent": 0,
    "growthOtherNonCurrentLiabilities": -0.25473621612437347,
    "growthTotalNonCurrentLiabilities": -0.012319927971188475,
    "growthOtherLiabilities": 0,
    "growthTotalLiabilities": 0.1627985987376559,
    "growthCommonStock": 0.04290544771446462,
    "growthRetainedEarnings": -3.0528988573846805,
    "growthAccumulatedOtherComprehensiveIncomeLoss": 0.14781368821292776,
    "growthOthertotalStockholdersEquity": 0,
    "growthTotalStockholdersEquity": -0.14627930682976553,
    "growthTotalLiabilitiesAndStockholdersEquity": 0.1006236203756197,
    "growthTotalInvestments": -0.006032508079952305,
    "growthTotalDebt": 0.052564558161573086,
    "growthNetDebt": 0.012503465849826378
  },
  ...
]
```

*financial growth*

https://financialmodelingprep.com/api/v3/financial-growth/AAPL?period=quarter&apikey=4KbPA5SFE5xuvmiYYxGd0rz163fMbyzN&limit=120

Response:

```json
[
  {
    "symbol": "AAPL",
    "date": "2024-09-28",
    "calendarYear": "2024",
    "period": "Q4",
    "revenueGrowth": 0.10670692609907084,
    "grossProfitGrowth": 0.1058773123645345,
    "ebitgrowth": 0.16720574313663616,
    "operatingIncomeGrowth": 0.16720574313663616,
    "netIncomeGrowth": -0.31294293174188736,
    "epsgrowth": -0.3071428571428571,
    "epsdilutedGrowth": -0.3071428571428571,
    "weightedAverageSharesGrowth": -0.009661227154046997,
    "weightedAverageSharesDilutedGrowth": -0.006862183940435915,
    "dividendsperShareGrowth": -0.013835729233359888,
    "operatingCashFlowGrowth": -0.07093353662762492,
    "freeCashFlowGrowth": -0.10499120080877672,
    "tenYRevenueGrowthPerShare": 2.5256388904617553,
    "fiveYRevenueGrowthPerShare": 0.7550701329182128,
    "threeYRevenueGrowthPerShare": 0.23750810611953302,
    "tenYOperatingCFGrowthPerShare": 2.1655616484035556,
    "fiveYOperatingCFGrowthPerShare": 0.5943514850685873,
    "threeYOperatingCFGrowthPerShare": 0.44232762175626916,
    "tenYNetIncomeGrowthPerShare": 1.7227244692969284,
    "fiveYNetIncomeGrowthPerShare": 0.27480980522889353,
    "threeYNetIncomeGrowthPerShare": -0.22080002294104545,
    "tenYShareholdersEquityGrowthPerShare": -0.20128935048338498,
    "fiveYShareholdersEquityGrowthPerShare": -0.2548476815498045,
    "threeYShareholdersEquityGrowthPerShare": -0.0190757333397355,
    "tenYDividendperShareGrowthPerShare": 1.103590506264014,
    "fiveYDividendperShareGrowthPerShare": 0.29457852213629754,
    "threeYDividendperShareGrowthPerShare": 0.1356418855251837,
    "receivablesGrowth": 0.5343972945427592,
    "inventoryGrowth": 0.18183292781832927,
    "assetGrowth": 0.1006236203756197,
    "bookValueperShareGrowth": -0.13795085421437844,
    "debtGrowth": 0.052564558161573086,
    "rdexpenseGrowth": -0.03010242318261304,
    "sgaexpensesGrowth": 0.03212025316455696
  },
  ...
]
```

*Owner's earnings*

https://financialmodelingprep.com/api/v4/owner_earnings?symbol=AAPL&apikey=4KbPA5SFE5xuvmiYYxGd0rz163fMbyzN

Response:

```json
[
  {
    "symbol": "AAPL",
    "date": "2024-09-28",
    "averagePPE": 0.13969,
    "maintenanceCapex": -2149203920,
    "ownersEarnings": 24661796080,
    "growthCapex": -758796080,
    "ownersEarningsPerShare": 1.62
  },
  {
    "symbol": "AAPL",
    "date": "2024-06-29",
    "averagePPE": 0.13969,
    "maintenanceCapex": -1595033800,
    "ownersEarnings": 27262966200,
    "growthCapex": -555966200,
    "ownersEarningsPerShare": 1.78
  },
  ...
]
```

# Implementations

1. Update db_creation.sql to create new tables.
- use financial_growth table to store
  * financial growth metrics
  * income statement growth
  * balance sheet growth
  * cash flow growth
- use owner_earnings table to store
  * owner's earnings 

The table schema should be created according to the response json examples above.

2. Create a new crawler to crawl the growth metrics. Name it GrowthMetricsCrawler. It should reuse the structures of other FMP crawlers including MetricsCrawler, PriceCrawler, and SymbolCrawler.

3. Update fmp_crawl.py to add the new crawler to the overall crawling process.
