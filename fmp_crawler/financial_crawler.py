from .base_fmp_crawler import BaseFMPCrawler
import time
import asyncio
from tqdm import tqdm


class FinancialCrawler(BaseFMPCrawler):
    def get_symbols_to_crawl(self):
        cursor = self.db.cursor()
        cursor.execute('''
            SELECT symbol FROM stock_symbol 
            WHERE exchange_short_name IN ('NYSE', 'NASDAQ', 'AMEX')
            AND type = 'stock'
        ''')
        return [row['symbol'] for row in cursor.fetchall()]

    async def crawl_income_statement(self, symbol: str):
        data = await self.make_request(
            f'income-statement/{symbol}',
            {'period': 'quarter', 'limit': 120}
        )

        if not data:
            self.logger.error(f"Failed to fetch income statement for {symbol}")
            return

        cursor = self.db.cursor()
        for statement in data:
            self.check_missing_fields(
                statement,
                ['date', 'period', 'revenue', 'netIncome', 'eps'],
                symbol
            )

            cursor.execute('''
                INSERT OR REPLACE INTO income_statement 
                (symbol, date, period, calendar_year, reported_currency,
                 revenue, cost_of_revenue, gross_profit, gross_profit_ratio,
                 research_and_development_expenses, general_and_administrative_expenses,
                 selling_and_marketing_expenses, selling_general_and_administrative_expenses,
                 operating_expenses, operating_income, operating_income_ratio,
                 ebitda, ebitda_ratio, net_income, net_income_ratio,
                 eps, eps_diluted, weighted_average_shares_outstanding,
                 weighted_average_shares_outstanding_diluted, link)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, statement.get('date'), statement.get('period'),
                statement.get('calendarYear'), statement.get(
                    'reportedCurrency'),
                statement.get('revenue'), statement.get('costOfRevenue'),
                statement.get('grossProfit'), statement.get(
                    'grossProfitRatio'),
                statement.get('researchAndDevelopmentExpenses'),
                statement.get('generalAndAdministrativeExpenses'),
                statement.get('sellingAndMarketingExpenses'),
                statement.get('sellingGeneralAndAdministrativeExpenses'),
                statement.get('operatingExpenses'), statement.get(
                    'operatingIncome'),
                statement.get(
                    'operatingIncomeRatio'), statement.get('ebitda'),
                statement.get('ebitdaratio'), statement.get('netIncome'),
                statement.get('netIncomeRatio'), statement.get('eps'),
                statement.get('epsdiluted'),
                statement.get('weightedAverageShsOut'),
                statement.get('weightedAverageShsOutDil'),
                statement.get('link')
            ))

    async def crawl_balance_sheet(self, symbol: str):
        data = await self.make_request(
            f'balance-sheet-statement/{symbol}',
            {'period': 'quarter', 'limit': 120}
        )

        if not data:
            self.logger.error(f"Failed to fetch balance sheet for {symbol}")
            return

        cursor = self.db.cursor()
        for statement in data:
            self.check_missing_fields(
                statement,
                ['date', 'period', 'totalAssets', 'totalLiabilities'],
                symbol
            )

            cursor.execute('''
                INSERT OR REPLACE INTO balance_sheet 
                (symbol, date, period, calendar_year, reported_currency,
                 cash_and_cash_equivalents, short_term_investments,
                 cash_and_short_term_investments, net_receivables, inventory,
                 total_current_assets, property_plant_equipment_net,
                 goodwill, intangible_assets, long_term_investments,
                 total_non_current_assets, total_assets, accounts_payables,
                 short_term_debt, total_current_liabilities, long_term_debt,
                 total_non_current_liabilities, total_liabilities,
                 total_stockholders_equity, total_equity,
                 total_liabilities_and_stockholders_equity,
                 total_investments, total_debt, net_debt, link)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, statement.get('date'), statement.get('period'),
                statement.get('calendarYear'), statement.get(
                    'reportedCurrency'),
                statement.get('cashAndCashEquivalents'),
                statement.get('shortTermInvestments'),
                statement.get('cashAndShortTermInvestments'),
                statement.get('netReceivables'), statement.get(
                    'inventory'),
                statement.get('totalCurrentAssets'),
                statement.get('propertyPlantEquipmentNet'),
                statement.get('goodwill'), statement.get(
                    'intangibleAssets'),
                statement.get('longTermInvestments'),
                statement.get('totalNonCurrentAssets'),
                statement.get('totalAssets'), statement.get(
                    'accountPayables'),
                statement.get('shortTermDebt'),
                statement.get('totalCurrentLiabilities'),
                statement.get('longTermDebt'),
                statement.get('totalNonCurrentLiabilities'),
                statement.get('totalLiabilities'),
                statement.get('totalStockholdersEquity'),
                statement.get('totalEquity'),
                statement.get('totalLiabilitiesAndStockholdersEquity'),
                statement.get('totalInvestments'),
                statement.get('totalDebt'), statement.get('netDebt'),
                statement.get('link')
            ))

    async def crawl_cash_flow(self, symbol: str):
        data = await self.make_request(
            f'cash-flow-statement/{symbol}',
            {'period': 'quarter', 'limit': 120}
        )

        if not data:
            self.logger.error(f"Failed to fetch cash flow for {symbol}")
            return

        cursor = self.db.cursor()
        for statement in data:
            self.check_missing_fields(
                statement,
                ['date', 'period', 'netIncome', 'operatingCashFlow'],
                symbol
            )

            cursor.execute('''
                INSERT OR REPLACE INTO cash_flow 
                (symbol, date, period, calendar_year, reported_currency,
                 net_income, depreciation_and_amortization,
                 stock_based_compensation, change_in_working_capital,
                 accounts_receivables, inventory, accounts_payables,
                 net_cash_provided_by_operating_activities,
                 investments_in_property_plant_and_equipment,
                 acquisitions_net, purchases_of_investments,
                 sales_maturities_of_investments,
                 net_cash_used_for_investing_activities,
                 debt_repayment, common_stock_repurchased,
                 dividends_paid,
                 net_cash_used_provided_by_financing_activities,
                 net_change_in_cash, operating_cash_flow,
                 capital_expenditure, free_cash_flow, link)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol, statement.get('date'), statement.get('period'),
                statement.get('calendarYear'), statement.get(
                    'reportedCurrency'),
                statement.get('netIncome'),
                statement.get('depreciationAndAmortization'),
                statement.get('stockBasedCompensation'),
                statement.get('changeInWorkingCapital'),
                statement.get('accountsReceivables'),
                statement.get('inventory'), statement.get(
                    'accountsPayables'),
                statement.get('netCashProvidedByOperatingActivities'),
                statement.get('investmentsInPropertyPlantAndEquipment'),
                statement.get('acquisitionsNet'),
                statement.get('purchasesOfInvestments'),
                statement.get('salesMaturitiesOfInvestments'),
                statement.get('netCashUsedForInvestingActivites'),
                statement.get('debtRepayment'),
                statement.get('commonStockRepurchased'),
                statement.get('dividendsPaid'),
                statement.get('netCashUsedProvidedByFinancingActivities'),
                statement.get('netChangeInCash'),
                statement.get('operatingCashFlow'),
                statement.get('capitalExpenditure'),
                statement.get('freeCashFlow'),
                statement.get('link')
            ))

    async def crawl(self):
        self.logger.info("Starting financial crawling...")
        start_time = time.time()

        symbols = self.get_symbols_to_crawl()

        for symbol in tqdm(symbols, desc="Crawling financials"):
            await self.crawl_income_statement(symbol)
            await self.crawl_balance_sheet(symbol)
            await self.crawl_cash_flow(symbol)
            self.db.commit()

        elapsed = time.time() - start_time
        self.logger.info(
            f"Financial crawling completed in {elapsed:.2f} seconds")


async def main():
    crawler = FinancialCrawler()
    try:
        await crawler.crawl()
    finally:
        crawler.close()

if __name__ == "__main__":
    asyncio.run(main())
