from .base_fmp_crawler import BaseFMPCrawler
import time
import asyncio
import logging
from tqdm import tqdm


class FinancialCrawler(BaseFMPCrawler):
    async def crawl_income_statement(self, symbol: str):
        if self.skip_existing:
            cursor = self.db.cursor()
            # Check if the record already exists
            cursor.execute("SELECT 1 FROM income_statement WHERE symbol = ? LIMIT 1", 
                           (symbol,))
            if cursor.fetchone():
                return

        data = await self.make_request(
            f'income-statement/',
            {'period': 'quarter', 'limit': 120, 'symbol': symbol}
        )

        if not data:
            logging.warning(f"Failed to fetch income statement for {symbol}")
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
                (symbol, date, filing_date, accepted_date, period, fiscal_year, reported_currency, cik,
                 revenue, cost_of_revenue, gross_profit,
                 research_and_development_expenses, general_and_administrative_expenses,
                 selling_and_marketing_expenses, selling_general_and_administrative_expenses,
                 other_expenses, operating_expenses, cost_and_expenses,
                 net_interest_income, interest_income, interest_expense,
                 depreciation_and_amortization, ebitda, ebit,
                 non_operating_income_excluding_interest, operating_income, total_other_income_expenses_net,
                 income_before_tax, income_tax_expense, net_income_from_continuing_operations,
                 net_income_from_discontinued_operations, other_adjustments_to_net_income,
                 net_income, net_income_deductions, bottom_line_net_income,
                 eps, eps_diluted, weighted_average_shares_outstanding,
                 weighted_average_shares_outstanding_diluted)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                statement.get('date'),
                statement.get('filingDate'),
                statement.get('acceptedDate'),
                statement.get('period'),
                statement.get('fiscalYear'),
                statement.get('reportedCurrency'),
                statement.get('cik'),
                statement.get('revenue'),
                statement.get('costOfRevenue'),
                statement.get('grossProfit'),
                statement.get('researchAndDevelopmentExpenses'),
                statement.get('generalAndAdministrativeExpenses'),
                statement.get('sellingAndMarketingExpenses'),
                statement.get('sellingGeneralAndAdministrativeExpenses'),
                statement.get('otherExpenses'),
                statement.get('operatingExpenses'),
                statement.get('costAndExpenses'),
                statement.get('netInterestIncome'),
                statement.get('interestIncome'),
                statement.get('interestExpense'),
                statement.get('depreciationAndAmortization'),
                statement.get('ebitda'),
                statement.get('ebit'),
                statement.get('nonOperatingIncomeExcludingInterest'),
                statement.get('operatingIncome'),
                statement.get('totalOtherIncomeExpensesNet'),
                statement.get('incomeBeforeTax'),
                statement.get('incomeTaxExpense'),
                statement.get('netIncomeFromContinuingOperations'),
                statement.get('netIncomeFromDiscontinuedOperations'),
                statement.get('otherAdjustmentsToNetIncome'),
                statement.get('netIncome'),
                statement.get('netIncomeDeductions'),
                statement.get('bottomLineNetIncome'),
                statement.get('eps'),
                statement.get('epsDiluted'),
                statement.get('weightedAverageShsOut'),
                statement.get('weightedAverageShsOutDil')
            ))

    async def crawl_balance_sheet(self, symbol: str):
        if self.skip_existing:
            cursor = self.db.cursor()
            # Check if the record already exists
            cursor.execute("SELECT 1 FROM balance_sheet WHERE symbol = ? LIMIT 1", 
                           (symbol,))
            if cursor.fetchone():
                return

        data = await self.make_request(
            f'balance-sheet-statement/',
            {'period': 'quarter', 'limit': 120, 'symbol': symbol}
        )

        if not data:
            logging.warning(f"Failed to fetch balance sheet for {symbol}")
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
                (symbol, date, filing_date, accepted_date, period, fiscal_year, reported_currency, cik,
                 cash_and_cash_equivalents, short_term_investments, cash_and_short_term_investments,
                 net_receivables, accounts_receivables, other_receivables, inventory, prepaids,
                 other_current_assets, total_current_assets, property_plant_equipment_net,
                 goodwill, intangible_assets, goodwill_and_intangible_assets, long_term_investments,
                 tax_assets, other_non_current_assets, total_non_current_assets, other_assets,
                 total_assets, total_payables, account_payables, other_payables, accrued_expenses,
                 short_term_debt, capital_lease_obligations_current, tax_payables, deferred_revenue,
                 other_current_liabilities, total_current_liabilities, long_term_debt,
                 capital_lease_obligations_non_current, deferred_revenue_non_current,
                 deferred_tax_liabilities_non_current, other_non_current_liabilities,
                 total_non_current_liabilities, other_liabilities, capital_lease_obligations,
                 total_liabilities, treasury_stock, preferred_stock, common_stock, retained_earnings,
                 additional_paid_in_capital, accumulated_other_comprehensive_income_loss,
                 other_total_stockholders_equity, total_stockholders_equity, total_equity,
                 minority_interest, total_liabilities_and_total_equity, total_investments,
                 total_debt, net_debt)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                statement.get('date'),
                statement.get('filingDate'),
                statement.get('acceptedDate'),
                statement.get('period'),
                statement.get('fiscalYear'),
                statement.get('reportedCurrency'),
                statement.get('cik'),
                statement.get('cashAndCashEquivalents'),
                statement.get('shortTermInvestments'),
                statement.get('cashAndShortTermInvestments'),
                statement.get('netReceivables'),
                statement.get('accountsReceivables'),
                statement.get('otherReceivables'),
                statement.get('inventory'),
                statement.get('prepaids'),
                statement.get('otherCurrentAssets'),
                statement.get('totalCurrentAssets'),
                statement.get('propertyPlantEquipmentNet'),
                statement.get('goodwill'),
                statement.get('intangibleAssets'),
                statement.get('goodwillAndIntangibleAssets'),
                statement.get('longTermInvestments'),
                statement.get('taxAssets'),
                statement.get('otherNonCurrentAssets'),
                statement.get('totalNonCurrentAssets'),
                statement.get('otherAssets'),
                statement.get('totalAssets'),
                statement.get('totalPayables'),
                statement.get('accountPayables'),
                statement.get('otherPayables'),
                statement.get('accruedExpenses'),
                statement.get('shortTermDebt'),
                statement.get('capitalLeaseObligationsCurrent'),
                statement.get('taxPayables'),
                statement.get('deferredRevenue'),
                statement.get('otherCurrentLiabilities'),
                statement.get('totalCurrentLiabilities'),
                statement.get('longTermDebt'),
                statement.get('capitalLeaseObligationsNonCurrent'),
                statement.get('deferredRevenueNonCurrent'),
                statement.get('deferredTaxLiabilitiesNonCurrent'),
                statement.get('otherNonCurrentLiabilities'),
                statement.get('totalNonCurrentLiabilities'),
                statement.get('otherLiabilities'),
                statement.get('capitalLeaseObligations'),
                statement.get('totalLiabilities'),
                statement.get('treasuryStock'),
                statement.get('preferredStock'),
                statement.get('commonStock'),
                statement.get('retainedEarnings'),
                statement.get('additionalPaidInCapital'),
                statement.get('accumulatedOtherComprehensiveIncomeLoss'),
                statement.get('otherTotalStockholdersEquity'),
                statement.get('totalStockholdersEquity'),
                statement.get('totalEquity'),
                statement.get('minorityInterest'),
                statement.get('totalLiabilitiesAndTotalEquity'),
                statement.get('totalInvestments'),
                statement.get('totalDebt'),
                statement.get('netDebt')
            ))

    async def crawl_cash_flow(self, symbol: str):
        if self.skip_existing:
            cursor = self.db.cursor()
            # Check if the record already exists
            cursor.execute("SELECT 1 FROM cash_flow WHERE symbol = ? LIMIT 1", 
                           (symbol,))
            if cursor.fetchone():
                return

        data = await self.make_request(
            f'cash-flow-statement/',
            {'period': 'quarter', 'limit': 120, 'symbol': symbol}
        )

        if not data:
            logging.warning(f"Failed to fetch cash flow for {symbol}")
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
                (symbol, date, filing_date, accepted_date, period, fiscal_year, reported_currency, cik,
                 net_income, depreciation_and_amortization, deferred_income_tax,
                 stock_based_compensation, change_in_working_capital,
                 accounts_receivables, inventory, accounts_payables, other_working_capital,
                 other_non_cash_items, net_cash_provided_by_operating_activities,
                 investments_in_property_plant_and_equipment,
                 acquisitions_net, purchases_of_investments,
                 sales_maturities_of_investments, other_investing_activities,
                 net_cash_provided_by_investing_activities,
                 net_debt_issuance, long_term_net_debt_issuance, short_term_net_debt_issuance,
                 net_stock_issuance, net_common_stock_issuance, common_stock_issuance,
                 common_stock_repurchased, net_preferred_stock_issuance,
                 net_dividends_paid, common_dividends_paid, preferred_dividends_paid,
                 other_financing_activities, net_cash_provided_by_financing_activities,
                 effect_of_forex_changes_on_cash, net_change_in_cash,
                 cash_at_end_of_period, cash_at_beginning_of_period,
                 operating_cash_flow, capital_expenditure, free_cash_flow,
                 income_taxes_paid, interest_paid)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                symbol,
                statement.get('date'),
                statement.get('filingDate'),
                statement.get('acceptedDate'),
                statement.get('period'),
                statement.get('fiscalYear'),
                statement.get('reportedCurrency'),
                statement.get('cik'),
                statement.get('netIncome'),
                statement.get('depreciationAndAmortization'),
                statement.get('deferredIncomeTax'),
                statement.get('stockBasedCompensation'),
                statement.get('changeInWorkingCapital'),
                statement.get('accountsReceivables'),
                statement.get('inventory'),
                statement.get('accountsPayables'),
                statement.get('otherWorkingCapital'),
                statement.get('otherNonCashItems'),
                statement.get('netCashProvidedByOperatingActivities'),
                statement.get('investmentsInPropertyPlantAndEquipment'),
                statement.get('acquisitionsNet'),
                statement.get('purchasesOfInvestments'),
                statement.get('salesMaturitiesOfInvestments'),
                statement.get('otherInvestingActivities'),
                statement.get('netCashProvidedByInvestingActivities'),
                statement.get('netDebtIssuance'),
                statement.get('longTermNetDebtIssuance'),
                statement.get('shortTermNetDebtIssuance'),
                statement.get('netStockIssuance'),
                statement.get('netCommonStockIssuance'),
                statement.get('commonStockIssuance'),
                statement.get('commonStockRepurchased'),
                statement.get('netPreferredStockIssuance'),
                statement.get('netDividendsPaid'),
                statement.get('commonDividendsPaid'),
                statement.get('preferredDividendsPaid'),
                statement.get('otherFinancingActivities'),
                statement.get('netCashProvidedByFinancingActivities'),
                statement.get('effectOfForexChangesOnCash'),
                statement.get('netChangeInCash'),
                statement.get('cashAtEndOfPeriod'),
                statement.get('cashAtBeginningOfPeriod'),
                statement.get('operatingCashFlow'),
                statement.get('capitalExpenditure'),
                statement.get('freeCashFlow'),
                statement.get('incomeTaxesPaid'),
                statement.get('interestPaid')
            ))

    async def crawl(self):
        logging.info("Starting financial crawling...")

        symbols = self.get_symbols_to_crawl()

        for symbol in tqdm(symbols, desc="Crawling financials"):
            await self.crawl_income_statement(symbol)
            await self.crawl_balance_sheet(symbol)
            await self.crawl_cash_flow(symbol)
            self.db.commit()


async def main():
    crawler = FinancialCrawler()
    try:
        await crawler.crawl()
    finally:
        crawler.close()

if __name__ == "__main__":
    asyncio.run(main())
