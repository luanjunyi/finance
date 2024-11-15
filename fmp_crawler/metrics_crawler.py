import asyncio
from typing import Dict, List, Optional
from datetime import datetime
import logging
import time
from tqdm import tqdm

from .base_fmp_crawler import BaseFMPCrawler


class MetricsCrawler(BaseFMPCrawler):
    async def crawl_symbol_metrics(self, symbol: str):
        metrics = await self.make_request(
            f'key-metrics/{symbol}',
            {'period': 'quarter', 'limit': 120}
        )

        ratios = await self.make_request(
            f'ratios/{symbol}',
            {'period': 'quarter', 'limit': 120}
        )

        if not metrics or not ratios:
            self.logger.error(f"Failed to fetch metrics/ratios for {symbol}")
            with open('missing_data.log', 'a') as log_file:
                if not metrics:
                    log_file.write(f"{symbol}: Missing metrics\n")
                if not ratios:
                    log_file.write(f"{symbol}: Missing ratios\n")
            return

        cursor = self.db.cursor()

        # Combine metrics and ratios by date
        metrics_dict = {m['date']: m for m in metrics}
        for ratio in ratios:
            date = ratio['date']
            if date in metrics_dict:
                combined = metrics_dict[date]
                # Add non-duplicate ratio fields
                for key, value in ratio.items():
                    if key not in combined:
                        combined[key] = value

                try:
                    # Insert combined metrics
                    cursor.execute('''
                        INSERT OR REPLACE INTO metrics (
                            symbol, date, calendar_year, period,
                            revenue_per_share, net_income_per_share,
                            operating_cash_flow_per_share, free_cash_flow_per_share,
                            cash_per_share, book_value_per_share,
                            tangible_book_value_per_share, shareholders_equity_per_share,
                            interest_debt_per_share, debt_to_equity, debt_to_assets,
                            net_debt_to_EBITDA, current_ratio, interest_coverage,
                            income_quality, payout_ratio, sga_to_revenue,
                            rnd_to_revenue, intangibles_to_total_assets,
                            capex_to_operating_cash_flow, capex_to_revenue,
                            capex_to_depreciation, stock_based_compensation_to_revenue,
                            graham_number, roic, return_on_tangible_assets,
                            graham_net_net, working_capital, tangible_asset_value,
                            net_current_asset_value, invested_capital,
                            days_sales_outstanding, days_payables_outstanding,
                            days_inventory_onhand, receivables_turnover,
                            payables_turnover, inventory_turnover, roe,
                            capex_per_share, quick_ratio, cash_ratio,
                            operating_cycle, cash_conversion_cycle,
                            gross_profit_margin, operating_profit_margin,
                            pretax_profit_margin, net_profit_margin,
                            effective_tax_rate, return_on_assets,
                            return_on_capital_employed, net_income_per_ebt,
                            ebt_per_ebit, ebit_per_revenue, debt_ratio,
                            long_term_debt_to_capitalization, total_debt_to_capitalization,
                            cash_flow_to_debt_ratio, company_equity_multiplier,
                            fixed_asset_turnover, asset_turnover,
                            operating_cash_flow_sales_ratio,
                            free_cash_flow_operating_cash_flow_ratio,
                            cash_flow_coverage_ratios, short_term_coverage_ratios,
                            capital_expenditure_coverage_ratio,
                            dividend_paid_and_capex_coverage_ratio,
                            price_fair_value
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        combined.get('date'),
                        combined.get('calendarYear'),
                        combined.get('period'),
                        combined.get('revenuePerShare'),
                        combined.get('netIncomePerShare'),
                        combined.get('operatingCashFlowPerShare'),
                        combined.get('freeCashFlowPerShare'),
                        combined.get('cashPerShare'),
                        combined.get('bookValuePerShare'),
                        combined.get('tangibleBookValuePerShare'),
                        combined.get('shareholdersEquityPerShare'),
                        combined.get('interestDebtPerShare'),
                        combined.get('debtToEquity'),
                        combined.get('debtToAssets'),
                        combined.get('netDebtToEBITDA'),
                        combined.get('currentRatio'),
                        combined.get('interestCoverage'),
                        combined.get('incomeQuality'),
                        combined.get('payoutRatio'),
                        combined.get('salesGeneralAndAdministrativeToRevenue'),
                        combined.get('researchAndDevelopmentToRevenue'),
                        combined.get('intangiblesToTotalAssets'),
                        combined.get('capexToOperatingCashFlow'),
                        combined.get('capexToRevenue'),
                        combined.get('capexToDepreciation'),
                        combined.get('stockBasedCompensationToRevenue'),
                        combined.get('grahamNumber'),
                        combined.get('roic'),
                        combined.get('returnOnTangibleAssets'),
                        combined.get('grahamNetNet'),
                        combined.get('workingCapital'),
                        combined.get('tangibleAssetValue'),
                        combined.get('netCurrentAssetValue'),
                        combined.get('investedCapital'),
                        combined.get('daysSalesOutstanding'),
                        combined.get('daysPayablesOutstanding'),
                        combined.get('daysOfInventoryOnHand'),
                        combined.get('receivablesTurnover'),
                        combined.get('payablesTurnover'),
                        combined.get('inventoryTurnover'),
                        combined.get('roe'),
                        combined.get('capexPerShare'),
                        combined.get('quickRatio'),
                        combined.get('cashRatio'),
                        combined.get('operatingCycle'),
                        combined.get('cashConversionCycle'),
                        combined.get('grossProfitMargin'),
                        combined.get('operatingProfitMargin'),
                        combined.get('pretaxProfitMargin'),
                        combined.get('netProfitMargin'),
                        combined.get('effectiveTaxRate'),
                        combined.get('returnOnAssets'),
                        combined.get('returnOnCapitalEmployed'),
                        combined.get('netIncomePerEBT'),
                        combined.get('ebtPerEbit'),
                        combined.get('ebitPerRevenue'),
                        combined.get('debtRatio'),
                        combined.get('longTermDebtToCapitalization'),
                        combined.get('totalDebtToCapitalization'),
                        combined.get('cashFlowToDebtRatio'),
                        combined.get('companyEquityMultiplier'),
                        combined.get('fixedAssetTurnover'),
                        combined.get('assetTurnover'),
                        combined.get('operatingCashFlowSalesRatio'),
                        combined.get('freeCashFlowOperatingCashFlowRatio'),
                        combined.get('cashFlowCoverageRatios'),
                        combined.get('shortTermCoverageRatios'),
                        combined.get('capitalExpenditureCoverageRatio'),
                        combined.get('dividendPaidAndCapexCoverageRatio'),
                        combined.get('priceFairValue')
                    ))
                except Exception as e:
                    self.logger.error(
                        f"Error inserting metrics for {symbol} on {date}: {str(e)}")

    async def crawl(self):
        self.logger.info("Starting metrics crawling...")
        start_time = time.time()

        symbols = self.get_symbols_to_crawl()

        for symbol in tqdm(symbols, desc="Crawling metrics"):
            await self.crawl_symbol_metrics(symbol)
            self.db.commit()

        elapsed = time.time() - start_time
        self.logger.info(
            f"Metrics crawling completed in {elapsed:.2f} seconds")


async def main():
    crawler = MetricsCrawler()
    await crawler.crawl()
    crawler.close()

if __name__ == "__main__":
    asyncio.run(main())
