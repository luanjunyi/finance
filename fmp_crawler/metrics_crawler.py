import asyncio
import logging
from tqdm import tqdm

from .base_fmp_crawler import BaseFMPCrawler


class MetricsCrawler(BaseFMPCrawler):
    async def crawl_symbol_metrics(self, symbol: str):
        metrics = await self.make_request(
            f'key-metrics',
            {'period': 'quarter', 'limit': 120, 'symbol': symbol}
        )

        ratios = await self.make_request(
            f'ratios',
            {'period': 'quarter', 'limit': 120, 'symbol': symbol}
        )

        if not metrics or not ratios:
            logging.error(f"Failed to fetch metrics/ratios for {symbol}")
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
                            symbol, date, fiscal_year, period, reported_currency,
                            gross_profit_margin, ebitda_margin, ebit_margin,
                            operating_profit_margin, pretax_profit_margin, net_profit_margin,
                            continuous_operations_profit_margin, bottom_line_profit_margin,
                            receivables_turnover, payables_turnover, inventory_turnover,
                            fixed_asset_turnover, asset_turnover, working_capital_turnover_ratio,
                            current_ratio, quick_ratio, cash_ratio, solvency_ratio,
                            price_to_earnings_ratio, price_to_earnings_growth_ratio,
                            forward_price_to_earnings_growth_ratio, price_to_book_ratio,
                            price_to_sales_ratio, price_to_free_cash_flow_ratio,
                            price_to_operating_cash_flow_ratio, 
                            price_to_fair_value, enterprise_value_multiple, ev_to_sales,
                            ev_to_operating_cash_flow, ev_to_free_cash_flow, ev_to_ebitda,
                            debt_to_assets_ratio, debt_to_equity_ratio, debt_to_capital_ratio,
                            long_term_debt_to_capital_ratio, financial_leverage_ratio,
                            debt_to_market_cap, net_debt_to_ebitda,
                            operating_cash_flow_ratio, operating_cash_flow_sales_ratio,
                            free_cash_flow_operating_cash_flow_ratio, debt_service_coverage_ratio,
                            interest_coverage_ratio, short_term_operating_cash_flow_coverage_ratio,
                            operating_cash_flow_coverage_ratio, capital_expenditure_coverage_ratio, 
                            dividend_paid_and_capex_coverage_ratio,
                            dividend_payout_ratio, dividend_yield, dividend_yield_percentage,
                            dividend_per_share, revenue_per_share, net_income_per_share,
                            interest_debt_per_share, cash_per_share, book_value_per_share,
                            tangible_book_value_per_share, shareholders_equity_per_share,
                            operating_cash_flow_per_share, capex_per_share, free_cash_flow_per_share,
                            net_income_per_ebt, ebt_per_ebit, effective_tax_rate,
                            income_quality, return_on_assets, operating_return_on_assets, return_on_equity,
                            return_on_invested_capital, return_on_tangible_assets,
                            return_on_capital_employed, earnings_yield, free_cash_flow_yield,
                            tax_burden, interest_burden, working_capital, invested_capital,
                            tangible_asset_value, net_current_asset_value, graham_number,
                            graham_net_net, days_sales_outstanding,
                            days_payables_outstanding, days_of_inventory_outstanding,
                            operating_cycle, cash_conversion_cycle, sga_to_revenue,
                            stock_based_compensation_to_revenue, research_and_developement_to_revenue,
                            capex_to_revenue, intangibles_to_total_assets,
                            capex_to_operating_cash_flow, capex_to_depreciation,
                            sales_general_and_administrative_to_revenue,
                            market_cap, enterprise_value, average_receivables,
                            average_payables, average_inventory, free_cash_flow_to_equity,
                            free_cash_flow_to_firm
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        symbol,
                        combined.get('date'),
                        combined.get('fiscalYear'),
                        combined.get('period'),
                        combined.get('reportedCurrency'),
                        combined.get('grossProfitMargin'),
                        combined.get('ebitdaMargin'),
                        combined.get('ebitMargin'),
                        combined.get('operatingProfitMargin'),
                        combined.get('pretaxProfitMargin'),
                        combined.get('netProfitMargin'),
                        combined.get('continuousOperationsProfitMargin'),
                        combined.get('bottomLineProfitMargin'),
                        combined.get('receivablesTurnover'),
                        combined.get('payablesTurnover'),
                        combined.get('inventoryTurnover'),
                        combined.get('fixedAssetTurnover'),
                        combined.get('assetTurnover'),
                        combined.get('workingCapitalTurnoverRatio'),
                        combined.get('currentRatio'),
                        combined.get('quickRatio'),
                        combined.get('cashRatio'),
                        combined.get('solvencyRatio'),
                        combined.get('priceToEarningsRatio'),
                        combined.get('priceToEarningsGrowthRatio'),
                        combined.get('forwardPriceToEarningsGrowthRatio'),
                        combined.get('priceToBookRatio'),
                        combined.get('priceToSalesRatio'),
                        combined.get('priceToFreeCashFlowRatio'),
                        combined.get('priceToOperatingCashFlowRatio'),
                        combined.get('priceToFairValue'),
                        combined.get('enterpriseValueMultiple'),
                        combined.get('evToSales'),
                        combined.get('evToOperatingCashFlow'),
                        combined.get('evToFreeCashFlow'),
                        combined.get('evToEBITDA'),
                        combined.get('debtToAssetsRatio'),
                        combined.get('debtToEquityRatio'),
                        combined.get('debtToCapitalRatio'),
                        combined.get('longTermDebtToCapitalRatio'),
                        combined.get('financialLeverageRatio'),
                        combined.get('debtToMarketCap'),
                        combined.get('netDebtToEBITDA'),
                        combined.get('operatingCashFlowRatio'),
                        combined.get('operatingCashFlowSalesRatio'),
                        combined.get('freeCashFlowOperatingCashFlowRatio'),
                        combined.get('debtServiceCoverageRatio'),
                        combined.get('interestCoverageRatio'),
                        combined.get('shortTermOperatingCashFlowCoverageRatio'),
                        combined.get('operatingCashFlowCoverageRatio'),
                        combined.get('capitalExpenditureCoverageRatio'),
                        combined.get('dividendPaidAndCapexCoverageRatio'),
                        combined.get('dividendPayoutRatio'),
                        combined.get('dividendYield'),
                        combined.get('dividendYieldPercentage'),
                        combined.get('dividendPerShare'),
                        combined.get('revenuePerShare'),
                        combined.get('netIncomePerShare'),
                        combined.get('interestDebtPerShare'),
                        combined.get('cashPerShare'),
                        combined.get('bookValuePerShare'),
                        combined.get('tangibleBookValuePerShare'),
                        combined.get('shareholdersEquityPerShare'),
                        combined.get('operatingCashFlowPerShare'),
                        combined.get('capexPerShare'),
                        combined.get('freeCashFlowPerShare'),
                        combined.get('netIncomePerEBT'),
                        combined.get('ebtPerEbit'),
                        combined.get('effectiveTaxRate'),
                        combined.get('incomeQuality'),
                        combined.get('returnOnAssets'),
                        combined.get('operatingReturnOnAssets'),
                        combined.get('returnOnEquity'),
                        combined.get('returnOnInvestedCapital'),
                        combined.get('returnOnTangibleAssets'),
                        combined.get('returnOnCapitalEmployed'),
                        combined.get('earningsYield'),
                        combined.get('freeCashFlowYield'),
                        combined.get('taxBurden'),
                        combined.get('interestBurden'),
                        combined.get('workingCapital'),
                        combined.get('investedCapital'),
                        combined.get('tangibleAssetValue'),
                        combined.get('netCurrentAssetValue'),
                        combined.get('grahamNumber'),
                        combined.get('grahamNetNet'),
                        combined.get('daysOfSalesOutstanding'),
                        combined.get('daysOfPayablesOutstanding'),
                        combined.get('daysOfInventoryOutstanding'),
                        combined.get('operatingCycle'),
                        combined.get('cashConversionCycle'),
                        combined.get('salesGeneralAndAdministrativeToRevenue'),
                        combined.get('stockBasedCompensationToRevenue'),
                        combined.get('researchAndDevelopementToRevenue'),
                        combined.get('capexToRevenue'),
                        combined.get('intangiblesToTotalAssets'),
                        combined.get('capexToOperatingCashFlow'),
                        combined.get('capexToDepreciation'),
                        combined.get('salesGeneralAndAdministrativeToRevenue'),
                        combined.get('marketCap'),
                        combined.get('enterpriseValue'),
                        combined.get('averageReceivables'),
                        combined.get('averagePayables'),
                        combined.get('averageInventory'),
                        combined.get('freeCashFlowToEquity'),
                        combined.get('freeCashFlowToFirm')
                    ))
                except Exception as e:
                    logging.error(
                        f"Error inserting metrics for {symbol} on {date}: {str(e)}")

    async def crawl(self):
        logging.info("Starting metrics crawling...")

        symbols = self.get_symbols_to_crawl()

        for symbol in tqdm(symbols, desc="Crawling metrics"):
            await self.crawl_symbol_metrics(symbol)
            self.db.commit()




async def main():
    crawler = MetricsCrawler()
    await crawler.crawl()
    crawler.close()

if __name__ == "__main__":
    asyncio.run(main())
