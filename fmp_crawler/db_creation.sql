-- Stock Symbol table
CREATE TABLE IF NOT EXISTS stock_symbol (
  symbol VARCHAR(10) PRIMARY KEY,
  name VARCHAR(255),
  exchange VARCHAR(255),
  exchange_short_name VARCHAR(255),
  type VARCHAR(255)
);
-- Daily Price table
CREATE TABLE IF NOT EXISTS daily_price (
  symbol VARCHAR(10) PRIMARY KEY,
  date DATE NOT NULL,
  open DECIMAL(10, 2),
  high DECIMAL(10, 2),
  low DECIMAL(10, 2),
  close DECIMAL(10, 2),
  adjusted_close DECIMAL(10, 2),
  volume BIGINT,
  PRIMARY KEY (symbol, date)
);
CREATE INDEX IF NOT EXISTS idx_daily_price_symbol_date ON daily_price(symbol, date);
-- Income Statement table
CREATE TABLE IF NOT EXISTS income_statement (
  symbol VARCHAR(10) PRIMARY KEY,
  date DATE NOT NULL,
  period VARCHAR(10) NOT NULL,
  calendar_year VARCHAR(4),
  reported_currency VARCHAR(3),
  revenue DECIMAL(15, 2),
  cost_of_revenue DECIMAL(15, 2),
  gross_profit DECIMAL(15, 2),
  gross_profit_ratio DECIMAL(10, 4),
  research_and_development_expenses DECIMAL(15, 2),
  general_and_administrative_expenses DECIMAL(15, 2),
  selling_and_marketing_expenses DECIMAL(15, 2),
  selling_general_and_administrative_expenses DECIMAL(15, 2),
  operating_expenses DECIMAL(15, 2),
  operating_income DECIMAL(15, 2),
  operating_income_ratio DECIMAL(10, 4),
  ebitda DECIMAL(15, 2),
  ebitda_ratio DECIMAL(10, 4),
  net_income DECIMAL(15, 2),
  net_income_ratio DECIMAL(10, 4),
  eps DECIMAL(10, 2),
  eps_diluted DECIMAL(10, 2),
  weighted_average_shares_outstanding DECIMAL(15, 0),
  weighted_average_shares_outstanding_diluted DECIMAL(15, 0),
  link VARCHAR(255),
  PRIMARY KEY (symbol, date, period)
);
CREATE INDEX IF NOT EXISTS idx_income_statement_symbol_date ON income_statement(symbol, date);
-- Balance Sheet table
CREATE TABLE IF NOT EXISTS balance_sheet (
  symbol VARCHAR(10) PRIMARY KEY,
  date DATE NOT NULL,
  period VARCHAR(10) NOT NULL,
  calendar_year VARCHAR(4),
  reported_currency VARCHAR(3),
  cash_and_cash_equivalents DECIMAL(15, 2),
  short_term_investments DECIMAL(15, 2),
  cash_and_short_term_investments DECIMAL(15, 2),
  net_receivables DECIMAL(15, 2),
  inventory DECIMAL(15, 2),
  total_current_assets DECIMAL(15, 2),
  property_plant_equipment_net DECIMAL(15, 2),
  goodwill DECIMAL(15, 2),
  intangible_assets DECIMAL(15, 2),
  long_term_investments DECIMAL(15, 2),
  total_non_current_assets DECIMAL(15, 2),
  total_assets DECIMAL(15, 2),
  accounts_payables DECIMAL(15, 2),
  short_term_debt DECIMAL(15, 2),
  total_current_liabilities DECIMAL(15, 2),
  long_term_debt DECIMAL(15, 2),
  total_non_current_liabilities DECIMAL(15, 2),
  total_liabilities DECIMAL(15, 2),
  total_stockholders_equity DECIMAL(15, 2),
  total_equity DECIMAL(15, 2),
  total_liabilities_and_stockholders_equity DECIMAL(15, 2),
  total_investments DECIMAL(15, 2),
  total_debt DECIMAL(15, 2),
  net_debt DECIMAL(15, 2),
  link VARCHAR(255),
  PRIMARY KEY (symbol, date, period)
);
CREATE INDEX IF NOT EXISTS idx_balance_sheet_symbol_date ON balance_sheet(symbol, date);
-- Cash Flow table
CREATE TABLE IF NOT EXISTS cash_flow (
  symbol VARCHAR(10) PRIMARY KEY,
  date DATE NOT NULL,
  period VARCHAR(10) NOT NULL,
  calendar_year VARCHAR(4),
  reported_currency VARCHAR(3),
  net_income DECIMAL(15, 2),
  depreciation_and_amortization DECIMAL(15, 2),
  stock_based_compensation DECIMAL(15, 2),
  change_in_working_capital DECIMAL(15, 2),
  accounts_receivables DECIMAL(15, 2),
  inventory DECIMAL(15, 2),
  accounts_payables DECIMAL(15, 2),
  net_cash_provided_by_operating_activities DECIMAL(15, 2),
  investments_in_property_plant_and_equipment DECIMAL(15, 2),
  acquisitions_net DECIMAL(15, 2),
  purchases_of_investments DECIMAL(15, 2),
  sales_maturities_of_investments DECIMAL(15, 2),
  net_cash_used_for_investing_activities DECIMAL(15, 2),
  debt_repayment DECIMAL(15, 2),
  common_stock_repurchased DECIMAL(15, 2),
  dividends_paid DECIMAL(15, 2),
  net_cash_used_provided_by_financing_activities DECIMAL(15, 2),
  net_change_in_cash DECIMAL(15, 2),
  operating_cash_flow DECIMAL(15, 2),
  capital_expenditure DECIMAL(15, 2),
  free_cash_flow DECIMAL(15, 2),
  link VARCHAR(255),
  PRIMARY KEY (symbol, date, period)
);
CREATE INDEX IF NOT EXISTS idx_cash_flow_symbol_date ON cash_flow(symbol, date);