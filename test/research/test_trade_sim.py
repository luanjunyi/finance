import pytest
from datetime import datetime
import pandas as pd
import os
from research.trade_sim import Backtest
from unittest.mock import patch, MagicMock
import tempfile
import csv

@pytest.fixture
def backtest():
    return Backtest(initial_fund=1000000)

@pytest.fixture
def mock_price_loader():
    with patch('research.trade_sim.FMPPriceLoader') as mock:
        price_loader = mock.return_value
        # Mock get_last_available_price to return predictable values
        price_loader.get_last_available_price.return_value = (100.0, '2024-01-01')
        price_loader.get_next_available_price.return_value = (100.0, '2024-01-02')
        yield price_loader

@pytest.fixture
def sample_trading_ops():
    return [
        ['AAPL', '2024-01-01', 'BUY', '0.5'],
        ['AAPL', '2024-01-02', 'SELL', '0.5'],
        ['GOOGL', '2024-01-03', 'BUY', '0.3']
    ]

def test_portfolio_value_calculation(backtest, mock_price_loader):
    backtest.price_loader = mock_price_loader
    backtest.cash = 500000
    backtest.portfolio = {'AAPL': 1000, 'GOOGL': 500}
    
    date = datetime.strptime('2024-01-01', '%Y-%m-%d').date()
    value = backtest.get_portfolio_value(date)
    
    # Expected: cash + (AAPL shares * price) + (GOOGL shares * price)
    expected_value = 500000 + (1000 * 100.0) + (500 * 100.0)
    assert value == expected_value

def test_buy_operation(backtest, mock_price_loader, sample_trading_ops):
    backtest.price_loader = mock_price_loader
    end_day = datetime.strptime('2024-01-04', '%Y-%m-%d').date()
    
    # Run simulation with just the buy operation
    backtest.run([sample_trading_ops[0]], end_day)
    
    # Check if shares were bought correctly
    assert 'AAPL' in backtest.portfolio
    expected_shares = int((1000000 * 0.5) / 100.0)  # 50% of initial fund at $100 per share
    assert backtest.portfolio['AAPL'] == expected_shares
    assert backtest.cash == 1000000 - (expected_shares * 100.0)

def test_sell_operation(backtest, mock_price_loader):
    backtest.price_loader = mock_price_loader
    end_day = datetime.strptime('2024-01-02', '%Y-%m-%d').date()
    
    # Create buy and sell operations
    operations = [
        ['AAPL', '2024-01-01', 'BUY', '1.0'],   # Buy with full portfolio
        ['AAPL', '2024-01-02', 'SELL', '0.5']    # Sell half
    ]
    
    # Run both operations in sequence
    backtest.run(operations, end_day)
    
    # Calculate expected shares
    # Initial buy: should buy shares worth ~1M at $100/share = 10000 shares
    # Then sell half = 5000 shares remaining
    expected_shares = int(backtest.initial_fund / 100.0) // 2  # Half of initial shares
    
    # Verify the portfolio state after both operations
    assert 'AAPL' in backtest.portfolio
    assert backtest.portfolio['AAPL'] == expected_shares  # Should have half the initial shares
    assert backtest.cash >= backtest.initial_fund * 0.45  # Should have recovered significant cash from selling

def test_read_trading_ops():
    # Create a temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, newline='') as f:
        writer = csv.writer(f)
        writer.writerows([
            ['AAPL', '2024-01-01', 'BUY', '0.5'],
            ['GOOGL', '2024-01-02', 'BUY', '0.3']
        ])
        temp_file = f.name
    
    try:
        ops = Backtest.read_trading_ops(temp_file)
        assert len(ops) == 2
        assert ops[0] == ['AAPL', '2024-01-01', 'BUY', '0.5']
        assert ops[1] == ['GOOGL', '2024-01-02', 'BUY', '0.3']
    finally:
        os.unlink(temp_file)  # Clean up temporary file

def test_invalid_sell(backtest, mock_price_loader):
    backtest.price_loader = mock_price_loader
    end_day = datetime.strptime('2024-01-02', '%Y-%m-%d').date()
    
    # Try to sell stock we don't own
    sell_op = ['AAPL', '2024-01-02', 'SELL', '0.5']
    
    with pytest.raises(Exception) as exc_info:
        backtest.run([sell_op], end_day)
    assert "No shares of AAPL to sell" in str(exc_info.value)

def test_portfolio_returns(backtest, mock_price_loader):
    backtest.price_loader = mock_price_loader
    end_day = datetime.strptime('2024-01-04', '%Y-%m-%d').date()
    
    # Mock prices for all the necessary calls
    mock_price_loader.get_last_available_price.side_effect = lambda sym, date, price_type: (
        (100.0, '2024-01-01') if date.strftime('%Y-%m-%d') <= '2024-01-01' else (110.0, '2024-01-04')
    )
    
    # Buy operation using 50% of portfolio
    buy_op = ['AAPL', '2024-01-01', 'BUY', '0.5']
    final_value = backtest.run([buy_op], end_day, return_history=False)
    
    # Calculate expected return:
    # 1. Initial investment of 500k (50% of 1M) buys 5000 shares at $100
    # 2. Final value = 500k (remaining cash) + (5000 shares * $110)
    # 3. Return = Final value / Initial value = (500k + 550k) / 1M = 1.05
    expected_final_value = 1.05
    
    assert abs(final_value - expected_final_value) < 0.0001  # Account for floating point precision

def test_portfolio_returns_multiple_buys(backtest, mock_price_loader):
    backtest.price_loader = mock_price_loader
    end_day = datetime.strptime('2024-01-04', '%Y-%m-%d').date()
    
    # Mock prices: AAPL starts at 100, ends at 110; GOOGL starts at 150, ends at 180
    mock_price_loader.get_last_available_price.side_effect = lambda sym, date, price_type: (
        (100.0, '2024-01-01') if date.strftime('%Y-%m-%d') <= '2024-01-01' and sym == 'AAPL'
        else (110.0, '2024-01-04') if sym == 'AAPL'
        else (150.0, '2024-01-01') if date.strftime('%Y-%m-%d') <= '2024-01-01'
        else (180.0, '2024-01-04')
    )
    
    # Buy operations: 30% AAPL, 40% GOOGL
    operations = [
        ['AAPL', '2024-01-01', 'BUY', '0.3'],
        ['GOOGL', '2024-01-01', 'BUY', '0.4']
    ]
    
    final_value = backtest.run(operations, end_day, return_history=False)
    
    # Calculate expected return:
    # Initial portfolio: $1M
    # AAPL investment: 300k -> 3000 shares -> worth 330k at end
    # GOOGL investment: 400k -> 2666 shares -> worth ~480k at end
    # Remaining cash: 300k
    # Final value = (330k + 480k + 300k) / 1M ≈ 1.11
    expected_final_value = 1.11
    
    assert abs(final_value - expected_final_value) < 0.01

def test_portfolio_returns_buy_sell_buy(backtest, mock_price_loader):
    backtest.price_loader = mock_price_loader
    end_day = datetime.strptime('2024-01-04', '%Y-%m-%d').date()
    
    # Mock a price sequence for AAPL: 100 -> 120 -> 110
    prices = {
        '2024-01-01': {'AAPL': 100.0},
        '2024-01-02': {'AAPL': 120.0},
        '2024-01-04': {'AAPL': 110.0}
    }
    
    mock_price_loader.get_last_available_price.side_effect = lambda sym, date, price_type: (
        (prices[date.strftime('%Y-%m-%d')][sym], date.strftime('%Y-%m-%d'))
    )
    
    # Operations: Buy -> Sell at peak -> Buy again
    operations = [
        ['AAPL', '2024-01-01', 'BUY', '1.0'],    # Buy all at 100
        ['AAPL', '2024-01-02', 'SELL', 'ALL'],   # Sell all at 120
        ['AAPL', '2024-01-02', 'BUY', '0.5']     # Buy back half at 120
    ]
    
    final_value = backtest.run(operations, end_day, return_history=False)
    
    # Calculate expected return:
    # 1. Initial buy: $1M buys 10000 shares at $100
    # 2. Sell all: Get $1.2M at $120 (20% gain)
    # 3. Buy back: Spend $600k to get 5000 shares at $120
    # 4. Final value: $600k cash + (5000 shares * $110) = $1.15M
    expected_final_value = 1.15  # 15% total return
    
    assert abs(final_value - expected_final_value) < 0.01

def test_portfolio_returns_gradual_sells(backtest, mock_price_loader):
    backtest.price_loader = mock_price_loader
    end_day = datetime.strptime('2024-01-04', '%Y-%m-%d').date()
    
    # Mock rising prices: 100 -> 110 -> 120 -> 130
    prices = {
        '2024-01-01': {'AAPL': 100.0},
        '2024-01-02': {'AAPL': 110.0},
        '2024-01-03': {'AAPL': 120.0},
        '2024-01-04': {'AAPL': 130.0}
    }
    
    mock_price_loader.get_last_available_price.side_effect = lambda sym, date, price_type: (
        (prices[date.strftime('%Y-%m-%d')][sym], date.strftime('%Y-%m-%d'))
    )
    
    # Operations: Buy all, then gradually sell
    operations = [
        ['AAPL', '2024-01-01', 'BUY', '1.0'],     # Buy all at 100
        ['AAPL', '2024-01-02', 'SELL', '0.3'],    # Sell 30% at 110
        ['AAPL', '2024-01-03', 'SELL', '0.5'],    # Sell 50% of remainder at 120
        ['AAPL', '2024-01-04', 'SELL', '0.2']     # Sell 20% of remainder at 130
    ]
    
    final_value = backtest.run(operations, end_day, return_history=False)
    
    # Calculate expected return:
    # 1. Buy 10000 shares at $100 each ($1M)
    # 2. Sell 3000 shares at $110 ($330k cash, 7000 shares left)
    # 3. Sell 3500 shares at $120 ($420k more cash, 3500 shares left)
    # 4. Sell 700 shares at $130 ($91k more cash, 2800 shares left)
    # Final state:
    # - Cash: $841k ($330k + $420k + $91k)
    # - Remaining shares: 2800 worth $364k at $130
    # Total value: $1.205M
    expected_final_value = 1.205  # 20.5% total return
    
    assert abs(final_value - expected_final_value) < 0.01

def test_portfolio_returns_price_gaps(backtest, mock_price_loader):
    backtest.price_loader = mock_price_loader
    end_day = datetime.strptime('2024-01-04', '%Y-%m-%d').date()
    
    # Mock prices with gaps (no data for Jan 2-3)
    prices = {
        '2024-01-01': {'AAPL': 100.0},
        '2024-01-04': {'AAPL': 120.0}
    }
    
    def mock_price(sym, date, price_type):
        date_str = date.strftime('%Y-%m-%d')
        if date_str in prices and sym in prices[date_str]:
            return prices[date_str][sym], date_str
        # Return last available price
        if date_str > '2024-01-01' and date_str < '2024-01-04':
            return 100.0, '2024-01-01'
        raise Exception(f"No price data for {sym} on {date_str}")
    
    mock_price_loader.get_last_available_price.side_effect = mock_price
    
    operations = [
        ['AAPL', '2024-01-01', 'BUY', '1.0'],
        ['AAPL', '2024-01-02', 'SELL', '0.5'],  # Should use Jan 1 price
        ['AAPL', '2024-01-03', 'BUY', '0.3']    # Should use Jan 1 price
    ]
    
    final_value = backtest.run(operations, end_day, return_history=False)
    
    # Calculate expected return:
    # 1. Buy 10000 shares at $100 ($1M)
    # 2. Sell 5000 shares at $100 ($500k cash, 5000 shares left)
    # 3. Buy 3000 shares at $100 (using $300k, $200k cash left, 8000 shares)
    # 4. Final: $200k cash + (8000 shares * $120) = $1.16M
    expected_final_value = 1.16
    
    assert abs(final_value - expected_final_value) < 0.01

def test_portfolio_returns_same_day_trades(backtest, mock_price_loader):
    backtest.price_loader = mock_price_loader
    end_day = datetime.strptime('2024-01-04', '%Y-%m-%d').date()
    
    # Mock stable price
    mock_price_loader.get_last_available_price.return_value = (100.0, '2024-01-01')
    
    # Multiple trades on the same day
    operations = [
        ['AAPL', '2024-01-01', 'BUY', '0.4'],
        ['AAPL', '2024-01-01', 'SELL', '0.5'],  # Sell half of the first buy
        ['AAPL', '2024-01-01', 'BUY', '0.3'],   # Buy more
        ['AAPL', '2024-01-01', 'SELL', '0.2']   # Sell a bit more
    ]
    
    final_value = backtest.run(operations, end_day, return_history=False)
    
    # Verify that same-day trades don't compound returns incorrectly
    # Each trade should be at the same price point
    assert abs(final_value - 1.0) < 0.01  # Should be ~1.0 as price hasn't changed

def test_portfolio_returns_tiny_trades(backtest, mock_price_loader):
    backtest.price_loader = mock_price_loader
    end_day = datetime.strptime('2024-01-04', '%Y-%m-%d').date()
    
    # Mock prices with small changes
    mock_price_loader.get_last_available_price.side_effect = lambda sym, date, price_type: (
        (100.001, '2024-01-01') if date.strftime('%Y-%m-%d') == '2024-01-01'
        else (100.002, '2024-01-02') if date.strftime('%Y-%m-%d') == '2024-01-02'
        else (100.003, '2024-01-04')
    )
    
    # Very small trades
    operations = [
        ['AAPL', '2024-01-01', 'BUY', '0.001'],    # Tiny initial buy
        ['AAPL', '2024-01-02', 'BUY', '0.0005'],   # Even smaller buy
        ['AAPL', '2024-01-02', 'SELL', '0.0001']   # Microscopic sell
    ]
    
    final_value = backtest.run(operations, end_day, return_history=False)
    
    # Should handle tiny trades without floating point errors
    assert 0.99 <= final_value <= 1.01

def test_portfolio_returns_extreme_moves(backtest, mock_price_loader):
    backtest.price_loader = mock_price_loader
    end_day = datetime.strptime('2024-01-04', '%Y-%m-%d').date()
    
    # Mock extreme price movements
    prices = {
        '2024-01-01': {'AAPL': 100.0, 'GOOGL': 150.0},
        '2024-01-02': {'AAPL': 1.0, 'GOOGL': 1500.0},    # AAPL crashes 99%, GOOGL 10x
        '2024-01-04': {'AAPL': 0.1, 'GOOGL': 3000.0}     # AAPL down 90% more, GOOGL doubles
    }
    
    mock_price_loader.get_last_available_price.side_effect = lambda sym, date, price_type: (
        (prices[date.strftime('%Y-%m-%d')][sym], date.strftime('%Y-%m-%d'))
    )
    
    operations = [
        ['AAPL', '2024-01-01', 'BUY', '0.5'],     # Buy AAPL at 100
        ['GOOGL', '2024-01-01', 'BUY', '0.5'],    # Buy GOOGL at 150
        ['AAPL', '2024-01-02', 'SELL', 'ALL'],    # Sell AAPL at 1
        ['GOOGL', '2024-01-02', 'SELL', '0.5']    # Sell half GOOGL at 1500
    ]
    
    final_value = backtest.run(operations, end_day, return_history=False)
    
    # Calculate expected return:
    # 1. Initial: $500k in each stock
    # 2. AAPL: 5000 shares -> worth $5k after crash -> sell all
    # 3. GOOGL: 3333 shares -> worth $5M after 10x -> sell half (1666 shares)
    # 4. Final state:
    #    - Cash: $5k (AAPL) + $2.5M (GOOGL half-sale)
    #    - Remaining GOOGL: 1666 shares * $3000 = $5M
    # Total value: $7.505M / $1M initial = 7.505
    expected_final_value = 7.505
    
    assert abs(final_value - expected_final_value) < 0.01

def test_portfolio_returns_insufficient_funds(backtest, mock_price_loader):
    backtest.price_loader = mock_price_loader
    end_day = datetime.strptime('2024-01-04', '%Y-%m-%d').date()
    
    # Mock stable prices
    mock_price_loader.get_last_available_price.return_value = (100.0, '2024-01-01')
    
    # Try to buy more than we have funds for
    operations = [
        ['AAPL', '2024-01-01', 'BUY', '0.8'],    # Use 80% of funds
        ['GOOGL', '2024-01-01', 'BUY', '0.8'],   # Try to use another 80%
        ['MSFT', '2024-01-01', 'BUY', '0.8']     # And another 80%
    ]
    
    final_value = backtest.run(operations, end_day, return_history=False)
    
    # Despite attempting to buy 240% of portfolio value,
    # the final value should still be valid
    assert 0.99 <= final_value <= 1.01  # Should be ~1.0 as price hasn't changed

def test_portfolio_returns_rebalancing(backtest, mock_price_loader):
    backtest.price_loader = mock_price_loader
    end_day = datetime.strptime('2024-01-04', '%Y-%m-%d').date()
    
    # Mock prices with diverging performance
    prices = {
        '2024-01-01': {'AAPL': 100.0, 'GOOGL': 100.0, 'MSFT': 100.0},
        '2024-01-02': {'AAPL': 90.0, 'GOOGL': 110.0, 'MSFT': 105.0},
        '2024-01-03': {'AAPL': 85.0, 'GOOGL': 120.0, 'MSFT': 108.0},
        '2024-01-04': {'AAPL': 80.0, 'GOOGL': 130.0, 'MSFT': 110.0}
    }
    
    mock_price_loader.get_last_available_price.side_effect = lambda sym, date, price_type: (
        (prices[date.strftime('%Y-%m-%d')][sym], date.strftime('%Y-%m-%d'))
    )
    
    # Complex rebalancing operations
    operations = [
        # Initial equal allocation
        ['AAPL', '2024-01-01', 'BUY', '0.33'],
        ['GOOGL', '2024-01-01', 'BUY', '0.33'],
        ['MSFT', '2024-01-01', 'BUY', '0.33'],
        
        # Rebalance day 2: reduce AAPL, increase GOOGL
        ['AAPL', '2024-01-02', 'SELL', '0.5'],
        ['GOOGL', '2024-01-02', 'BUY', '0.2'],
        
        # Rebalance day 3: exit AAPL, split between GOOGL and MSFT
        ['AAPL', '2024-01-03', 'SELL', 'ALL'],
        ['GOOGL', '2024-01-03', 'BUY', '0.1'],
        ['MSFT', '2024-01-03', 'BUY', '0.1']
    ]
    
    final_value = backtest.run(operations, end_day, return_history=False)
    
    # Portfolio should have benefited from:
    # 1. Reducing exposure to falling AAPL
    # 2. Increasing exposure to rising GOOGL
    # 3. Maintaining moderate exposure to stable MSFT
    assert final_value > 1.0  # Should have positive return
    assert abs(final_value - 1.13) < 0.02  # Approximately 13% return