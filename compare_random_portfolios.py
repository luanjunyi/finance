import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from generate_random_ops import RandomOperationsGenerator
from trade_sim import backtest_trading, read_trading_ops
import logging
import argparse


def generate_and_evaluate_portfolios(num_variants=20, initial_fund=1000000, target_ops_file='ap_ops.csv', end_date=None):
    # Set up logging to suppress excessive warnings
    logging.basicConfig(level=logging.INFO)

    # Get end date from the last operation if not provided
    if end_date is None:
        target_ops = pd.read_csv(target_ops_file, header=None)
        end_day = datetime.strptime(target_ops.iloc[-1, 1], '%Y-%m-%d').date()
    else:
        end_day = datetime.strptime(end_date, '%Y-%m-%d').date()

    # Initialize the random operations generator
    target_ops_df = pd.read_csv(target_ops_file, names=['symbol', 'date', 'action', 'amount'])
    start_date = target_ops_df['date'].min()
    end_date = end_day.strftime('%Y-%m-%d')
    
    generator = RandomOperationsGenerator(start_date, end_date)

    # Generate and run random variants
    random_returns = []

    # Run original AP ops simulation first to get the timeline
    trading_ops = read_trading_ops(target_ops_file)
    ap_values, ap_dates = backtest_trading(
        trading_ops, initial_fund, end_day, return_history=True)
    ap_return = (ap_values[-1] - initial_fund) / initial_fund * 100

    plt.figure(figsize=(12, 8))

    # Plot AP portfolio with actual values
    plt.plot(ap_dates, ap_values, color='red',
             linewidth=2, label='AP Portfolio')

    # Generate and plot random variants
    for i in range(num_variants):
        # Generate random operations using the generator
        logging.info(f"Generating random operations for variant {i}")
        random_ops_df = generator.generate_operations(target_ops_file)
        random_ops = random_ops_df.values.tolist()

        # Run simulation with history
        random_values, dates = backtest_trading(
            random_ops, initial_fund, end_day, return_history=True)
        random_return = (random_values[-1] - initial_fund) / initial_fund * 100
        random_returns.append(random_return)

        # Plot with actual values
        plt.plot(dates, random_values, color='gray',
                 alpha=0.3, label='_nolegend_')

    plt.title('Portfolio Performance Comparison')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    random_returns = np.array(random_returns)

    print(f"\nResults Summary:")
    print(f"AP Portfolio Return: {ap_return:.2f}%")
    print(f"Random Portfolios Mean Return: {np.mean(random_returns):.2f}%")
    print(f"Random Portfolios Std Dev: {np.std(random_returns):.2f}%")
    print(f"Random Portfolios Min Return: {np.min(random_returns):.2f}%")
    print(f"Random Portfolios Max Return: {np.max(random_returns):.2f}%")
    print(
        f"Random Portfolios 1% Percentile: {np.percentile(random_returns, 1):.2f}%")
    print(
        f"Random Portfolios 5% Percentile: {np.percentile(random_returns, 5):.2f}%")
    print(
        f"Random Portfolios 10% Percentile: {np.percentile(random_returns, 10):.2f}%")
    print(
        f"Random Portfolios 25% Percentile: {np.percentile(random_returns, 25):.2f}%")
    print(
        f"Random Portfolios 50% Percentile: {np.percentile(random_returns, 50):.2f}%")
    print(
        f"Random Portfolios 75% Percentile: {np.percentile(random_returns, 75):.2f}%")
    print(
        f"Random Portfolios 90% Percentile: {np.percentile(random_returns, 90):.2f}%")
    print(
        f"Random Portfolios 99% Percentile: {np.percentile(random_returns, 99):.2f}%")

    return random_returns


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare random portfolios against a target portfolio')
    parser.add_argument('--target-ops-file', type=str, default='ap_ops.csv',
                        help='Path to the target operations file')
    parser.add_argument('--end-date', type=str,
                        help='End date for simulation (YYYY-MM-DD). If not provided, uses last date in ops file')
    parser.add_argument('--num-variants', type=int, default=20,
                        help='Number of random portfolio variants to generate')

    args = parser.parse_args()

    random_returns = generate_and_evaluate_portfolios(
        num_variants=args.num_variants,
        target_ops_file=args.target_ops_file,
        end_date=args.end_date
    )
