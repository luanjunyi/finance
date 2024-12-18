import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from trade_sim import Backtest
import logging
import argparse


def compare_with_many(random_tradings, initial_fund=1000000,
                      benchmarks={}, end_day=datetime.now().date()):
    """Compare random portfolios against multiple benchmark portfolios.
    
    Args:
        random_tradings (list): List of DataFrames, each containing trading operations
        initial_fund (float): Initial investment amount
        benchmarks (dict): Dictionary mapping benchmark names to their operations DataFrames
        end_date (str, optional): End date for simulation in 'YYYY-MM-DD' format
    
    Returns:
        numpy.ndarray: Array of returns for all comparison portfolios
    """

    bt = Backtest(initial_fund=initial_fund)

    if type(end_day) == str:
        end_day = datetime.strptime(end_day, '%Y-%m-%d').date()

    comparison_returns = []
    plt.figure(figsize=(12, 8))

    # Plot each benchmark portfolio first
    colors = ['red', 'blue', 'green', 'purple', 'orange']  # Add more colors if needed
    for i, (name, ops_df) in enumerate(benchmarks.items()):
        benchmark_ops = ops_df.values.tolist()
        benchmark_values, benchmark_dates = bt.run(
            benchmark_ops, end_day, return_history=True)
        benchmark_return = (benchmark_values[-1] - initial_fund) / initial_fund * 100

        # Plot benchmark with actual values
        plt.plot(benchmark_dates, benchmark_values, color=colors[i % len(colors)],
                linewidth=2, label=f'{name} ({benchmark_return:.1f}%)')

    # Plot each random comparison portfolio
    for i, ops_df in enumerate(random_tradings):
        logging.info(f"Processing comparison portfolio {i}")
        comparison_ops = ops_df.values.tolist()

        # Run simulation with history
        comparison_values, dates = bt.run(
            comparison_ops, end_day, return_history=True)
        comparison_return = (comparison_values[-1] - initial_fund) / initial_fund * 100
        comparison_returns.append(comparison_return)

        # Plot with actual values
        plt.plot(dates, comparison_values, color='gray',
                alpha=0.3, label='_nolegend_')

    plt.title('Portfolio Performance Comparison')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    comparison_returns = np.array(comparison_returns)

    print(f"\nResults Summary:")
    print(f"Comparison Portfolios Mean Return: {np.mean(comparison_returns):.2f}%")
    print(f"Comparison Portfolios Std Dev: {np.std(comparison_returns):.2f}%")
    print(f"Comparison Portfolios Min Return: {np.min(comparison_returns):.2f}%")
    print(f"Comparison Portfolios Max Return: {np.max(comparison_returns):.2f}%")
    print(
        f"Comparison Portfolios 1% Percentile: {np.percentile(comparison_returns, 1):.2f}%")
    print(
        f"Comparison Portfolios 5% Percentile: {np.percentile(comparison_returns, 5):.2f}%")
    print(
        f"Comparison Portfolios 10% Percentile: {np.percentile(comparison_returns, 10):.2f}%")
    print(
        f"Comparison Portfolios 25% Percentile: {np.percentile(comparison_returns, 25):.2f}%")
    print(
        f"Comparison Portfolios 50% Percentile: {np.percentile(comparison_returns, 50):.2f}%")
    print(
        f"Comparison Portfolios 75% Percentile: {np.percentile(comparison_returns, 75):.2f}%")
    print(
        f"Comparison Portfolios 90% Percentile: {np.percentile(comparison_returns, 90):.2f}%")
    print(
        f"Comparison Portfolios 99% Percentile: {np.percentile(comparison_returns, 99):.2f}%")

    return comparison_returns


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare portfolios against a target portfolio')
    parser.add_argument('--target-ops-file', type=str, default='ap_ops.csv',
                        help='Path to the target operations file')
    parser.add_argument('--num-variants', type=int, default=20,
                        help='Number of random variants to generate')
    parser.add_argument('--initial-fund', type=float,
                        default=1000000, help='Initial investment amount')
    parser.add_argument('--end-date', type=str,
                        help='End date for simulation (YYYY-MM-DD)')

    args = parser.parse_args()
    
    # Note: This main function needs to be updated to work with the new compare_with_many function
    print("Please provide a list of comparison operations to use this script.")
