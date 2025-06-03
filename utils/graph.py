import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from typing import List

def per_group_return_graph(data, cut_column, min_value, max_value, num_bins=40, extra_title=""):
    assert cut_column in data.columns
    assert 'return' in data.columns
    assert 'spx_return' in data.columns
    assert 'win_spx' in data.columns
    assert 'symbol' in data.columns

    NUM_BINS = num_bins
    data = data[data[cut_column].between(min_value, max_value)].copy().reset_index(drop=True)
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.dropna()

    max_value = data[cut_column].max()
    min_value = data[cut_column].min()
    interval = (max_value - min_value) / NUM_BINS

    
    labels = [f'[{min_value + i * interval:.2f}, {min_value + (i+1)*interval:.2f}]' for i in range(NUM_BINS)]
    round = 2
    while len(set(labels)) < NUM_BINS:
        round += 1
        labels = [f'[{min_value + i * interval:.{round}f}, {min_value + (i+1)*interval:.{round}f}]' for i in range(NUM_BINS)]

    x = pd.cut(data[cut_column], bins=NUM_BINS, labels=labels)
    y = data.groupby(x, observed=False).agg({"spx_return": "mean", "return": "mean", "symbol": "count", "win_spx": "mean"}).reset_index()

    plt.figure(figsize=(20, 8))
    bars = plt.bar(y[cut_column], y['return'], edgecolor='black')
    plt.plot(y[cut_column], y['win_spx'], '-o', color='#0c1ef2', linewidth=1, markersize=7, zorder=3)

    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, 
                0.001, 
                f'n={y["symbol"].iloc[i]} | r = {y["return"].iloc[i] * 100:.2f}%', 
                ha='center', 
                va='bottom', 
                rotation=90, 
                fontsize=9,
                fontweight='bold')
    plt.xticks(rotation=90)
    plt.ylabel("Net Return")
    plt.title(f"{cut_column} and price return, spx average return {data['spx_return'].mean() * 100:.2f}% | {extra_title}")
    plt.show()
    

def price_to_fcf_graph(data, num_bins=40, max_price_to_fcf=100):
    assert 'price_to_fcf' in data.columns
    assert 'return' in data.columns
    assert 'spx_return' in data.columns
    assert 'win_spx' in data.columns
    assert 'symbol' in data.columns

    NUM_BINS = num_bins
    MAX_PRICE_TO_FCF = max_price_to_fcf
    interval = 2 * MAX_PRICE_TO_FCF / NUM_BINS

    labels = [f'[{i * interval - MAX_PRICE_TO_FCF}, {(i+1)*interval - MAX_PRICE_TO_FCF}]' for i in range(NUM_BINS)]
    
    fcf = data[data.price_to_fcf.abs() < MAX_PRICE_TO_FCF]
    fcf = fcf.replace([np.inf, -np.inf], np.nan)
    fcf = fcf.dropna()

    x = pd.cut(fcf['price_to_fcf'], bins=NUM_BINS, labels=labels).rename('price_to_fcf_bin')
    y = fcf.groupby(x, observed=False).agg({"spx_return": "mean", "return": "mean", "symbol": "count", "win_spx": "mean"}).reset_index()

    plt.figure(figsize=(20, 8))
    bars = plt.bar(y['price_to_fcf_bin'], y['return'], edgecolor='black')
    plt.plot(y['price_to_fcf_bin'], y['win_spx'], '-o', color='#0c1ef2', linewidth=1, markersize=7, zorder=3)

    for i, bar in enumerate(bars):
        plt.text(bar.get_x() + bar.get_width()/2, 
                0.001, 
                f'n={y["symbol"].iloc[i]} | r = {y["return"].iloc[i] * 100:.2f}%', 
                ha='center', 
                va='bottom', 
                rotation=90, 
                fontsize=9,
                fontweight='bold')
    plt.xticks(rotation=90)
    plt.ylabel("Net Return")
    plt.title(f"Price to FCF and price return, spx average return {data['spx_return'].mean() * 100:.2f}%")
    plt.show()

def price_history_graph(data: List[pd.DataFrame], title=""):
    """
    data: List[pd.DataFrame] where each DataFrame is a price series
    For each series S,
      - We use S.base_date as the date when the position is built.
      - Each S.price is then transformed to the multiple of S.base_date's price.
      - We use S.color for the color of the line, defaut to gray.
    
    The function samples data points to improve performance when plotting.
    """
    # Maximum number of points to plot per series
    MAX_POINTS_PER_SERIES = 300
    data = data[['symbol', 'date', 'price', 'color', 'base_date']].copy()
    plt.figure(figsize=(20, 8))
    for s in data.symbol.unique():
        d = data[data.symbol == s].copy().dropna()
        base_date = d.base_date.iloc[0]
        base_price = d.price[d.date == base_date].iloc[0]

        # Assert that base_date exists in the data
        assert (d.base_date == base_date).all(), f"Multiple base dates found in data for symbol {s}: {d.base_date.unique()}"
        
        if len(d) > MAX_POINTS_PER_SERIES:
            d = d.sample(n=MAX_POINTS_PER_SERIES, random_state=42)
        d = d.sort_values('date', ascending=True)
        
        y = [np.log(p / base_price) for p in d.price]
        x = d.date.apply(lambda ds: (pd.to_datetime(ds).date() - pd.to_datetime(base_date).date()).days)
        if s == '^SPX':
            plt.plot(x, y, color=d.color.iloc[0], linewidth=2, linestyle='-')
        else:
            plt.plot(x, y, color=d.color.iloc[0])

    plt.ylabel("Relative Price")
    plt.xlabel("Time")
    plt.title(title)
    