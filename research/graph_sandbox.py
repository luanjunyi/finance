# %% [markdown]
# Random graph for intuitions
# %%
import pdb
from research.interday_trading import InterdayTrading
from fmp_fetch import FMPOnline
from fmp_fetch.fmp_api import FMPAPI
from utils.graph import price_history_graph

api = FMPAPI()
fmp = FMPOnline()
t = InterdayTrading('2013-01-01', '2023-01-05')
# %%
from datetime import timedelta
from tqdm import tqdm

date_range = pd.date_range(start='2010-06-01', end='2019-06-01', freq=timedelta(days=365))
year_data = []
for date in tqdm(date_range):
    while not t._is_trading_day(pd.to_datetime(date).date()):
        date = (pd.to_datetime(date) + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
    date = pd.to_datetime(date).strftime('%Y-%m-%d')
    current = t.build_features_for_date(pd.to_datetime(date).date(), use_return_after_days=365)
    current['spx_return'] = fmp.get_index_price('^GSPC', current['return_date'].iloc[0]) / fmp.get_index_price('^GSPC', date) - 1
    current['return'] = current['return_day_price'] / current['price'] - 1
    current['win_spx'] = current['return'] - current['spx_return']
    year_data.append(current)

data = pd.concat(year_data)
d = data.copy()
# %% sadfaf
from utils.graph import price_to_fcf_graph
d = data.copy()
price_to_fcf_graph(d)

# %%
from utils.graph import per_group_return_graph
per_group_return_graph(d, 'price_to_fcf', -100, 100)

# %%
d['m3_log'] = np.log(d.m3 + 1)
d.m3_log.describe()
d[d.m3_log.abs() > 9]
per_group_return_graph(d, 'm3', -0.6, 1.6)

# %%
per_group_return_graph(d, 'm12', -0.9, 3)

# %%
per_group_return_graph(d[d.price_to_fcf.between(0, 4)], 'm3', -0.6, 3)

# %%
d.price_to_ncav.describe()

# %%
per_group_return_graph(d[d.price_to_fcf.between(0, 3)], 'price_to_ncav', 0.5, 0.7)

# %%
from fmp_data import Dataset
ph = Dataset(symbol=d['symbol'].unique().tolist() + ['^SPX'] , metrics={'adjusted_close': 'price'}, start_date='2017-01-01', end_date='2019-12-31').get_data()
ph = ph.merge(d[['symbol', 'date', 'price_to_fcf', 'price_to_ncav', 'win_spx', 'opm_12m']], 
    on='symbol', how='left', suffixes=('_history', '_sample'))
ph['color'] = np.where(ph['win_spx'] > 0, 'red', 'grey')
ph.loc[ph.symbol == '^SPX', 'date_sample'] = sample['date'].iloc[0]
ph.loc[ph.symbol == '^SPX', 'color'] = 'green'
ph['base_date'] = ph['date_sample'].apply(lambda x: x.strftime('%Y-%m-%d'))
ph.rename(columns={'date_history': 'date'}, inplace=True)

price_history_graph(ph, 'Operation margin > 20%')

# %%
d15 = data[data['date'] == '2014-12-31'].copy()
d15 = d15[(d15.price_to_fcf > 0) & (d15.price_to_fcf < 2)]
ph15 = Dataset(symbol=d15['symbol'].unique().tolist() + ['^SPX'] , metrics={'adjusted_close': 'price'}, start_date='2013-01-01', end_date='2015-12-31').get_data()
ph15 = ph15.merge(d15[['symbol', 'date', 'price_to_fcf', 'price_to_ncav', 'win_spx', 'opm_12m', 'm12']], 
    on='symbol', how='left', suffixes=('_history', '_sample'))
ph15['color'] = np.where(ph15['m12'].between(0, 0.05), 'red', 'grey')
ph15.loc[ph15.symbol == '^SPX', 'date_sample'] = d15['date'].iloc[0]
ph15.loc[ph15.symbol == '^SPX', 'color'] = 'black'
ph15['base_date'] = ph15['date_sample']
ph15.rename(columns={'date_history': 'date'}, inplace=True)

price_history_graph(ph15, 'M12 > 5%')


# %%
d15['win_spx'].describe()

# %%
d15[d15.price_to_fcf.between(0, 1)]['win_spx'].describe()
# %%
for x in year_data:
    date = x['date'].iloc[0]
    print(f"{date}: {x[x.price_to_fcf.between(0, 1)]['win_spx'].mean():.2f}, {x['win_spx'].mean():.2f}")
# %%
