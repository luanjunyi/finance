import pandas as pd
from datetime import datetime

# Update file path names for clarity
closed_positions_file = '~/Downloads/ap_op_closed_positions.csv'
open_positions_file = '~/Downloads/ap_op_open_positions.csv'

# Read both files
df_closed = pd.read_csv(closed_positions_file)
df_open = pd.read_csv(open_positions_file)

# Process closed positions (existing logic)
df_closed['Picked'] = pd.to_datetime(df_closed['Picked'])
df_closed['Closed'] = pd.to_datetime(df_closed['Closed'])
df_closed['Symbol'] = df_closed['Symbol'].str.replace('SMCI*', 'SMCI')

# Create buy/sell records for closed positions
buy_records_closed = pd.DataFrame({
    'Symbol': df_closed['Symbol'],
    'Date': df_closed['Picked'].dt.strftime('%Y-%m-%d'),
    'Action': 'BUY',
    'Amount': '0.02'
})

sell_records_closed = pd.DataFrame({
    'Symbol': df_closed['Symbol'],
    'Date': df_closed['Closed'].dt.strftime('%Y-%m-%d'),
    'Action': 'SELL',
    'Amount': 'ALL'
})

# Process open positions
df_open['Picked'] = pd.to_datetime(df_open['Picked'])
buy_records_open = pd.DataFrame({
    'Symbol': df_open['Symbol'],
    'Date': df_open['Picked'].dt.strftime('%Y-%m-%d'),
    'Action': 'BUY',
    'Amount': '0.02'
})

# Combine all records and sort
result = pd.concat([buy_records_closed, sell_records_closed, buy_records_open])
result = result.sort_values('Date')

# Convert to CSV string without header and index
output = result.to_csv(index=False, header=False)

print(output)
