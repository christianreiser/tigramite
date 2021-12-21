import pandas as pd

from chris_exploration.preprocessing import remove_nan_seq_from_top_and_bot

df = pd.read_csv('./data/daily_summaries_all (copy).csv', sep=",")
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')

# select columns
df = df[['Date', 'SleepEfficiency', 'LowLatitude']]

df = remove_nan_seq_from_top_and_bot(df)
# check for nans
print('nans found in:')
