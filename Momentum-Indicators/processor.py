import pandas as pd
import numpy as np
from typing import Callable

FILE_PATH = "/home/lawre/Southwest-Securities-Quant-Internship-Summer-2025/Momentum-Indicators/chgpct_of_stock_D.csv"

def get_df(file: str):
    df = pd.read_csv(file)
    df = df.rename(columns = {"Unnamed: 0": "date"})
    long_df = df.melt(id_vars=["date"], var_name="stockid", value_name="value")
    long_df['value'] = long_df['value'].astype('float32')
    return long_df

def calculate_stats(df: pd.DataFrame):
    df = df.sort_values(['stockid', 'date']).copy()
    result = df[['date', 'stockid']].copy()

    calculate_rolling_features(df, result, 'cum_ret', lambda x: x.sum(), [20, 40, 80])
    calculate_rolling_features(df, result, 'ret_stdev', lambda x: x.std(ddof=1))
    calculate_rolling_features(df, result, 'pos_ret_stdev', lambda x: x[x >= 0].std(ddof=1))
    calculate_rolling_features(df, result, 'neg_ret_stdev', lambda x: x[x < 0].std(ddof=1))

    return result

def calculate_rolling_features(df: pd.DataFrame, result: pd.DataFrame, feature_name: str, function: Callable[[pd.Series], float], windows: list[int] = [20, 40, 60]):
    for length in windows:
        if feature_name == 'cum_ret':
            result[f'{feature_name}_{length}'] = df.groupby('stockid')['value'].rolling(window=length).sum().reset_index(level=0, drop=True)
        elif feature_name == 'ret_stdev':
            result[f'{feature_name}_{length}'] = df.groupby('stockid')['value'].rolling(window=length).std(ddof=1).reset_index(level=0, drop=True)
        elif feature_name == 'pos_ret_stdev':
            result[f'{feature_name}_{length}'] = df['value'].where(df['value'] >= 0).groupby(df['stockid']).rolling(window=length, min_periods=2).std(ddof=1).reset_index(level=0, drop=True)
        else:
            result[f'{feature_name}_{length}'] = df['value'].where(df['value'] < 0).groupby(df['stockid']).rolling(window=length, min_periods=2).std(ddof=1).reset_index(level=0, drop=True)

df = get_df(FILE_PATH)

stock_stats = calculate_stats(df)
stock_stats.to_csv("stock_stats.csv", index=False)
print(stock_stats)