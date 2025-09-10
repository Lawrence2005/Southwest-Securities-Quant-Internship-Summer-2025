import pandas as pd
import numpy as np
from typing import Callable
from numba import njit

from scipy.stats import spearmanr, pearsonr

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

FILE_PATH1: str = "/home/lawre/Southwest-Securities-Quant-Internship-Summer-2025/Momentum-Indicators/chgpct_of_stock_D.csv"
FILE_PATH2: str = "/home/lawre/Southwest-Securities-Quant-Internship-Summer-2025/Momentum-Indicators/stock_price_vol_d.txt"
CHUNK: int = 700000

def get_df(file1: str = FILE_PATH1, file2: str = FILE_PATH2, chunk_size: int = CHUNK):
    df = pd.read_csv(file1)
    df = df.rename(columns = {"Unnamed: 0": "date"})

    long_df = df.melt(id_vars=["date"], var_name="stockid", value_name="return")
    long_df['return'] = long_df['return'].astype('float32')
    long_df['date'] = pd.to_datetime(long_df['date'])

    turnover_df = pd.read_feather(file2)
    turnover_df['turnoverrate'] = turnover_df['turnoverrate'].astype('float32')
    turnover_df['date'] = pd.to_datetime(turnover_df['date'])

    chunks = []
    for start in range(0, len(long_df), chunk_size):
        chunk = long_df.iloc[start:start + chunk_size]

        merged = chunk.merge(
            turnover_df[['StockID', 'date', 'turnoverrate']],
            left_on=['stockid', 'date'],
            right_on=['StockID', 'date'],
            how='left'
        ).drop(columns=['StockID'])

        chunks.append(merged)

    long_df = pd.concat(chunks, ignore_index=True)

    long_df.to_csv("dataset.csv", index=False)

    return long_df

def calculate_stats(df: pd.DataFrame):
    df = df.sort_values(['stockid', 'date']).copy()
    result = df[['date', 'stockid', 'return']].copy()

    calculate_rolling_features(df, result, 'cum_ret', lambda x: x.sum(), [20, 40, 80])
    calculate_rolling_features(df, result, 'ret_stdev', lambda x: x.std(ddof=1))
    calculate_rolling_features(df, result, 'pos_ret_stdev', lambda x: x[x >= 0].std(ddof=1))
    calculate_rolling_features(df, result, 'neg_ret_stdev', lambda x: x[x < 0].std(ddof=1))

    calculate_turnover_features(df, result)

    result.to_csv("stock_stats_mini.csv", index=False)

    return result

def calculate_rolling_features(df: pd.DataFrame, result: pd.DataFrame, feature_name: str, function: Callable[[pd.Series], float], windows: list[int] = [20, 40, 60]):
    for length in windows:
        if feature_name == 'cum_ret':
            result[f'{feature_name}_{length}'] = df.groupby('stockid')['return'].rolling(window=length).sum().reset_index(level=0, drop=True).astype('float32').round(4)
        elif feature_name == 'ret_stdev':
            result[f'{feature_name}_{length}'] = df.groupby('stockid')['return'].rolling(window=length).std(ddof=1).reset_index(level=0, drop=True).astype('float32').round(4)
        elif feature_name == 'pos_ret_stdev':
            result[f'{feature_name}_{length}'] = df['return'].where(df['return'] >= 0).groupby(df['stockid']).rolling(window=length, min_periods=2).std(ddof=1).reset_index(level=0, drop=True).astype('float32').round(4)
        else:
            result[f'{feature_name}_{length}'] = df['return'].where(df['return'] < 0).groupby(df['stockid']).rolling(window=length, min_periods=2).std(ddof=1).reset_index(level=0, drop=True).astype('float32').round(4)

def calculate_turnover_features(df: pd.DataFrame, result: pd.DataFrame, window: int = 20):
    result_cols = [f'trnovr{i}_avg_ret' for i in range(1, 6)]

    result_df = df.groupby('stockid', group_keys=False).apply(
        lambda g: pd.DataFrame(
            calculate_turnover_returns(g['turnoverrate'].to_numpy(), g['return'].to_numpy(), window),
            index=g.index,
            columns=result_cols
        )
    )
    
    for col in result_cols:
        result[col] = result_df[col].values.astype('float32').round(4)

@njit
def calculate_turnover_returns(turnovers, returns, window):
    result = np.full((len(turnovers), 5), np.nan)
    
    for i in range(window - 1, len(turnovers)):
        window_turnover, window_returns = turnovers[i-window+1:i+1], returns[i-window+1:i+1]

        mask_valid = ~np.isnan(window_turnover) & ~np.isnan(window_returns)
        window_turnover = window_turnover[mask_valid]
        window_returns = window_returns[mask_valid]

        if len(window_turnover) < 5 or len(window_returns) < 5:
            continue

        q = np.quantile(window_turnover, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        for j in range(5):
            mask = (window_turnover >= q[j]) & (window_turnover <= q[j+1]) if j == 0 else (window_turnover > q[j]) & (window_turnover <= q[j+1])
            result[i, j] = window_returns[mask].mean() if mask.any() else np.nan
    return result

def winsorize_series(s, lower_q=0.01, upper_q=0.99):
    if s.isna().all():
        return s

    low = s.quantile(lower_q)
    high = s.quantile(upper_q)
    return s.clip(lower=low, upper=high)

def calculate_IC(df: pd.DataFrame, indicators: list[str]):
    grouped = df.groupby("date", sort=False)

    ic_pearson = {}
    ic_spearman = {}
    for i in indicators:
        pearsons, spearmans = [], []
        
        for _, g in grouped:
            data = g[[i, "return"]].dropna()
            x = winsorize_series(data[i])
            y = data["return"].shift(-1)
            x, y = x[~y.isna()], y[~y.isna()]

            if len(data) < 5 or x.std(ddof=0) == 0 or y.std(ddof=0) == 0:
                pearsons.append(np.nan)
                spearmans.append(np.nan)
                continue

            pearson_corr = np.corrcoef(x, y)[0, 1]
            pearsons.append(pearson_corr)

            spearman_corr, _ = spearmanr(x, y)
            spearmans.append(spearman_corr)

        ic_pearson[i], ic_spearman[i] = pearsons, spearmans

    dates = df["date"].drop_duplicates().reset_index(drop=True)

    ic_pearson_df = pd.DataFrame(ic_pearson, index=dates).astype("float32").round(4)
    ic_spearman_df = pd.DataFrame(ic_spearman, index=dates).astype("float32").round(4)

    ic_pearson_df.to_csv("ic_pearson.csv")
    ic_spearman_df.to_csv("ic_spearman.csv")

    return ic_pearson_df, ic_spearman_df

def backtest(df: pd.DataFrame, indicators: list[str], n_groups: int = 5):
    df['date'] = pd.to_datetime(df['date'])

    trade_days = df['date'].drop_duplicates().sort_values()
    trade_days_index = pd.Index(trade_days)
    rebalance_dates = trade_days.groupby(trade_days.dt.to_period('M')).min().tolist()

    for indicator in indicators:
        indicator_rets = []

        for start in rebalance_dates: 
            idx = trade_days_index.get_loc(start)
            if idx + 19 >= len(trade_days):
                continue

            end = trade_days.iloc[idx + 19]

            snapshot_indicator = df.loc[df['date'] == start, ['stockid', indicator]].dropna()
            snapshot_return = df.loc[df['date'] == end, ['stockid', 'cum_ret_20']].dropna().rename(columns={'cum_ret_20': 'period_ret_20'})

            merged = snapshot_indicator.merge(snapshot_return, on='stockid', how='inner')
            if len(merged) < 8:
                continue

            try:
                merged['tile'] = pd.qcut(merged[indicator], q=n_groups, labels=False) + 1
            except ValueError:
                merged['rank'] = merged[indicator].rank(method='first')
                merged['tile'] = pd.qcut(merged['rank'], q=n_groups, labels=False) + 1

            tile_rets = merged.groupby('tile')['period_ret_20'].mean()
            tile_rets['date'] = start
            indicator_rets.append(tile_rets)
            
        if not indicator_rets:
            continue
        
        res = pd.DataFrame(indicator_rets).set_index('date').sort_index(axis=1)
        res.columns = [f'Q{col}' for col in res.columns]
        
        nav = (1+res.fillna(0)).cumprod(skipna=True)

        res.to_csv('factor_monthly_return.csv')
        nav.to_csv('factor_monthly_nav.csv')

    return res, nav

def backtest_visaulized(factor: str, ic_p_df: pd.DataFrame, ic_s_df: pd.DataFrame, month_ret_df: pd.DataFrame, month_nav_df: pd.DataFrame):
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ic_p_df.index, ic_p_df[factor], label = 'Pearson IC', color='blue', linewidth=1)
    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.axhline(y=ic_p_df[factor].mean(), color='orange', linestyle='-.', linewidth=3)
    ax1.set_title("Pearson IC (with mean) vs Time")
    ax1.set_ylabel("Pearson Information Coefficient")
    ax1.set_xlabel("Date")
    ax1.set_xticks(ic_p_df.index[::30])
    ax1.set_xticklabels([d for d in ic_p_df.index[::30]], rotation=45, ha="right")

    ax2 = fig.add_subplot(gs[0, 1], sharex=ax1)
    ax2.plot(ic_s_df.index, ic_s_df[factor], label = 'Spearman IC', color='purple', linewidth=1)
    ax2.axhline(y=0, color='red', linestyle='--')
    ax2.axhline(y=ic_s_df[factor].mean(), color='orange', linestyle='-.', linewidth=3)
    ax2.set_title("Spearman IC (with mean) vs Time")
    ax2.set_ylabel("Spearman Information Coefficient")
    ax2.set_xlabel("Date")
    ax2.set_xticks(ic_s_df.index[::30])
    ax2.set_xticklabels([d for d in ic_s_df.index[::30]], rotation=45, ha="right")

    ax3 = fig.add_subplot(gs[1, 0])
    for col in month_ret_df.columns:
        ax3.plot(month_ret_df.index, month_ret_df[col], marker='o', label=col)
    ax3.axhline(y=0, color='red', linestyle='--')
    ax3.set_title("Group Monthly Returns vs Time")
    ax3.set_ylabel("Monthly Return")
    ax3.set_xlabel("Date")
    ax3.legend()

    ax4 = fig.add_subplot(gs[1, 1])
    for col in month_nav_df.columns:
        ax4.plot(month_nav_df.index, month_nav_df[col], marker='o', label=col)
    ax4.axhline(y=1, color='red', linestyle='--')
    ax4.set_title("Group NAVs vs Time")
    ax4.set_ylabel("Net Asset Value")
    ax4.set_xlabel("Date")
    ax4.legend()

    plt.tight_layout()
    plt.savefig(f'backtest_{factor}.png')

    return fig

# price_df = get_df()
# price_df = pd.read_csv("/home/lawre/Southwest-Securities-Quant-Internship-Summer-2025/Momentum-Indicators/dataset.csv")  # For quicker debugging

# stat_df = pd.read_csv("/home/lawre/Southwest-Securities-Quant-Internship-Summer-2025/Momentum-Indicators/stock_stats.csv")
stat_df_small = pd.read_csv("/home/lawre/Southwest-Securities-Quant-Internship-Summer-2025/Momentum-Indicators/stock_stats_mini.csv")

factor = 'trnovr3_avg_ret'
ic_p_df, ic_sp_df = calculate_IC(stat_df_small, [factor])
bt_ret, bt_nav = backtest(stat_df_small, [factor], 3)
backtest_visaulized(factor, ic_p_df, ic_sp_df, bt_ret, bt_nav)