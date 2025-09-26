import pandas as pd
import numpy as np
from typing import Callable
from numba import njit

from scipy.stats import spearmanr, pearsonr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

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

    calculate_rolling_features(df, result, 'cum_ret', [20, 40, 80])
    calculate_rolling_features(df, result, 'ret_stdev')
    calculate_rolling_features(df, result, 'pos_ret_stdev')
    calculate_rolling_features(df, result, 'neg_ret_stdev')
    calculate_rolling_features(df, result, 'turnover_z', windows = [10])

    calculate_turnover_features(df, result)

    result.to_csv("stock_stats.csv", index=False)

    return result

def calculate_rolling_features(df: pd.DataFrame, result: pd.DataFrame, feature_name: str, windows: list[int] = [20, 40, 60]):
    for length in windows:
        if feature_name == 'cum_ret':
            result[f'{feature_name}_{length}'] = df.groupby('stockid')['return'].rolling(window=length).sum().reset_index(level=0, drop=True).astype('float32').round(4)
        elif feature_name == 'ret_stdev':
            result[f'{feature_name}_{length}'] = df.groupby('stockid')['return'].rolling(window=length).std(ddof=1).reset_index(level=0, drop=True).astype('float32').round(4)
        elif feature_name == 'pos_ret_stdev':
            result[f'{feature_name}_{length}'] = df['return'].where(df['return'] >= 0).groupby(df['stockid']).rolling(window=length, min_periods=2).std(ddof=1).reset_index(level=0, drop=True).astype('float32').round(4)
        elif feature_name == 'neg_ret_stdev':
            result[f'{feature_name}_{length}'] = df['return'].where(df['return'] < 0).groupby(df['stockid']).rolling(window=length, min_periods=2).std(ddof=1).reset_index(level=0, drop=True).astype('float32').round(4)
        else:
            means = df.groupby('stockid')['turnoverrate'].rolling(window=10, min_periods=5).mean().reset_index(level=0, drop=True).astype('float32').round(4)
            stdevs = df.groupby('stockid')['turnoverrate'].rolling(window=10, min_periods=5).std(ddof=1).reset_index(level=0, drop=True).astype('float32').round(4)
            result[f'{feature_name}_{length}'] = (np.sign(df['return']) * (df['turnoverrate'] - means) / stdevs).astype('float32').round(4)

def half_life_weights(n: int, h: int = 4):
    k = np.arange(n - 1, -1, -1)
    return 0.5 ** (k/h)

def calculate_turnover_features(df: pd.DataFrame, result: pd.DataFrame, window: int = 20, num_groups: int = 4):
    result_cols = [f'trnovr{i}_avg_ret' for i in range(1, num_groups + 1)]

    result_df = df.groupby('stockid', group_keys=False).apply(
        lambda g: pd.DataFrame(
            calculate_turnover_returns(g['turnoverrate'].to_numpy(), g['return'].to_numpy(), window, num_groups),
            index=g.index,
            columns=result_cols
        )
    )
    
    for col in result_cols:
        result[col] = result_df[col].values.astype('float32').round(4)

# @njit
def calculate_turnover_returns(turnovers, returns, window, num_groups):
    result = np.full((len(turnovers), num_groups), np.nan)
    
    for i in range(window - 1, len(turnovers)):
        window_turnover, window_returns = turnovers[i-window+1:i+1], returns[i-window+1:i+1]

        mask_valid = ~np.isnan(window_turnover) & ~np.isnan(window_returns)
        window_turnover = window_turnover[mask_valid]
        window_returns = window_returns[mask_valid]

        if len(window_turnover) < num_groups or len(window_returns) < num_groups:
            continue

        q = np.quantile(window_turnover, np.linspace(0, 1, num_groups + 1))
        for j in range(num_groups):
            mask = (window_turnover >= q[j]) & (window_turnover <= q[j+1]) if j == 0 else (window_turnover > q[j]) & (window_turnover <= q[j+1])

            group_returns = window_returns[mask]
            if len(group_returns) > 0:
                w = half_life_weights(len(group_returns))
                result[i, j] = np.sum(group_returns * w)
            else:
                result[i, j] = np.nan
    return result

def kmeans_regime(df: pd.DataFrame, n_clusters: int = 3):
    df['log_return'] = np.log(df['close']).diff()
    df['momentum'] = df['close'].pct_change(20)
    df['ret_stdev'] = df['return'].rolling(60).std(ddof=1).astype('float32').round(4)
    df['ret_vol_adj'] = df['return'] / df['ret_stdev']

    # Standardize each feature
    features = df[['log_return', 'momentum', 'ret_vol_adj']].dropna()
    scaled = StandardScaler().fit_transform(features)

    models = [KMeans(n, init='k-means++').fit(scaled) for n in np.arange(1, 7)]

    plt.plot(np.arange(1, 7), [m.inertia_ for m in models], label='Inertia')
    plt.legend(loc='best')
    plt.title('K-Means Elbow Graph')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Inertia')
    plt.savefig('K-Means Elbow Graph.png')

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df.loc[features.index, 'regime'] = kmeans.fit_predict(scaled)

    cluster_means = features.copy()
    cluster_means['cluster'] = kmeans.labels_

    sns.pairplot(cluster_means, hue='cluster')
    plt.savefig('K-Means Performance.png')

    return df, cluster_means

def assign_regime(df: pd.DataFrame, regime_map: dict[int, str] = {0: 'Bull', 1: 'Bear', 2: 'Reversal', 3: 'Other'}):
    df['regime'] = df['regime'].map(regime_map)
    return df

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

    cum_ic_pearson_df = ic_pearson_df.cumsum().round(4)
    cum_ic_spearman_df = ic_spearman_df.cumsum().round(4)

    ic_pearson_df.to_csv("ic_pearson.csv")
    ic_spearman_df.to_csv("ic_spearman.csv")
    cum_ic_pearson_df.to_csv("cum_ic_pearson.csv")
    cum_ic_spearman_df.to_csv("cum_ic_spearman.csv")

    return ic_pearson_df, ic_spearman_df, cum_ic_pearson_df, cum_ic_spearman_df

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

def shade_regime(df: pd.DataFrame, ax: plt.Axes):
    ymin, ymax = ax.get_ylim()
    df = df.set_index('date')

    ax.fill_between(df.index, ymin, ymax,
                 where=(df['regime'] == 'Bull'), color='green', alpha=0.3, label='Bull')
    ax.fill_between(df.index, ymin, ymax,
                    where=(df['regime'] == 'Bear'), color='red', alpha=0.3, label='Bear')
    ax.fill_between(df.index, ymin, ymax,
                    where=(df['regime'] == 'Reversal'), color='grey', alpha=0.3, label='Reversal')

def backtest_visaulized(factor: str, regime_df: pd.DataFrame, ic_p_df: pd.DataFrame, ic_s_df: pd.DataFrame, cum_ic_p_df: pd.DataFrame, cum_ic_s_df: pd.DataFrame, month_ret_df: pd.DataFrame, month_nav_df: pd.DataFrame, industry: str = ''):
    fig = plt.figure(figsize=(20, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = ax1.twinx()
    ax1.plot(cum_ic_p_df.index, cum_ic_p_df[factor], label='Cumulative IC', color='blue', linewidth=1)
    ax1.set_ylabel('Cumulative IC', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2.bar(ic_p_df.index, ic_p_df[factor], label='Daily IC', color='black', alpha = 0.5)
    ax2.axhline(y=ic_p_df[factor].mean(), color='orange', linestyle='-.', linewidth=3)
    ax2.set_ylabel('Daily IC')
    ax2.tick_params(axis='y')

    shade_regime(regime_df, ax1)

    ax1.axhline(y=0, color='red', linestyle='--')
    ax1.set_title(f"Pearson IC (mean = {ic_p_df[factor].mean() * 100: .2f}%) vs Time")
    ax1.set_ylabel("Pearson Information Coefficient")
    ax1.set_xlabel("Date")
    ax1.set_xticks(ic_p_df.index[::30])
    ax1.set_xticklabels([d for d in ic_p_df.index[::30]], rotation=45, ha="right")

    ax3 = fig.add_subplot(gs[0, 1], sharex=ax1)
    ax4 = ax3.twinx()
    ax3.plot(cum_ic_s_df.index, cum_ic_s_df[factor], label = 'Cumulative IC', color='purple', linewidth=1)
    ax3.set_ylabel('Cumulative IC', color='purple')
    ax3.tick_params(axis='y', labelcolor='purple')

    ax4.bar(ic_s_df.index, ic_s_df[factor], label="Daily IC", color="black", alpha = 0.5)
    ax4.axhline(y=ic_s_df[factor].mean(), color='orange', linestyle='-.', linewidth=3)
    ax4.set_ylabel('Daily IC')
    ax4.tick_params(axis='y')

    shade_regime(regime_df, ax3)

    ax3.axhline(y=0, color='red', linestyle='--')
    ax3.set_title(f"Spearman IC (mean = {ic_s_df[factor].mean() * 100: .2f}%) vs Time")
    ax3.set_ylabel("Spearman Information Coefficient")
    ax3.set_xlabel("Date")
    ax3.set_xticks(ic_s_df.index[::30])
    ax3.set_xticklabels([d for d in ic_s_df.index[::30]], rotation=45, ha="right")

    ax5 = fig.add_subplot(gs[1, 0])
    for col in month_ret_df.columns:
        ax5.plot(month_ret_df.index, month_ret_df[col], marker='o', label=col)
    ax5.axhline(y=0, color='red', linestyle='--')
    ax5.set_title("Group Monthly Returns")
    ax5.set_ylabel("Monthly Return")
    ax5.set_xlabel("Date")
    ax5.legend()

    ax6 = fig.add_subplot(gs[1, 1])
    for col in month_nav_df.columns:
        ax6.plot(month_nav_df.index, month_nav_df[col], marker='o', label=col)
    ax6.axhline(y=1, color='red', linestyle='--')
    ax6.set_title("Group NAVs")
    ax6.set_ylabel("Net Asset Value")
    ax6.set_xlabel("Date")
    ax6.legend()

    plt.tight_layout()
    plt.savefig(f'backtest_{factor}.png')

    return fig

industry_df = pd.read_csv('dataset.csv')

# price_df = pd.read_csv("/home/lawre/Southwest-Securities-Quant-Internship-Summer-2025/Momentum-Indicators/dataset.csv") # price_df = get_df()

sse_comp_df = pd.read_csv('sse_comp_index.csv')
regime_df = sse_comp_df
regime_df['regime'] = np.where(regime_df['return'] >= 0, 'Bull', 'Bear')

# regime_df, metrics = kmeans_regime(sse_comp_df, 3)
# regime_df = assign_regime(regime_df)

# # stat_df = pd.read_csv("/home/lawre/Southwest-Securities-Quant-Internship-Summer-2025/Momentum-Indicators/stock_stats.csv")
# stat_df_small = pd.read_csv("/home/lawre/Southwest-Securities-Quant-Internship-Summer-2025/Momentum-Indicators/stock_stats_mini.csv")

# stat_df_small["cum_ret_40_posvol"] = stat_df_small["cum_ret_40"] / stat_df_small['pos_ret_stdev_60']

# ind = 1000042205 # 汽车: 1000042194; 房地产：1000042202；银行：1000042205； 计算机：1000042213

# industry_df = entire_df[entire_df['industry'] == ind].copy()

# industry_df['cum_ret_20'] = industry_df.groupby('stockid')['return'].rolling(20).sum().reset_index(level=0, drop=True).astype('float32').round(4)
# industry_df['cum_ret_80'] = industry_df.groupby('stockid')['return'].rolling(80).sum().reset_index(level=0, drop=True).astype('float32').round(4)
industry_df['cum_ret_20'] = industry_df.groupby('stockid')['return'].transform(lambda x: (1 + x).rolling(20).apply(np.prod, raw=True) - 1).astype('float32').round(4)
# industry_df['cum_ret_40'] = industry_df.groupby('stockid')['return'].transform(lambda x: (1 + x).rolling(40).apply(np.prod, raw=True) - 1).astype('float32').round(4)
industry_df['cum_ret_80'] = industry_df.groupby('stockid')['return'].transform(lambda x: (1 + x).rolling(80).apply(np.prod, raw=True) - 1).astype('float32').round(4)
industry_df['ret_stdev'] = industry_df.groupby('stockid')['return'].rolling(60).std(ddof=1).reset_index(level=0, drop=True).astype('float32').round(4)
# industry_df["cum_ret_60_prior_vol"] = (industry_df["cum_ret_80"] - industry_df["cum_ret_20"]) / industry_df['ret_stdev']
# industry_df["cum_ret_60_prior"] = (industry_df["cum_ret_80"] - industry_df["cum_ret_20"])
# industry_df['ma_20'] = industry_df.groupby('stockid')['return'].rolling(window=20).mean().reset_index(level=0, drop=True).astype('float32').round(4)
# industry_df['ret-ma20'] = industry_df['return'] - industry_df['ma_20']
industry_df['cum_ret_80_vol'] = industry_df['cum_ret_80'] / industry_df['ret_stdev']

industry_df = industry_df[industry_df['date'] >= '2023-01-01']
regime_df = regime_df[regime_df['date'] >= '2023-01-01']

factor = 'cum_ret_80_vol'
ic_p_df, ic_sp_df, cum_ic_p_df, cum_ic_sp_df = calculate_IC(industry_df, [factor])
bt_ret, bt_nav = backtest(industry_df, [factor], 3)
backtest_visaulized(factor, regime_df, ic_p_df, ic_sp_df, cum_ic_p_df, cum_ic_sp_df, bt_ret, bt_nav)