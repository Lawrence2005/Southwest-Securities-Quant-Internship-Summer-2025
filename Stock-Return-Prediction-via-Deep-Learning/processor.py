import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr

import matplotlib.pyplot as pl

DATA_FILE_NAME = "stock_price_vol_d.txt"

# df = pd.read_feather(DATA_FILE_NAME)  # Feather file
# df = df.drop(columns=['induID2', 'indup'])  # Drop non-features
# df.to_csv("stock_data.csv", index=False, encoding="utf-8")

# ------------------------------------------------------------迷你测试仿数据--------------------------------------------------------------------------------
dates = pd.date_range("2013-01-01", "2024-12-31", freq="D")
# 股票列表
stocks = ["000001.SZ", "000002.SZ", "000003.SZ"]
# 特征列
features = ["open", "high", "low", "close", "amount"]
data = []
for stock in stocks:
    stock_name = "平安银行" if stock == "000001.SZ" else "万科A"
    base_price = 10 if stock == "000001.SZ" else 20
    
    for d in dates:
        open_p = base_price + np.random.randn()  # 随机开盘价
        close_p = open_p + np.random.randn() * 0.5
        high_p = max(open_p, close_p) + abs(np.random.randn()) * 0.2
        low_p = min(open_p, close_p) - abs(np.random.randn()) * 0.2
        volume = np.random.randint(1000, 5000)
        data.append([stock, stock_name, d, open_p, high_p, low_p, close_p, volume])
# 转 DataFrame
df = pd.DataFrame(data, columns=["StockID", "stockName", "date"] + features)
# ------------------------------------------------------------迷你测试仿数据--------------------------------------------------------------------------------

ALL_STOCKS = df['StockID'].unique()
FEATURES = ['open', 'high', 'low', 'close', 'amount']
LOOKBACK =40
CUTOFF_DATE = "2023-12-31"
MAX_WINDOW = 2000

def build_sample(df, dates, lookback, features):
    X = []
    for i in range(lookback, len(dates)+1):
        sample = df[df['date'].isin(dates[i-lookback:i])]

        pivoted = sample.pivot(index='date', columns='StockID', values=features)

        pivoted_arr = []
        for stock in pivoted.columns.get_level_values(1).unique():
            stock_data = pivoted[[(f, stock) for f in features if (f, stock) in pivoted.columns]].values
            stock_data = stock_data[:-1, :]  # remove the last row
            pivoted_arr.append(stock_data)

        X.append(np.stack(pivoted_arr, axis=0))
    
    return np.array(X)

def build_label(df, dates, lookback):
    Y = []
    for i in range(lookback, len(dates)+1):
        last_day = dates[i-1]
        Y.append(df[df['date']==last_day]['label'].values)
    return np.array(Y)

def process_data(df, cutoff_date = CUTOFF_DATE, tickers=ALL_STOCKS, lookback=LOOKBACK, features=FEATURES, max_window=MAX_WINDOW):
    df = df[df['StockID'].isin(tickers)]  # Filter by stock IDs

    df = df.ffill()

    df['close_shifted'] = df.groupby('StockID')['close'].shift(-20)
    df['label'] = df['close_shifted'] / df['close'] - 1
    df = df.drop(columns=['close_shifted'])
    df['label'] = df.groupby('date')['label'].transform(lambda x: (x - x.mean()) / x.std(ddof=0))

    df['date'] = pd.to_datetime(df['date'])
    cutoff_date = pd.to_datetime(cutoff_date)

    trainval_dates = df[(df['date'] >= (cutoff_date - pd.DateOffset(years=10))) & (df['date'] < cutoff_date)]['date'].unique()

    split_idx = int(len(trainval_dates) * 0.8)
    train_dates = trainval_dates[:split_idx]
    val_dates   = trainval_dates[split_idx:]

    test_dates = df[df['date'].between("2024-01-01", "2024-12-31")]['date'].unique()

    if max_window is not None:
        train_dates = train_dates[-max_window:] if len(train_dates) > max_window else train_dates
        test_dates = test_dates[:max_window] if len(test_dates) > max_window else test_dates

    X_train = build_sample(df, train_dates, lookback, features)
    X_val = build_sample(df, val_dates, lookback, features)
    X_test = build_sample(df, test_dates, lookback, features)

    Y_train = build_label(df, train_dates, lookback)
    Y_val   = build_label(df, val_dates, lookback)
    Y_test  = build_label(df, test_dates, lookback)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test

class StockSampleDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype=torch.float32)  # X: [num_samples, num_stocks, seq_len, features]
        self.Y = torch.tensor(Y, dtype=torch.float32)  # Y: [num_samples, num_stocks]
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]  # [num_stocks, seq_len, features], [num_stocks]

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :]).squeeze()

class TransformerModel(nn.Module):
    def __init__(self, input_dim=len(FEATURES), hidden_dim=64, num_heads=4, num_layers=2):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        out = self.transformer(x)
        return self.fc(out[:, -1, :]).squeeze()

def train_model(model, X_train, Y_train, X_val, Y_val, epochs=5, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 创建 dataset 和 dataloader，每个 batch 是一个 sample
    train_loader = DataLoader(StockSampleDataset(X_train, Y_train), batch_size=1, shuffle=True)
    val_loader   = DataLoader(StockSampleDataset(X_val, Y_val), batch_size=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.squeeze(0).to(device)  # [num_stocks, seq_len, features]
            y_batch = y_batch.squeeze(0).to(device)  # [num_stocks]

            optimizer.zero_grad()
            pred = model(X_batch)  # [num_stocks]
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_losses.append(train_loss / len(train_loader))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.squeeze(0).to(device)
                y_batch = y_batch.squeeze(0).to(device)
                pred = model(X_batch)
                val_loss += criterion(pred, y_batch).item()
        
        val_losses.append(val_loss / len(val_loader))

    return model, train_losses, val_losses

class MetricsMonitor:
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true).flatten()
        self.y_pred = np.array(y_pred).flatten()

    def compute_metrics(self):
        mse = mean_squared_error(self.y_true, self.y_pred)
        mae = mean_absolute_error(self.y_true, self.y_pred)
        r2 = r2_score(self.y_true, self.y_pred)
        ic = np.corrcoef(self.y_pred, self.y_true)[0, 1]
        rank_ic, _ = spearmanr(self.y_pred, self.y_true)
        bias = np.mean(self.y_pred - self.y_true)

        metrics = {
            "MSE": mse,
            "MAE": mae,
            "R2": r2,
            "IC": ic,
            "RankIC": rank_ic,
            "Bias": bias
        }
        return pd.DataFrame(metrics, index=["Value"]).T

    def plot_loss_curves(self, train_losses=None, val_losses=None):
        if train_losses is None or val_losses is None:
            print("⚠️ No training/validation loss provided.")
            return
        plt.plot(train_losses, label="Train Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Training vs Validation Loss")
        plt.savefig("loss_curve.png")

    def plot_pred_vs_true(self, n=200):
        plt.plot(self.y_true[:n], label="True")
        plt.plot(self.y_pred[:n], label="Pred")
        plt.legend()
        plt.title(f"Prediction vs True (first {n} samples)")
        plt.savefig("Pred_vs_True.png")

    def plot_residuals(self):
        residuals = self.y_true - self.y_pred
        plt.hist(residuals, bins=50, alpha=0.7)
        plt.title("Residual Distribution")
        plt.savefig("Residuals.png")

    def plot_scatter(self):
        plt.scatter(self.y_true, self.y_pred, alpha=0.5)
        min_val, max_val = min(self.y_true.min(), self.y_pred.min()), max(self.y_true.max(), self.y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], "r--")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title("Pred vs True Scatter")
        plt.savefig("Scatter.png")

    def plot_rolling_ic(self, window=50):
        rolling_ic = [np.corrcoef(self.y_pred[i:i+window], self.y_true[i:i+window])[0,1] 
                      for i in range(len(self.y_pred)-window)]
        plt.plot(rolling_ic)
        plt.title(f"Rolling IC (window={window})")
        plt.xlabel("Index")
        plt.ylabel("IC")
        plt.savefig("Rolling_IC.png")

    def run_all(self, train_losses=None, val_losses=None):
        print("📊 Metrics Summary Done")
        self.plot_loss_curves(train_losses, val_losses)
        self.plot_pred_vs_true()
        self.plot_residuals()
        self.plot_scatter()
        self.plot_rolling_ic()

X_tr, Y_tr, X_val, Y_val, X_te, Y_te = process_data(df, max_window=None)
X_te, Y_te = X_te[~np.isnan(Y_te).any(axis=1)], Y_te[~np.isnan(Y_te).any(axis=1)]

model = TransformerModel(input_dim=len(FEATURES), hidden_dim=64, num_layers=2)
model, train_loss, val_loss = train_model(model, X_tr[:10], Y_tr[:10], X_val[:5], Y_val[:5], epochs=5, lr=1e-3)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
test_loader = DataLoader(StockSampleDataset(X_te, Y_te), batch_size=1)

model.eval()  # 切换到评估模式
preds, truths = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.squeeze(0).to(device)  # [num_stocks, seq_len, features]
        y_batch = y_batch.squeeze(0).to(device)  # [num_stocks]

        pred = model(X_batch)  # [num_stocks]
        preds.extend(pred.cpu().tolist())
        truths.extend(y_batch.cpu().tolist())

monitor = MetricsMonitor(y_true=truths, y_pred=preds)

monitor.run_all(train_losses=train_loss, val_losses=val_loss)

# ------------------------------------------------------------------------------------------------------------------------------------------
# import time

# start = time.time()

# X_tr, Y_tr, X_val, Y_val, X_te, Y_te = process_data(df, max_window=None)
# print("X_train shape:", X_tr.shape)
# print("Y_ttrain shape:", Y_tr.shape)

# end = time.time()

# length = end - start
# print("It took", length, "seconds!")