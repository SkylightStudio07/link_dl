# https://towardsdatascience.com/cryptocurrency-price-prediction-using-deep-learning-70cfca50dd3a
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import os
import torch
import pandas as pd
import numpy as np

BASE_PATH = str(Path(__file__).resolve().parent.parent.parent)  # BASE_PATH: /Users/yhhan/git/link_dl
import sys
sys.path.append(BASE_PATH)


class CryptoCurrencyDataset(Dataset):
  def __init__(self, X, y, is_regression=True):
    self.X = X
    self.y = y
    assert len(self.X) == len(self.y)

  def __len__(self):
    return len(self.X)

  def __getitem__(self, idx):
    X = self.X[idx]
    y = self.y[idx]
    return X, y

  def __str__(self):
    s = "Data Size: {0}, Input Shape: {1}, Target Shape: {2}".format(
      len(self.X), self.X.shape, self.y.shape
    )
    return s


def _load_and_clean_btc_csv(csv_path: str) -> pd.DataFrame:
  """
  Loads yfinance-exported CSV that may contain header-like rows:
    row0: Ticker, BTC-KRW, BTC-KRW, ...
    row1: Date,   NaN,     NaN,    ...
    row2~: actual data rows (Date in 'Price' column)
  Keeps raw CSV unchanged; cleans only in memory.
  Returns a DataFrame with columns: Date, Open, High, Low, Close, Volume
  """
  df = pd.read_csv(csv_path)

  # Expected columns in your raw CSV: Price, Close, High, Low, Open, Volume
  # If Date column is missing but Price exists, treat Price as Date and drop header-like rows.
  if 'Date' not in df.columns and 'Price' in df.columns:
    df = df[~df['Price'].isin(['Ticker', 'Date'])].copy()
    df = df.rename(columns={'Price': 'Date'})

  # Keep only required columns (avoid any stray columns)
  need_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
  missing = [c for c in need_cols if c not in df.columns]
  if missing:
    raise ValueError(f"CSV is missing required columns: {missing}")

  df = df[need_cols].copy()

  # Parse Date
  df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

  # Force numeric for features (string/과학표기/콤마 포함 모두 처리)
  for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
    df[c] = pd.to_numeric(
        df[c].astype(str).str.replace(',', ''),
        errors='coerce'
    )
  df = df.dropna().reset_index(drop=True)

  # ==================== 문제 3 ====================
  df["Next_Open"] = df["Open"].shift(-1)
  df = df.dropna().reset_index(drop=True)
  # ===============================================

  return df


def get_cryptocurrency_data(
    sequence_size=21, validation_size=150, test_size=30,
    target_column='Close', y_normalizer=1.0e7, is_regression=True
):
  btc_krw_path = os.path.join("BTC_KRW_2025_11_30.csv")

  # 1) Load & clean (no file modification)
  df = _load_and_clean_btc_csv(btc_krw_path)

  # 2) Split date list for plotting/labels (aligned with cleaned rows)
  date_list = df['Date'].tolist()

  # 3) Make feature matrix as float32 numpy (object dtype 절대 안 나오게 고정)
  feature_cols = ["Open", "High", "Low", "Close", "Volume", "Next_Open"]
  all_data = df[feature_cols].to_numpy(dtype=np.float32)
  if target_column not in feature_cols:
    raise ValueError(f"target_column must be one of {feature_cols}, got {target_column}")

  all_data = df[feature_cols].to_numpy(dtype=np.float32)
  target_idx = feature_cols.index(target_column)

  # 4) Compute sizes
  row_size = all_data.shape[0]
  data_size = row_size - sequence_size
  train_size = data_size - (validation_size + test_size)
  if train_size <= 0:
    raise ValueError(
      f"Not enough data. row_size={row_size}, sequence_size={sequence_size}, "
      f"validation_size={validation_size}, test_size={test_size}"
    )

  row_cursor = 0

  # ============================= TRAIN =============================
  X_train_list = []
  y_train_regression_list = []
  y_train_classification_list = []
  y_train_date = []

  for idx in range(0, train_size):
    sequence_data = all_data[idx: idx + sequence_size]  # (sequence_size, 5) float32
    X_train_list.append(torch.from_numpy(sequence_data))

    y_t = float(all_data[idx + sequence_size, target_idx])
    y_prev = float(all_data[idx + sequence_size - 1, target_idx])

    y_train_regression_list.append(y_t)
    y_train_classification_list.append(1 if y_t >= y_prev else 0)
    y_train_date.append(date_list[idx + sequence_size])

    row_cursor += 1

  X_train = torch.stack(X_train_list, dim=0).to(torch.float)
  y_train_regression = torch.tensor(y_train_regression_list, dtype=torch.float32) / y_normalizer
  y_train_classification = torch.tensor(y_train_classification_list, dtype=torch.int64)

  m = X_train.mean(dim=0, keepdim=True)
  s = X_train.std(dim=0, keepdim=True)
  X_train = (X_train - m) / s

  # =========================== VALIDATION ==========================
  X_validation_list = []
  y_validation_regression_list = []
  y_validation_classification_list = []
  y_validation_date = []

  for idx in range(row_cursor, row_cursor + validation_size):
    sequence_data = all_data[idx: idx + sequence_size]
    X_validation_list.append(torch.from_numpy(sequence_data))

    y_t = float(all_data[idx + sequence_size, target_idx])
    y_prev = float(all_data[idx + sequence_size - 1, target_idx])

    y_validation_regression_list.append(y_t)
    y_validation_classification_list.append(1 if y_t >= y_prev else 0)
    y_validation_date.append(date_list[idx + sequence_size])

    row_cursor += 1

  X_validation = torch.stack(X_validation_list, dim=0).to(torch.float)
  y_validation_regression = torch.tensor(y_validation_regression_list, dtype=torch.float32) / y_normalizer
  y_validation_classification = torch.tensor(y_validation_classification_list, dtype=torch.int64)
  X_validation = (X_validation - m) / s

  # ============================== TEST =============================
  X_test_list = []
  y_test_regression_list = []
  y_test_classification_list = []
  y_test_date = []

  for idx in range(row_cursor, row_cursor + test_size):
    sequence_data = all_data[idx: idx + sequence_size]
    X_test_list.append(torch.from_numpy(sequence_data))

    y_t = float(all_data[idx + sequence_size, target_idx])
    y_prev = float(all_data[idx + sequence_size - 1, target_idx])

    y_test_regression_list.append(y_t)
    # 원본 코드가 test에서만 '>'를 쓰고 있었길래 그대로 유지
    y_test_classification_list.append(1 if y_t > y_prev else 0)
    y_test_date.append(date_list[idx + sequence_size])

    row_cursor += 1

  X_test = torch.stack(X_test_list, dim=0).to(torch.float)
  y_test_regression = torch.tensor(y_test_regression_list, dtype=torch.float32) / y_normalizer
  y_test_classification = torch.tensor(y_test_classification_list, dtype=torch.int64)
  X_test = (X_test - m) / s

  if is_regression:
    return (
      X_train, X_validation, X_test,
      y_train_regression, y_validation_regression, y_test_regression,
      y_train_date, y_validation_date, y_test_date
    )
  else:
    return (
      X_train, X_validation, X_test,
      y_train_classification, y_validation_classification, y_test_classification,
      y_train_date, y_validation_date, y_test_date
    )


if __name__ == "__main__":
  is_regression = False

  X_train, X_validation, X_test, y_train, y_validation, y_test, y_train_date, y_validation_date, y_test_date \
    = get_cryptocurrency_data(
      sequence_size=10, validation_size=100, test_size=10,
      target_column='Close', y_normalizer=1.0e7, is_regression=is_regression
    )

  train_crypto_currency_dataset = CryptoCurrencyDataset(X=X_train, y=y_train, is_regression=is_regression)
  validation_crypto_currency_dataset = CryptoCurrencyDataset(X=X_validation, y=y_validation, is_regression=is_regression)
  test_crypto_currency_dataset = CryptoCurrencyDataset(X=X_test, y=y_test, is_regression=is_regression)

  train_data_loader = DataLoader(
    dataset=train_crypto_currency_dataset,
    batch_size=32,
    shuffle=True,
    drop_last=True
  )

  for idx, batch in enumerate(train_data_loader):
    input, target = batch
    print("{0} - {1}: {2}, {3}".format(idx, input.shape, target.shape, target))
