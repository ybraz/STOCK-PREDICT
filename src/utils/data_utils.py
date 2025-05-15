from __future__ import annotations
import os
from typing import Tuple
import numpy as np
import pandas as pd
import yfinance as yf

def download_prices(symbol: str, start: str, end: str, cache_dir: str = "data/raw") -> pd.DataFrame:
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{symbol}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, parse_dates=["Date"])
        df.set_index("Date", inplace=True)
    else:
        df = yf.download(symbol, start=start, end=end, auto_adjust=True)

        # Se vier MultiIndex, corrige:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)  # Pega apenas o primeiro nível: 'Close', 'High', etc.

        df.columns.name = None
        df.reset_index(inplace=True)
        df.to_csv(cache_path, index=False)
        df.set_index("Date", inplace=True)

    # Agora filtra as colunas corretas
    expected_cols = ["Close", "Volume", "High", "Low", "Open"]
    available_cols = [col for col in expected_cols if col in df.columns]

    if "Close" not in available_cols:
        raise ValueError(f"O ativo {symbol} não possui coluna 'Close' disponível.")

    df = df[available_cols]

    return df

def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def train_val_test_split(series: np.ndarray, val_size: float, test_size: float):
    n_total = len(series)
    n_test = int(n_total * test_size)
    n_val = int(n_total * val_size)
    train, val, test = (
        series[:n_total - n_val - n_test],
        series[n_total - n_val - n_test: n_total - n_test],
        series[n_total - n_test:],
    )
    return train, val, test
