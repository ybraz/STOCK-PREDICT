"""Funções auxiliares para download e pré-processamento dos dados."""
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
        df = pd.read_csv(cache_path, index_col="Date", parse_dates=True)
    else:
        df = yf.download(symbol, start=start, end=end)
        df.index.name = "Date"
        df.to_csv(cache_path)

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
