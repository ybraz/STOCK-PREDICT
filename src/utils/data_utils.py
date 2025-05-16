#!/usr/bin/env python3
"""
Dados + indicadores técnicos  •  modelo LSTM com Dropout
e hyper-tuning via Keras-Tuner.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from keras_tuner import RandomSearch
from pandas.tseries.offsets import BDay
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

BASE = Path(__file__).resolve().parents[2]
CACHE = BASE / "data" / "raw"
MODELS = BASE / "models"
CACHE.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# ── funções de indicadores simples ─────────────────────────────
def sma(series: pd.Series, n: int):
    return series.rolling(n).mean()

def ema(series: pd.Series, n: int):
    return series.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, n: int = 14):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = -delta.clip(upper=0).rolling(n).mean()
    rs = up / (down + 1e-8)
    return 100 - 100 / (1 + rs)

def macd(series: pd.Series):
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    line = ema12 - ema26
    signal = line.ewm(span=9, adjust=False).mean()
    hist = line - signal
    return line, signal, hist

# ── download & cache ───────────────────────────────────────────
def _cache(symbol): return CACHE / f"{symbol}.csv"

def download_prices(symbol: str, start="2005-01-01",
                    end: str | None = None) -> pd.DataFrame:
    if end is None:
        end = (pd.Timestamp.utcnow().normalize() - BDay(1)).date().isoformat()
    p = _cache(symbol)
    if p.exists():
        df = pd.read_csv(p, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce")
        if not df.empty and df.index[-1].date().isoformat() >= end:
            return df[df.index <= end]
    df = yf.download(symbol, start=start, end=end, interval="1d",
                     auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"Sem dados para {symbol}")
    df.to_csv(p)
    return df

# ── feature engineering ───────────────────────────────────────
def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=str.title)
    df_feat = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    # médias móveis
    df_feat["SMA_5"] = sma(df_feat["Close"], 5)
    df_feat["SMA_10"] = sma(df_feat["Close"], 10)
    df_feat["SMA_20"] = sma(df_feat["Close"], 20)
    df_feat["EMA_10"] = ema(df_feat["Close"], 10)
    # RSI & MACD
    df_feat["RSI_14"] = rsi(df_feat["Close"])
    macd_line, macd_sig, macd_hist = macd(df_feat["Close"])
    df_feat["MACD"] = macd_line
    df_feat["MACD_Signal"] = macd_sig
    df_feat["MACD_Hist"] = macd_hist
    # retorno alvo
    df_feat["Return"] = df_feat["Close"] / df_feat["Open"] - 1
    df_feat = df_feat.dropna()
    return df_feat

def fetch_recent_features(symbol: str, seq_len: int) -> pd.DataFrame:
    df = build_feature_frame(download_prices(symbol))
    return df.iloc[-seq_len:].drop(columns=["Return"])

# ── sequências e escalers ─────────────────────────────────────
def make_sequences(df: pd.DataFrame, seq_len: int
                   ) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    X_cols = df.columns.drop("Return")
    scaler_x = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(df[X_cols].values)
    scaler_y = MinMaxScaler(feature_range=(-1, 1))
    y_scaled = scaler_y.fit_transform(df["Return"].values.reshape(-1, 1))
    X, y = [], []
    for i in range(seq_len, len(df)):
        X.append(X_scaled[i - seq_len:i])
        y.append(y_scaled[i])
    return np.array(X), np.array(y), scaler_x, scaler_y

# ── modelo base ────────────────────────────────────────────────
def build_model(seq_len: int, n_features: int,
                lstm_units=64, dropout=0.2, lr=1e-3):
    m = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(lstm_units, return_sequences=True),
        Dropout(dropout),
        LSTM(lstm_units // 2),
        Dropout(dropout),
        Dense(1, activation="tanh"),
    ])
    m.compile(optimizer=Adam(lr), loss="mse")
    return m

# ── hyper-tuning & treino final ───────────────────────────────
def train_and_save(symbol: str, seq_len=180, epochs=30,
                   start="2005-01-01", trials=10):
    df = build_feature_frame(download_prices(symbol, start=start))
    X, y, scaler_x, scaler_y = make_sequences(df, seq_len)
    n_features = X.shape[2]

    # hypermodel closure recebe n_features fixo
    def model_builder(hp):
        units   = hp.Choice("units", [32, 64, 128])
        dropout = hp.Choice("dropout", [0.1, 0.2, 0.3])
        lr      = hp.Choice("lr", [1e-3, 5e-4, 1e-4])
        return build_model(seq_len, n_features, units, dropout, lr)

    tuner = RandomSearch(
        model_builder,
        objective="val_loss",
        max_trials=trials,
        directory="tuner",
        project_name=f"{symbol}_ret",
        overwrite=True,
    )
    tuner.search(X, y, validation_split=0.2, epochs=20, verbose=0)
    best_hp = tuner.get_best_hyperparameters(1)[0]
    model = tuner.hypermodel.build(best_hp)

    es = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    model.fit(X, y, epochs=epochs, batch_size=32,
              validation_split=0.2, callbacks=[es], verbose=2)

    model.save(MODELS / f"lstm_{symbol}.keras")
    joblib.dump({"x": scaler_x, "y": scaler_y}, MODELS / f"scalers_{symbol}.pkl")

def load_model_and_scalers(symbol: str):
    return (
        load_model(MODELS / f"lstm_{symbol}.keras"),
        joblib.load(MODELS / f"scalers_{symbol}.pkl"),
    )