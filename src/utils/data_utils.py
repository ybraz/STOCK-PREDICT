#!/usr/bin/env python3
"""
Módulo para previsão de preços de ativos financeiros utilizando LSTM com Dropout,
engenharia de atributos baseada em indicadores técnicos e _hyper-tuning_ via Keras-Tuner.

Desenvolvido para uso acadêmico, com foco em boas práticas, modularização, documentação
e clareza de código.
"""

from __future__ import annotations
from pathlib import Path
from typing import Tuple
import os

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
from tensorflow.keras.callbacks import TensorBoard

# ─────────────── Configuração de diretórios para cache e modelos ───────────────
BASE = Path(__file__).resolve().parents[2]
CACHE = BASE / "data" / "raw"
MODELS = BASE / "models"
CACHE.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)

# ─────────────── Indicadores Técnicos ───────────────
def sma(series: pd.Series, n: int) -> pd.Series:
    """
    Calcula a média móvel simples.
    """
    return series.rolling(n).mean()

def ema(series: pd.Series, n: int) -> pd.Series:
    """
    Calcula a média móvel exponencial.
    """
    return series.ewm(span=n, adjust=False).mean()

def rsi(series: pd.Series, n: int = 14) -> pd.Series:
    """
    Calcula o RSI (Índice de Força Relativa).
    """
    delta = series.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = -delta.clip(upper=0).rolling(n).mean()
    rs = up / (down + 1e-8)
    return 100 - 100 / (1 + rs)

def macd(series: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calcula o MACD (linha, sinal, histograma).
    """
    ema12 = ema(series, 12)
    ema26 = ema(series, 26)
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist

# ─────────────── Download e Cache de Dados ───────────────
def _cache(symbol: str) -> Path:
    return CACHE / f"{symbol}.csv"

def download_prices(symbol: str, start: str = "2005-01-01", end: str | None = None) -> pd.DataFrame:
    """
    Faz download dos preços históricos de um ativo usando o Yahoo Finance,
    com mecanismo de cache para evitar downloads repetidos.
    """
    if end is None:
        end = (pd.Timestamp.utcnow().normalize() - BDay(1)).date().isoformat()
    p = _cache(symbol)
    if p.exists():
        df = pd.read_csv(p, index_col=0)
        df.index = pd.to_datetime(df.index, errors="coerce")
        if not df.empty and df.index[-1].date().isoformat() >= end:
            return df[df.index <= end]
    df = yf.download(symbol, start=start, end=end, interval="1d", auto_adjust=False, progress=False)
    if df.empty:
        raise RuntimeError(f"Sem dados para {symbol}")
    df.to_csv(p)
    return df

# ─────────────── Engenharia de Atributos ───────────────
def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constrói o dataframe de features, incluindo indicadores técnicos e o retorno alvo.
    """
    df = df.rename(columns=str.title)
    df_feat = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    # Médias móveis
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
    # Retorno alvo (target)
    df_feat["Return"] = df_feat["Close"] / df_feat["Open"] - 1
    df_feat = df_feat.dropna()
    return df_feat

def fetch_recent_features(symbol: str, seq_len: int) -> pd.DataFrame:
    """
    Busca as features mais recentes para previsão (sem coluna alvo).
    """
    df = build_feature_frame(download_prices(symbol))
    return df.iloc[-seq_len:].drop(columns=["Return"])

# ─────────────── Sequenciamento e Escalonamento ───────────────
def make_sequences(
    df: pd.DataFrame, seq_len: int
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    Constrói sequências para treinamento de LSTM e aplica _scaling_.
    Retorna as sequências de entrada, alvo e os _scalers_ usados.
    """
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

# ─────────────── Modelo LSTM ───────────────
def build_model(
    seq_len: int,
    n_features: int,
    lstm_units: int = 64,
    dropout: float = 0.2,
    lr: float = 1e-3
) -> Sequential:
    """
    Constrói e compila o modelo LSTM para previsão.
    """
    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(lstm_units, return_sequences=True),
        Dropout(dropout),
        LSTM(lstm_units // 2),
        Dropout(dropout),
        Dense(1, activation="tanh"),
    ])
    model.compile(optimizer=Adam(lr), loss="mse")
    return model

# ─────────────── Treinamento e Hyper-Tuning ───────────────
def train_and_save(
    symbol: str,
    seq_len: int = 180,
    epochs: int = 30,
    start: str = "2005-01-01",
    trials: int = 10
):
    """
    Realiza o treinamento do modelo LSTM, incluindo _hyper-tuning_ com Keras-Tuner,
    e salva o modelo e escalers.
    Também gera logs para visualização no TensorBoard.
    """
    df = build_feature_frame(download_prices(symbol, start=start))
    X, y, scaler_x, scaler_y = make_sequences(df, seq_len)
    n_features = X.shape[2]

    def model_builder(hp):
        """
        Builder para o Keras-Tuner, parametrizando unidades, dropout e learning rate.
        """
        units = hp.Choice("units", [32, 64, 128])
        dropout = hp.Choice("dropout", [0.1, 0.2, 0.3])
        lr = hp.Choice("lr", [1e-3, 5e-4, 1e-4])
        return build_model(seq_len, n_features, units, dropout, lr)

    # Busca hiperparâmetros ideais com RandomSearch
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

    # Adiciona callback para TensorBoard (curvas de loss, etc.)
    log_dir = os.path.join("logs", f"{symbol}_{pd.Timestamp.now():%Y%m%d_%H%M%S}")
    tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)
    es = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(
        X, y,
        epochs=epochs,
        batch_size=32,
        validation_split=0.2,
        callbacks=[es, tensorboard_cb],
        verbose=2
    )

    # Salva modelo e escalers
    model.save(MODELS / f"lstm_{symbol}.keras")
    joblib.dump({"x": scaler_x, "y": scaler_y}, MODELS / f"scalers_{symbol}.pkl")

def load_model_and_scalers(symbol: str):
    """
    Carrega modelo treinado e escalers a partir do disco.
    """
    return (
        load_model(MODELS / f"lstm_{symbol}.keras"),
        joblib.load(MODELS / f"scalers_{symbol}.pkl"),
    )