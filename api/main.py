#!/usr/bin/env python3
"""
API de previsão de retorno diário de ativos, utilizando features técnicos e OHLCV.
Endpoints:
    - POST /predict: Faz previsão de retorno diário e preço futuro.
    - POST /train: Treina o modelo LSTM com os dados mais recentes.
    - GET  /metrics: Exibe métricas Prometheus (latência, total de requisições).
"""

from datetime import datetime, timedelta, time as dtime
from typing import Tuple, Optional
import numpy as np
import pytz
import yfinance as yf
from fastapi import FastAPI, HTTPException
from prometheus_client import Summary, Counter
from prometheus_fastapi_instrumentator import Instrumentator

from api.schemas import PredictRequest, PredictResponse, TrainRequest, TrainResponse
from src.utils.data_utils import fetch_recent_features, load_model_and_scalers

DEFAULT_SEQ = 180  # Tamanho padrão da sequência de entrada LSTM
app = FastAPI(title="Tech-Return Predictor")
LAT = Summary("prediction_latency_seconds", "Latency")
REQ = Counter("requests_total", "Total requests")

# Inicializa e anexa o instrumentador Prometheus (já registra métricas padrão)
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)


def _market_hours(symbol: str) -> Tuple[pytz.BaseTzInfo, dtime, dtime]:
    if symbol.upper().endswith(".SA"):
        return pytz.timezone("America/Sao_Paulo"), dtime(10, 0), dtime(17, 0)
    return pytz.timezone("America/New_York"), dtime(9, 30), dtime(16, 0)


def _today_open(symbol: str, tz: pytz.BaseTzInfo) -> Optional[float]:
    today = datetime.now(tz).date()
    df = yf.download(
        symbol,
        start=today,
        end=today + timedelta(days=1),
        interval="1d",
        progress=False
    )
    if df.empty:
        return None
    return float(df["Open"].iloc[0])


@app.post("/predict", response_model=PredictResponse)
@LAT.time()
def predict(r: PredictRequest):
    REQ.inc()
    tz, h_open, h_close = _market_hours(r.symbol)
    now = datetime.now(tz)
    if now.weekday() >= 5:
        raise HTTPException(400, "Mercado fechado (fim de semana).")
    if not (h_open <= now.time() < h_close):
        raise HTTPException(400, "Fora do horário de pregão do ativo.")

    open_price = _today_open(r.symbol, tz)
    if open_price is None:
        raise HTTPException(400, "Preço de abertura indisponível para o ativo.")

    X_df = fetch_recent_features(r.symbol, r.seq_length or DEFAULT_SEQ)
    model, scalers = load_model_and_scalers(r.symbol)
    X_scaled = scalers["x"].transform(X_df.values).reshape(1, -1, X_df.shape[1])

    ret_scaled = model.predict(X_scaled, verbose=0)[0]
    ret = scalers["y"].inverse_transform(ret_scaled.reshape(-1, 1))[0][0]

    return PredictResponse(
        next_price=round(open_price * (1 + ret), 4),
        expected_return_pct=round(ret * 100, 2),
    )


@app.post("/train", response_model=TrainResponse)
def train(r: TrainRequest):
    try:
        model_path, scaler_x_path, scaler_y_path = train_model(r.symbol, r.start_date, r.end_date)
        return TrainResponse(
            message=f"Modelo treinado com sucesso para o ativo {r.symbol}.",
            model_path=model_path,
            scaler_x_path=scaler_x_path,
            scaler_y_path=scaler_y_path,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, f"Erro interno: {str(e)}")
