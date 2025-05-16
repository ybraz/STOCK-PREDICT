#!/usr/bin/env python3
"""
API - prevê retorno diário com features técnicos + OHLCV.
"""
from datetime import datetime, timedelta, time as dtime
import numpy as np
import pytz
import yfinance as yf
from fastapi import FastAPI, HTTPException
from prometheus_client import Summary, Counter, make_asgi_app

from api.schemas import PredictRequest, PredictResponse
from src.utils.data_utils import fetch_recent_features, load_model_and_scalers

DEFAULT_SEQ = 180
app = FastAPI(title="Tech-Return Predictor")
LAT = Summary("prediction_latency_seconds", "Latency")
REQ = Counter("requests_total", "Total requests")


def _market_hours(sym: str):
    if sym.upper().endswith(".SA"):
        return pytz.timezone("America/Sao_Paulo"), dtime(10, 0), dtime(17, 0)
    return pytz.timezone("America/New_York"), dtime(9, 30), dtime(16, 0)


def _today_open(symbol: str, tz):
    today = datetime.now(tz).date()
    df = yf.download(symbol, start=today, end=today + timedelta(days=1),
                     interval="1d", progress=False)
    return None if df.empty else float(df["Open"].iloc[0])


@app.post("/predict", response_model=PredictResponse)
@LAT.time()
def predict(r: PredictRequest):
    REQ.inc()
    tz, h_open, h_close = _market_hours(r.symbol)
    now = datetime.now(tz)
    if now.weekday() >= 5:
        raise HTTPException(400, "Mercado fechado.")
    if not (h_open <= now.time() < h_close):
        raise HTTPException(400, "Fora do horário de pregão.")

    open_price = _today_open(r.symbol, tz)
    if open_price is None:
        raise HTTPException(400, "Preço de abertura indisponível.")

    X_df = fetch_recent_features(r.symbol, r.seq_length or DEFAULT_SEQ)
    model, scalers = load_model_and_scalers(r.symbol)
    X_scaled = scalers["x"].transform(X_df.values).reshape(1, -1, X_df.shape[1])
    ret_scaled = model.predict(X_scaled, verbose=0)[0]
    ret = scalers["y"].inverse_transform(ret_scaled.reshape(-1, 1))[0][0]
    return PredictResponse(
        next_price=round(open_price * (1 + ret), 4),
        expected_return_pct=round(ret * 100, 2),   # agora em %
    )


app.mount("/metrics", make_asgi_app())