#!/usr/bin/env python3
"""
API de previsão de retorno diário de ativos, utilizando features técnicos e OHLCV.
Endpoints:
    - POST /predict: Faz previsão de retorno diário e preço futuro.
    - GET  /metrics: Exibe métricas Prometheus (latência, total de requisições).
    - POST /train: Treina o modelo LSTM com os dados mais recentes.

Dependências:
    - FastAPI, yfinance, prometheus_client, numpy, pytz
    - src.utils.data_utils (funções auxiliares e carregamento do modelo)
    - api.schemas (Pydantic para request e response)
"""

from datetime import datetime, timedelta, time as dtime
from typing import Tuple, Optional
import numpy as np
import pytz
import yfinance as yf
from fastapi import FastAPI, HTTPException
from prometheus_client import Summary, Counter, make_asgi_app

from api.schemas import PredictRequest, PredictResponse
from src.utils.data_utils import fetch_recent_features, load_model_and_scalers

DEFAULT_SEQ = 180  # Tamanho padrão da sequência de entrada LSTM
app = FastAPI(title="Tech-Return Predictor")
LAT = Summary("prediction_latency_seconds", "Latency")
REQ = Counter("requests_total", "Total requests")

def _market_hours(symbol: str) -> Tuple[pytz.BaseTzInfo, dtime, dtime]:
    """
    Retorna o timezone e horários de pregão do ativo informado.
    """
    if symbol.upper().endswith(".SA"):
        # B3 (Brasil)
        return pytz.timezone("America/Sao_Paulo"), dtime(10, 0), dtime(17, 0)
    # Default: NYSE/NASDAQ (EUA)
    return pytz.timezone("America/New_York"), dtime(9, 30), dtime(16, 0)

def _today_open(symbol: str, tz: pytz.BaseTzInfo) -> Optional[float]:
    """
    Busca o preço de abertura do dia corrente para o ativo no fuso horário do mercado.
    """
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
    """
    Realiza a previsão do próximo preço de fechamento e retorno percentual esperado.

    Requisição: PredictRequest (symbol, seq_length, open_price)
    Resposta:   PredictResponse (next_price, expected_return_pct)

    Regras de validação:
      - Apenas durante o horário de pregão do ativo (de acordo com o timezone e o dia da semana)
      - Preço de abertura deve estar disponível
    """
    REQ.inc()
    tz, h_open, h_close = _market_hours(r.symbol)
    now = datetime.now(tz)
    if now.weekday() >= 5:  # 5=sábado, 6=domingo
        raise HTTPException(400, "Mercado fechado (fim de semana).")
    if not (h_open <= now.time() < h_close):
        raise HTTPException(400, "Fora do horário de pregão do ativo.")

    open_price = _today_open(r.symbol, tz)
    if open_price is None:
        raise HTTPException(400, "Preço de abertura indisponível para o ativo.")

    # Engenharia de features para previsão
    X_df = fetch_recent_features(r.symbol, r.seq_length or DEFAULT_SEQ)
    model, scalers = load_model_and_scalers(r.symbol)
    X_scaled = scalers["x"].transform(X_df.values).reshape(1, -1, X_df.shape[1])

    # Previsão do retorno
    ret_scaled = model.predict(X_scaled, verbose=0)[0]
    ret = scalers["y"].inverse_transform(ret_scaled.reshape(-1, 1))[0][0]

    # Monta resposta
    return PredictResponse(
        next_price=round(open_price * (1 + ret), 4),
        expected_return_pct=round(ret * 100, 2),  # valor percentual
    )

@app.post("/train", response_model=TrainResponse)
def train(r: TrainRequest):
    """
    Treina ou atualiza o modelo para o ativo fornecido.
    """
    try:
        # Treinar o modelo
        model_path, scaler_x_path, scaler_y_path = train_model(r.symbol, r.start_date, r.end_date)

        # Retornar os caminhos dos arquivos salvos
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

# Exposição de métricas para Prometheus
app.mount("/metrics", make_asgi_app())