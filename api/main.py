from fastapi import FastAPI, HTTPException
from prometheus_client import Summary, make_asgi_app, Counter
import joblib
import numpy as np
from tensorflow.keras.models import load_model
from .schemas import PricesRequest, PredictionResponse
from src.utils.data_utils import download_prices
import os

REQUEST_TIME = Summary("request_processing_seconds", "Tempo de processamento das previsões")
TOTAL_REQUESTS = Counter("total_prediction_requests", "Número total de previsões")

app = FastAPI(title="Stock LSTM Predictor", version="3.0.0")
app.mount("/metrics", make_asgi_app())

def load_model_and_scaler(symbol: str):
    model_path = f"models/lstm_{symbol}.h5"
    scaler_path = f"models/scaler_{symbol}.pkl"

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise HTTPException(status_code=404, detail=f"Modelo ou scaler para {symbol} não encontrados. Treine primeiro.")

    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

@app.post("/predict", response_model=PredictionResponse)
@REQUEST_TIME.time()
def predict(req: PricesRequest):
    TOTAL_REQUESTS.inc()

    try:
        model, scaler = load_model_and_scaler(req.symbol)
        df = download_prices(req.symbol, "2018-01-01", "2025-01-01")
        close_prices = df["Close"].dropna().values
    except Exception:
        raise HTTPException(status_code=400, detail="Erro ao buscar dados ou carregar modelo.")

    if len(close_prices) < req.seq_length:
        raise HTTPException(status_code=400, detail="Dados insuficientes para previsão.")

    last_prices = close_prices[-req.seq_length:]
    last_prices = last_prices.reshape(-1, 1)

    scaled = scaler.transform(last_prices)
    X_input = np.expand_dims(scaled, axis=0)

    pred_scaled = model.predict(X_input)[0][0]
    next_price = float(scaler.inverse_transform([[pred_scaled]])[0][0])

    return PredictionResponse(next_price=next_price)

@app.get("/health")
def health():
    return {"status": "ok"}
