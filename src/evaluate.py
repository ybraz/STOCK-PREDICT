"""Avalia o modelo salvo nas partições de validação e teste."""
from __future__ import annotations
import argparse
import os
import numpy as np
import joblib
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from utils.data_utils import download_prices, create_sequences, train_val_test_split

load_dotenv()

DEFAULTS = dict(
    symbol=os.getenv("SYMBOL", "DIS"),
    start_date=os.getenv("START_DATE", "2010-01-01"),
    end_date=os.getenv("END_DATE", "2025-01-01"),
    seq_length=int(os.getenv("SEQ_LENGTH", 180)),
    test_size=float(os.getenv("TEST_SIZE", 0.2)),
    val_size=float(os.getenv("VAL_SIZE", 0.1)),
)

def main(**kwargs):
    params = {**DEFAULTS, **{k: v for k, v in kwargs.items() if v is not None}}

    df = download_prices(params["symbol"], params["start_date"], params["end_date"])
    features = df[["Close", "Volume", "High", "Low"]].dropna().values

    scaler = joblib.load(f"models/scaler_{params['symbol']}.pkl")
    scaled_features = scaler.transform(features)

    train, val, test = train_val_test_split(scaled_features, params["val_size"], params["test_size"])
    X_test, y_test = create_sequences(np.concatenate([train[-params["seq_length"]:], val, test]), params["seq_length"])

    model = load_model(f"models/lstm_{params['symbol']}.h5")
    preds_scaled = model.predict(X_test)

    # Só queremos avaliar a feature "Close" (posição 0)
    preds = scaler.inverse_transform(np.hstack([preds_scaled, np.zeros((preds_scaled.shape[0], 3))]))[:, 0]
    y_true = scaler.inverse_transform(np.hstack([y_test[:, 0].reshape(-1, 1), np.zeros((y_test.shape[0], 3))]))[:, 0]

    mae = mean_absolute_error(y_true, preds)
    rmse = mean_squared_error(y_true, preds, squared=False)
    mape = np.mean(np.abs((y_true - preds) / y_true)) * 100

    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        arg_type = type(v)
        parser.add_argument(f"--{k}", type=arg_type, default=None)
    args = vars(parser.parse_args())
    main(**args)
