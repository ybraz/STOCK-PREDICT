"""Script de treino do modelo LSTM, agora profissionalizado."""
from __future__ import annotations
import argparse
import os
import numpy as np
import joblib
from dotenv import load_dotenv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from src.utils.data_utils import download_prices, create_sequences, train_val_test_split
from sklearn.preprocessing import MinMaxScaler

load_dotenv()

DEFAULTS = dict(
    symbol=os.getenv("SYMBOL", "DIS"),
    start_date=os.getenv("START_DATE", "2010-01-01"),
    end_date=os.getenv("END_DATE", "2025-01-01"),
    seq_length=int(os.getenv("SEQ_LENGTH", 180)),
    test_size=float(os.getenv("TEST_SIZE", 0.2)),
    val_size=float(os.getenv("VAL_SIZE", 0.1)),
    epochs=int(os.getenv("EPOCHS", 200)),
    batch_size=int(os.getenv("BATCH_SIZE", 32)),
)

def build_model(input_shape: tuple[int, int]) -> Sequential:
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["mae"])
    return model

def main(**kwargs):
    params = {**DEFAULTS, **{k: v for k, v in kwargs.items() if v is not None}}
    print("Parâmetros:", params)

    df = download_prices(params["symbol"], params["start_date"], params["end_date"])

    # Vamos usar Close, Volume, High, Low
    features = df[["Close", "Volume", "High", "Low"]].dropna().values

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)

    train, val, test = train_val_test_split(scaled_features, params["val_size"], params["test_size"])

    X_train, y_train = create_sequences(train, params["seq_length"])
    X_val, y_val = create_sequences(np.concatenate([train[-params["seq_length"]:], val]), params["seq_length"])

    model = build_model((params["seq_length"], X_train.shape[2]))

    os.makedirs("models", exist_ok=True)
    model_checkpoint_path = f"models/lstm_{params['symbol']}.h5"
    scaler_path = f"models/scaler_{params['symbol']}.pkl"

    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
        ModelCheckpoint(model_checkpoint_path, save_best_only=True),
    ]

    model.fit(
        X_train,
        y_train[:, 0],  # vamos prever a primeira feature: Close
        validation_data=(X_val, y_val[:, 0]),
        epochs=params["epochs"],
        batch_size=params["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )

    joblib.dump(scaler, scaler_path)
    print(f"Treino concluído. Modelo salvo em {model_checkpoint_path}, scaler salvo em {scaler_path}.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        arg_type = type(v)
        parser.add_argument(f"--{k}", type=arg_type, default=None)
    args = vars(parser.parse_args())
    main(**args)
