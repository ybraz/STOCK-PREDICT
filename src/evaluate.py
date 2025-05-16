#!/usr/bin/env python3
"""
Avaliação da performance de um modelo LSTM para previsão de retorno diário de ativos,
usando as métricas MAE, RMSE, MAPE e SMAPE sobre o retorno (Close / Open − 1).

Uso:
    PYTHONPATH=. python src/evaluate.py AAPL --seq 180 --split 0.8

Autor: [Seu Nome]
Data: [Data Atual]
"""

import argparse
from typing import Tuple
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.utils.data_utils import (
    download_prices,
    build_feature_frame,
    make_sequences,
    load_model_and_scalers,
)

def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Calcula o MAPE (Mean Absolute Percentage Error), robusto contra divisão por zero.
    """
    den = np.where(np.abs(y_true) < epsilon, epsilon, y_true)
    return np.mean(np.abs((y_true - y_pred) / den)) * 100

def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Calcula o SMAPE (Symmetric Mean Absolute Percentage Error).
    """
    return np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon)) * 100

def evaluate_model(
    symbol: str,
    seq_len: int = 180,
    split: float = 0.8
) -> Tuple[float, float, float, float]:
    """
    Avalia o modelo LSTM salvo para um determinado ativo, retornando as métricas MAE, RMSE, MAPE e SMAPE.
    
    Args:
        symbol (str): Ticker do ativo a ser avaliado.
        seq_len (int): Comprimento da janela LSTM.
        split (float): Fração da base usada para treino (o restante é para teste).
    
    Returns:
        Tuple[float, float, float, float]: MAE, RMSE, MAPE, SMAPE.
    """
    # Monta base de teste
    df_full = build_feature_frame(download_prices(symbol))
    split_idx = int(len(df_full) * split)
    test_df = df_full.iloc[split_idx - seq_len :]  # inclui janela de contexto para LSTM

    X_test, y_test_scaled, _, _ = make_sequences(test_df, seq_len)
    model, scalers = load_model_and_scalers(symbol)

    # Previsão e inversão de escala
    pred_scaled = model.predict(X_test, verbose=0)
    y_test = scalers["y"].inverse_transform(y_test_scaled)
    pred = scalers["y"].inverse_transform(pred_scaled)

    # Métricas
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    mape = calculate_mape(y_test, pred)
    smape = calculate_smape(y_test, pred)

    return mae, rmse, mape, smape

def main():
    """
    Ponto de entrada principal para parsing dos argumentos e avaliação do modelo.
    """
    parser = argparse.ArgumentParser(
        description="Avalia um modelo LSTM salvo para previsão de retorno de ativos financeiros."
    )
    parser.add_argument(
        "symbol", type=str, help="Ticker do ativo (ex: AAPL, PETR4.SA)"
    )
    parser.add_argument(
        "--seq", type=int, default=180, help="Comprimento da janela LSTM (default: 180)"
    )
    parser.add_argument(
        "--split", type=float, default=0.8, help="Fração da base de treino (default: 0.8)"
    )
    args = parser.parse_args()

    # Avaliação
    mae, rmse, mape, smape = evaluate_model(args.symbol, args.seq, args.split)

    # Relatório
    print(f"===== Avaliação {args.symbol} =====")
    print(f"MAE   : {mae:.6f}  (~{mae*100:.2f} p.p.)")
    print(f"RMSE  : {rmse:.6f}  (~{rmse*100:.2f} p.p.)")
    print(f"MAPE  : {mape:.2f} %")
    print(f"SMAPE : {smape:.2f} %")

if __name__ == "__main__":
    main()