#!/usr/bin/env python3
"""
Avalia a qualidade do modelo em MAE, RMSE, MAPE e SMAPE
sobre o retorno diário (Close / Open − 1).

Uso:
    PYTHONPATH=. python src/evaluate.py AAPL --seq 180 --split 0.8
"""
import argparse
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.utils.data_utils import (
    download_prices,
    build_feature_frame,
    make_sequences,
    load_model_and_scalers,
)

# ────────────────────────────────
# Argumentos de CLI
# ────────────────────────────────
p = argparse.ArgumentParser()
p.add_argument("symbol")
p.add_argument("--seq", type=int, default=180, help="Comprimento da janela LSTM")
p.add_argument("--split", type=float, default=0.8, help="Fraç. treino (0–1)")
args = p.parse_args()

# ────────────────────────────────
# Monta base de teste
# ────────────────────────────────
df_full = build_feature_frame(download_prices(args.symbol))
split_idx = int(len(df_full) * args.split)
test_df = df_full.iloc[split_idx - args.seq :]

X_test, y_test_scaled, _, _ = make_sequences(test_df, args.seq)
model, scalers = load_model_and_scalers(args.symbol)

# ────────────────────────────────
# Previsão e inversão de escala
# ────────────────────────────────
pred_scaled = model.predict(X_test, verbose=0)
y_test = scalers["y"].inverse_transform(y_test_scaled)
pred = scalers["y"].inverse_transform(pred_scaled)

# ────────────────────────────────
# Métricas
# ────────────────────────────────
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

# MAPE robusto: evita divisão por zero
epsilon = 1e-8
den = np.where(np.abs(y_test) < epsilon, epsilon, y_test)
mape = np.mean(np.abs((y_test - pred) / den)) * 100

# SMAPE (opcional, mas útil)
smape = (
    np.mean(np.abs(y_test - pred) / (np.abs(y_test) + np.abs(pred) + epsilon))
    * 100
)

# ────────────────────────────────
# Relatório
# ────────────────────────────────
print(f"===== Avaliação {args.symbol} =====")
print(f"MAE   : {mae:.6f}  (~{mae*100:.2f} p.p.)")
print(f"RMSE  : {rmse:.6f}  (~{rmse*100:.2f} p.p.)")
print(f"MAPE  : {mape:.2f} %")
print(f"SMAPE : {smape:.2f} %")