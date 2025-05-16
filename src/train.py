#!/usr/bin/env python3
import argparse
from src.utils.data_utils import train_and_save

p = argparse.ArgumentParser()
p.add_argument("symbol")
p.add_argument("--seq", type=int, default=180)
p.add_argument("--epochs", type=int, default=50)
p.add_argument("--start", default="2005-01-01")
p.add_argument("--trials", type=int, default=10, help="Keras-Tuner trials")
a = p.parse_args()

train_and_save(a.symbol, seq_len=a.seq, epochs=a.epochs,
               start=a.start, trials=a.trials)
print("Modelo treinado e salvo com hiperparâmetros otimizados.")