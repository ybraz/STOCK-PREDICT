#!/usr/bin/env python3
"""
Script de linha de comando para treinar e salvar um modelo LSTM otimizado
para previsão de preços de ativos, utilizando hyper-tuning com Keras-Tuner.

Uso:
    python train.py SYMBOL [--seq 180] [--epochs 50] [--start 2005-01-01] [--trials 10]

Exemplo:
    python train.py PETR4.SA --seq 180 --epochs 50 --start 2010-01-01 --trials 20
"""

import argparse
from src.utils.data_utils import train_and_save

def main():
    """
    Faz o parsing dos argumentos de linha de comando e executa o treinamento do modelo,
    salvando os resultados na pasta de modelos.
    """
    parser = argparse.ArgumentParser(
        description="Treina e salva um modelo LSTM para previsão de retornos diários de um ativo financeiro."
    )
    parser.add_argument(
        "symbol",
        type=str,
        help="Ticker do ativo (exemplo: PETR4.SA)"
    )
    parser.add_argument(
        "--seq",
        type=int,
        default=180,
        help="Tamanho da janela de sequência de entrada para o LSTM (default: 180 dias)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Número de épocas para o treinamento (default: 50)"
    )
    parser.add_argument(
        "--start",
        type=str,
        default="2005-01-01",
        help="Data inicial para download dos dados históricos (formato: AAAA-MM-DD, default: 2005-01-01)"
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Número de tentativas para o Keras-Tuner hyper-tuning (default: 10)"
    )
    args = parser.parse_args()

    # Treinamento e salvamento do modelo
    train_and_save(
        symbol=args.symbol,
        seq_len=args.seq,
        epochs=args.epochs,
        start=args.start,
        trials=args.trials
    )
    print("Modelo treinado e salvo com hiperparâmetros otimizados.")

if __name__ == "__main__":
    main()